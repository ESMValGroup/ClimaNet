from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ray import tune
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from climanet.dataset import DataLoaderConfig, DatasetConfig, STDataset
from climanet.predict import PredictionConfig, predict_monthly_var
from climanet.utils import (
    compute_masked_loss,
    data_preparation,
    save_model,
    setup_logging,
)


@dataclass
class TrainConfig:
    """Configuration for training the model."""

    calculate_residuals: bool = True
    num_epoch: int = 100
    patience: int = 10
    accumulation_steps: int = 1
    optimizer_lr: float = 1e-3
    device: str = "cpu"
    verbose: bool = False
    verbose_epoch_interval: int = 20
    tune_checkpoint: bool = False
    store_model: bool = True


def _move_batch_to_device(batch: dict, device: str):
    use_cuda = device == "cuda"
    return {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}


def _run_one_batch(model: torch.nn.Module, batch: dict, accumulation_steps, device):
    batch = _move_batch_to_device(batch, device)
    pred = model(
        batch["daily_patch"],
        batch["daily_mask_patch"],
        batch["daily_timef_patch"],
        batch["land_mask_patch"],
        batch["geo_pos_embedding_patch"],
        batch["scale_feature_patch"],
        batch["padded_days_mask"],
    )  # (B, M, H, W)

    # Compute masked loss
    loss = compute_masked_loss(pred, batch["monthly_patch"], batch["land_mask_patch"])
    scaled_loss = loss * (1.0 / accumulation_steps)
    scaled_loss.backward()

    return loss


def _train_data_preparation(
        input_data,
        monthly_data,
        land_mask,
        calculate_residuals=True,
        is_hourly=False,
        dataset_patch_size=(1, 16, 16),
        dataset_stride=None,
):
    # prepare data
    (input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features) = (
        data_preparation(
            input_data,
            monthly_data,
            calculate_residuals=calculate_residuals,
            is_hourly=is_hourly,
        )
    )

    return STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        monthly_da=monthly_da,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        land_mask=land_mask,
        patch_size=dataset_patch_size,
        stride=dataset_stride,
    )


def _train_one_year(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_data_year,
    monthly_data_year,
    land_mask,
    accumulation_counter,
    calculate_residuals=True,
    is_hourly=False,
    device="cpu",
    dataset_patch_size=(1, 16, 16),
    dataset_stride=None,
    accumulation_steps=1,
    dataloader_batch_size=32,
    dataloader_shuffle=True,
    dataloader_num_workers=0,
):
    dataset = _train_data_preparation(
        input_data_year,
        monthly_data_year,
        land_mask,
        calculate_residuals=calculate_residuals,
        is_hourly=is_hourly,
        dataset_patch_size=dataset_patch_size,
        dataset_stride=dataset_stride,
    )

    use_cuda = device == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        shuffle=dataloader_shuffle,
        pin_memory=use_cuda,
        num_workers=dataloader_num_workers,  # for data loading
        persistent_workers=False,
    )

    total_loss = 0.0
    num_batches_year = len(dataloader)

    for batch in dataloader:
        loss = _run_one_batch(model, batch, accumulation_steps, device)

        # Track unscaled loss for logging
        total_loss += loss.detach()
        accumulation_counter += 1

        if accumulation_counter == accumulation_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0

    del dataloader
    del dataset
    return total_loss, num_batches_year, accumulation_counter


def _run_validation(
    model,
    input_data_validation,
    monthly_data_validation,
    land_mask,
    dataset_config,
    dataloader_config,
    training_config,
    run_dir,
):
    prediction_config = PredictionConfig(
        calculate_residuals=training_config.calculate_residuals,
        device=training_config.device,
        save_predictions=False,
        return_loss=True,
        return_numpy=False,
        verbose=False,
    )
    # Store train loss for gap calculation
    avg_val_loss = predict_monthly_var(
        model=model,
        input_data=input_data_validation,
        monthly_data=monthly_data_validation,
        land_mask=land_mask,
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        prediction_config=prediction_config,
        run_dir=run_dir,
    )

    return avg_val_loss


def _load_checkpoint(model, optimizer):
    loaded_checkpoint = tune.get_checkpoint()
    with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
        loaded_checkpoint_dir = Path(loaded_checkpoint_dir).resolve()
        checkpoint = torch.load(loaded_checkpoint_dir / "checkpoint.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def train_monthly_model(
    model: torch.nn.Module,
    input_data_train,
    monthly_data_train,
    dataloader_config: DataLoaderConfig,
    dataset_config: DatasetConfig,
    training_config: TrainConfig,
    input_data_validation=None,
    monthly_data_validation=None,
    land_mask=None,
    run_dir: str = ".",
):
    """Train the model to predict monthly data from daily data.
    Args:
        model: the PyTorch model to train
        input_data_train: xarray Dataset containing daily/hourly data
        monthly_data_train: xarray Dataset containing monthly data
        dataloader_config: configuration for the DataLoader
        dataset_config: configuration for the Dataset
        training_config: configuration for the training process
        input_data_validation: xarray Dataset containing daily/hourly validation data (optional)
        monthly_data_validation: xarray Dataset containing monthly validation data (optional)
        land_mask: xarray Dataarray containing land mask (optional)
        run_dir: directory to save logs and model checkpoints
    Returns:
        The trained model.
    """
    # Initialize the model
    model = model.to(training_config.device)

    # Set up logging
    writer = setup_logging(run_dir)

    # Set the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.optimizer_lr, weight_decay=1e-2
    )
    best_loss = float("inf")
    counter = 0
    best_state_dict = None  # Store best model state

    if tune.get_checkpoint():
        _load_checkpoint(model, optimizer)

    # Add scheduler - reduces LR instead of stopping immediately
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=training_config.patience // 2,  # Reduce LR before early stop triggers
        min_lr=1e-7,
    )

    # get years from training data
    years = np.unique(monthly_data_train.time.dt.year)

    model.train()
    for epoch in range(training_config.num_epoch):
        epoch_loss = 0.0
        total_num_batches = 0
        accumulation_counter = 0

        optimizer.zero_grad()

        for year in years:
            input_data_year = input_data_train[dataset_config.var_name].sel(
                time=str(year)
            )
            monthly_data_year = monthly_data_train[dataset_config.var_name].sel(
                time=str(year)
            )

            loss_year, num_batches_year, accumulation_counter = _train_one_year(
                model=model,
                optimizer=optimizer,
                input_data_year=input_data_year,
                monthly_data_year=monthly_data_year,
                land_mask=land_mask,
                accumulation_counter=accumulation_counter,
                calculate_residuals=training_config.calculate_residuals,
                is_hourly=dataset_config.is_hourly,
                device=training_config.device,
                dataset_patch_size=dataset_config.patch_size,
                dataset_stride=dataset_config.stride,
                accumulation_steps=training_config.accumulation_steps,
                dataloader_batch_size=dataloader_config.batch_size,
                dataloader_shuffle=dataloader_config.shuffle,
                dataloader_num_workers=dataloader_config.num_workers,
            )
            epoch_loss += loss_year
            total_num_batches += num_batches_year

        # Flush remaining accumulated gradients after ALL years
        if accumulation_counter != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0

        # Calculate average epoch loss
        avg_train_loss = epoch_loss.item() / total_num_batches
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        avg_epoch_loss = avg_train_loss  # Initially use training loss

        # Validation loss (optional)
        if input_data_validation is not None:
            avg_val_loss = _run_validation(
                model,
                input_data_validation,
                monthly_data_validation,
                land_mask,
                dataset_config,
                dataloader_config,
                training_config,
                run_dir,
            )
            avg_epoch_loss = avg_val_loss

            writer.add_scalar("Loss/validation", avg_val_loss, epoch)

            if (
                training_config.verbose
                and epoch % training_config.verbose_epoch_interval == 0
            ):
                gap = avg_val_loss - avg_train_loss
                print(f"Epoch {epoch}: gap between train and val loss: {gap:.6f}")

        # Step scheduler
        scheduler.step(avg_epoch_loss)

        # Early stopping check
        # Consider improvement only if loss decreases more than a small threshold
        if avg_epoch_loss < best_loss - 1e-4:
            best_loss = avg_epoch_loss
            best_state_dict = {k: v.detach() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        # Log to TensorBoard
        writer.add_scalar("Loss/best", best_loss, epoch)

        if (
            training_config.verbose
            and epoch % training_config.verbose_epoch_interval == 0
        ):
            print(f"Epoch {epoch}: best_loss = {best_loss:.6f}")

        # Only stop if LR is at minimum AND no improvement
        current_lr = optimizer.param_groups[0]["lr"]
        if counter >= training_config.patience and current_lr <= scheduler.min_lrs[0]:
            writer.add_text("Training", f"Early stop at epoch {epoch}", epoch)
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if training_config.tune_checkpoint:
        # Save the model and optimizer state for Ray Tune checkpointing
        save_model(model, optimizer, run_dir, filename="checkpoint.pt", verbose=False)
        tune.report(
            {"loss": best_loss}, checkpoint=tune.Checkpoint.from_directory(run_dir)
        )

    # Close the writer when done
    writer.close()

    if training_config.verbose:
        print(f"Training complete. Best loss: {best_loss:.6f}")

    if training_config.store_model:
        save_model(model, optimizer, run_dir, verbose=training_config.verbose)

    return model
