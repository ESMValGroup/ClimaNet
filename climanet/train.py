from pathlib import Path

import numpy as np
import torch
import xarray as xr
from ray import tune
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from climanet.dataset import DataLoaderConfig, DatasetConfig, STDataset

from climanet.predict import predict_monthly_var
from climanet.utils import compute_masked_loss, data_preparation, save_model, setup_logging
from dataclasses import dataclass

@dataclass
class TrainConfig:
    calculate_residuals: bool = True
    is_hourly: bool = False
    var_name: str ="tos"
    num_epoch: int = 100
    patience: int = 10
    accumulation_steps: int = 1
    optimizer_lr: float = 1e-3
    device: str = "cpu"
    verbose: bool = True
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


def _train_one_year(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch_loss,
        epoch_num_batches,
        input_data_year,
        monthly_data_year,
        land_mask,
        training_config: TrainConfig,
        dataloader_config: DataLoaderConfig,
        dataset_config: DatasetConfig,
):
    # prepare data
    (
        input_da,
        input_da_nan_mask,
        monthly_da,
        padded_days_mask,
        time_features
    ) = data_preparation(
        input_data_year,
        monthly_data_year,
        calculate_residuals=training_config.calculate_residuals,
        is_hourly=training_config.is_hourly,
    )
    device = training_config.device
    use_cuda = device == "cuda"

    dataset = STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        monthly_da=monthly_da,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        land_mask=land_mask["lsm"],
        patch_size=dataset_config.spatial_patch_size,
        stride=dataset_config.spatial_stride,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config.batch_size,
        shuffle=dataloader_config.shuffle,
        pin_memory=use_cuda,
        num_workers=dataloader_config.num_workers,  # for data loading
        persistent_workers=False,
    )

    year_num_batches = 0
    total_loss = epoch_loss
    total_num_batches = epoch_num_batches

    for i, batch in enumerate(dataloader):
        loss = _run_one_batch(model, batch, training_config.accumulation_steps, device)

        # Track unscaled loss for logging
        total_loss += loss.detach()

        # Update weights every accumulation_steps batches
        if (i + 1) % training_config.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_num_batches += 1
        year_num_batches += 1

    # Handle remaining gradients if num_batches is not divisible by accumulation_steps
    if year_num_batches % training_config.accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    del dataloader
    del dataset
    return total_loss, total_num_batches


def _run_validation(epoch):
    # Store train loss for gap calculation
    _, avg_val_loss = predict_monthly_var(
        model,
        input_data_validation,
        monthly_data_validation,
        land_mask,
        batch_size=batch_size,
        device=device,
        return_numpy=False,
        save_predictions=False,
        return_loss=True,
        verbose=False,
        run_dir=run_dir,
        dataloader_num_workers=dataloader_num_workers,
        var_name=var_name,
        spatial_patch_size=spatial_patch_size,
        stride=stride,
    )
    writer.add_scalar("Loss/validation", avg_val_loss, epoch)

    if training_config.verbose and epoch % training_config.verbose_epoch_interval == 0:
        gap = avg_val_loss - avg_train_loss
        print(f"Epoch {epoch}: gap between train and val loss: {gap:.6f}")

    return avg_val_loss

def train_monthly_model(
    model: torch.nn.Module,
    input_data_train,
    monthly_data_train,
    input_data_validation=None,
    monthly_data_validation=None,
    land_mask=None,
    dataloader_config: DataLoaderConfig = DataLoaderConfig(),
    dataset_config: DatasetConfig = DatasetConfig(),
    training_config: TrainConfig = TrainConfig(),
    run_dir: str = ".",
):
    """Train the model to predict monthly data from daily data.
    Args:
        model: the PyTorch model to train
        dataset: Dataset object containing the training data
        shuffle: whether to shuffle the data each epoch
        batch_size: number of samples per batch
        num_epoch: number of epochs to train
        patience: number of epochs to wait for improvement before early stopping
        accumulation_steps: number of batches to accumulate gradients over before updating weights
        optimizer_lr: learning rate for the optimizer
        run_dir: directory to save logs and model
        store_model: whether to save the best model to disk
        device: device to run training on ("cpu" or "cuda")
        verbose: whether to print training progress
        dataloader_num_workers: how many subprocesses to use for data loading.
            See torch DataLoader docs for details.
        verbose_epoch_interval: how often to print training progress (in epochs)
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
        loaded_checkpoint = tune.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            loaded_checkpoint_dir = Path(loaded_checkpoint_dir).resolve()
            checkpoint = torch.load(loaded_checkpoint_dir / "checkpoint.pt")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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
        num_batches = 0

        optimizer.zero_grad()

        for year in years:
            input_data_year = input_data_train[training_config.var_name].sel(time=str(year))
            monthly_data_year = monthly_data_train[training_config.var_name].sel(time=str(year))

            epoch_loss, num_batches = _train_one_year(
                model=model,
                optimizer=optimizer,
                epoch_loss=epoch_loss,
                epoch_num_batches=num_batches,
                input_data_year=input_data_year,
                monthly_data_year=monthly_data_year,
                land_mask=land_mask,
                training_config=training_config,
                dataloader_config=dataloader_config,
                dataset_config=dataset_config,
            )

        # Calculate average epoch loss
        avg_train_loss = epoch_loss.item() / num_batches
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        avg_epoch_loss = avg_train_loss  # Initially use training loss

        # Validation loss (optional)
        if input_data_validation is not None:
            avg_epoch_loss = _run_validation(epoch)

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

        if training_config.verbose and epoch % training_config.verbose_epoch_interval == 0:
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
