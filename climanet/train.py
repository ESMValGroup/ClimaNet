import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from ray import tune
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from climanet.dataset import DataLoaderConfig
from climanet.predict import PredictionConfig, predict_monthly_var
from climanet.utils import (
    compute_masked_loss,
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
    store_logs: bool = True


def _move_batch_to_device(batch: dict, device: str):
    use_cuda = device == "cuda"
    return {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}


def _run_one_batch(model: torch.nn.Module, batch: dict, device):
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
    return compute_masked_loss(pred, batch["monthly_patch"], batch["land_mask_patch"])


def _load_checkpoint(model, optimizer, loaded_checkpoint):
    with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
        loaded_checkpoint_dir = Path(loaded_checkpoint_dir).resolve()
        checkpoint = torch.load(loaded_checkpoint_dir / "checkpoint.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def train_monthly_model(
    model: torch.nn.Module,
    dataset_train,
    dataloader_config: DataLoaderConfig,
    training_config: TrainConfig,
    dataset_validation=None,
    run_dir: str = ".",
):
    """Train the model to predict monthly data from daily data.
    Args:
        model: the PyTorch model to train
        dataset_train: the training dataset
        dataloader_config: configuration for the data loader, see DataLoaderConfig for details.
        training_config: configuration for the training process, see TrainConfig for details.
        dataset_validation: the validation dataset, if provided, will be used to compute validation loss.
        run_dir: directory to save logs and model checkpoints
    Returns:
        The trained model.
    """
    device = training_config.device
    # Initialize the model
    model = model.to(device)

    # Set up logging
    if training_config.store_logs:
        writer = setup_logging(run_dir)

    # Create data loader

    use_cuda = device == "cuda"
    dataloader = DataLoader(
        dataset_train,
        batch_size=dataloader_config.batch_size,
        shuffle=dataloader_config.shuffle,
        pin_memory=use_cuda,
        num_workers=dataloader_config.num_workers,  # for data loading
        persistent_workers=dataloader_config.persistent_workers,  # keep workers alive between epochs
        multiprocessing_context=dataloader_config.multiprocessing_context,
    )
    num_batches = len(dataloader)

    # Set the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.optimizer_lr, weight_decay=1e-2
    )

    best_loss = float("inf")
    counter = 0
    best_state_dict = None  # Store best model state

    if training_config.tune_checkpoint:
        checkpoint = tune.get_checkpoint()
        if checkpoint is not None:
            _load_checkpoint(model, optimizer, checkpoint)

    # Add scheduler - reduces LR instead of stopping immediately
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=training_config.patience // 2,  # Reduce LR before early stop triggers
        min_lr=1e-7,
    )

    model.train()
    for epoch in range(training_config.num_epoch):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            loss = _run_one_batch(model, batch, device)

            # Scale loss for gradient accumulation
            scaled_loss = loss * (1.0 / training_config.accumulation_steps)
            scaled_loss.backward()

            # Track unscaled loss for logging
            epoch_loss += loss.detach()

            # Update weights every accumulation_steps batches
            if (i + 1) % training_config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Handle remaining gradients if num_batches is not divisible by accumulation_steps
        if (i + 1) % training_config.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Calculate average epoch loss
        avg_train_loss = epoch_loss.item() / num_batches
        avg_epoch_loss = avg_train_loss  # Initially use training loss

        if training_config.store_logs:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # Validation loss (optional)
        if dataset_validation is not None:
            prediction_config = PredictionConfig(
                calculate_residuals=training_config.calculate_residuals,
                device=training_config.device,
                save_predictions=False,
                return_loss=True,
                return_numpy=False,
                verbose=False,
                store_logs=False,
            )
            # Store train loss for gap calculation
            avg_val_loss = predict_monthly_var(
                model,
                dataset_validation,
                dataloader_config=dataloader_config,
                prediction_config=prediction_config,
                run_dir=run_dir,
            )
            avg_epoch_loss = avg_val_loss

            if training_config.store_logs:
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
        if training_config.store_logs:
            writer.add_scalar("Loss/best", best_loss, epoch)

        if (
            training_config.verbose
            and epoch % training_config.verbose_epoch_interval == 0
        ):
            print(f"Epoch {epoch}: best_loss = {best_loss:.6f}")

        # Only stop if LR is at minimum AND no improvement
        current_lr = optimizer.param_groups[0]["lr"]
        if counter >= training_config.patience and current_lr <= scheduler.min_lrs[0]:
            if training_config.store_logs:
                writer.add_text("Training", f"Early stop at epoch {epoch}", epoch)
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if training_config.tune_checkpoint:
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)

            # Save the model and optimizer state for Ray Tune checkpointing
            save_model(model, optimizer, checkpoint_path, filename="checkpoint.pt", verbose=False)
            tune.report(
                {"loss": best_loss}, checkpoint=tune.Checkpoint.from_directory(checkpoint_path)
            )

    # Close the writer when done
    if training_config.store_logs:
        writer.close()

    if training_config.verbose:
        print(f"Training complete. Best loss: {best_loss:.6f}")

    if training_config.store_model:
        save_model(model, optimizer, run_dir, verbose=training_config.verbose)

    return model
