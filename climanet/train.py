import copy
from pathlib import Path
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch


def _setup_logging(log_dir: str) -> SummaryWriter:
    """Set up TensorBoard logging directory and writer."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir)


def _compute_masked_loss(
    pred: torch.Tensor, target: torch.Tensor, land_mask: torch.Tensor
) -> torch.Tensor:
    """Compute L1 loss masked to ocean pixels only."""
    ocean = (~land_mask).to(pred.device).unsqueeze(1).float()
    loss = torch.nn.functional.l1_loss(pred, target, reduction="none") * ocean

    num = loss.sum(dim=(-2, -1))
    denom = ocean.sum(dim=(-2, -1)).clamp_min(1)

    return (num / denom).mean()


def _save_model(model: torch.nn.Module, run_dir: str, verbose: bool) -> None:
    """Save model state and config to disk."""
    model_path = Path(run_dir) / "best_model.pth"
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": model.config},
        model_path,
    )
    if verbose:
        print(f"Model saved to {model_path}")


def train_monthly_model(
    model: torch.nn.Module,
    dataset: Dataset,
    shuffle: bool = True,
    batch_size: int = 2,
    num_epoch: int = 100,
    patience: int = 10,
    accumulation_steps: int = 1,
    optimizer_lr: float = 1e-3,
    run_dir: str = ".",
    save_model: bool = True,
    device: str = "cpu",
    verbose: bool = True,
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
        save_model: whether to save the best model to disk
        device: device to run training on ("cpu" or "cuda")
        verbose: whether to print training progress
    """

    # Initialize the model
    model = model.to(device)
    decoder = model.decoder
    with torch.no_grad():
        decoder.bias.copy_(torch.from_numpy(dataset.daily_mean))
        decoder.scale.copy_(torch.from_numpy(dataset.daily_std) + 1e-6)

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
    )

    # Set up logging
    writer = _setup_logging(run_dir)

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
    best_loss = float("inf")
    counter = 0
    best_state_dict = None  # Store best model state

    # Add scheduler - reduces LR instead of stopping immediately
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=patience // 2,  # Reduce LR before early stop triggers
        min_lr=1e-7,
    )

    model.train()
    for epoch in range(num_epoch):
        epoch_loss = 0.0

        optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # Batch prediction
            pred = model(
                batch["daily_patch"],
                batch["daily_mask_patch"],
                batch["daily_timef_patch"],
                batch["land_mask_patch"],
                batch["padded_days_mask"],
            )  # (B, M, H, W)

            # Compute masked loss
            loss = _compute_masked_loss(
                pred, batch["monthly_patch"], batch["land_mask_patch"]
            )

            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            # Track unscaled loss for logging
            epoch_loss += loss.item()

            # Update weights every accumulation_steps batches
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Handle remaining gradients if num_batches is not divisible by accumulation_steps
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / (i + 1)

        # Step scheduler
        scheduler.step(avg_epoch_loss)

        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        writer.add_scalar("Loss/best", best_loss, epoch)

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}: best_loss = {best_loss:.6f}")

        # Only stop if LR is at minimum AND no improvement
        current_lr = optimizer.param_groups[0]["lr"]
        if counter >= patience and current_lr <= scheduler.min_lrs[0]:
            writer.add_text("Training", f"Early stop at epoch {epoch}", epoch)
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Close the writer when done
    writer.close()

    if verbose:
        print(f"Training complete. Best loss: {best_loss:.6f}")

    if save_model:
        _save_model(model, run_dir, verbose)

    return model
