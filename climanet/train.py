import copy
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch

from climanet.predict import predict_monthly_var
from climanet.utils import setup_logging, compute_masked_loss, save_model


def train_monthly_model(
    model: torch.nn.Module,
    dataset: Dataset,
    validation_dataset: Dataset | None = None,
    shuffle: bool = True,
    batch_size: int = 2,
    num_epoch: int = 100,
    patience: int = 10,
    accumulation_steps: int = 1,
    optimizer_lr: float = 1e-3,
    run_dir: str = ".",
    store_model: bool = True,
    device: str = "cpu",
    verbose: bool = True,
    dataloader_num_workers: int = 2,
    training_threads: int = None,

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
    """
    # check if dataset has indices attribute for stats calculation
    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else None
    mean, std = base_dataset.compute_stats(indices)

    # Initialize the model
    model = model.to(device)

    decoder = model.module.decoder if hasattr(model, 'module') else model.decoder
    with torch.no_grad():
        decoder.bias.copy_(torch.from_numpy(mean))
        decoder.scale.copy_(torch.from_numpy(std) + 1e-6)

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=dataloader_num_workers, # for data loading
        persistent_workers=True,  # keep workers alive between epochs
    )

    # Set up logging
    writer = setup_logging(run_dir)

    # Set the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_lr, weight_decay=1e-2
    )
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
    for epoch in range(num_epoch + 1):
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
            loss = compute_masked_loss(
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
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)

        # Validation loss (optional)
        if validation_dataset is not None:
            # Store train loss for gap calculation
            avg_train_loss = avg_epoch_loss

            _, avg_epoch_loss = predict_monthly_var(
                model,
                validation_dataset,
                batch_size=batch_size,
                device=device,
                return_numpy=False,
                save_predictions=False,
                return_loss=True,
                verbose=False,
                run_dir=run_dir,
                dataloader_num_workers=dataloader_num_workers,
            )
            writer.add_scalar("Loss/validation", avg_epoch_loss, epoch)

            if verbose and epoch % 20 == 0:
                gap = avg_epoch_loss - avg_train_loss
                print(f"Epoch {epoch}: gap between train and val loss: {gap:.6f}")

        # Step scheduler
        scheduler.step(avg_epoch_loss)

        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        writer.add_scalar("Loss/best", best_loss, epoch)

        # Early stopping check
        # Consider improvement only if loss decreases more than a small threshold
        if avg_epoch_loss < best_loss - 1e-4:
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

    if store_model:
        save_model(model, run_dir, verbose)

    return model
