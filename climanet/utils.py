from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset
import xarray as xr
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def regrid_to_boundary_centered_grid(
    da: xr.DataArray,
    roll = False
) -> xr.DataArray:
    """
    Interpolates a DataArray from its current center-based grid onto a new
    grid whose coordinates are derived from user-specified boundaries.

    Includes robust handling for 0-360 vs -180-180 longitude domains.

    Assumes dimensions are named 'lat' and 'lon'.
    """
    print("Starting regridding process...")

    # --- 0. Longitude Domain Check and Correction ---

    input_lon = da['longitude']

    # Check if roll for 0-360 to -180-180 is requested
    if roll:
        print("Applying cyclic roll to -180 to 180...")

        # Calculate the index closest to 180 degrees
        lon_diff = np.abs(input_lon - 180.0)
        # We need to roll such that the 180-degree line is moved to the edge
        # and the new array starts near -180
        roll_amount = int(lon_diff.argmin().item() + (input_lon.size / 2)) % input_lon.size

        # Roll the DataArray and its coordinates
        da = da.roll(longitude=roll_amount, roll_coords=True)

        # Correct the longitude coordinate values: shift values > 180 down by 360
        new_lon_coords = da['longitude'].where(da['longitude'] <= 180, da['longitude'] - 360)

        # Assign the corrected and sorted coordinates
        da = da.assign_coords(longitude=new_lon_coords).sortby('longitude')
        print(f"Longitudes adjusted. New range: {da['longitude'].min().item():.2f} "
              f"to {da['longitude'].max().item():.2f}")

    # --- 1. Define Target Grid Boundaries ---

    # Target latitude boundaries: -90.0 up to 90.0 in 0.25 degree steps
    # (721 points)
    lat_bnds = np.linspace(-90.0, 90.0, 721)

    # Target longitude boundaries: -180.0 up to 180.0 in 0.25 degree steps
    # (1441 points)
    lon_bnds = np.linspace(-180.0, 180.0, 1441)

    # --- 2. Calculate New Grid Centers (Coordinates) ---

    # New latitude centers are the average of consecutive boundaries
    # (720 points)
    new_lats = (lat_bnds[:-1] + lat_bnds[1:]) / 2.0

    # New longitude centers are the average of consecutive boundaries
    # (1440 points)
    new_lons = (lon_bnds[:-1] + lon_bnds[1:]) / 2.0

    # --- 3. Interpolate the Data ---

    # Use linear interpolation (suitable for gappy data) to map data onto the
    # new centers. xarray handles the NaNs automatically.
    da_regridded = da.interp(
        latitude=new_lats,
        longitude=new_lons,
        method="linear"
    )

    print(f"Regridding complete. New dimensions: {da_regridded.dims}")
    return da_regridded


def add_month_day_dims(
    daily_ts: xr.DataArray,    # (time, H, W) daily
    monthly_ts: xr.DataArray,  # (time, H, W) monthly
    time_dim: str = "time",
    spatial_dims: Tuple[str, str] = ("lat", "lon")
):
    """ Reshape daily and monthly data to have explicit month (M) and day (T) dimensions.

    Here we assume maximum 31 days in a month, and invalid day entries will be
    padded with NaN.

    Returns
    -------
    daily_m : xr.DataArray - dims: (M, T, H, W)
    monthly_m : xr.DataArray - dims: (M, H, W)
    padded_days_mask : xr.DataArray - dims: (M, T=31), bool, True where day is padded
    """
    # Month key as integer YYYYMM
    dkey = daily_ts[time_dim].dt.year * 100 + daily_ts[time_dim].dt.month
    mkey = monthly_ts[time_dim].dt.year * 100 + monthly_ts[time_dim].dt.month

    # Unique month keys preserving order
    _, idx = np.unique(dkey.values, return_index=True)
    month_keys = dkey.values[np.sort(idx)]

    # Add M (month key) and T (day of month) coordinates to daily data
    daily_indexed = (
        daily_ts
        .assign_coords(M=(time_dim, dkey.values), T=(time_dim, daily_ts[time_dim].dt.day.values))
        .set_index({time_dim: ("M", "T")})
        .unstack(time_dim)
        .reindex(T=np.arange(1, 32), M=month_keys)
    )
    # Force dim order: (M, T, H, W) (and keep any other non-time dims after M,T)
    other_dims = [d for d in daily_ts.dims if d != time_dim]  # e.g. ["H", "W"]
    daily_indexed = daily_indexed.transpose("M", "T", *other_dims)

    # Build padded days mask from daily_indexed (NaN locations)
    padded_days_mask = ~daily_indexed.notnull().any(dim=spatial_dims)

    # Align monthly data to same month keys/order
    monthly_m = (
        monthly_ts
        .assign_coords(M=(time_dim, mkey.values))
        .swap_dims({time_dim: "M"})
        .drop_vars(time_dim)
        .sel(M=month_keys)
    )

    return daily_indexed, monthly_m, padded_days_mask


def pred_to_numpy(pred, orig_H=None, orig_W=None, land_mask=None):
    """
    pred: (B, M, H_pad,W_pad) or (B, H, W) torch tensor
    orig_H/W: original sizes before padding (optional)
    land_mask: (B, H_pad,W_pad) or (B, H,W) bool; if given, land will be set to NaN
    returns: (H,W) numpy array
    """
    # crop to original size if provided
    if orig_H is not None and orig_W is not None:
        pred = pred[..., :orig_H, :orig_W]
        if land_mask is not None:
            land_mask = land_mask[..., :orig_H, :orig_W]

    # set land to NaN (broadcast mask across batch)
    if land_mask is not None:
        pred = pred.clone().to(torch.float32)
        land_mask = land_mask.bool()
        land_mask = land_mask.unsqueeze(1) # (B, H,W) -> (B, 1, H, W) for broadcasting
        pred = torch.where(land_mask, torch.full_like(pred, float("nan")), pred)

    return pred.detach().cpu().numpy()


def calc_stats(data: xr.DataArray, time_unit="MS", spatial_dims=("lat", "lon")):
    averaged = data.resample(time=time_unit).mean(skipna=True)
    mean = averaged.mean(dim=spatial_dims, skipna=True).values
    std = averaged.std(dim=spatial_dims, skipna=True).values
    return mean, std


def train_monthly_model(
        model: torch.nn.Module,
        dataset: Dataset,
        decoder_stats: Tuple[np.ndarray, np.ndarray],
        batch_size=2,
        num_epoch=100,
        patience=10,
        accumulation_steps=1,
        optimizer_lr=1e-3,
        log_dir=".",
        save_model=True,
        device="cpu",
        verbose=True
    ):
    """ Train the model to predict monthly data from daily data.
    Args:
        model: the PyTorch model to train
        dataset: Dataset object containing the training data
        decoder_stats: Tuple of (mean, std) for the decoder
        batch_size: number of samples per batch
        num_epoch: number of epochs to train
        patience: number of epochs to wait for improvement before early stopping
        accumulation_steps: number of batches to accumulate gradients over before updating weights
        optimizer_lr: learning rate for the optimizer
        log_dir: directory to save logs
        save_model: whether to save the best model to disk
        device: device to run training on ("cpu" or "cuda")
        verbose: whether to print training progress
    """

    # Initialize the model
    model = model.to(device)
    mean, std = decoder_stats
    decoder = model.decoder
    with torch.no_grad():
        decoder.bias.copy_(torch.from_numpy(mean))
        decoder.scale.copy_(torch.from_numpy(std) + 1e-6)  # small epsilon to avoid zero

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
    )

    # Initialize TensorBoard writer
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
    best_loss = float("inf")
    counter = 0

    model.train()
    for epoch in range(num_epoch):
        epoch_loss = 0.0

        optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # Get batch data
            daily_batch = batch["daily_patch"]
            daily_mask = batch["daily_mask_patch"]
            monthly_target = batch["monthly_patch"]
            land_mask = batch["land_mask_patch"]
            padded_days_mask = batch["padded_days_mask"]

            # Batch prediction
            pred = model(daily_batch, daily_mask, land_mask, padded_days_mask)  # (B, M, H, W)

            # Mask out land pixels
            ocean = (~land_mask).to(pred.device).unsqueeze(1).float()  # (B, M=1, H, W) bool
            loss = torch.nn.functional.l1_loss(pred, monthly_target, reduction="none")
            loss = loss * ocean

            num = loss.sum(dim=(-2, -1))  # (B, M)
            denom = ocean.sum(dim=(-2, -1)).clamp_min(1)  # (B, 1)

            loss_per_month = num / denom
            loss = loss_per_month.mean()

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

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Loss/best', best_loss, epoch)

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            counter = 0
        else:
            counter += 1

        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}: best_loss = {best_loss:.6f}")

        if counter >= patience:
            writer.add_text('Training', f'Early stop at epoch {epoch}', epoch)
            break

    # Close the writer when done
    writer.close()

    if verbose:
        print(f"Training complete. Best loss: {best_loss:.6f}")

    if save_model:
        model_path = Path(log_dir) / "best_model.pth"
        torch.save(
            {"model_state_dict": model.state_dict(), "model_config": model.config}, model_path
        )
        if verbose:
            print(f"Model saved to {model_path}")

    return model
