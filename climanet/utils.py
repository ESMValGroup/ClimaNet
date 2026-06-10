from pathlib import Path
import random
from typing import Tuple
import numpy as np
import xarray as xr
import torch

from torch.utils.tensorboard import SummaryWriter


def regrid_to_boundary_centered_grid(da: xr.DataArray, roll=False) -> xr.DataArray:
    """
    Interpolates a DataArray from its current center-based grid onto a new
    grid whose coordinates are derived from user-specified boundaries.

    Includes robust handling for 0-360 vs -180-180 longitude domains.

    Assumes dimensions are named 'lat' and 'lon'.
    """
    print("Starting regridding process...")

    # --- 0. Longitude Domain Check and Correction ---

    input_lon = da["longitude"]

    # Check if roll for 0-360 to -180-180 is requested
    if roll:
        print("Applying cyclic roll to -180 to 180...")

        # Calculate the index closest to 180 degrees
        lon_diff = np.abs(input_lon - 180.0)
        # We need to roll such that the 180-degree line is moved to the edge
        # and the new array starts near -180
        roll_amount = (
            int(lon_diff.argmin().item() + (input_lon.size / 2)) % input_lon.size
        )

        # Roll the DataArray and its coordinates
        da = da.roll(longitude=roll_amount, roll_coords=True)

        # Correct the longitude coordinate values: shift values > 180 down by 360
        new_lon_coords = da["longitude"].where(
            da["longitude"] <= 180, da["longitude"] - 360
        )

        # Assign the corrected and sorted coordinates
        da = da.assign_coords(longitude=new_lon_coords).sortby("longitude")
        print(
            f"Longitudes adjusted. New range: {da['longitude'].min().item():.2f} "
            f"to {da['longitude'].max().item():.2f}"
        )

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
    da_regridded = da.interp(latitude=new_lats, longitude=new_lons, method="linear")

    print(f"Regridding complete. New dimensions: {da_regridded.dims}")
    return da_regridded


def add_month_day_dims(
    daily_ts: xr.DataArray,  # (time, H, W) daily
    monthly_ts: xr.DataArray,  # (time, H, W) monthly
    time_dim: str = "time",
    spatial_dims: Tuple[str, str] = ("lat", "lon"),
):
    """Reshape daily and monthly data to have explicit month (M) and day (T) dimensions.

    Here we assume maximum 31 days in a month, and invalid day entries will be
    padded with NaN.

    Returns
    -------
    daily_m : xr.DataArray - dims: (M, T, H, W)
    monthly_m : xr.DataArray - dims: (M, H, W)
    padded_days_mask : xr.DataArray - dims: (M, T=31), bool, True where day is padded
    time_features : xr.DataArray - dims: (M, T, 2)
    """
    # Month key as integer YYYYMM
    dkey = daily_ts[time_dim].dt.year * 100 + daily_ts[time_dim].dt.month
    mkey = monthly_ts[time_dim].dt.year * 100 + monthly_ts[time_dim].dt.month

    # Unique month keys preserving order
    _, idx = np.unique(dkey.values, return_index=True)
    month_keys = dkey.values[np.sort(idx)]

    # Add M (month key) and T (day of month) coordinates to daily data
    daily_indexed = (
        daily_ts.assign_coords(
            M=(time_dim, dkey.values), T=(time_dim, daily_ts[time_dim].dt.day.values)
        )
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
        monthly_ts.assign_coords(M=(time_dim, mkey.values))
        .swap_dims({time_dim: "M"})
        .drop_vars(time_dim)
        .sel(M=month_keys)
    )

    # Build aligned datetime array (M,T)
    time_da = daily_ts[time_dim]

    #time_indexed is (M,T) with NaT for padded days
    time_indexed = (
        time_da.assign_coords(M=(time_dim, dkey.values),
                              T=(time_dim, time_da.dt.day.values))
        .set_index({time_dim: ("M", "T")})
        .unstack(time_dim)
        .reindex(T=np.arange(1,32), M=month_keys)
    )

    #determine day-of-year (doy) [and hour-of-day (hod) if applicable], fill NaT with 0 inplace
    # here we choose to use the tropical year length (365.2422 day, which we round to 365.24) as the
    # period to return to the position of the sun relative to the Earth
    doy_period = 365.24
    hod_period = 24.0

    doy = time_indexed.dt.dayofyear.fillna(0)

    if "hour" in dir(time_indexed.dt):
        hod = time_indexed.dt.hour.fillna(0)
    else:
        hod = xr.zeros_like(doy)

    #create phase from day and hod
    doy_phase = 2*np.pi*doy/doy_period
    hod_phase = 2*np.pi*hod/hod_period


    #Stack cyclic encodings into time_features (M,T,2)
    time_features = xr.concat([doy_phase,hod_phase],
                              dim="feature"
                              ).transpose("M","T","feature")

    return daily_indexed, monthly_m, padded_days_mask, time_features


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
        land_mask = land_mask.unsqueeze(1)  # (B, H,W) -> (B, 1, H, W) for broadcasting
        pred = torch.where(land_mask, torch.full_like(pred, float("nan")), pred)

    return pred.detach().cpu().numpy()


def calc_stats(arr: np.ndarray, mean_axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and std along the specified axis, ignoring NaNs.

    Args:
        arr: Input array containing NaNs to ignore. shape is (M, T, H, W)
        mean_axis: Axis along which to compute mean and std (default is 0 for month)
    Returns:
        mean: Mean values along the specified axis, shape (M,)
        std: Standard deviation along the specified axis, shape (M,)
    """
    axes_to_reduce = tuple(i for i in range(arr.ndim) if i != mean_axis)

    mean = np.nanmean(arr, axis=axes_to_reduce)  # shape: (M,)
    std = np.nanstd(arr, axis=axes_to_reduce)  # shape: (M,)
    return mean, std


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str) -> SummaryWriter:
    """Set up TensorBoard logging directory and writer."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir)


def compute_masked_loss(
    pred: torch.Tensor, target: torch.Tensor, land_mask: torch.Tensor
) -> torch.Tensor:
    """Compute L1 loss masked to ocean pixels only."""
    ocean = (~land_mask).to(pred.device).unsqueeze(1)

    # Mask for valid (non-NaN) target values
    valid = ~torch.isnan(target)
    target = torch.nan_to_num(target, nan=0.0)

    mask = ocean & valid
    loss = torch.nn.functional.l1_loss(pred, target, reduction="none")
    loss = loss * mask

    num = loss.sum(dim=(-2, -1))
    denom = mask.sum(dim=(-2, -1)).clamp_min(1)

    return (num / denom).mean()


def save_model(model: torch.nn.Module, run_dir: str, verbose: bool) -> None:
    """Save model state and config to disk."""
    model_path = Path(run_dir) / "best_model.pth"
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": model.config},
        model_path,
    )
    if verbose:
        print(f"Model saved to {model_path}")


def add_month_hour_dims(
    hourly_ts: xr.DataArray,  # (time, H, W) hourly
    monthly_ts: xr.DataArray,  # (time, H, W) monthly
    time_dim: str = "time",
    spatial_dims: Tuple[str, str] = ("lat", "lon"),
):
    """Reshape hourly and monthly data to have explicit month (M) and hour (T) dimensions.

    Here we assume maximum 31 days in a month with 24 hours per day = 744 hours maximum.
    Invalid hour entries will be padded with NaN.

    Returns
    -------
    hourly_m : xr.DataArray - dims: (M, T=744, H, W)
    monthly_m : xr.DataArray - dims: (M, H, W)
    padded_hours_mask : xr.DataArray - dims: (M, T=744), bool, True where hour is padded
    time_features : xr.DataArray - dims: (M, T=744, 2)
    """
    # Month key as integer YYYYMM
    hkey = hourly_ts[time_dim].dt.year * 100 + hourly_ts[time_dim].dt.month
    mkey = monthly_ts[time_dim].dt.year * 100 + monthly_ts[time_dim].dt.month

    # Unique month keys preserving order
    _, idx = np.unique(hkey.values, return_index=True)
    month_keys = hkey.values[np.sort(idx)]

    # Create hour-of-month coordinate (1-744)
    # hour_of_month = (day_of_month - 1) * 24 + hour_of_day + 1
    day_of_month = hourly_ts[time_dim].dt.day.values
    hour_of_day = hourly_ts[time_dim].dt.hour.values
    hour_of_month = (day_of_month - 1) * 24 + hour_of_day + 1

    # Add M (month key) and T (hour of month) coordinates to hourly data
    hourly_indexed = (
        hourly_ts.assign_coords(
            M=(time_dim, hkey.values),
            T=(time_dim, hour_of_month)
        )
        .set_index({time_dim: ("M", "T")})
        .unstack(time_dim)
        .reindex(T=np.arange(1, 745), M=month_keys)  # 744 = 31 days * 24 hours
    )
    # Force dim order: (M, T, H, W)
    other_dims = [d for d in hourly_ts.dims if d != time_dim]
    hourly_indexed = hourly_indexed.transpose("M", "T", *other_dims)

    # Build padded hours mask from hourly_indexed (NaN locations)
    padded_hours_mask = ~hourly_indexed.notnull().any(dim=spatial_dims)

    # Align monthly data to same month keys/order
    monthly_m = (
        monthly_ts.assign_coords(M=(time_dim, mkey.values))
        .swap_dims({time_dim: "M"})
        .drop_vars(time_dim)
        .sel(M=month_keys)
    )

    # Build aligned datetime array (M, T)
    time_da = hourly_ts[time_dim]

    # time_indexed is (M, T) with NaT for padded hours
    time_indexed = (
        time_da.assign_coords(
            M=(time_dim, hkey.values),
            T=(time_dim, hour_of_month)
        )
        .set_index({time_dim: ("M", "T")})
        .unstack(time_dim)
        .reindex(T=np.arange(1, 745), M=month_keys)
    )

    # Determine day-of-year (doy) and hour-of-day (hod)
    doy_period = 365.24
    hod_period = 24.0

    doy = time_indexed.dt.dayofyear.fillna(0)
    hod = time_indexed.dt.hour.fillna(0)

    # Create phase from day and hour
    doy_phase = 2 * np.pi * doy / doy_period
    hod_phase = 2 * np.pi * hod / hod_period

    # Stack cyclic encodings into time_features (M, T, 2)
    time_features = xr.concat(
        [doy_phase, hod_phase],
        dim="feature"
    ).transpose("M", "T", "feature")

    return hourly_indexed, monthly_m, padded_hours_mask, time_features
