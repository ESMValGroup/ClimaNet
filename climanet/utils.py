from pathlib import Path
import random
from typing import Tuple
import numpy as np
import xarray as xr
import torch
import psutil

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker


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

    # time_indexed is (M,T) with NaT for padded days
    time_indexed = (
        time_da.assign_coords(
            M=(time_dim, dkey.values), T=(time_dim, time_da.dt.day.values)
        )
        .set_index({time_dim: ("M", "T")})
        .unstack(time_dim)
        .reindex(T=np.arange(1, 32), M=month_keys)
    )

    # determine day-of-year (doy) [and hour-of-day (hod) if applicable], fill NaT with 0 inplace
    # here we choose to use the tropical year length (365.2422 day, which we round to 365.24) as the
    # period to return to the position of the sun relative to the Earth
    doy_period = 365.24
    hod_period = 24.0

    doy = time_indexed.dt.dayofyear.fillna(0)

    if "hour" in dir(time_indexed.dt):
        hod = time_indexed.dt.hour.fillna(0)
    else:
        hod = xr.zeros_like(doy)

    # create phase from day and hod
    doy_phase = 2 * np.pi * doy / doy_period
    hod_phase = 2 * np.pi * hod / hod_period

    # Stack cyclic encodings into time_features (M,T,2)
    time_features = xr.concat([doy_phase, hod_phase], dim="feature").transpose(
        "M", "T", "feature"
    )

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


def configure_compute_resources(
    model: torch.nn.Module,
    device: str,
    compute_threads: int,
    dataloader_num_workers: int,
) -> torch.nn.Module:
    """Configure model for multi-GPU and set CPU thread usage for compute (training or prediction).

    Args:
        model: the PyTorch model to configure
        device: device to run on ("cpu" or "cuda")
        compute_threads: number of threads to use for compute when device is CPU.
            If None, it will be set automatically.
        dataloader_num_workers: how many subprocesses to use for data loading.
            See torch DataLoader docs for details.
    Returns:
        The model, potentially wrapped in DataParallel if using multiple GPUs.
    """
    if device == "cpu":
        if compute_threads is None:
            total_cpus = psutil.cpu_count()
            # keep 1 for main thread
            compute_threads = max(1, total_cpus - dataloader_num_workers - 1)
        torch.set_num_threads(compute_threads)
    elif device == "cuda":
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)
    return model


def plot_results(
    target, predictions, label="SST K", title=("Target", "Prediction"), error=False
):
    fig, axs = plt.subplots(
        nrows=len(target.time), ncols=2, figsize=(10, 8), constrained_layout=True
    )

    for t in range(len(target.time)):
        # Select data for this timestep
        target_t = target.isel(time=t)
        pred_t = predictions.isel(time=t)

        # Shared color scale for this row
        target_min, target_max = target_t.min().compute(), target_t.max().compute()
        pred_min, pred_max = pred_t.min().compute(), pred_t.max().compute()

        abs_max = max(abs(target_min), abs(target_max), abs(pred_min), abs(pred_max))

        norm = None
        cmap = "RdBu_r"
        if error:
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
            cmap = "RdBu_r"

        # Left: truth
        _ = target_t.plot(ax=axs[t, 0], cmap=cmap, norm=norm, add_colorbar=False)

        # Right: prediction
        im1 = pred_t.plot(ax=axs[t, 1], cmap=cmap, norm=norm, add_colorbar=False)
        title_1, title_2 = title
        axs[t, 0].set_title(
            f"{title_1}, month={target.time.dt.strftime('%Y-%m-%d').values[t]}"
        )
        axs[t, 1].set_title(
            f"{title_2}, month={target.time.dt.strftime('%Y-%m-%d').values[t]}"
        )

        # One shared colorbar for the row
        cbar = fig.colorbar(im1, ax=axs[t, :], orientation="vertical", shrink=0.9)

        cbar.set_label(label)

    plt.show()


def plot_histograms(
    target, predictions, label="SST K", legend_labels=("Target", "Prediction"), bins=30
):
    """Plot histograms of target and predictions in the same figure for comparison."""
    fig, axs = plt.subplots(
        nrows=len(target.time),
        ncols=1,
        figsize=(8, 4 * len(target.time)),
        constrained_layout=True,
    )

    # Handle single timestep case
    if len(target.time) == 1:
        axs = axs.reshape(1, -1)

    for t in range(len(target.time)):
        target_t = target.isel(time=t)
        pred_t = predictions.isel(time=t)

        # Target histogram
        axs[t].hist(
            target_t.values.flatten(), bins=bins, alpha=0.7, color="blue", density=True
        )
        axs[t].set_xlabel(label)
        axs[t].set_ylabel("Frequency")
        axs[t].grid(True, alpha=0.3)

        # Prediction histogram (overlaid)
        axs[t].hist(
            pred_t.values.flatten(), bins=bins, alpha=0.7, color="orange", density=True
        )
        axs[t].legend(legend_labels)
        axs[t].set_title(
            f"Histogram {legend_labels[0]} vs {legend_labels[1]}, month={target.time.dt.strftime('%Y-%m-%d').values[t]}"
        )

    plt.show()


def plot_nobs_vs_err(
    nobs: xr.DataArray, err_baseline: xr.DataArray, err_predictions: xr.DataArray
):
    """Plot number of observations vs error for each month.

    The three inputs are expected to be xarray DataArrays with dimensions (time, lat, lon).
    They should share the same spatial and temporal coordinates.

    Parameters
    ----------
    nobs : xr.DataArray
        Number of observations per grid cell per month. Dimensions: (time, lat, lon)
    err_baseline : xr.DataArray
        Baseline error per grid cell per month. Dimensions: (time, lat, lon)
    err_predictions : xr.DataArray
        Prediction error per grid cell per month. Dimensions: (time, lat, lon)
    """

    fig, axes = plt.subplots(nobs.sizes["time"], 1, figsize=(5 * nobs.sizes["time"], 8))
    if nobs.sizes["time"] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.set_title(f"Month = {i}")

        # Get unique number of observations for this month, ignoring NaNs and zeros
        n_obs_unique = np.unique(nobs.isel(time=i).values.flatten())
        n_obs_unique = n_obs_unique[~np.isnan(n_obs_unique)]
        n_obs_unique = n_obs_unique.astype(int)
        n_obs_unique = n_obs_unique[n_obs_unique > 0]

        err_by_n_obs_baseline = []
        err_by_n_obs_predictions = []

        for n_obs in n_obs_unique:
            # Baseline error
            err_arr = (
                err_baseline.isel(time=i)
                .where(nobs.isel(time=i) == n_obs)
                .values.flatten()
            )
            err_arr = err_arr[~np.isnan(err_arr)]
            if len(err_arr) == 0:
                err_arr = np.array([np.nan])
            err_by_n_obs_baseline.append(np.abs(err_arr))

            # Prediction error
            err_arr = (
                err_predictions.isel(time=i)
                .where(nobs.isel(time=i) == n_obs)
                .values.flatten()
            )
            err_arr = err_arr[~np.isnan(err_arr)]
            if len(err_arr) == 0:
                err_arr = np.array([np.nan])
            err_by_n_obs_predictions.append(np.abs(err_arr))

        h1 = ax.violinplot(
            err_by_n_obs_baseline,
            positions=n_obs_unique,
            showmedians=True,
            showextrema=True,
            points=500,
        )
        h2 = ax.violinplot(
            err_by_n_obs_predictions,
            positions=n_obs_unique,
            showmedians=True,
            showextrema=True,
            points=500,
        )

        # Style: thinner outlines + less prominent extrema
        for body in h1["bodies"]:
            body.set_facecolor("tab:blue")
            body.set_edgecolor("tab:blue")
            body.set_alpha(0.45)
            body.set_linewidth(0.5)

        for body in h2["bodies"]:
            body.set_facecolor("tab:orange")
            body.set_edgecolor("tab:orange")
            body.set_alpha(0.45)
            body.set_linewidth(0.5)

        for h in (h1, h2):
            h["cmedians"].set_linewidth(0.9)
            h["cmedians"].set_alpha(0.9)

            h["cbars"].set_linewidth(0.35)
            h["cbars"].set_alpha(0.2)
            h["cmins"].set_linewidth(0.35)
            h["cmins"].set_alpha(0.2)
            h["cmaxes"].set_linewidth(0.35)
            h["cmaxes"].set_alpha(0.2)

        ax.set_xlabel("Number of Daily Observations")
        ax.set_ylabel("Log Absolute Error (K)")

        # Non-linear y-axis: keeps detail near 0 and compresses larger values.
        ax.set_yscale("symlog", linthresh=0.05, linscale=0.8, base=10)

        # Show major ticks as plain decimals instead of scientific/log notation.
        ax.yaxis.set_major_locator(
            mticker.SymmetricalLogLocator(base=10, linthresh=0.05)
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y:.3f}".rstrip("0").rstrip("."))
        )
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        ax.legend(
            [h1["bodies"][0], h2["bodies"][0]],
            ["Baseline", "Prediction"],
            loc="upper right",
        )

        plt.tight_layout()
