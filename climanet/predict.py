from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from climanet.st_encoder_decoder import SpatioTemporalModel
import xarray as xr
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def _setup_logging(log_dir: str) -> SummaryWriter:
    """Set up TensorBoard logging directory and writer."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir)


def _save_netcdf(predictions: np.ndarray, dataset: Dataset, save_dir: str):
    """Helper function to convert predictions to xarray and save as netCDF."""
    B, M, H, W = predictions.shape

    lats = dataset.monthly_da.coords["lat"].values
    lons = dataset.monthly_da.coords["lon"].values
    times = dataset.monthly_da.coords["time"].values

    full_predictions = np.empty((M, len(lats), len(lons)), dtype=predictions.dtype)
    for i, (lat_start, lon_start) in enumerate(dataset.patch_indices):
        full_predictions[:, lat_start : lat_start + H, lon_start : lon_start + W] = (
            predictions[i]
        )

    data_vars = {
        "predictions": (("time", "lat", "lon"), full_predictions),
    }

    coords = {
        "time": times,
        "lat": lats,
        "lon": lons,
    }

    ds_pred = xr.Dataset(data_vars=data_vars, coords=coords)

    for t in times:
        time_str = np.datetime_as_string(t, unit="D").replace("-", "")
        ds_pred.sel(time=[t]).to_netcdf(f"{save_dir}/{time_str}_predictions.nc")
    return ds_pred


def _load_model(model_path: str, device: str):
    """Helper function to load a model from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SpatioTemporalModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def predict_monthly_var(
    model: torch.nn.Module | str,
    dataset: Dataset,
    batch_size: int = 2,
    return_numpy: bool = True,
    save_predictions: bool = True,
    device: str = "cpu",
    log_dir: str = ".",
    verbose: bool = True,
):
    """
    Predicts monthly variable values using a trained model and a provided dataset.

    Args:
        model: A trained PyTorch model or a path to a saved model file.
        dataset: A PyTorch Dataset containing the input data for prediction.
        batch_size: The number of samples to process in each batch during prediction.
        return_numpy: If True, returns predictions as a NumPy array.
            Otherwise, returns a PyTorch tensor.
        save_predictions: If True, convert the predictions to xarray and
            save to disk as netCDF files and return the xarray Dataset.
        device: The device to run the predictions on (e.g., 'cpu' or 'cuda').
        log_dir: Directory to save log files and predictions.
        verbose: If True, prints progress information during prediction.
    Returns:
        A NumPy array, PyTorch tensor, or xarray Dataset containing the predicted values.
    """
    # Load the model if a path is provided
    if isinstance(model, str):
        model = _load_model(model, device)

    model.to(device)
    model.eval()

    use_cuda = device == "cuda"
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=use_cuda
    )

    # Initialize an empty list to store predictions
    M = dataset.monthly_np.shape[0]
    H, W = dataset.patch_size
    all_predictions = torch.empty(len(dataset), M, H, W)

    # Set up logging
    writer = _setup_logging(log_dir)

    with torch.no_grad():
        idx = 0
        for i, batch in enumerate(dataloader):
            # Move batch to the appropriate device
            predictions = model(
                batch["daily_patch"].to(device, non_blocking=use_cuda),
                batch["daily_mask_patch"].to(device, non_blocking=use_cuda),
                batch["land_mask_patch"].to(device, non_blocking=use_cuda),
                batch["padded_days_mask"].to(device, non_blocking=use_cuda),
            )
            all_predictions[idx : idx + predictions.size(0)] = predictions.cpu()
            idx += predictions.size(0)

            if verbose:
                print(f"Processed batch {i + 1}/{len(dataloader)}")

            writer.add_scalar("Progress/Batch", i + 1, idx)

    if return_numpy:
        all_predictions = all_predictions.numpy()

    if save_predictions:
        if not return_numpy:
            all_predictions = all_predictions.cpu().numpy()
        all_predictions = _save_netcdf(all_predictions, dataset, log_dir)

        if verbose:
            print(f"Predictions saved to '{log_dir}'")

        writer.add_text("Info", f"Predictions saved to '{log_dir}'")

    # Close the writer when done
    writer.close()

    return all_predictions
