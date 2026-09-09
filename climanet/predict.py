from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from climanet.dataset import DataLoaderConfig
from climanet.utils import (
    compute_masked_loss,
    load_model,
    setup_logging,
)


@dataclass
class PredictionConfig:
    """Configuration for making predictions with the model."""

    calculate_residuals: bool = True
    return_numpy: bool = True
    save_predictions: bool = True
    return_loss: bool = False
    device: str = "cpu"
    verbose: bool = False
    store_logs: bool = True


def _save_netcdf(
    predictions: np.ndarray, dataset: Dataset, save_dir: str, residuals: bool = False
):
    """Helper function to convert predictions to xarray and save as netCDF."""
    _, M, H, W = predictions.shape

    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(dataset))

    lats = base_dataset.monthly_da.coords["lat"].values
    lons = base_dataset.monthly_da.coords["lon"].values
    times = base_dataset.monthly_da.coords["M"].values
    var_name = base_dataset.monthly_da.name

    full_predictions = np.full(
        (len(times), len(lats), len(lons)), np.nan, dtype=predictions.dtype
    )
    for i, data_idx in enumerate(indices):
        month_start, lat_start, lon_start = base_dataset.crop_indices[data_idx]
        full_predictions[
            month_start : month_start + M,
            lat_start : lat_start + H,
            lon_start : lon_start + W,
        ] = predictions[i]

    data_vars = {
        var_name: (("time", "lat", "lon"), full_predictions),
    }

    coords = {
        "time": times,
        "lat": lats,
        "lon": lons,
    }

    ds_pred = xr.Dataset(data_vars=data_vars, coords=coords)

    for t in times:
        time_str = np.datetime_as_string(t, unit="M").replace("-", "")

        file_name = f"{save_dir}/{time_str}_{var_name}_prediction.nc"
        if residuals:
            file_name = f"{save_dir}/{time_str}_{var_name}_prediction_residual.nc"

        ds_pred.sel(time=[t]).to_netcdf(file_name)


def _move_batch_to_device(batch: dict, device: str):
    use_cuda = device == "cuda"
    return {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}


def _run_one_batch(model: torch.nn.Module, batch: dict, device: str):
    batch = _move_batch_to_device(batch, device)
    pred = model(
        batch["input_data"],
        batch["input_data_mask"],
        batch["input_data_timef"],
        batch["land_mask"],
        batch["geo_pos_embedding"],
        batch["scale_feature"],
        batch["padded_days_mask"],
    )  # (B, M, H, W)

    # Compute masked loss
    loss = compute_masked_loss(pred, batch["monthly_data"], batch["land_mask"])
    return loss, pred


def predict_monthly_var(
    model: torch.nn.Module | str,
    dataset: Dataset,
    dataloader_config: DataLoaderConfig,
    prediction_config: PredictionConfig,
    run_dir: str = ".",
):
    """
    Predicts monthly variable values using a trained model and a provided dataset.

    Args:
        model: A trained PyTorch model or a path to a saved model file.
        dataset: A PyTorch Dataset containing the input data for prediction.
        dataloader_config: Configuration for the DataLoader, see DataLoaderConfig for details.
        prediction_config: Configuration for the prediction process, see PredictionConfig for details.
        run_dir: Directory to save log files and predictions.
    Returns:
        A NumPy array, PyTorch tensor, or xarray Dataset containing the predicted values.
        If return_loss is True, it also returns the average loss over the dataset.
    """
    device = prediction_config.device
    # Load the model if a path is provided
    if isinstance(model, str | Path):
        model = load_model(model, device)

    model.to(device)
    model.eval()

    use_cuda = device == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config.batch_size,
        shuffle=dataloader_config.shuffle,
        pin_memory=use_cuda,
        num_workers=dataloader_config.num_workers,  # for data loading
        persistent_workers=dataloader_config.persistent_workers,  # keep workers alive between epochs
        multiprocessing_context=dataloader_config.multiprocessing_context,
    )
    num_batches = len(dataloader)

    # Set up logging
    if prediction_config.store_logs:
        writer = setup_logging(run_dir)

    # Initialize an empty list to store predictions
    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset

    M, H, W = base_dataset.crop_size
    all_predictions = torch.empty(len(dataset), M, H, W, device=device)

    # Set up logging
    if prediction_config.store_logs:
        writer = setup_logging(run_dir)

    with torch.inference_mode():
        idx = 0
        average_loss = 0.0
        for i, batch in enumerate(dataloader):
            loss, predictions = _run_one_batch(model, batch, device)
            average_loss += loss.detach()

            all_predictions[idx : idx + predictions.size(0)] = predictions.detach()
            idx += predictions.size(0)

            if prediction_config.verbose:
                print(
                    f"Processed batch {i + 1}/{num_batches}, with loss: {loss.item():.4f}"
                )

        average_loss = average_loss.item() / num_batches

    if prediction_config.verbose:
        print(f"Average loss over all batches: {average_loss:.4f}")

    if prediction_config.store_logs:
        writer.add_scalar("Loss/Average", average_loss)

    # Close the writer when done
    if prediction_config.store_logs:
        writer.close()

    if prediction_config.return_numpy:
        all_predictions = all_predictions.cpu().numpy()
        if prediction_config.return_loss:
            all_predictions = (all_predictions, average_loss)

    if prediction_config.save_predictions:
        if not prediction_config.return_numpy:
            all_predictions = all_predictions.cpu().numpy()
        all_predictions = _save_netcdf(
            all_predictions,
            dataset,
            run_dir,
            residuals=prediction_config.calculate_residuals,
        )

    if prediction_config.return_loss:
        all_predictions = average_loss

    return all_predictions
