from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from climanet.dataset import DataLoaderConfig, DatasetConfig, STDataset
from climanet.utils import (
    compute_masked_loss,
    data_preparation,
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
    verbose: bool = True


def _save_netcdf(predictions: np.ndarray, dataset: Dataset, save_dir: str):
    """Helper function to convert predictions to xarray and save as netCDF."""
    _, M, H, W = predictions.shape

    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(dataset))

    lats = base_dataset.monthly_da.coords["lat"].values
    lons = base_dataset.monthly_da.coords["lon"].values
    times = base_dataset.monthly_da.coords["time"].values

    full_predictions = np.full(
        (len(times), len(lats), len(lons)), np.nan, dtype=predictions.dtype
    )
    for i, patch_idx in enumerate(indices):
        month_start, lat_start, lon_start = base_dataset.patch_indices[patch_idx]
        full_predictions[
            month_start : month_start + M,
            lat_start : lat_start + H,
            lon_start : lon_start + W,
        ] = predictions[i]

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


def _move_batch_to_device(batch: dict, device: str):
    use_cuda = device == "cuda"
    return {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}


def _run_one_batch(model: torch.nn.Module, batch: dict, device: str):
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
    return loss, pred


def _predict_one_year(
    model: torch.nn.Module,
    input_data_year,
    monthly_data_year,
    land_mask,
    calculate_residuals=True,
    is_hourly=False,
    dataset_patch_size=(1, 16, 16),
    dataset_stride=None,
    dataloader_batch_size=32,
    dataloader_shuffle=True,
    dataloader_num_workers=0,
    device: str = "cpu",
    run_dir: str = ".",
    verbose: bool = True,
    save_predictions: bool = True,
):
    # prepare data
    (input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features) = (
        data_preparation(
            input_data_year,
            monthly_data_year,
            calculate_residuals=calculate_residuals,
            is_hourly=is_hourly,
        )
    )

    dataset = STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        monthly_da=monthly_da,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        land_mask=land_mask["lsm"],
        patch_size=dataset_patch_size,
        stride=dataset_stride,
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

    # Initialize an empty list to store predictions
    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset

    M, H, W = base_dataset.patch_size
    all_predictions = torch.empty(len(dataset), M, H, W, device=device)

    idx = 0
    total_loss = 0.0
    total_num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        loss, predictions = _run_one_batch(model, batch, device)
        total_loss += loss.detach()

        all_predictions[idx : idx + predictions.size(0)] = predictions.detach()
        idx += predictions.size(0)

        if verbose:
            print(
                f"Processed batch {i + 1}/{len(dataloader)}, with loss: {loss.item():.4f}"
            )

    all_predictions = all_predictions.cpu().numpy()
    if save_predictions:
        all_predictions = _save_netcdf(all_predictions, dataset, run_dir)

        if verbose:
            print(f"Predictions saved to '{run_dir}'")

    del dataloader
    del dataset

    return total_loss, all_predictions, total_num_batches


def predict_monthly_var(
    model: torch.nn.Module | str,
    input_data,
    monthly_data,
    dataset_config: DatasetConfig,
    dataloader_config: DataLoaderConfig,
    prediction_config: PredictionConfig,
    land_mask=None,
    run_dir: str = ".",
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
        return_loss: If True, also return the average loss over the dataset.
        device: The device to run the predictions on (e.g., 'cpu' or 'cuda').
        run_dir: Directory to save log files and predictions.
        verbose: If True, prints progress information during prediction.
        dataloader_num_workers: how many subprocesses to use for data loading.
            See torch DataLoader docs for details.
    Returns:
        A NumPy array, PyTorch tensor, or xarray Dataset containing the predicted values.
        If return_loss is True, it also returns the average loss over the dataset.
    """
    # Load the model if a path is provided
    if isinstance(model, str | Path):
        model = load_model(model, prediction_config.device)

    model.to(prediction_config.device)
    model.eval()

    # Set up logging
    writer = setup_logging(run_dir)

    # get years from training data
    years = np.unique(monthly_data.time.dt.year)

    # create a nympy array to store all predictions
    all_predictions = []

    with torch.inference_mode():
        total_loss = 0.0
        total_num_batches = 0

        for year in years:
            input_data_year = input_data[dataset_config.var_name].sel(time=str(year))
            monthly_data_year = monthly_data[dataset_config.var_name].sel(
                time=str(year)
            )
            loss_year, predictions_year, num_batches_year = _predict_one_year(
                model=model,
                input_data_year=input_data_year,
                monthly_data_year=monthly_data_year,
                land_mask=land_mask,
                calculate_residuals=prediction_config.calculate_residuals,
                is_hourly=dataset_config.is_hourly,
                dataset_patch_size=dataset_config.spatial_patch_size,
                dataset_stride=dataset_config.spatial_stride,
                dataloader_batch_size=dataloader_config.batch_size,
                dataloader_shuffle=dataloader_config.shuffle,
                dataloader_num_workers=dataloader_config.num_workers,
                device=prediction_config.device,
                run_dir=run_dir,
                verbose=prediction_config.verbose,
                save_predictions=prediction_config.save_predictions,
            )
            total_loss += loss_year
            total_num_batches += num_batches_year

            if prediction_config.return_numpy:
                all_predictions.append(predictions_year)

        average_loss = total_loss.item() / total_num_batches
    if prediction_config.verbose:
        print(f"Average loss over all batches: {average_loss:.4f}")
    writer.add_scalar("Loss/Average", average_loss)

    # Close the writer when done
    writer.close()

    if prediction_config.return_numpy:
        return np.stack(all_predictions, axis=0)

    if prediction_config.return_loss:
        return average_loss

    return None
