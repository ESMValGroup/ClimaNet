import numpy as np
from torch.utils.data import Dataset
from climanet.st_encoder_decoder import SpatioTemporalModel
import xarray as xr
import torch
from torch.utils.data import DataLoader
from climanet.utils import setup_logging, compute_masked_loss


def _save_netcdf(predictions: np.ndarray, dataset: Dataset, save_dir: str):
    """Helper function to convert predictions to xarray and save as netCDF."""
    B, M, H, W = predictions.shape

    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(dataset))

    lats = base_dataset.monthly_da.coords["lat"].values
    lons = base_dataset.monthly_da.coords["lon"].values
    times = base_dataset.monthly_da.coords["time"].values

    full_predictions = np.full(
        (M, len(lats), len(lons)), np.nan, dtype=predictions.dtype
    )
    for i, patch_idx in enumerate(indices):
        lat_start, lon_start = base_dataset.patch_indices[patch_idx]
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
    return_loss: bool = False,
    device: str = "cpu",
    run_dir: str = ".",
    verbose: bool = True,
    dataloader_num_workers: int = 2,
    predict_threads: int | None = None,
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
    if isinstance(model, str):
        model = _load_model(model, device)

    model.to(device)
    model.eval()

    use_cuda = device == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=dataloader_num_workers,# for data loading
        persistent_workers=True,  # keep workers alive between epochs
    )

    # Initialize an empty list to store predictions
    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset

    M = base_dataset.monthly_np.shape[0]
    H, W = base_dataset.patch_size
    all_predictions = torch.empty(len(dataset), M, H, W)

    # Set up logging
    writer = setup_logging(run_dir)

    with torch.no_grad():
        idx = 0
        average_loss = 0.0
        for i, batch in enumerate(dataloader):
            # Move batch to the appropriate device
            predictions = model(
                batch["daily_patch"].to(device, non_blocking=use_cuda),
                batch["daily_mask_patch"].to(device, non_blocking=use_cuda),
                batch["daily_timef_patch"].to(device, non_blocking=use_cuda),
                batch["land_mask_patch"].to(device, non_blocking=use_cuda),
                batch["geo_pos_embedding_patch"].to(device, non_blocking=use_cuda),
                batch["scale_feature_patch"].to(device, non_blocking=use_cuda),
                batch["padded_days_mask"].to(device, non_blocking=use_cuda),
            )

            # Compute masked loss
            loss = compute_masked_loss(
                predictions,
                batch["monthly_patch"].to(device, non_blocking=use_cuda),
                batch["land_mask_patch"].to(device, non_blocking=use_cuda),
            )
            average_loss += loss.item()

            all_predictions[idx : idx + predictions.size(0)] = predictions.cpu()
            idx += predictions.size(0)

            if verbose:
                print(
                    f"Processed batch {i + 1}/{len(dataloader)}, with loss: {loss.item():.4f}"
                )

            writer.add_scalar("Progress/Batch", i + 1, idx)

    average_loss = average_loss / len(dataloader)

    if verbose:
        print(f"Average loss over all batches: {average_loss:.4f}")
    writer.add_scalar("Loss/Average", average_loss)

    if return_numpy:
        all_predictions = all_predictions.numpy()

    if save_predictions:
        if not return_numpy:
            all_predictions = all_predictions.cpu().numpy()
        all_predictions = _save_netcdf(all_predictions, dataset, run_dir)

        if verbose:
            print(f"Predictions saved to '{run_dir}'")

        writer.add_text("Info", f"Predictions saved to '{run_dir}'")

    # Close the writer when done
    writer.close()

    if return_loss:
        all_predictions = (all_predictions, average_loss)

    return all_predictions
