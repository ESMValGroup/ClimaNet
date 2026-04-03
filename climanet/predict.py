from typing import Tuple

import numpy as np
from torch.utils.data import Dataset
from climanet.st_encoder_decoder import SpatioTemporalModel
import xarray as xr
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def _save_netcdf(predictions: np.ndarray, dataset: Dataset, output_path: str):
    """Helper function to convert predictions to xarray and save as netCDF."""
    data_vars = {
    "predictions": (("time", "lat", "lon"), predictions),
    }

    coords = {
        "time": dataset.monthly_da["time"].values,
        "lat": dataset.monthly_da.coords["lat"].values,
        "lon": dataset.monthly_da.coords["lon"].values,
    }

    da_pred = xr.Dataset(data_vars=data_vars, coords=coords)
    da_pred.to_netcdf(output_path)



def predict_monthly_var(
    model: torch.nn.Module | str,
    dataset: Dataset,
    batch_size: int = 2,
    return_numpy: bool = True,
    save_predictions: bool = True,
    device: str ="cpu",
    log_dir: str = ".",
    verbose: bool = True,
):
    """
    Predicts monthly variable values using a trained model and a provided dataset.

    Args:
        model: A trained PyTorch model or a path to a saved model file.
        dataset: A PyTorch Dataset containing the input data for prediction.
        batch_size: The number of samples to process in each batch during prediction.
        return_numpy: If True, returns predictions as a NumPy array. Otherwise, returns a PyTorch tensor.
        save_predictions: If True, convert the predictions to xarray and save to disk as netCDF files.
        device: The device to run the predictions on (e.g., 'cpu' or 'cuda').
        log_dir: Directory to save log files and predictions.
        verbose: If True, prints progress information during prediction.
    Returns:
        A NumPy array or PyTorch tensor containing the predicted values.
    """
    # Load the model if a path is provided
    if isinstance(model, str):
        checkpoint = torch.load(model, map_location=device, weights_only=False)
        model = SpatioTemporalModel(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    use_cuda = device == "cuda"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=use_cuda)

    # Initialize an empty list to store predictions
    all_predictions = torch.empty(len(dataset), *dataset.monthly_np.shape)

    with torch.no_grad():
        idx = 0
        for i, batch in enumerate(dataloader):
            inputs = batch.to(device, non_blocking=use_cuda)
            predictions = model(
                inputs["daily_patch"],
                inputs["daily_mask_patch"],
                inputs["land_mask_patch"],
                inputs["padded_days_mask"],
            )
            all_predictions[idx:idx + predictions.size(0)] = predictions.cpu()
            idx += predictions.size(0)

            if verbose:
                print(f"Processed batch {i + 1}/{len(dataloader)}")

    if return_numpy:
        all_predictions = all_predictions.numpy()

    if save_predictions:
        output_path = f"{log_dir}/monthly_predictions.nc"
        _save_netcdf(all_predictions, dataset, output_path)
        if verbose:
            print(f"Predictions saved to {output_path}")

    return all_predictions