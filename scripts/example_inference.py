#!/usr/bin/env python3
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import torch.nn.functional
import xarray as xr
from torch.utils.data import DataLoader

from climanet import STDataset
from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.utils import pred_to_numpy


def main():
    # Data files
    data_folder = Path(
        "/work/bd0854/b380103/eso4clima/output/v1.0/concatenated/"
    )  # HPC
    # data_folder = Path("../../data/output/") # local
    daily_files = sorted(data_folder.rglob("20*_day_ERA5_masked_ts.nc"))
    monthly_files = sorted(data_folder.rglob("20*_mon_ERA5_full_ts.nc"))
    daily_files.sort()
    monthly_files.sort()

    # Land surface
    lsm_file = "/home/b/b383704/eso4clima/train_twoyears/era5_lsm_bool.nc"  # HPC
    # lsm_file = data_folder / "era5_lsm_bool.nc" # local

    # Path to the trained model
    model_save_path = Path("./models/spatio_temporal_model.pth")

    # Load full dataset
    daily_files = sorted(data_folder.rglob("20*_day_ERA5_masked_ts.nc"))
    monthly_files = sorted(data_folder.rglob("20*_mon_ERA5_full_ts.nc"))
    patch_size_training = 80
    daily_data = xr.open_mfdataset(daily_files)
    monthly_data = xr.open_mfdataset(monthly_files)

    daily_data = xr.open_mfdataset(
        daily_files,
        combine="by_coords",
        chunks={
            "time": 1,
            "lat": patch_size_training * 2,
            "lon": patch_size_training * 2,
        },
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=False,
    )

    monthly_data = xr.open_mfdataset(
        monthly_files,
        combine="by_coords",
        chunks={
            "time": 1,
            "lat": patch_size_training * 2,
            "lon": patch_size_training * 2,
        },
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=False,
    )

    lsm_mask = xr.open_dataset(lsm_file)

    # Load the trained model
    model = SpatioTemporalModel()
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Calculate prediction and attach to monthly_data xr.Dataset
    dataset_pred = STDataset(
        daily_da=daily_data["ts"],
        monthly_da=monthly_data["ts"],
        land_mask=lsm_mask["lsm"],
        patch_size=(daily_data.sizes["lat"], daily_data.sizes["lon"]),
    )
    dataloader_pred = DataLoader(
        dataset_pred,
        batch_size=len(dataset_pred),
        pin_memory=False,
    )
    full_batch = next(iter(dataloader_pred))
    daily_batch = full_batch["daily_patch"]
    daily_mask = full_batch["daily_mask_patch"]
    land_mask_patch = full_batch["land_mask_patch"][0, ...]
    padded_days_mask = full_batch["padded_days_mask"]
    with torch.no_grad():
        pred = model(daily_batch, daily_mask, land_mask_patch, padded_days_mask)
    monthly_prediction = pred_to_numpy(pred, land_mask=land_mask_patch)[0]
    monthly_data["ts_pred"] = (("time", "lat", "lon"), monthly_prediction)

    # Save the xr.Dataset with predictions
    predictions_save_path = Path("./predicted_data/predictions.nc")
    predictions_save_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_data.to_netcdf(predictions_save_path)
    print(f"Saved predictions to: {predictions_save_path}")


if __name__ == "__main__":
    main()
