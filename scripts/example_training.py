#!/usr/bin/env python3
from pathlib import Path
import xarray as xr
from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.train import train_monthly_model
from climanet import STDataset


def main():
    # Data settings
    data_folder = Path(
        "/work/bd0854/b380103/eso4clima/output/v1.0/concatenated/"
    )  # Data folder
    lsm_file = "/home/b/b383704/eso4clima/train_twoyears/era5_lsm_bool.nc"  # Path to land-sea mask file (local)
    patch_size_training = 200  # Spatial patch size for the training samples (lat, lon)
    # Must be divisible by the model patch size

    # Training settings
    patch_size_model = (1, 4, 4)  # Size of model encoder (time, lat, lon).
    overlap = 1  # Overlap between patches (in pixels).
    num_months = 24  # Number of months to predict (model output channels)
    batch_size = 10  # Number of samples per batch in training
    num_epoch = 501  # Maximum number of epochs to train
    patience = 10  # Number of epochs to wait for improvement before early stopping
    accumulation_steps = 2  # Number of batches to accumulate gradients over
    run_dir = "./runs"  # Directory to save logs and model checkpoints

    # Get list of daily and monthly files, sort by time
    daily_files = sorted(data_folder.rglob("20*_day_ERA5_masked_ts.nc"))
    monthly_files = sorted(data_folder.rglob("20*_mon_ERA5_full_ts.nc"))

    # Open datasets with chunks
    # The chunk sizes are chosen as twice the sample patch size
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

    # create the model
    model = SpatioTemporalModel(
        patch_size=patch_size_model, overlap=overlap, num_months=num_months
    )

    # Make a dataset
    dataset = STDataset(
        daily_da=daily_data["ts"],
        monthly_da=monthly_data["ts"],
        land_mask=lsm_mask["lsm"],
        patch_size=(patch_size_training, patch_size_training),
    )

    # Train the model
    # Results will be saved to runs/best_model.pth
    _ = train_monthly_model(
        model,
        dataset,
        batch_size=batch_size,
        num_epoch=num_epoch,
        patience=patience,
        accumulation_steps=accumulation_steps,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
