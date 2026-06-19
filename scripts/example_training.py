#!/usr/bin/env python3
from pathlib import Path
import xarray as xr
import torch
import torch.nn.functional
from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.utils import (
    set_seed,
    configure_compute_resources,
)
from climanet.train import train_monthly_model
from climanet import STDataset

from torch.utils.data import random_split


def main():
    # Data settings
    # Data folder
    data_folder = Path("/work/bd0854/b380103/eso4clima/output/concatenated/")
    # Path to land-sea mask file (need to setup in the experiment directory)
    lsm_file = "/home/b/b383704/eso4clima/data/era5_lsm_bool.nc"
    # Must be divisible by the model patch size
    # Default input data has 720x1440 spatial dimensions

    # Training settings
    patch_size_model = (1, 4, 4)  # Size of model encoder (time, lat, lon).
    num_patches = (30, 30)  # Number of patches in spatial dimensions
    spatial_patch_size = (
        patch_size_model[1] * num_patches[0],
        patch_size_model[2] * num_patches[1],
    )  # Spatial dimensions of the input data
    stride = (spatial_patch_size[0] // 5, spatial_patch_size[1] // 5)
    overlap = 2  # Overlap between patches (in pixels).
    num_months = 2  # Number of months to predict (model output channels)
    embed_dim = 128
    dropout = 0.2
    hidden = 128
    batch_size = 50  # Number of samples per batch in training
    num_epoch = 100  # Maximum number of epochs to train
    accumulation_steps = 2  # Number of batches to accumulate gradients over
    sh_embed_dim = 96
    sh_order_L = 10
    compute_threads = 96
    dataloader_num_workers = 32
    run_dir = "./runs"  # Directory to save logs and model checkpoints

    # Get list of daily and monthly files, sort by time
    daily_files = sorted(data_folder.rglob("20*day_ERA5dc_masked_tos.nc"))
    monthly_files = sorted(data_folder.rglob("20*mon_ERA5dc_full_tos.nc"))

    # Set seed for reproducibility
    set_seed()

    # Open datasets with chunks
    # The chunk sizes are chosen as twice the sample patch size
    daily_data = xr.open_mfdataset(
        daily_files,
        combine="by_coords",
        chunks={
            "time": 1,
            "lat": spatial_patch_size[0] * 2,
            "lon": spatial_patch_size[1] * 2,
        },
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=False,
    )
    daily_data = daily_data.chunk(
        {"time": 1, "lat": spatial_patch_size[0] * 2, "lon": spatial_patch_size[1] * 2}
    )  # Mannually chunk the dataset after opening

    monthly_data = xr.open_mfdataset(
        monthly_files,
        combine="by_coords",
        chunks={
            "time": 1,
            "lat": spatial_patch_size[0] * 2,
            "lon": spatial_patch_size[1] * 2,
        },
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=False,
    )
    monthly_data = monthly_data.chunk(
        {"time": 1, "lat": spatial_patch_size[0] * 2, "lon": spatial_patch_size[1] * 2}
    )  # Mannually chunk the dataset after opening

    lsm_mask = xr.open_dataset(lsm_file)

    # Use Monthly residuals as target
    daily_subset_averaged = daily_data.resample(time="MS").mean(skipna=True)
    daily_subset_averaged["time"] = monthly_data["time"]
    monthly_subset_res = monthly_data - daily_subset_averaged

    # create the model
    print("Creating the model...")
    model = SpatioTemporalModel(
        patch_size=patch_size_model,
        overlap=overlap,
        num_months=num_months,
        embed_dim=embed_dim,
        dropout=dropout,
        hidden=hidden,
    )

    # Make a dataset
    print("Creating the dataset...")
    dataset = STDataset(
        daily_da=daily_data["tos"],
        monthly_da=monthly_subset_res["tos"],
        land_mask=lsm_mask["lsm"],
        patch_size=spatial_patch_size,  # based on the patch_size in model
        stride=stride,
        sh_embed_dim=sh_embed_dim,
        sh_order_L=sh_order_L,
    )
    print(f"Total length training dataset: {len(dataset)}")

    # create train test data
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.6 * len(dataset))
    validation_size = int(0.3 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_size, validation_size, test_size], generator=generator
    )
    print(
        f"Train dataset length: {len(train_dataset)}, Validation dataset length: {len(validation_dataset)}, Test dataset length: {len(test_dataset)}"
    )

    # Device and resources
    model = configure_compute_resources(
        model,
        device="cpu",
        compute_threads=compute_threads,
        dataloader_num_workers=dataloader_num_workers,
    )

    # Train the model
    # Results will be saved to runs/best_model.pth
    print("Starting training...")
    _ = train_monthly_model(
        model,
        train_dataset,
        validation_dataset=validation_dataset,
        batch_size=batch_size,
        num_epoch=num_epoch,
        accumulation_steps=accumulation_steps,
        run_dir=run_dir,
        dataloader_num_workers=dataloader_num_workers,
    )


if __name__ == "__main__":
    main()
