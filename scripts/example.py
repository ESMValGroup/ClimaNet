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


def _preprocess_roi(ds, lon_subset, lat_subset):
    return ds[["ts"]].sel(lon=lon_subset, lat=lat_subset)


def main():
    # Data files
    data_folder = Path("/work/bd0854/b380103/eso4clima/output/v1.0/concatenated/") # HPC
    # data_folder = Path("../../data/output/") # local
    daily_files = sorted(data_folder.rglob("20*_day_ERA5_masked_ts.nc"))
    monthly_files = sorted(data_folder.rglob("20*_mon_ERA5_full_ts.nc"))
    daily_files.sort()
    monthly_files.sort()

    # Land surface
    lsm_file = "/home/b/b383704/eso4clima/train_twoyears/era5_lsm_bool.nc" # HPC
    # lsm_file = data_folder / "era5_lsm_bool.nc" # local
    

    # # Load full dataset
    # daily_files = sorted(data_folder.rglob("20*_day_ERA5_masked_ts.nc"))
    # monthly_files = sorted(data_folder.rglob("20*_mon_ERA5_full_ts.nc"))
    # patch_size_training = 80
    # daily_data = xr.open_mfdataset(daily_files)
    # monthly_data = xr.open_mfdataset(monthly_files)

    # Uncomment following for a partial debugging
    lon_subset = slice(-10, 10)
    lat_subset = slice(-5, 5)
    patch_size_training = 20
    daily_data = xr.open_mfdataset(
        daily_files,
        combine="by_coords",
        preprocess=lambda ds: _preprocess_roi(ds, lon_subset, lat_subset),
        chunks={"time": 1, "lat": patch_size_training * 2, "lon": patch_size_training * 2},
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=False,
    )

    monthly_data = xr.open_mfdataset(
        monthly_files,
        combine="by_coords",
        preprocess=lambda ds: _preprocess_roi(ds, lon_subset, lat_subset),
        chunks={"time": 1, "lat": patch_size_training * 2, "lon": patch_size_training * 2},
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=False,
    )
    
    lsm_mask = xr.open_dataset(lsm_file)
    lsm_mask = lsm_mask.rename({"latitude": "lat", "longitude": "lon"})[["lsm"]].sel(
        lon=lon_subset, lat=lat_subset
    )

    # Compute monthly climatology stats without persisting the full (time, lat, lon) monthly field
    monthly_ts = daily_data["ts"].resample(time="MS").mean(skipna=True)
    mean = monthly_ts.mean(dim=["lat", "lon"], skipna=True).compute().values
    std = monthly_ts.std(dim=["lat", "lon"], skipna=True).compute().values
    print(f"mean: {mean}, std: {std}")

    # Make a dataset
    dataset = STDataset(
        daily_da=daily_data["ts"],
        monthly_da=monthly_data["ts"],
        land_mask=lsm_mask["lsm"],
        patch_size=(patch_size_training, patch_size_training),
    )

    # Initialize training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpatioTemporalModel(
        embed_dim=128,
        patch_size=(1, 2, 2),
        overlap=2,
        max_months=monthly_data.sizes["time"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    decoder = model.decoder
    with torch.no_grad():
        decoder.bias.copy_(torch.from_numpy(mean))
        decoder.scale.copy_(torch.from_numpy(std) + 1e-6)

    # Make a dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=False,
    )

    # Training process
    best_loss = float("inf")
    patience = 10
    counter = 0
    model.train()
    for epoch in range(101):
        for batch in dataloader:
            optimizer.zero_grad()

            daily_batch = batch["daily_patch"]
            daily_mask = batch["daily_mask_patch"]
            monthly_target = batch["monthly_patch"]
            land_mask_patch = batch["land_mask_patch"][0, ...]
            padded_days_mask = batch["padded_days_mask"]

            pred = model(daily_batch, daily_mask, land_mask_patch, padded_days_mask)

            ocean = (~land_mask_patch).to(pred.device)
            ocean = ocean[None, None, :, :]

            loss = (
                torch.nn.functional.l1_loss(pred, monthly_target, reduction="none")
                * ocean
            )
            loss_per_month = loss.sum(dim=(-2, -1)) / ocean.sum(dim=(-2, -1))
            loss = loss_per_month.mean()

            loss.backward()
            optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0

        if epoch % 20 == 0:
            print(f"The loss is {best_loss} at epoch {epoch}")
        else:
            counter += 1
            if counter >= patience:
                print(
                    f"No improvement for {patience} epochs, stopping early at epoch {epoch}."
                )
                break

    print("training done!")
    print(f"Final loss: {loss.item()}")

    # Calculate prediction and error
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
    monthly_target = full_batch["monthly_patch"]
    land_mask_patch = full_batch["land_mask_patch"][0, ...]
    padded_days_mask = full_batch["padded_days_mask"]
    model.eval()
    with torch.no_grad():
        pred = model(daily_batch, daily_mask, land_mask_patch, padded_days_mask)
    monthly_prediction = pred_to_numpy(pred, land_mask=land_mask_patch)[0]
    monthly_data["ts_pred"] = (("time", "lat", "lon"), monthly_prediction)
    target = monthly_data["ts"].where(~lsm_mask["lsm"].values)
    err = target - monthly_data["ts_pred"]

    # Save the trained model
    model_save_path = Path("./models/spatio_temporal_model.pth")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    # Save the xr.Dataset with predictions
    predictions_save_path = Path("./predicted_data/predictions.nc")
    predictions_save_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_data.to_netcdf(predictions_save_path)
    print(f"Saved model to: {model_save_path}")
    print(f"Saved predictions to: {predictions_save_path}")

    # Plot and save inspections
    plot_path = Path("./figures/") # local
    plot_path.mkdir(parents=True, exist_ok=True)
    # 1) Prediction (t=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly_data["ts_pred"].isel(time=0).plot(ax=ax)
    fig.savefig(plot_path / "ts_pred_t0.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Target (t=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly_data["ts"].where(~lsm_mask["lsm"].values).isel(time=0).plot(ax=ax)
    fig.savefig(plot_path / "ts_target_t0.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) Error (t=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    err.isel(time=0).plot(ax=ax)
    fig.savefig(plot_path / "err_t0.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4) Error (t=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    err.isel(time=1).plot(ax=ax)
    fig.savefig(plot_path / "err_t1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
