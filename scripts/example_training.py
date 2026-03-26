#!/usr/bin/env python3
"""Example training script"""

from pathlib import Path
import torch
import torch.nn.functional
import xarray as xr
from torch.utils.data import DataLoader

from climanet import STDataset
from climanet.st_encoder_decoder import SpatioTemporalModel

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    # Compute monthly climatology stats without persisting the full (time, lat, lon) monthly field
    monthly_ts = daily_data["ts"].resample(time="MS").mean(skipna=True)
    mean = monthly_ts.mean(dim=["lat", "lon"], skipna=True).compute().values
    std = monthly_ts.std(dim=["lat", "lon"], skipna=True).compute().values
    logger.info(f"mean: {mean}, std: {std}")

    # Make a dataset
    dataset = STDataset(
        daily_da=daily_data["ts"],
        monthly_da=monthly_data["ts"],
        land_mask=lsm_mask["lsm"],
        patch_size=(patch_size_training, patch_size_training),
    )

    # Initialize training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_size = (1, 4, 4)
    overlap = 1
    model = SpatioTemporalModel(patch_size=patch_size, overlap=overlap, num_months=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    decoder = model.decoder
    with torch.no_grad():
        decoder.bias.copy_(torch.from_numpy(mean))
        decoder.scale.copy_(torch.from_numpy(std) + 1e-6)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,  
        shuffle=True,
        pin_memory=False,
    )

    best_loss = float("inf")
    patience = 10  # stop if no improvement for <patience> epochs
    counter = 0

    # Effective batch size = batch_size (fit in memory) * accumulation_steps
    accumulation_steps = 2

    # Training loop with DataLoader
    model.train()
    for epoch in range(501):
        epoch_loss = 0.0
        
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            # Get batch data
            daily_batch = batch["daily_patch"]
            daily_mask = batch["daily_mask_patch"]
            monthly_target = batch["monthly_patch"]
            land_mask = batch["land_mask_patch"]
            padded_days_mask = batch["padded_days_mask"]

            # Batch prediction
            pred = model(daily_batch, daily_mask, land_mask, padded_days_mask)  # (B, M, H, W)
            
            # Mask out land pixels
            ocean = (~land_mask).to(pred.device).unsqueeze(1).float()  # (B, M=1, H, W) bool    
            loss = torch.nn.functional.l1_loss(pred, monthly_target, reduction="none") 
            loss = loss * ocean
            
            num = loss.sum(dim=(-2, -1))  # (B, M)
            denom = ocean.sum(dim=(-2, -1)).clamp_min(1)  # (B, 1)
            
            loss_per_month = num / denom   
            loss = loss_per_month.mean()

            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            # Track unscaled loss for logging
            epoch_loss += loss.item()

            # Update weights every accumulation_steps batches
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Handle remaining gradients if num_batches is not divisible by accumulation_steps
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / (i + 1)
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            counter = 0
        else:
            counter += 1
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: best_loss = {best_loss:.6f}")
        
        if counter >= patience:
            print(f"No improvement for {patience} epochs, stopping early at epoch {epoch}.")
            break

    logger.info("training done!")
    logger.info(f"Final loss: {loss.item()}")

    # Save the trained model with config
    checkpoint = {
        "config": model.config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss.item(),
    }
    model_save_path = Path("./models/spatio_temporal_model.pth")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_save_path)
    logger.info(f"Checkpoint saved to {model_save_path}")


if __name__ == "__main__":
    main()
