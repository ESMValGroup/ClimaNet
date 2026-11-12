import argparse
from pathlib import Path

import ray
import xarray as xr

from climanet.dataset import DataLoaderConfig, STDataset
from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.train import TrainConfig, train_monthly_model
from climanet.utils import (
    compute_anom_scale,
    configure_compute_resources,
    read_st_data,
    set_seed,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default=Path("./run_dir").resolve(),
    )
    parser.add_argument(
        "--prepared-data-dir",
        type=str,
        default=Path("./data").resolve(),
    )
    parser.add_argument(
        "--tune-dir",
        type=str,
        default=Path("./data").resolve(),
    )
    parser.add_argument(
        "--lsm-dir",
        type=str,
        default=Path("./data").resolve(),
    )
    args = parser.parse_args()

    var_name = "tos"
    device = "cuda"
    prepared_data_dir = Path(args.prepared_data_dir).resolve()
    lsm_dir = Path(args.lsm_dir).resolve()
    tune_dir = Path(args.tune_dir).resolve()
    run_dir = Path(args.run_dir).resolve()

    # Load the best hyperparameters from tuning
    analysis = ray.tune.ExperimentAnalysis(str(tune_dir))
    best_result = analysis.get_best_trial("loss", "min")
    best_config = best_result.config

    # set the random seed for reproducibility
    set_seed()

    # Build dataset for training and validation
    lsm_file_path = lsm_dir / "era5_lsm_bool.nc"
    lsm_mask = xr.open_dataset(lsm_file_path)["lsm"]  # make sure is dask array

    dataset_crop_size = (1, 40, 40)
    model_patch_size = (1, best_config["patch_size"], best_config["patch_size"])

    train_years = [2018, 2019, 2020]
    data = [read_st_data(data_path=f"{prepared_data_dir}/{year}", var_name=var_name) for year in train_years]
    input_das, input_da_nan_masks, monthly_das, padded_days_masks, time_features_list = zip(*data)

    input_da = xr.concat(input_das, dim="M")
    input_da_nan_mask = xr.concat(input_da_nan_masks, dim="M")
    monthly_da = xr.concat(monthly_das, dim="M")
    padded_days_mask = xr.concat(padded_days_masks, dim="M")
    time_features = xr.concat(time_features_list, dim="M")

    # calculate anom_scale
    anom_scale = compute_anom_scale(input_da, input_da_nan_mask, lsm_mask)
    print("Anomaly scale:", anom_scale)

    # normalize monthly data for training
    STD_TRAIN = float(monthly_da.std(skipna=True).compute())
    monthly_da_norm = monthly_da / STD_TRAIN
    print(f"STD_TRAIN: {STD_TRAIN}")

    dataset_train = STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        monthly_da=monthly_da_norm,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        land_mask=lsm_mask,
        crop_size=dataset_crop_size,
        stride=None,
        model_patch_size=model_patch_size,
        verbose=False,
        load_lazy=True,
        augment=True,
        p_aug=0.5,
        drop_range=(0.1, 0.5),
        min_obs=5,
        target_scale=STD_TRAIN,
    )

    validation_year = [2021]
    data = [read_st_data(data_path=f"{prepared_data_dir}/{year}", var_name=var_name) for year in validation_year]
    input_das, input_da_nan_masks, monthly_das, padded_days_masks, time_features_list = zip(*data)

    input_da = xr.concat(input_das, dim="M")
    input_da_nan_mask = xr.concat(input_da_nan_masks, dim="M")
    monthly_da = xr.concat(monthly_das, dim="M")
    padded_days_mask = xr.concat(padded_days_masks, dim="M")
    time_features = xr.concat(time_features_list, dim="M")

    # normalize monthly data for validation
    monthly_da_norm = monthly_da / STD_TRAIN

    dataset_validation = STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        monthly_da=monthly_da_norm,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        land_mask=lsm_mask,
        crop_size=dataset_crop_size,
        stride=None, # no overlap for validation
        model_patch_size=model_patch_size,
        verbose=False,
        load_lazy=True,
    )

    # Build the dataloader config
    dataloader_num_workers = 16  # adjust if needed
    use_cuda = device == "cuda"
    dataloader_config = DataLoaderConfig(
        batch_size=108, # adjust if OOM issue, dividable by length of dataset
        shuffle=True,  # this is only for training and not for validation
        num_workers=dataloader_num_workers,
        pin_memory=use_cuda,
        persistent_workers=True,
        device=device,
        multiprocessing_context="spawn",
    )

    # Build the model with the best hyperparameters from tuning
    model = SpatioTemporalModel(
        patch_size=model_patch_size,
        embed_dim=best_config["embed_dim"],
        dropout=best_config["dropout"],
        hidden=best_config["hidden"],
        encoder_proj_dim=best_config["encoder_proj_dim"],
        anom_scale=anom_scale,
        target_scale=STD_TRAIN,
    )

    # move the model to GPU and configure compute resources
    model = configure_compute_resources(
        model,
        device=device,
        compute_threads=None,  # on gpu, it is not used
        dataloader_num_workers=dataloader_num_workers
    )

    # Training configuration
    training_config = TrainConfig(
        calculate_residuals=False,
        num_epoch=101,
        patience=20,
        accumulation_steps=1,
        optimizer_lr=best_config["optimizer_lr"],
        optimizer_weight_decay=best_config["optimizer_weight_decay"],
        scheduler_min_lr=1e-6,
        scheduler_lr_factor=0.1,
        device=device,
        verbose=True,
        verbose_epoch_interval=10,
        tune_checkpoint=False,
        store_model=True,
        store_logs=True,
    )

    trained_model = train_monthly_model(
        model=model,
        dataset_train=dataset_train,
        dataloader_config=dataloader_config,
        training_config=training_config,
        dataset_validation=dataset_validation,
        run_dir=run_dir,
    )
