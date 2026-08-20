import argparse
from pathlib import Path

import ray
import xarray as xr

from climanet.dataset import DataLoaderConfig, STDataset
from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.train import TrainConfig, train_monthly_model
from climanet.utils import configure_compute_resources, read_st_data, set_seed


def _build_dataset(
    prepared_data_dir: Path,
    years: list[int],
    var_name: str,
    land_mask: xr.DataArray,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int],
) -> STDataset:

    data = [read_st_data(data_path=f"{prepared_data_dir}/{year}", var_name=var_name) for year in years]
    input_das, input_da_nan_masks, monthly_das, padded_days_masks, time_features_list = zip(*data)

    input_da = xr.concat(input_das, dim="M")
    input_da_nan_mask = xr.concat(input_da_nan_masks, dim="M")
    monthly_da = xr.concat(monthly_das, dim="M")
    padded_days_mask = xr.concat(padded_days_masks, dim="M")
    time_features = xr.concat(time_features_list, dim="M")

    return STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        monthly_da=monthly_da,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        land_mask=land_mask,
        patch_size=patch_size,
        stride=stride,
        sh_embed_dim=96,
        sh_order_L=10,
        verbose=False,
        load_lazy=True,
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

    dataset_patch_size = (1, 40, 40)
    dataset_stride = (20, 20)

    train_years = [2018, 2019, 2020]
    dataset_train = _build_dataset(
        prepared_data_dir=prepared_data_dir,
        years=train_years,
        var_name=var_name,
        land_mask=lsm_mask,
        patch_size=dataset_patch_size,
        stride=dataset_stride,
    )

    validation_year = [2021]
    dataset_validation = _build_dataset(
        prepared_data_dir=prepared_data_dir,
        years=validation_year,
        var_name=var_name,
        land_mask=lsm_mask,
        patch_size=dataset_patch_size,
        stride=dataset_stride,
    )

    # Build the dataloader config
    dataloader_num_workers = 32  # adjust if needed
    dataloader_config = DataLoaderConfig(
        batch_size=best_config["batch_config"]["batch_size"], # adjust if OOM issue
        shuffle=True,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
        device="cuda",
        multiprocessing_context="spawn",
    )

    # Build the model with the best hyperparameters from tuning
    patch_size = (1, best_config["patch_size"], best_config["patch_size"])
    overlap = best_config["overlap"]
    embed_dim = best_config["embed_dim"]
    dropout = best_config["dropout"]
    hidden = best_config["hidden"]
    spatial_depth = best_config["spatial_depth"]
    spatial_heads = best_config["spatial_heads"]

    model = SpatioTemporalModel(
        patch_size=patch_size,
        overlap=overlap,
        embed_dim=embed_dim,
        dropout=dropout,
        hidden=hidden,
        spatial_depth=spatial_depth,
        spatial_heads=spatial_heads,
    )

    # move the model to GPU and configure compute resources
    model = configure_compute_resources(
        model,
        device="cuda",
        compute_threads=None,  # on gpu, it is not used
        dataloader_num_workers=dataloader_num_workers
    )

    # Training configuration
    training_config = TrainConfig(
        calculate_residuals=True,
        num_epoch=101,
        patience=10,
        accumulation_steps=best_config["batch_config"]["accumulation_steps"],
        optimizer_lr=best_config["optimizer_lr"],
        device="cuda",
        verbose=False,
        verbose_epoch_interval=20,
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
