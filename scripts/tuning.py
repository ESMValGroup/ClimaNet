import argparse
import time
from pathlib import Path

import ray
import xarray as xr
from ray import tune

from climanet.tune import run_tune
from climanet.utils import compute_anom_scale, read_st_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage-path",
        type=str,
        default=Path("./tune_results").resolve(),
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--data-dir-train",
        type=str,
        default=Path("./data").resolve(),
    )
    parser.add_argument(
        "--data-dir-validation",
        type=str,
        default=Path("./data").resolve(),
    )
    parser.add_argument(
        "--lsm-dir",
        type=str,
        default=Path("./data").resolve(),
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="auto",
    )
    args = parser.parse_args()

    data_folder_train = Path(args.data_dir_train).resolve()
    data_folder_validation = Path(args.data_dir_validation).resolve()
    lsm_folder = Path(args.lsm_dir).resolve()
    var_name = "tos"

    land_mask_data = xr.open_dataset(lsm_folder / "era5_lsm_bool.nc")["lsm"]

    data_crop_size = (1, 40, 40)
    data_config_train = {
        "input_data_dir": data_folder_train,
        "land_mask_data": land_mask_data,  # dont use ray.put()
        "crop_size": data_crop_size,
        "var_name": var_name,
    }

    data_config_validation = {
        "input_data_dir": data_folder_validation,
        "land_mask_data": land_mask_data,  # dont use ray.put()
        "crop_size": data_crop_size,
        "var_name": var_name,
    }
    # Prepare train dataset
    input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features = (
        read_st_data(
            data_path=data_config_train["input_data_dir"], var_name=data_config_train["var_name"]
        )
    )

    # calculate anom_scale
    lsm_mask = data_config_train["land_mask_data"]
    anom_scale = compute_anom_scale(input_da, input_da_nan_mask, lsm_mask)
    print("Anomaly scale:", anom_scale)

    # normalize monthly data for training
    STD_TRAIN = float(monthly_da.std(skipna=True).compute())
    print(f"STD_TRAIN: {STD_TRAIN}")

    # dont use ray.put() (i.e. object store) when data is large
    static_args = {
        "data_config_train": data_config_train,
        "data_config_validation": data_config_validation,
        "anom_scale": anom_scale,
        "STD_TRAIN": STD_TRAIN,
        "is_hourly": True,
        "var_name": var_name,
        "max_num_epochs": 100,
        "num_trials": 50,  # this is num_samples in ray.tune.TuneConfig
        "cpu_per_trial": 10,
        "gpu_per_trial": 1,
        "run_dir": args.storage_path,
        "device": "cuda",
        "dataloader_num_workers": 2,
        "dataloader_persistent_workers": True,
        "dataloader_multiprocessing_context": "spawn",
        "num_epoch": 100,
        "max_concurrent_trials": args.num_nodes * 3,  # less than GPUs per node (4) avoid OOM
        "experiment_name": "sst_01",
    }

    # parameters to tune
    tune_config = {
        "patch_size": tune.choice([4, 8]),
        "embed_dim": tune.choice([64, 96, 128]),
        "dropout": tune.choice([0.0, 0.05, 0.1]),
        "hidden": tune.choice([64, 96, 128]),
        "encoder_proj_dim": tune.choice([16, 32, 64]),
        "optimizer_lr": tune.loguniform(1e-4, 1e-1),
        "optimizer_weight_decay": tune.loguniform(1e-5, 1e-1),
        "batch_config": tune.grid_search(
            [
                {"batch_size": 108, "accumulation_steps": 1},
            ]
        ),
    }

    # Start Ray Tune for distributed training on several nodes
    print("Start ray init in python script....")
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    time.sleep(30)  # wait for nodes
    results = run_tune(tune_config, static_args)

    ray.shutdown()
