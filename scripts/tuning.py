import argparse
import time
from pathlib import Path

import ray
import xarray as xr
from ray import tune

from climanet.tune import run_tune

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

    data_config_train = {
        "input_data_dir": data_folder_train,
        "land_mask_data": ray.put(land_mask_data),
        "load_lazy": False,  # one year fits in memory
        "patch_size": (1, 40, 40),
        "stride": (20, 20),
    }

    data_config_validation = {
        "input_data_dir": data_folder_validation,
        "land_mask_data": ray.put(land_mask_data),
        "load_lazy": False,  # one year fits in memory
        "patch_size": (1, 40, 40),
        "stride": (20, 20),
    }

    # dont use ray.put() (i.e. object store) when data is large
    static_args = {
        "data_config_train": data_config_train,
        "data_config_validation": data_config_validation,
        "is_hourly": True,
        "var_name": var_name,
        "max_num_epochs": 100,
        "num_trials": 50,  # this is num_samples in ray.tune.TuneConfig
        "cpu_per_trial": 10,
        "gpu_per_trial": 1,
        "run_dir": args.storage_path,
        "device": "cuda",
        "dataloader_num_workers": 4,
        "dataloader_persistent_workers": True,
        "dataloader_multiprocessing_context": None,  # load_lazy is False
        "num_epoch": 100,
        "max_concurrent_trials": args.num_nodes * 2,  # less than GPUs per node (4) avoid OOM
        "experiment_name": "climanet_tune",
    }

    # parameters to tune
    tune_config = {
        "patch_size": tune.choice([2, 4, 8]),
        "overlap": tune.choice([0, 1, 2]),
        "embed_dim": tune.choice([32, 64, 128]),
        "dropout": tune.choice([0.0, 0.1, 0.2]),
        "hidden": tune.choice([32, 64, 128]),
        "spatial_depth": tune.choice([1, 2, 3]),
        "spatial_heads": tune.choice([2, 4, 8]),
        "optimizer_lr": tune.loguniform(1e-3, 1e-1),
        "batch_config": tune.grid_search([
            {"batch_size": 100, "accumulation_steps": 1},
            {"batch_size": 200, "accumulation_steps": 2},
            {"batch_size": 400, "accumulation_steps": 2},
        ]),
    }

    # Start Ray Tune for distributed training on several nodes
    print("Start ray init in python script....")
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    time.sleep(30)  # wait for nodes
    results = run_tune(tune_config, static_args)

    ray.shutdown()
