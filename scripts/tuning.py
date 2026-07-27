import argparse
from pathlib import Path

from ray import tune
import ray

from climanet.tune import run_tune
from climanet.utils import data_split

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
        "--data-dir",
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

    data_folder = Path(args.data_dir).resolve()
    lsm_folder = Path(args.lsm_dir).resolve()
    var_name = "tos"

    hourly_files = data_split(
        data_folder,
        filename_pattern=f"*_hr_ERA5dc_masked_{var_name}.nc",
        train_range=(2020, 2020),
        validation_range=(2021, 2021),
        test_range=(2022, 2022),
    )
    monthly_files = data_split(
        data_folder,
        filename_pattern=f"*_mon_ERA5dc_full_{var_name}.nc",
        train_range=(2020, 2020),
        validation_range=(2021, 2021),
        test_range=(2022, 2022),
    )
    data_config_train = {
        "input_filenames": hourly_files["train"],
        "monthly_filenames": monthly_files["train"],
        "landmask_filename": lsm_folder / "era5_lsm_bool.nc",
        "var_name": var_name,
        "patch_size": (1, 40, 40),  # based on the patch_size in model
        "stride": (20, 20),  # data agumentation by overlapping patches
    }
    data_config_validation = {
        "input_filenames": hourly_files["validation"],
        "monthly_filenames": monthly_files["validation"],
        "landmask_filename": lsm_folder / "era5_lsm_bool.nc",
        "var_name": var_name,
        "patch_size": (1, 40, 40),  # based on the patch_size in model
        "stride": (20, 20),  # data agumentation by overlapping patches
    }

    # dont use ray.put() (i.e. object store) when data is large
    static_args = {
        "max_num_epochs": 100,
        "num_trials": 200,  # this is num_samples in ray.tune.TuneConfig
        "cpu_per_trial": 30,
        "gpu_per_trial": 1,
        "run_dir": args.storage_path,
        "device": "cuda",
        "dataloader_num_workers": 8,
        "data_config_train": data_config_train,
        "data_config_validation": data_config_validation,
        "num_epoch": 100,
        "max_concurrent_trials": args.num_nodes * 2,  # less than GPUs per node (4) avoid OOM
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
    print("Start ray ....")
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    results = run_tune(tune_config, static_args)

    ray.shutdown()
