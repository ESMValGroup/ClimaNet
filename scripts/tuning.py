import argparse
from pathlib import Path

from ray import tune
import ray

from climanet.tune import run_tune, tune_data_preparation
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
    hourly_files = data_split(
        data_folder,
        filename_pattern="*_hr_ERA5dc_masked_tos.nc",
        train_range=(2018, 2020),
        validation_range=(2021, 2021),
        test_range=(2022, 2022),
    )
    monthly_files = data_split(
        data_folder,
        filename_pattern="*_mon_ERA5dc_masked_tos.nc",
        train_range=(2018, 2020),
        validation_range=(2021, 2021),
        test_range=(2022, 2022),
    )
    data_config_train = {
        "input_filenames": hourly_files["train"],
        "monthly_filenames": monthly_files["train"],
        "landmask_filename": lsm_folder / "era5_lsm_bool.nc",
        "var_name": "tos",
        "patch_size": (2, 40, 40),  # based on the patch_size in model
        "stride": (1, 30, 30),  # data agumentation by overlapping patches
    }
    data_config_validation = {
        "input_filenames": hourly_files["validation"],
        "monthly_filenames": monthly_files["validation"],
        "landmask_filename": lsm_folder / "era5_lsm_bool.nc",
        "var_name": "tos",
        "patch_size": (1, 40, 40),
        "stride": (1, 30, 30),  # data agumentation by overlapping patches
    }

    tune_config = {
        "max_num_epochs": 100,
        "num_trials": 10,
        "cpu_per_trial": 4,
        "gpu_per_trial": 1,
        "run_dir": args.storage_path,
        "device": "cuda",
        "dataloader_num_workers": 4,
        "train_dataset": tune_data_preparation(data_config_train),
        "validation_dataset": tune_data_preparation(data_config_validation),
        "num_epoch": 100,
        # parameters to tune
        "patch_size": tune.grid_search([2, 4, 8]),
        "overlap": tune.grid_search([0, 1, 2]),
        "embed_dim": tune.grid_search([64, 128, 256]),
        "dropout": tune.grid_search([0.0, 0.1, 0.2]),
        "hidden": tune.grid_search([128, 256, 512]),
        "spatial_depth": tune.grid_search([1, 2, 3]),
        "spatial_heads": tune.grid_search([2, 4, 8]),
        "optimizer_lr": tune.loguniform(1e-3, 1e-1),
        "batch_size": tune.grid_search([200, 400, 800]),  # based on GPU memory
        "accumulation_steps": tune.grid_search([1, 2, 4]),  # based on batch_size
        "max_concurrent_trials": args.num_nodes * 4,  # GPUs per node (4)
    }

    # Start Ray Tune for distributed training on several nodes
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    results = run_tune(tune_config)

    ray.shutdown()
