import argparse
from pathlib import Path

from ray import tune

from climanet.tune import run_tune
from climanet.utils import data_split

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-dir",
        type=str,
        default=Path("./tune_results"),
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=Path("./data"),
    )
    args = parser.parse_args()

    data_folder = Path(args.data_dir)
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

    tune_config = {
        "max_num_epochs": 100,
        "num_trials": 100,
        "cpu_per_trial": 4,
        "gpu_per_trial": 1,
        "run_dir": args.local_dir,
        "device": "cuda",
        "dataloader_num_workers": 4,
        "data_config_train": {
            "input_filenames": hourly_files["train"],
            "monthly_filenames": monthly_files["train"],
            "landmask_filename": data_folder / "era5_lsm_bool.nc",
            "var_name": "tos",
        },
        "data_config_validation": {
            "input_filenames": hourly_files["validation"],
            "monthly_filenames": monthly_files["validation"],
            "landmask_filename": data_folder / "era5_lsm_bool.nc",
            "var_name": "tos",
        },
        "num_epoch": 100,
        # parameters to tune
        "patch_size": tune.choice([2, 4, 8, 16]),
        "overlap": tune.choice([0, 1, 2]),
        "embed_dim": tune.choice([64, 128, 256]),
        "dropout": tune.choice([0.1, 0.2, 0.3]),
        "hidden": tune.choice([128, 256, 512]),
        "spatial_depth": tune.choice([1, 2, 3]),
        "spatial_heads": tune.choice([2, 4, 8]),
        "optimizer_lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([200, 400, 800]),
        "accumulation_steps": tune.choice([100, 200, 400]),
        "max_concurrent_trials": args.num_nodes * 4,  # GPUs per node (4)
    }

    results = run_tune(tune_config)
