import argparse
from pathlib import Path

import xarray as xr
from ray import tune

from climanet.dataset import DataLoaderConfig, STDataset
from climanet.predict import PredictionConfig, predict_monthly_var
from climanet.utils import data_preparation, read_st_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the best Ray Tune checkpoint, prepare the test data, and evaluate the "
            "trained model on the 2023 test period."
        )
    )
    parser.add_argument(
        "--experiment-path",
        type=Path,
        required=True,
        help="Path to the Ray Tune experiment directory containing the checkpoint.",
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        required=True,
        help="Directory containing the test NetCDF files.",
    )
    parser.add_argument(
        "--lsm-file-path",
        type=Path,
        required=True,
        help="Path to the land-sea mask NetCDF file.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("./run_dir_tune_test").resolve(),
        help="Directory used for the evaluation run and saved logs.",
    )
    parser.add_argument(
        "--var-name",
        type=str,
        default="tos",
        help="Variable name to evaluate in the NetCDF files.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2022",
        help="Year pattern to include in the test files (e.g. 2022).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_path = args.experiment_path.resolve()
    test_data_dir = args.test_data_dir.resolve()
    lsm_file_path = args.lsm_file_path.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if not experiment_path.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist: {experiment_path}"
        )
    if not test_data_dir.exists():
        raise FileNotFoundError(f"Test data directory does not exist: {test_data_dir}")
    if not lsm_file_path.exists():
        raise FileNotFoundError(f"LSM file does not exist: {lsm_file_path}")

    daily_files = list(
        test_data_dir.glob(f"{args.year}*_hr_ERA5dc_masked_{args.var_name}*.nc")
    )
    monthly_files = list(
        test_data_dir.glob(f"{args.year}*_mon_ERA5dc_masked_{args.var_name}*.nc")
    )

    if not daily_files:
        raise FileNotFoundError(
            f"No daily test files found for year '{args.year}' in '{test_data_dir}'"
        )
    if not monthly_files:
        raise FileNotFoundError(
            f"No monthly test files found for year '{args.year}' in '{test_data_dir}'"
        )

    print(f"Using daily files ({len(daily_files)}): {daily_files[:3]} ...")
    print(f"Using monthly files ({len(monthly_files)}): {monthly_files[:3]} ...")

    daily_data_test = xr.open_mfdataset(
        daily_files, combine="by_coords", parallel=False
    )
    monthly_data_test = xr.open_mfdataset(
        monthly_files, combine="by_coords", parallel=False
    )

    test_data_zarr_dir = run_dir / "test_data_zarr"
    test_data_zarr_dir.mkdir(parents=True, exist_ok=True)

    _ = data_preparation(
        daily_data_test[args.var_name],
        monthly_data_test[args.var_name],
        calculate_residuals=True,
        is_hourly=True,
        save_to_zarr=True,
        run_dir=test_data_zarr_dir,
    )

    input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features = (
        read_st_data(
            data_path=test_data_zarr_dir,
            var_name=args.var_name,
        )
    )

    lsm_mask = xr.open_dataset(lsm_file_path)

    num_patches = (10, 10)
    patch_size = (1, 4, 4)
    spatial_patch_size = (
        patch_size[1] * num_patches[0],
        patch_size[2] * num_patches[1],
    )
    stride = (spatial_patch_size[0] // 5, spatial_patch_size[1] // 5)

    dataset_test = STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        monthly_da=monthly_da,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        land_mask=lsm_mask["lsm"],
        patch_size=(1, *spatial_patch_size),
        stride=stride,
        sh_embed_dim=96,
        sh_order_L=10,
        verbose=True,
        load_lazy=False,
    )
    print(f"Created test dataset with {len(dataset_test)} patches.")

    analysis = tune.ExperimentAnalysis(str(experiment_path))
    best_result = analysis.get_best_trial("loss", "min")
    best_checkpoint = best_result.checkpoint
    model_path = Path(best_checkpoint.path) / "checkpoint.pt"
    print(f"Best checkpoint path: {model_path}")

    prediction_config = PredictionConfig(
        calculate_residuals=True,
        return_numpy=True,
        save_predictions=False,
        return_loss=True,
        device="cpu",
        verbose=False,
    )

    dataloader_config = DataLoaderConfig(
        batch_size=10,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        device="cpu",
        multiprocessing_context=None,
    )

    test_loss = predict_monthly_var(
        model=model_path,
        dataset=dataset_test,
        dataloader_config=dataloader_config,
        prediction_config=prediction_config,
        run_dir=run_dir,
    )

    print("Test loss:")
    print(test_loss)


if __name__ == "__main__":
    main()
