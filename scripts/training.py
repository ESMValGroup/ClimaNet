import argparse
from pathlib import Path

import xarray as xr

from climanet.dataset import DataLoaderConfig, STDataset
from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.train import TrainConfig, train_monthly_model
from climanet.utils import read_st_data, set_seed


# Best hyperparameters from tuning experiments (README).
BEST_PATCH_SIZE = 8
BEST_OVERLAP = 1
BEST_EMBED_DIM = 64
BEST_DROPOUT = 0.2
BEST_HIDDEN = 32
BEST_SPATIAL_DEPTH = 3
BEST_SPATIAL_HEADS = 2
BEST_OPTIMIZER_LR = 0.001787422899066508
BEST_ACCUMULATION_STEPS = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train ClimaNet on prepared yearly Zarr data using explicit year splits "
            "(e.g. 3 years train, 1 year validation)."
        )
    )
    parser.add_argument(
        "--prepared-data-dir",
        type=Path,
        required=True,
        help="Directory that contains one subdirectory per year with prepared Zarr files.",
    )
    parser.add_argument(
        "--lsm-file-path",
        type=Path,
        required=True,
        help="Path to land-sea mask NetCDF file.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("./runs").resolve(),
        help="Directory where logs/checkpoints are stored.",
    )
    parser.add_argument(
        "--var-name",
        type=str,
        default="tos",
        help="Variable name used in prepared Zarr files.",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        default=[2018, 2019, 2020],
        help="Years to concatenate for training.",
    )
    parser.add_argument(
        "--validation-year",
        type=int,
        default=2021,
        help="Year to use for validation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training.",
    )
    parser.add_argument("--num-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--accumulation-steps", type=int, default=BEST_ACCUMULATION_STEPS
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument(
        "--dataloader-persistent-workers",
        action="store_true",
        help="Enable persistent dataloader workers.",
    )
    return parser


def _read_year_data(
    prepared_data_dir: Path, year: int, var_name: str
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    year_dir = prepared_data_dir / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Prepared year folder does not exist: {year_dir}")
    return read_st_data(data_path=year_dir, var_name=var_name)


def _concat_years(
    prepared_data_dir: Path, years: list[int], var_name: str
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    year_data = [_read_year_data(prepared_data_dir, year, var_name) for year in years]
    input_da = xr.concat([x[0] for x in year_data], dim="M")
    input_da_nan_mask = xr.concat([x[1] for x in year_data], dim="M")
    monthly_da = xr.concat([x[2] for x in year_data], dim="M")
    padded_days_mask = xr.concat([x[3] for x in year_data], dim="M")
    time_features = xr.concat([x[4] for x in year_data], dim="M")
    return input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features


def _build_dataset(
    prepared_data_dir: Path,
    years: list[int],
    var_name: str,
    land_mask: xr.DataArray,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int],
    load_lazy: bool,
) -> STDataset:
    input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features = (
        _concat_years(prepared_data_dir, years, var_name)
    )
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
        verbose=True,
        load_lazy=load_lazy,
    )


def main() -> None:
    args = build_parser().parse_args()

    prepared_data_dir = args.prepared_data_dir.resolve()
    lsm_file_path = args.lsm_file_path.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if not prepared_data_dir.exists():
        raise FileNotFoundError(
            f"Prepared data directory does not exist: {prepared_data_dir}"
        )
    if not lsm_file_path.exists():
        raise FileNotFoundError(f"LSM file does not exist: {lsm_file_path}")

    all_years = [*args.train_years, args.validation_year]
    if len(set(all_years)) != len(all_years):
        raise ValueError(
            "Year splits must be distinct. Check train-years/validation-year."
        )

    model_patch_size = (1, BEST_PATCH_SIZE, BEST_PATCH_SIZE)
    num_patches = (10, 10)
    spatial_patch_size = (
        model_patch_size[1] * num_patches[0],
        model_patch_size[2] * num_patches[1],
    )
    dataset_patch_size = (1, *spatial_patch_size)
    stride = (spatial_patch_size[0] // 5, spatial_patch_size[1] // 5)

    set_seed()
    lsm_mask = xr.open_dataset(lsm_file_path)["lsm"]

    print(f"Train years: {args.train_years}")
    print(f"Validation year: {args.validation_year}")

    dataset_train = _build_dataset(
        prepared_data_dir=prepared_data_dir,
        years=args.train_years,
        var_name=args.var_name,
        land_mask=lsm_mask,
        patch_size=dataset_patch_size,
        stride=stride,
        load_lazy=True,
    )
    dataset_validation = _build_dataset(
        prepared_data_dir=prepared_data_dir,
        years=[args.validation_year],
        var_name=args.var_name,
        land_mask=lsm_mask,
        patch_size=dataset_patch_size,
        stride=stride,
        load_lazy=True,
    )

    print(f"Train dataset patches: {len(dataset_train)}")
    print(f"Validation dataset patches: {len(dataset_validation)}")

    model = SpatioTemporalModel(
        patch_size=model_patch_size,
        overlap=BEST_OVERLAP,
        embed_dim=BEST_EMBED_DIM,
        dropout=BEST_DROPOUT,
        hidden=BEST_HIDDEN,
        spatial_depth=BEST_SPATIAL_DEPTH,
        spatial_heads=BEST_SPATIAL_HEADS,
    )

    dataloader_config = DataLoaderConfig(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.dataloader_persistent_workers,
        device=args.device,
        multiprocessing_context="spawn",
    )

    training_config = TrainConfig(
        calculate_residuals=True,
        num_epoch=args.num_epoch,
        patience=10,
        accumulation_steps=args.accumulation_steps,
        optimizer_lr=BEST_OPTIMIZER_LR,
        device=args.device,
        verbose=False,
        verbose_epoch_interval=20,
        tune_checkpoint=False,
        store_model=True,
        store_logs=True,
    )

    print("Starting training...")
    _ = train_monthly_model(
        model=model,
        dataset_train=dataset_train,
        dataloader_config=dataloader_config,
        training_config=training_config,
        dataset_validation=dataset_validation,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
