import argparse
from pathlib import Path

import ray
import xarray as xr

from climanet.dataset import DataLoaderConfig, STDataset
from climanet.predict import PredictionConfig, predict_monthly_var
from climanet.utils import configure_compute_resources, load_model, read_st_data, set_seed


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
        "--train-dir",
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
    train_dir = Path(args.train_dir).resolve()
    run_dir = Path(args.run_dir).resolve()

    # set the random seed for reproducibility
    set_seed()

    # Load the trained model for prediction
    model_path = train_dir / "best_model.pth"
    model = load_model(model_path, device)
    model_patch_size = model.config["patch_size"]

    # Build dataset for training and validation
    lsm_file_path = lsm_dir / "era5_lsm_bool.nc"
    lsm_mask = xr.open_dataset(lsm_file_path)["lsm"]  # make sure is dask array

    predict_year = 2022
    data = read_st_data(data_path=f"{prepared_data_dir}/{predict_year}", var_name=var_name)
    input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features = zip(*data)

    monthly_shape = monthly_da.shape[1:]  # the whole dataset
    crop_size = (1, *monthly_shape)

    dataset_test = STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        monthly_da=monthly_da,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        land_mask=lsm_mask,
        crop_size=crop_size,
        stride=None,  # no overlap for prediction
        model_patch_size=model_patch_size,
        sh_embed_dim=96,
        sh_order_L=10,
        verbose=False,
        load_lazy=False,  # load all data into memory for prediction 1 year
    )

    # Build the dataloader config
    dataloader_num_workers = 10  # adjust if needed
    use_cuda = device == "cuda"
    dataloader_config = DataLoaderConfig(
        batch_size=100, # adjust if OOM issue
        shuffle=False,  # for prediction and reconstruction, it should be False
        num_workers=dataloader_num_workers,
        pin_memory=use_cuda,
        persistent_workers=True,
        device=device,
        multiprocessing_context=None,  # one year fits in memory
    )

    # move the model to GPU and configure compute resources
    model = configure_compute_resources(
        model,
        device=device,
        compute_threads=None,  # on gpu, it is not used
        dataloader_num_workers=dataloader_num_workers
    )

    # Training configuration# create prediction config
    prediction_config = PredictionConfig(
        calculate_residuals=True,
        return_numpy=False,
        save_predictions=True,
        return_loss=False,
        device=device,
        verbose=False,
        store_logs=False,
    )

    predictions = predict_monthly_var(
        model=model,
        dataset=dataset_test,
        dataloader_config=dataloader_config,
        prediction_config=prediction_config,
        run_dir=run_dir,
    )
