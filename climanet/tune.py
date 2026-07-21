from pathlib import Path

import ray
import xarray as xr

from ray.tune.schedulers import ASHAScheduler
from climanet.dataset import STDataset
from climanet.predict import predict_monthly_var
from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.train import train_monthly_model
from climanet.utils import set_seed


def _train(tune_config):
    """Helper function to train the model with Ray Tune."""

    device = tune_config["device"]
    dataloader_num_workers = tune_config["dataloader_num_workers"]
    train_dataset = ray.get(tune_config["train_dataset"])
    validation_dataset = ray.get(tune_config["validation_dataset"])
    patch_size = tune_config["patch_size"]
    overlap = tune_config["overlap"]
    embed_dim = tune_config["embed_dim"]
    dropout = tune_config["dropout"]
    hidden = tune_config["hidden"]
    spatial_depth = tune_config["spatial_depth"]
    spatial_heads = tune_config["spatial_heads"]
    run_dir = tune_config["run_dir"]
    num_epoch = tune_config["num_epoch"]

    set_seed()

    model = SpatioTemporalModel(
        patch_size=(1, patch_size, patch_size),
        overlap=overlap,
        embed_dim=embed_dim,
        dropout=dropout,
        hidden=hidden,
        spatial_depth=spatial_depth,
        spatial_heads=spatial_heads,
    )

    batch_size = tune_config["batch_config"]["batch_size"]
    accumulation_steps = tune_config["batch_config"]["accumulation_steps"]
    optimizer_lr = tune_config["optimizer_lr"]

    _ = train_monthly_model(
        model,
        train_dataset,
        validation_dataset=validation_dataset,
        batch_size=batch_size,
        num_epoch=num_epoch,
        accumulation_steps=accumulation_steps,
        optimizer_lr=optimizer_lr,
        device=device,
        run_dir=run_dir,
        dataloader_num_workers=dataloader_num_workers,
        store_model=False,
        verbose=False,
        tune_checkpoint=True,
    )


def tune_data_preparation(data_config: dict, is_hourly=True) -> STDataset:
    """Prepare the data for training and validation."""
    input_data = xr.open_mfdataset(data_config["input_filenames"], chunks=data_config.get("input_chunks"))
    monthly_data = xr.open_mfdataset(data_config["monthly_filenames"], chunks=data_config.get("monthly_chunks"))
    lsm_mask = xr.open_dataset(data_config["landmask_filename"], chunks=data_config.get("landmask_chunks"))

    # calculate residuals as target
    input_data_averaged = input_data.resample(time="MS").mean(skipna=True)
    input_data_averaged["time"] = monthly_data["time"]

    # Residuals
    monthly_data_res = monthly_data - input_data_averaged

    var_name = data_config["var_name"]

    dataset = STDataset(
        input_da=input_data[var_name],
        monthly_da=monthly_data_res[var_name],
        land_mask=lsm_mask["lsm"],
        patch_size=data_config["patch_size"],
        stride=data_config["stride"],
        sh_embed_dim=96,
        sh_order_L=10,
        is_hourly=is_hourly,
    )
    return dataset


def run_tune(tune_config: dict):
    """Run Ray Tune to find the best hyperparameters for the model.

    Args:
        tune_config: dictionary containing the hyperparameters to tune and their
            ranges and other config parameters.
    """
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=tune_config["max_num_epochs"],
        grace_period=1,
        reduction_factor=2,
    )

    tuner = ray.tune.Tuner(
        ray.tune.with_resources(
            ray.tune.with_parameters(_train),
            resources={
                "cpu": tune_config["cpu_per_trial"],
                "gpu": tune_config["gpu_per_trial"],
            },
        ),
        tune_config=ray.tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=tune_config["num_trials"],
            max_concurrent_trials=tune_config["max_concurrent_trials"],
        ),
        param_space=tune_config,
        run_config=ray.tune.RunConfig(storage_path=tune_config["run_dir"]),
    )
    results = tuner.fit()
    return results


def check_best_model(experiment_path: str | Path, test_dataset: STDataset, run_dir: str | Path):
    """Test the best model from a Ray Tune experiment.

    Args:
        experiment_path: path to the Ray Tune experiment directory
        test_dataset: Dataset object containing the test data
        run_dir: directory to save logs and model
    Returns:
        test_loss: the loss on the test dataset

    """
    if not ray.is_initialized():
        ray.init()
    analysis = ray.tune.ExperimentAnalysis(experiment_path)
    best_result = analysis.get_best_trial("loss", "min")

    batch_size = best_result.config["batch_size"]
    device = best_result.config["device"]
    dataloader_num_workers = best_result.config["dataloader_num_workers"]
    best_checkpoint = best_result.checkpoint

    model_path = Path(best_checkpoint.path) / "checkpoint.pt"

    _, test_loss = predict_monthly_var(
        model_path,
        test_dataset,
        batch_size=batch_size,
        device=device,
        return_numpy=False,
        save_predictions=False,
        return_loss=True,
        verbose=False,
        run_dir=run_dir,
        dataloader_num_workers=dataloader_num_workers,
    )
    return test_loss
