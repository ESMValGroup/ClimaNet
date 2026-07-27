import ray
import xarray as xr

from ray.tune.schedulers import ASHAScheduler
from climanet.dataset import STDataset
from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.train import train_monthly_model
from climanet.utils import set_seed

from pathlib import Path


def _train(tune_config, static_args):
    """Helper function to train the model with Ray Tune."""

    device = static_args["device"]
    dataloader_num_workers = static_args["dataloader_num_workers"]

    run_dir = static_args["run_dir"]
    num_epoch = static_args["num_epoch"]

    # dont use ray.put() and ray.get() (i.e. object store) when data is large
    train_dataset = tune_data_preparation(static_args["data_config_train"])
    validation_dataset = tune_data_preparation(static_args["data_config_validation"])

    patch_size = tune_config["patch_size"]
    overlap = tune_config["overlap"]
    embed_dim = tune_config["embed_dim"]
    dropout = tune_config["dropout"]
    hidden = tune_config["hidden"]
    spatial_depth = tune_config["spatial_depth"]
    spatial_heads = tune_config["spatial_heads"]

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
    input_data = xr.open_mfdataset(
        data_config["input_filenames"], chunks=data_config.get("input_chunks")
    )
    monthly_data = xr.open_mfdataset(
        data_config["monthly_filenames"], chunks=data_config.get("monthly_chunks")
    )
    lsm_mask = xr.open_dataset(
        data_config["landmask_filename"], chunks=data_config.get("landmask_chunks")
    )

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


def run_tune(tune_config: dict, static_args: dict):
    """Run Ray Tune to find the best hyperparameters for the model.

    Args:
        tune_config: dictionary containing the hyperparameters to tune and their
            ranges and other config parameters.
    """
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=static_args["max_num_epochs"],
        grace_period=1,
        reduction_factor=2,
    )

    experiment_name = static_args["experiment_name"]
    experiment_path = f"{static_args['run_dir']}/{experiment_name}"
    if Path(experiment_path).exists():
        tuner = ray.tune.Tuner.restore(
            experiment_path,
            resume_errored=True,
        )
    else:
        tuner = ray.tune.Tuner(
            ray.tune.with_resources(
                ray.tune.with_parameters(_train, static_args=static_args),
                resources={
                    "cpu": static_args["cpu_per_trial"],
                    "gpu": static_args["gpu_per_trial"],
                },
            ),
            tune_config=ray.tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=static_args["num_trials"],
                max_concurrent_trials=static_args["max_concurrent_trials"],
            ),
            param_space=tune_config,
            run_config=ray.tune.RunConfig(storage_path=static_args["run_dir"], name=experiment_name),
        )

    results = tuner.fit()
    return results
