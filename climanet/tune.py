from pathlib import Path

import ray
from ray.tune.schedulers import ASHAScheduler

from climanet.dataset import DataLoaderConfig, STDataset
from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.train import TrainConfig, train_monthly_model
from climanet.utils import read_st_data, set_seed


def _tune_data_preparation(data_config):

    # read zarr data
    input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features = (
        read_st_data(
            data_path=data_config["input_data_dir"], var_name=data_config["var_name"]
        )
    )

    return STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        monthly_da=monthly_da,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        land_mask=ray.get(data_config["land_mask_data"]),
        patch_size=data_config["patch_size"],  # based on the patch_size in model
        stride=data_config["stride"],
        sh_embed_dim=96,
        sh_order_L=10,
        verbose=False,
        load_lazy=data_config["load_lazy"],
    )


def _train(tune_config, static_args):
    """Helper function to train the model with Ray Tune."""
    run_dir = static_args["run_dir"]

    data_config_train = static_args.get("data_config_train")
    dataset_train = _tune_data_preparation(data_config_train)

    data_config_validation = static_args.get("data_config_validation")
    dataset_validation = _tune_data_preparation(data_config_validation)

    use_cuda = static_args["device"] == "cuda"
    dataloader_config = DataLoaderConfig(
        batch_size=tune_config["batch_config"]["batch_size"],
        shuffle=True,
        num_workers=static_args["dataloader_num_workers"],
        pin_memory=use_cuda,
        persistent_workers=static_args["dataloader_persistent_workers"],
        device=static_args["device"],
        multiprocessing_context=static_args["dataloader_multiprocessing_context"],
    )

    training_config = TrainConfig(
        calculate_residuals=True,
        num_epoch=static_args["num_epoch"],
        patience=10,
        accumulation_steps=tune_config["batch_config"]["accumulation_steps"],
        optimizer_lr=tune_config["optimizer_lr"],
        device=static_args["device"],
        verbose=False,
        verbose_epoch_interval=20,
        tune_checkpoint=True,
        store_model=False,
    )

    set_seed()

    patch_size = tune_config["patch_size"]
    overlap = tune_config["overlap"]
    embed_dim = tune_config["embed_dim"]
    dropout = tune_config["dropout"]
    hidden = tune_config["hidden"]
    spatial_depth = tune_config["spatial_depth"]
    spatial_heads = tune_config["spatial_heads"]
    model = SpatioTemporalModel(
        patch_size=(1, patch_size, patch_size),
        overlap=overlap,
        embed_dim=embed_dim,
        dropout=dropout,
        hidden=hidden,
        spatial_depth=spatial_depth,
        spatial_heads=spatial_heads,
    )

    _ = train_monthly_model(
        model=model,
        dataset_train=dataset_train,
        dataloader_config=dataloader_config,
        training_config=training_config,
        dataset_validation=dataset_validation,
        run_dir=run_dir,
    )


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
            run_config=ray.tune.RunConfig(
                storage_path=static_args["run_dir"], name=experiment_name
            ),
        )

    results = tuner.fit()
    return results
