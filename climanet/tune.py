from pathlib import Path

import ray
from ray.air.config import CheckpointConfig
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
        crop_size=data_config["crop_size"],  # based on the patch_size in model
        stride=data_config["stride"],
        model_patch_size=data_config["model_patch_size"],
        sh_embed_dim=96,
        sh_order_L=10,
        verbose=False,
        load_lazy=data_config["load_lazy"],
    )


def _train(tune_config, static_args):
    """Helper function to train the model with Ray Tune."""
    run_dir = static_args["run_dir"]
    patch_size = tune_config["patch_size"]

    set_seed()

    # Prepare train dataset
    data_config_train = static_args.get("data_config_train")
    data_config_train["model_patch_size"] = (1, patch_size, patch_size)
    dataset_train = _tune_data_preparation(data_config_train)

    # Prepare validation dataset
    data_config_validation = static_args.get("data_config_validation")
    data_config_validation["model_patch_size"] = (1, patch_size, patch_size)
    dataset_validation = _tune_data_preparation(data_config_validation)

    # Prepare dataloader configuration
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

    # Prepare training configuration
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
        store_logs=False,
    )

    # Set the model
    embed_dim = tune_config["embed_dim"]
    dropout = tune_config["dropout"]
    hidden = tune_config["hidden"]
    model = SpatioTemporalModel(
        patch_size=(1, patch_size, patch_size),
        embed_dim=embed_dim,
        dropout=dropout,
        hidden=hidden,
    )

    # Train the model
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
            ray.tune.with_resources(
                ray.tune.with_parameters(_train, static_args=static_args),
                resources={
                    "cpu": static_args["cpu_per_trial"],
                    "gpu": static_args["gpu_per_trial"],
                },
            ),
            experiment_path,
            ray.tune.with_resources(
                ray.tune.with_parameters(_train, static_args=static_args),
                resources={
                    "cpu": static_args["cpu_per_trial"],
                    "gpu": static_args["gpu_per_trial"],
                },
            ),
            resume_errored=True,
            resume_unfinished=True,
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
                storage_path=static_args["run_dir"],
                name=experiment_name,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="loss",
                    checkpoint_score_order="min",
                ),
            ),
        )

    results = tuner.fit()
    return results
