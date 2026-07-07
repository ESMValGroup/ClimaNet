from pathlib import Path
import tempfile

from ray import tune
from ray.tune.schedulers import ASHAScheduler
import torch

from climanet.predict import predict_monthly_var
from climanet.st_encoder_decoder import SpatioTemporalModel
from climanet.train import train_monthly_model
from climanet.utils import configure_compute_resources, save_model


def _train(tune_config):

    device = tune_config["device"]
    compute_threads = tune_config["compute_threads"]
    dataloader_num_workers = tune_config["dataloader_num_workers"]
    train_dataset = tune_config["train_dataset"]
    validation_dataset = tune_config["validation_dataset"]
    patch_size = tune_config["patch_size"]
    overlap = tune_config["overlap"]
    embed_dim = tune_config["embed_dim"]
    dropout = tune_config["dropout"]
    hidden = tune_config["hidden"]
    spatial_depth = tune_config["spatial_depth"]
    spatial_heads = tune_config["spatial_heads"]
    run_dir = tune_config["run_dir"]
    num_epoch = tune_config["num_epoch"]

    model = SpatioTemporalModel(
        patch_size=patch_size,
        overlap=overlap,
        embed_dim=embed_dim,
        dropout=dropout,
        hidden=hidden,
        spatial_depth=spatial_depth,
        spatial_heads=spatial_heads,
    )

    model = configure_compute_resources(
        model,
        device=device,
        compute_threads=compute_threads,
        dataloader_num_workers=dataloader_num_workers
    )

    batch_size = tune_config["batch_size"]
    accumulation_steps = tune_config["accumulation_steps"]
    optimizer_lr=tune_config["optimizer_lr"]

    trained_model = train_monthly_model(
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


def run_tune(
        tune_config: dict,
    ):

    max_t = tune_config["max_num_epochs"]
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=max_t,
        grace_period=1,
        reduction_factor=2)

    num_samples = tune_config["num_trials"]
    tuner = tune.Tuner(
        _train,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=tune_config,
    )
    results = tuner.fit()
    return results


def test_best_model(results, test_dataset, run_dir):
    best_result = results.get_best_result("loss", "min")
    best_checkpoint = best_result.checkpoint
    with best_checkpoint.as_directory() as best_checkpoint_dir:
        model_state = torch.load(best_checkpoint_dir / "checkpoint.pt")
        model_config = best_result.config
        model = SpatioTemporalModel(**model_config)
        model.load_state_dict(model_state)

    batch_size = best_result.config["batch_size"]
    device = best_result.config["device"]
    dataloader_num_workers = best_result.config["dataloader_num_workers"]

    _, test_loss = predict_monthly_var(
        model,
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
