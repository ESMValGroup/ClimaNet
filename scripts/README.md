# Scripts

## Structure

- `data_preparation.*`: Scripts for preparing the data for training, tuning and evaluation and saving them to Zarr storage with specific chunking strategies.
- `tuning.*`: Scripts for hyperparameter tuning.
- `run_best_tuned_model.*`: Scripts for running the best tuned model on the test set.

## Experiments

### Tuning experiments for SST variable

- datasplit: train set = 2020, validation set = 2021, test set = 2022
- path of tuning results: `/work/<account_id>/eso4clima/tune/sst_01`.
- test loss: 0.036662004509047774 (K)
- hyperparameters of the best model:
  ```
    {'patch_size': 8,
    'overlap': 1,
    'embed_dim': 64,
    'dropout': 0.2,
    'hidden': 32,
    'spatial_depth': 3,
    'spatial_heads': 2,
    'optimizer_lr': 0.001787422899066508,
    'batch_config': {'batch_size': 100, 'accumulation_steps': 2}}
 ```
