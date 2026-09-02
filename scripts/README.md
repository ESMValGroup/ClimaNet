# Scripts

## Structure

- `data_preparation.*`: Scripts for preparing the data for training, tuning and evaluation and saving them to Zarr storage with specific chunking strategies.
- `tuning.*`: Scripts for hyperparameter tuning.
- `run_best_tuned_model.*`: Scripts for running the best tuned model on the test set.
- `training.*`: Scripts for training the model with the best hyperparameters found in the tuning experiments.

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

### Training experiments

Use the best hyperparameters found in the tuning experiments to train the model
on the training set; three years 2018-2020 for training, 2021 for validation in
the training loop. Because three years of hourly data is too large to fit into
memory, we use `load_lazy` option in the dataset. This makes the training
process slower, but allows us to train on larger-than-memory datasets. The
training is done for 100 epochs, and the best model is saved based on the
validation loss. Note that the training is done only on one GPU node, including
4 GPUs.

The results are stored at `/work/<account_id>/eso4clima/train/sst_01/`.

