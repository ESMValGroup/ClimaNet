# Scripts

## Structure

- `data_preparation.*`: Scripts for preparing the data for training, tuning and evaluation. Mainly converting large netCDF files to Zarr storage with specific chunking strategies. This allows executing training for larger-than-memory datasets.
- `example_training.*`: example training script
- `tuning.*`: Scripts for hyperparameter tuning.
- `run_best_tuned_model.*`: Scripts for running the best tuned model on the test set.
- `logs`:
  - `eso4clima_24438134_subset.out`: example SLURM job output file of an execution on a subset of the global dataset. The dataset has two years of data (2020-2021) and the spatial coverage is from 30S to 30N and from 30W to 30E.
  - `eso4clima_24449471_full.out`: example SLURM job output file of an execution on the full dataset, two years of data (2020-2021) and almost global coverage (from 80S to 80N and from 179.99W to 179.99E). The training only executed for 1 hour and cuted off by SLURM time limit.

## Experiments

### Tuning experiments

- datasplit: train set = 2020, validation set = 2021, test set = 2022
- path of tuning results: `/work/<account_id>/eso4clima/tune/`.
- test loss: 0.036662004509047774
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

 - Use the best hyperparameters found in the tuning experiments to train the model on the training set and evaluate it on the test set.
 - Use three years 2018-2020 for training, 2021 for validation and 2022 for testing.
 - Prepared data is stored in `/work/<account_id>/eso4clima/preprocessed/sst/`.
 - Load one year data following example in `run_best_tuned_model.py` script. Then concatenate the three years as `xr.concat([da_2018, da_2019, da_2020], dim="M")`.
 - In dataloader, use `load_lazy=True`.
  

