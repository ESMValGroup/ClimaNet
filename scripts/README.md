# Scripts

## Structure

- `data_preparation.*`: Scripts for preparing the data for training, tuning and evaluation and saving them to Zarr storage with specific chunking strategies.
- `tuning.*`: Scripts for hyperparameter tuning.
- `training.*`: Scripts for training the model with the best hyperparameters found in the tuning experiments.
- `prediction.*`: Scripts for running predictions using the trained model.

## Experiments

### Tuning experiments for SST variable

- data split for training: train set = 2018-2020, validation set = 2021, test set = 2022
- data split for tuning: train set = 2018, validation set = 2019
- path of tuning results: `/work/<account_id>/eso4clima/tune/sst_01`.

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
