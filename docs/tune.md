#
## Hyperparameter tuning on HPC

In ClimaNet, we implement the hyperparameter tuning using [Ray
Tune](https://docs.ray.io/en/latest/tune/index.html). An example is implemented
in a Jupyter notebook, which can be found in the `notebooks` directory.

But here we provide some practical tips on hyperparameter tuning on HPC using
GPU nodes. The scripts including configs for hyperparameter tuning can be found
in the `scripts` directory.

### Data preparation

For hyperparameter tuning, we use three subsets of train/validation/test data.
The train and validation data are used in training workflow to calculate the
loss and validation metrics. We use continuous data for example year 2020 for
training, 2021 for validation, and 2022 for testing. The test data is used to
evaluate the model performance after training. The test data is not used in
training or validation. It is common that after hyperparameter tuning, the model
is retrained, and then evaluated on a test data. See the python script
`tuning.py` in the `scripts` directory.

### Static configuration of Ray Tune

Here an example of static configuration of Ray Tune. This parameters are not
used in the hyperparameter tuning. The parameters are set in the `tuning.py`
script in the `scripts` directory:

```python
static_args = {
        "max_num_epochs": 100,
        "num_trials": 50,  # this is num_samples in ray.tune.TuneConfig
        "cpu_per_trial": 10,
        "gpu_per_trial": 1,
        "run_dir": args.storage_path,
        "device": "cuda",
        "dataloader_num_workers": 4,
        "data_config_train": data_config_train,
        "data_config_validation": data_config_validation,
        "num_epoch": 100,
        "max_concurrent_trials": args.num_nodes * 2,  # less than GPUs per node (4) avoid OOM
        "experiment_name": "climanet_tune",
    }
```

- We dont use ray object store for data but we pass the data config to each
  trial. This is because the data is large and we want to avoid OOM. The data is
  loaded in each trial from disk.

- The `max_num_epochs` is set to be 100. This is the maximum number of epochs
  for each trial. The training will stop if the model converges before reaching
  this number. This is set based on our experiments. You can adjust this number
  based on your data and model.
 The `num_trials` is set to be 50. This is the number of trials for
  hyperparameter tuning. You can adjust this number based on your data and
  model. But for deep learning models, this is usually enough to explore the
  hyperparameter space.
- The `cpu_per_trial` is set to be 10 and `gpu_per_trial` is set to be 1. When
  using Ray Tune with multiple nodes, it is recommended to use 1 GPU for model
  training.
- The `run_dir` should be set to `SCRATCH` directory where nodes have access to,
  and it is used to store the results of each trial. Remember after tuning, to
  copy the results to project because the SCRATCH directory has usually an
  expiration date.
- The `dataloader_num_workers` is set to be 4. This should not be too large to
  avoid resource overload.
- The `max_concurrent_trials` is set to be less than the number of GPUs per node
  to avoid OOM. In this example, we have 4 GPUs per node, but we set
  `max_concurrent_trials` to 2 per node.

### Search space configuration of Ray Tune

Here an example of search space configuration of Ray Tune. This parameters are
used in the hyperparameter tuning. The parameters are set in the `tuning.py`
script in the `scripts` directory:

```python

tune_config = {
        "patch_size": tune.choice([2, 4, 8]),
        "overlap": tune.choice([0, 1, 2]),
        "embed_dim": tune.choice([32, 64, 128]),
        "dropout": tune.choice([0.0, 0.1, 0.2]),
        "hidden": tune.choice([32, 64, 128]),
        "spatial_depth": tune.choice([1, 2, 3]),
        "spatial_heads": tune.choice([2, 4, 8]),
        "optimizer_lr": tune.loguniform(1e-3, 1e-1),
        "batch_config": tune.grid_search([
            {"batch_size": 100, "accumulation_steps": 1},
            {"batch_size": 200, "accumulation_steps": 2},
            {"batch_size": 400, "accumulation_steps": 2},
        ]),
    }
```

We set the largest value in each parameter based on our experiments and
available GPU memory. The smallest value is set based on the minimum value that
we think is reasonable for the model. You can adjust these values based on your
data and model.

### UV environment

We use UV to create a virtual environment where `ClimaNet` and its dependencies
are installed. Due to known multiprocessing cleanup race in the newest versions
of python, pytorch multiprocessing/DataLoader workers and ray, you may get some
warnings when multiprocessing is done in background. To avoid this, you can use
a virtual environment with python 3.11. So in slurm script in the `scripts`
directory, you see something like because we want to specify which python
version to use:

```bash
UV_ENV="$HOME/climanet_py314"
```

### Slurm script

A sbatch script is provided in the `scripts` directory. You can adjust the
parameters in the script based on your data and model. The script will submit a
job to the HPC cluster and run the hyperparameter tuning using the python script
`tuning.py` in the `scripts` directory. The script will use the static
configuration and search space configuration specified in the `tuning.py`
script.

The script will also save the results of each trial in the `run_dir` specified
in the static configuration.

The SBATCH parameters are according to the [DKRZ Levante
limits](https://docs.dkrz.de/doc/levante/running-jobs/partitions-and-limits.html)
and [GPU
specifications](https://docs.dkrz.de/doc/levante/running-jobs/using-gpu-nodes.html).
You can adjust these parameters based on your cluster.

### Ray Tune distributed

For setting Ray Tune distributed, checkout this [Ray slurm
example](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm-template.html#slurm-template).
We dont use `ray symmetric-run` because it has issues with getting node ip
addresses. Instead, we use `ray start --head` on the first node and `ray start
--address` on the other nodes. We also set a few environment variables related
to timeout for different components of Ray. The environment variables are set in
the slurm script in the `scripts` directory.
