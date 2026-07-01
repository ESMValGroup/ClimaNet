#

## Using Jupyter Hub

Jupyterhub allows to run the Jupyter notebook directly on the DKRZ HPC system
Levante. JupyterHub is available at https://jupyterhub.dkrz.de for all DKRZ
users who have access to Levante and who are allowed to submit batch jobs.

Follow the step below to create a Jupyter kernel for ClimaNet. This only needs
to be done once, and the kernel will be available for all future Jupyter
sessions.

1. Install uv:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

2. Create a new conda environment and install ipykernel and climanet:

```bash
cd ClimaNet
uv venv
source .venv/bin/activate

uv sync
uv add ipykernel
```

3. Make the just created conda environment available as a notebook kernel by
   running the following command:

```bash
python -m ipykernel install --user --name climanet
```

4. Go to https://jupyterhub.dkrz.de and log in. Click the Start button in the
   column `Advanced` and select the project account and job specifications like
   node, cpu and gpus.

5. When opening a notebook or starting a new empty notebook, select the kernel
called "climanet". Stop the jupyter lab server when you're done with it to avoid
wasting computational resources. To do this, click File and the Hub Control
Panel and then click the red Stop button.

## Using slurm

See [Running Jobs with Slurm](https://docs.dkrz.de/doc/levante/running-jobs/index.html).