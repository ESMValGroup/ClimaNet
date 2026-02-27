# Execute training tasks on SLURM

1. Make a working directory

```sh
mkdir training
cd training
```

2. Clone this repo
```sh
git clone git@github.com:ESMValGroup/ClimaNet.git
```

3. Install uv for dependency management. Se [uv doc](https://docs.astral.sh/uv/getting-started/installation/).

4. Create a venv and install Python dependencies using uv
```sh
cd ClimaNet
```

```
uv sync
```

A `.venv` dir will appear

5. Copy the python script and slurm script into the working dir:

```sh
cp ClimaNet/scripts/example* .
```

6. Config `example.slurm`, in the `source ...` line, make sure the venv just created is activated.
   Note that the account is the ESO4CLIMA project account, which is shared by multiple users.

7. Config `example.py`, make sure the path of input data and land mask data is correct.

8. Execute the SLURM job
```sh
sbatch example.slurm
```