#!/bin/bash
#SBATCH --job-name=data_preparation
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --account=bd0854
#SBATCH --output=data_preparation_%j.out

set -euo pipefail
ulimit -s 204800

# Activate uv env
UV_ENV="/home/b/b383704/eso4clima/ClimaNet/.venv"
source "$UV_ENV/bin/activate"

# Set the scratch directory because they are avialable to all nodes
SCRATCH="/home/b/b383704/eso4clima/data/preprocessed_data"

# data directory (adjust this path to your data location)
DATA_DIR="/work/bd0854/b380103/eso4clima/output/sst/concatenated/"

python -u /home/b/b383704/eso4clima/data_preparation/data_preparation.py \
    --storage-path "$SCRATCH" \
    --data-dir "$DATA_DIR"

echo "Data preparation is done!"
