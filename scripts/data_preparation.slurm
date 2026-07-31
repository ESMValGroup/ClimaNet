#!/bin/bash
#SBATCH --job-name=data_preparation
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH --account=bd0854
#SBATCH --output=data_preparation_%j.out

set -euo pipefail
ulimit -s 204800

# Activate uv env
UV_ENV="$HOME/climanet_py314"
source "$UV_ENV/bin/activate"

# Set the scratch directory because they are avialable to all nodes
SCRATCH="/scratch/b/$USER/data_preparation/hourly"

# data directory (adjust this path to your data location)
DATA_DIR="$HOME/scratch/b/$USER/data"

python -u $HOME/ClimaNet/scripts/data_preparation.py \
    --storage-path "$SCRATCH" \
    --data-dir "$DATA_DIR"
