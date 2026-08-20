#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --account=bd0854
#SBATCH --output=training_%j.out

set -euo pipefail
ulimit -s 204800

# Activate uv env
UV_ENV="$HOME/climanet_py314"
source "$UV_ENV/bin/activate"

# Set the scratch directory because they are avialable to all nodes
RUN_DIR="/scratch/b/$USER/train"

# data directory (adjust this path to your data location)
PREPARED_DATA_DIR="/scratch/b/$USER/data"
TUNE_DIR="/scratch/b/$USER/tune"
LSM_DIR="/scratch/b/$USER/data"

echo "Starting training.py script..."
python -u $HOME/ClimaNet/scripts/training.py \
  --run-dir "$RUN_DIR" \
  --num-nodes "$SLURM_NNODES" \
  --prepared-data-dir "$PREPARED_DATA_DIR" \
  --tune-dir "$TUNE_DIR" \
  --lsm-dir "$LSM_DIR"

echo "**********Training script completed.************"
