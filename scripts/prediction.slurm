#!/bin/bash
#SBATCH --job-name=prediction
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH --account=bd0854
#SBATCH --output=prediction_%j.out

set -euo pipefail
ulimit -s 204800

# Activate uv env
UV_ENV="$HOME/climanet_py314"
source "$UV_ENV/bin/activate"

# Set the scratch directory because they are avialable to all nodes
RUN_DIR="/scratch/b/$USER/predict"

# data directory (adjust this path to your data location)
PREPARED_DATA_DIR="/scratch/b/$USER/data"
TRAIN_DIR="/scratch/b/$USER/train"
LSM_DIR="/scratch/b/$USER/data"
RAW_DATA_FOLDER="/scratch/b/$USER/output/sst/concatenated"

echo "Starting prediction.py script..."
python -u $HOME/ClimaNet/scripts/prediction.py \
  --run-dir "$RUN_DIR" \
  --prepared-data-dir "$PREPARED_DATA_DIR" \
  --train-dir "$TRAIN_DIR" \
  --lsm-dir "$LSM_DIR"\
  --raw-data-folder "$RAW_DATA_FOLDER"

echo "**********Prediction script completed.************"
