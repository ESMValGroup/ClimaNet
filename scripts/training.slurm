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

source /home/b/b383704/eso4clima/ClimaNet/.venv/bin/activate

# Prepared yearly zarr folders, e.g. /work/<account_id>/eso4clima/preprocessed/sst/
PREPARED_DATA_DIR="/home/b/b383704/eso4clima/data/preprocessed_data/"
LSM_FILE_PATH="/home/b/b383704/eso4clima/data/era5_lsm_bool.nc"
RUN_DIR="/home/b/b383704/eso4clima/train_three_years/run"

python -u /home/b/b383704/eso4clima/ClimaNet/scripts/training.py \
  --prepared-data-dir "$PREPARED_DATA_DIR" \
  --lsm-file-path "$LSM_FILE_PATH" \
  --run-dir "$RUN_DIR" \
  --var-name tos \
  --train-years 2018 2019 2020 \
  --validation-year 2021 \
  --device cuda \
  --batch-size 100 \
  --dataloader-num-workers 32 \
  --dataloader-persistent-workers

echo "==== Slurm accounting summary ${SLURM_JOB_ID} ===="
sstat --allsteps -j "$SLURM_JOB_ID" \
  --format=JobID,NTasks,AveCPU,AveRSS,MaxRSS,MaxVMSize,TresUsageInAve,TresUsageInMax \
  --parsable2