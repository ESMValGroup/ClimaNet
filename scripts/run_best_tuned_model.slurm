#!/bin/bash
#SBATCH --job-name=climanet_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --account=bd0854
#SBATCH --partition=compute
#SBATCH --output=climanet_eval_%j.out
#SBATCH --error=climanet_eval_%j.err

set -euo pipefail

source /home/b/b383704/eso4clima/ClimaNet/.venv/bin/activate

python -u /home/b/b383704/eso4clima/run_best_tuned_model/run_best_tuned_model.py \
  --experiment-path /work/bd0854/eso4clima/tune/sst_01 \
  --test-data-dir /work/bd0854/b380103/eso4clima/output/sst/concatenated/ \
  --lsm-file-path /home/b/b383704/eso4clima/data/era5_lsm_bool.nc \
  --run-dir /home/b/b383704/eso4clima/run_best_tuned_model/run_dir \
  --var-name tos \
  --year 2022

printf "\nFinished evaluation run.\n"
