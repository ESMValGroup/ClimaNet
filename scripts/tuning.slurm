#!/bin/bash
#SBATCH --job-name=tuning
#SBATCH --partition=g
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=bd0854
#SBATCH --output=eso4clima_%j.out

set -e
ulimit -s 204800

source ~/ClimaNet/.venv/bin/activate

python -u ~/ClimaNet/scripts/tuning.py