#!/bin/bash
#SBATCH --job-name=eso4clima
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=01:00:00
#SBATCH --account=bd0854

source /home/b/b383704/eso4clima/ClimaNet/.venv/bin/activate

python /home/b/b383704/eso4clima/train_twoyears/example.py

