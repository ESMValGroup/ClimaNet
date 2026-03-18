#!/bin/bash
#SBATCH --job-name=eso4clima
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=01:00:00
#SBATCH --account=bd0854

source /home/b/b383704/eso4clima/ClimaNet/.venv/bin/activate

# Run the training script
python /home/b/b383704/eso4clima/train_twoyears/example_training.py

# After training, you can run the inference script to generate predictions
# python /home/b/b383704/eso4clima/train_twoyears/example_inference.py

