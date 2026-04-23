#!/bin/bash
#SBATCH --job-name=eso4clima
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:30:00
#SBATCH --account=bd0854
#SBATCH --output=eso4clima_%j.out

source /home/b/b383704/eso4clima/ClimaNet/.venv/bin/activate

# Run the training script
python -u /home/b/b383704/eso4clima/train_twoyears/example_training.py

echo "==== Slurm accounting summary ${SLURM_JOB_ID} ===="
sstat --allsteps -j "$SLURM_JOB_ID" \
  --format=JobID,NTasks,AveCPU,AveRSS,MaxRSS,MaxVMSize,TresUsageInAve,TresUsageInMax \
  --parsable2