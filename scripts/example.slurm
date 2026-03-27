#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --account=bd0854
#SBATCH --output=eso4clima_%j.out

# Activate the virtual environment
# Change the path to your virtual environment
source /home/b/b383704/eso4clima/ClimaNet/.venv/bin/activate

# Run the training script
# Change the path to your training script
python /home/b/b383704/eso4clima/train_twoyears/example_training.py

echo "==== Slurm accounting summary ${SLURM_JOB_ID} ===="
sstat --allsteps -j "$SLURM_JOB_ID" \
  --format=JobID,NTasks,AveCPU,AveRSS,MaxRSS,MaxVMSize,TresUsageInAve,TresUsageInMax \
  --parsable2