#!/bin/bash
#SBATCH --job-name=tuning
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --account=bd0854
#SBATCH --output=tuning_%j.out

set -euo pipefail
ulimit -s 204800

# Activate uv env with python 3.11
UV_ENV="$HOME/climanet_py311"
source "$UV_ENV/bin/activate"

# Set the scratch directory because they are avialable to all nodes
SCRATCH="/scratch/b/$USER/tune_results"

# data directory (adjust this path to your data location)
DATA_DIR="$HOME/scratch/b/$USER/data"
LSM_DIR="$HOME/scratch/b/$USER/data"

# Get node list and ip
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

port=6379
head_node_ip="${head_node}:${port}"
export head_node_ip

echo "Ray head: $head_node $head_node_ip"

# Wait for all nodes to be ready
echo "Waiting for nodes to boot..."
sleep 180
echo "Checking nodes..."
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" hostname

# based on https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#ray-slurm-deploy
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    ray symmetric-run \
    --address "${head_node_ip}" \
    --num-cpus=128 \
    --num-gpus=4 \
    --object-store-memory 5000000000 \
    --temp-dir "$SCRATCH/ray_logs" \
    -- \
    python -u $HOME/ClimaNet/scripts/tuning.py \
        --storage-path "$SCRATCH" \
        --num-nodes "$SLURM_NNODES" \
        --ray-address "auto" \
        --data-dir "$DATA_DIR" \
        --lsm-dir "$LSM_DIR"
