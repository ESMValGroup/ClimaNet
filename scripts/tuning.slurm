#!/bin/bash
#SBATCH --job-name=tuning
#SBATCH --partition=gpu
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --account=bd0854
#SBATCH --output=tuning_%j.out

set -euo pipefail
ulimit -s 204800

# Activate uv env
UV_ENV="$HOME/.venv"
source "$UV_ENV/bin/activate"

# Set the scratch directory because they are avialable to all nodes
SCRATCH="/scratch/b/$USER/tune_results"

# data directory (adjust this path to your data location)
DATA_DIR="$HOME/scratch/b/$USER/data"
LSM_DIR="$HOME/scratch/b/$USER/data"

# Get node list
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

# First node is the Ray head
head_node="${nodes[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')

echo "Ray head: $head_node_ip"

# Start Ray head on the first node
# 6379 default port for Ray’s Redis/GCS service
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -lc "
        source '$UV_ENV/bin/activate' &&
        export OMP_NUM_THREADS=1 &&
        ray start --head \
        --node-ip-address=$head_node_ip \
        --port=6379 \
         --num-cpus=$SLURM_CPUS_PER_TASK \
         --num-gpus=4 \
         --block
    " &

sleep 15

# Start Ray workers on the remaining nodes
for node in "${nodes[@]:1}"; do
    srun --nodes=1 --ntasks=1 -w "$node" \
        bash -lc "
            source '$UV_ENV/bin/activate' &&
            export OMP_NUM_THREADS=1 &&
            ray start --address=$head_node_ip:6379 \
            --num-cpus=$SLURM_CPUS_PER_TASK \
            --num-gpus=4 \
            --block
        " &
done

sleep 15

echo "=== Ray sanity check ==="
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -lc "source '$UV_ENV/bin/activate' && python - <<'PY'
import os
import ray

address = os.environ.get('RAY_ADDRESS', '${head_node_ip}:6379')
ray.init(address=address, ignore_reinit_error=True)

resources = ray.available_resources()
print('Address:', address)
print('CPU:', resources.get('CPU'))
print('GPU:', resources.get('GPU'))
print('Resources:', resources)

ray.shutdown()
PY"

sleep 15

# Run the tuning script on the head node
python -u ~/scripts/tuning.py \
    --storage-path "$SCRATCH" \
    --num-nodes "$SLURM_NNODES" \
    --ray-address "$head_node_ip:6379" \
    --data-dir "$DATA_DIR" \
    --lsm-dir "$LSM_DIR"
