#!/bin/bash
#SBATCH --job-name=tuning
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=20
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

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

node_ip=$(srun --nodes=1 --ntasks=1 --nodelist="$head_node" hostname -i)

port=6379
head_node_ip="${node_ip}:${port}"
export head_node_ip

echo "Ray head: $head_node $head_node_ip"

# Wait for all nodes to be ready
echo "Waiting for nodes to boot..."
sleep 60
echo "Checking nodes..."
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" hostname

# based on https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#ray-slurm-deploy
export RAY_raylet_start_wait_time_s=600
export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=900
export RAY_gcs_server_request_timeout_seconds=600
export RAY_health_check_initial_delay_ms=30000
export RAY_GRACEFUL_SHUTDOWN_DRAIN_TIMEOUT_S=900
export RAY_gcs_server_port_wait_time_s=600
export RAY_gcs_rpc_server_connect_timeout_s=600

# Clean up any existing Ray files on all nodes
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    /usr/bin/bash -c 'rm -rf /dev/shm/ray* 2>/dev/null || true'

# Start Ray head on the first node
# 6379 default port for Ray’s Redis/GCS service
echo "Starting HEAD at $head_node $head_node_ip"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    /usr/bin/bash -c "
        source '$UV_ENV/bin/activate' &&
        ray start --head \
        --node-ip-address=$node_ip \
        --port="${port}" \
        --num-cpus=128 \
        --num-gpus=4 \
        --object-store-memory 5000000000 \
        --temp-dir "$SCRATCH/ray_logs" \
        --block
    " &

sleep 30

# Start Ray workers on the remaining nodes
for node in "${nodes_array[@]:1}"; do
    echo "Starting worker on node: $node"

    srun --nodes=1 --ntasks=1 -w "$node" \
        /usr/bin/bash -c "
            source '$UV_ENV/bin/activate' &&
            ray start --address=$head_node_ip \
            --num-cpus=128 \
            --num-gpus=4 \
            --object-store-memory 5000000000 \
            --temp-dir "$SCRATCH/ray_logs" \
            --block
        " &

    echo "Finished starting worker on node: $node"
    sleep 10
done

sleep 10

echo "Starting tuning.py script..."
python -u $HOME/ClimaNet/scripts/tuning.py \
    --storage-path "$SCRATCH" \
    --num-nodes "$SLURM_NNODES" \
    --ray-address "$head_node_ip" \
    --data-dir "$DATA_DIR" \
    --lsm-dir "$LSM_DIR"
