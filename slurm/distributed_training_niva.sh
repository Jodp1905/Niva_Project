#!/bin/bash
#
#
#SBATCH --job-name=distributed_training_niva
#SBATCH --time=120:00:00
#SBATCH --output=training_niva_distributed_%j.out
#SBATCH --error=training_niva_distributed_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 32

# Get allocated nodes in the job
nodes=($(scontrol show hostnames))
num_nodes=${#nodes[@]}
my_hostname=$(hostname)
my_ip=$(getent hosts $my_hostname | awk '{ print $1 }')

# Construct worker list
port=2222
worker_list=()
for node in "${nodes[@]}"; do
    node_ip=$(getent hosts $node | awk '{ print $1 }')
    worker_list+=("$node_ip:$port")
done

# Determine the node's index in the cluster
my_index=0
for i in "${!nodes[@]}"; do
    if [[ "$my_hostname" == "${nodes[$i]}" ]]; then
        my_index=$i
        break
    fi
done

# Build the TF_CONFIG environment variable
TF_CONFIG=$(
    cat <<EOF
{
    "cluster": {
        "worker": ["${worker_list[@]}"]
    },
    "task": {"type": "worker", "index": $my_index}
}
EOF
)

export TF_CONFIG
echo "TF_CONFIG for node $my_hostname (index $my_index): $TF_CONFIG"

exit 0

# Activate python virtual environment
source /home/jrisse/activate_venv.sh

# Run training
python3 /home/jrisse/niva/Niva_Project/scripts/training.py full_dataset_26_08_2024
