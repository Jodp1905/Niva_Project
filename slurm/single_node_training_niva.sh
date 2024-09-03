#!/bin/bash

PYTHON_VENV_PATH="/home/jrisse/venv-niva"
PYTHON_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/scripts"
SLURM_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/slurm"
PMI_PATH="/usr/lib/libpmi.so"

PORT=2222

# Get list of nodes
nodes=($(scontrol show hostnames))
num_nodes=${#nodes[@]}

# Get node hostname and IP address
my_hostname=$(hostname)
my_ip=$(hostname -I | awk '{print $1}')

# Construct TF worker list
worker_list=()
my_index=0
for node in "${nodes[@]}"; do
    if [[ "$node" == "$my_hostname" ]]; then
        node_ip=$my_ip
        my_index=${#worker_list[@]}
    else
        node_ip=$(getent hosts $node | awk '{print $1}')
    fi
    worker_list+=("\"$node_ip:$PORT\"")
done

# Build the TF_CONFIG environment variable from slurm parameters
worker_list_string=$(
    IFS=,
    echo "${worker_list[*]}"
)
TF_CONFIG=$(
    cat <<EOF
{
    "cluster": {
        "worker": [${worker_list_string[@]}]
    },
    "task": {"type": "worker", "index": $my_index}
}
EOF
)
export TF_CONFIG
echo "TF_CONFIG for node $my_hostname (index $my_index, IP $my_ip):"
echo $TF_CONFIG | jq .

# Activate python virtual environment
source $PYTHON_VENV_PATH/bin/activate
sleep 10

# Python training environment variables
export TRAINING_TYPE="MultiWorker"
export N_FOLDS_TO_RUN=1

# Run the training script
slurm_jobid=$SLURM_JOB_ID
training_name="distributed_training_${slurm_jobid}"
echo "Starting training $training_name on node $my_hostname"
python3 $PYTHON_SCRIPT_DIR/training.py $training_name
echo "Training $training_name completed on node $my_hostname"
