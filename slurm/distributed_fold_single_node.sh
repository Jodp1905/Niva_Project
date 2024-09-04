#!/bin/bash

PYTHON_VENV_PATH="/home/jrisse/venv-niva"
PYTHON_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/scripts"
SLURM_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/slurm"
NUM_FOLDS=10

# Activate python virtual environment
source $PYTHON_VENV_PATH/bin/activate

# Python training environment variables
my_index=$SLURM_NODEID
my_hostname=$(hostname)
echo "Node $my_hostname is running training on fold configuration $my_index"
export FOLD_LIST="$my_index"

# Run the training script
slurm_jobid=$SLURM_JOB_ID
training_name="distributed_training_${slurm_jobid}_fold_${my_index}"
echo "Starting training $training_name on node $my_hostname"
python3 $PYTHON_SCRIPT_DIR/training.py $training_name
echo "Training $training_name completed on node $my_hostname"
