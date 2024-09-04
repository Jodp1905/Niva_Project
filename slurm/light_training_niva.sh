#!/bin/bash
#
#
#SBATCH --job-name=training_niva
#SBATCH --time=120:00:00
#SBATCH --output=training_niva_light_%j.out
#SBATCH --error=training_niva_light_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 32

# Parameters
PYTHON_VENV_PATH="/home/jrisse/venv-niva"
PYTHON_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/scripts"

# Activate python virtual environment
source $PYTHON_VENV_PATH/bin/activate

# Set environment variables
export FOLD_LIST="0"
export NUM_EPOCHS=1

# Run training
export TZ="Europe/Paris"
date_str=$(date +%m%d%Y-%H%M)
if [ ! -z "$SLURM_JOB_ID" ]; then
    suffix="${SLURM_JOB_ID}_${date_str}"
else
    suffix="${date_str}"
fi
training_name="light_training_${suffix}"
python3 $PYTHON_SCRIPT_DIR/training.py $training_name
