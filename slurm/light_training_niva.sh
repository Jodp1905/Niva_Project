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
#SBATCH --nodelist=x440-05

# Parameters
PYTHON_VENV_PATH="/home/jrisse/venv-niva"
PYTHON_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/scripts"

# Activate python virtual environment
source $PYTHON_VENV_PATH/bin/activate

# Set environment variables
export N_FOLDS_TO_RUN=1
export NUM_EPOCHS=1

# Run training
if [ ! -z "$SLURM_JOB_ID" ]; then
    slurm_jobid=$SLURM_JOB_ID
else
    slurm_jobid=$(date +%s)
fi
training_name="light_training_${slurm_jobid}"
python3 $PYTHON_SCRIPT_DIR/training.py $training_name
