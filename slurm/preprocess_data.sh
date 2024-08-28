#!/bin/bash

# Job parameters
#SBATCH --job-name=data_process_niva
#SBATCH --time=24:00:00
#SBATCH --output=data_process_niva%j.out
#SBATCH --error=data_process_niva%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Parameters
PYTHON_VENV_PATH="/home/jrisse/venv-niva"
PYTHON_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/scripts"

# Activate python virtual environment
source $PYTHON_VENV_PATH/bin/activate

# Run preprocessing
python3 $PYTHON_SCRIPT_DIR/main_preprocessing.py
