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

# Activate python virtual environment
source /home/jrisse/activate_venv.sh

# Run preprocessing
# python3 /home/jrisse/niva/Niva_Project/scripts/main_preprocessing.py

# Set environment variables
export NUM_EPOCHS=1

# Run training
slurm_jobid=$SLURM_JOB_ID
training_name="light_training_${slurm_jobid}"
python3 /home/jrisse/niva/Niva_Project/scripts/training.py $training_name