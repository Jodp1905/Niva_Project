#!/bin/bash
#
#
#SBATCH --job-name=distributed_multiworker_training_niva
#SBATCH --time=120:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 32
#SBATCH --exclusive

LOG_DIR="/home/jrisse/niva/slurm_${SLURM_JOB_ID}_multiworker_${SLURM_JOB_NUM_NODES}nodes_logs"
mkdir -p $LOG_DIR

SLURM_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/slurm"

srun --output=$LOG_DIR/training_niva_distributed_%j_%N_%t.out \
    --error=$LOG_DIR/training_niva_distributed_%j_%N_%t.out \
    bash $SLURM_SCRIPT_DIR/distributed_multiworker_single_node.sh
