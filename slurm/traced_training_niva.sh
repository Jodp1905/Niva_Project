#!/bin/bash

# Job parameters
#SBATCH --job-name=training_niva
#SBATCH --time=24:00:00
#SBATCH --output=training_niva_traced_%j.out
#SBATCH --error=training_niva_traced_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Parameters
PYTHON_VENV_PATH="/home/jrisse/venv-niva"
PYTHON_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/scripts" # Has to be set for slurm jobs
DARSHAN_LOGS_DIR="/home/jrisse/darshan_logs"

# Training parameters
export N_FOLDS_TO_RUN=1
export NUM_EPOCHS=1

# Darshan Parameters
export DARSHAN_EXCLUDE_DIRS="${PYTHON_VENV_PATH}" # Exclude Python virtual environment
export DARSHAN_MODMEM=20000                       # 20 GB of memory allowed for Darshan instrumentation
export DARSHAN_ENABLE_NONMPI=1                    # Enable Darshan for non-MPI applications (Python)

# Modules setup (for lab machines)
module load Intel-oneAPI-HPC-Toolkit/mpi/latest || true

# Check environment variables
if [ -z "$NIVA_PROJECT_DATA_ROOT" ]; then
  echo "Error: Environment variable NIVA_PROJECT_DATA_ROOT is not set."
  echo "Please set it to the root directory of ai4boundary project data containing the 'sentinel2' directory."
  exit 1
else
  echo "Project data root directory: ${NIVA_PROJECT_DATA_ROOT}"
fi

if [ -z "$DARSHAN_LIBPATH" ]; then
  echo "Error: Environment variable DARSHAN_LIBPATH is not set."
  echo "Please set it to the path of the Darshan library (libdarshan.so)."
  exit 1
elif [ ! -f "$DARSHAN_LIBPATH" ]; then
  echo "Error: DARSHAN_LIBPATH is not a valid file: $DARSHAN_LIBPATH"
  exit 1
else
  echo "Darshan library found at ${DARSHAN_LIBPATH}"
fi

# Activate Python virtual environment
source "${PYTHON_VENV_PATH}/bin/activate"

# Setup output directory
job_id=$SLURM_JOB_ID
run_name="trace_training_${job_id}"
niva_project_data_root_sanitized="${NIVA_PROJECT_DATA_ROOT%/}"
output_dir="${niva_project_data_root_sanitized}/model/${run_name}"
output_dir=$(realpath "${output_dir}")
if [ -d "${output_dir}" ]; then
  echo "Directory already exists: ${output_dir}. Cleaning up."
  rm -r "${output_dir}"
fi
mkdir -p "${output_dir}"
echo "Output directory: ${output_dir}"

# Run training with Darshan tracing
training_script_name="training.py"
echo "Executing training with Darshan tracing for ${run_name}."
training_path=$(realpath "${PYTHON_SCRIPT_DIR}/${training_script_name}")
env LD_PRELOAD="${DARSHAN_LIBPATH}" python3 "${training_path}" "${run_name}"

# Logdir shenanigans
day=$(date +%-d)
month=$(date +%-m)
year=$(date +%-Y)
darshan_logdir="${DARSHAN_LOGS_DIR}/${year}/${month}/${day}"
echo "Darshan logs stored in ${darshan_logdir}"
