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

# Darshan Parameters
export DARSHAN_EXCLUDE_DIRS="${PYTHON_VENV_PATH}" # Exclude Python virtual environment
export DARSHAN_MODMEM=20000                       # 20 GB of memory allowed for Darshan instrumentation
export DARSHAN_ENABLE_NONMPI=1                    # Enable Darshan for non-MPI applications (Python)

# Modules setup (for lab machines)
module load Intel-oneAPI-HPC-Toolkit/mpi/latest || true

usage() {
  echo "Usage: traced_training.sh [-b] <run_name>"
  echo "  -b    Build dataset option"
  echo "  <run_name>  Name of the run, used for output files"
  echo ""
  echo "Required environment variables:"
  echo "  NIVA_PROJECT_DATA_ROOT  Root directory for project data"
  echo "  DARSHAN_LIBPATH  Path to libdarshan.so, used for preloading Python scripts"
  exit 1
}

# Initialize flags
build_flag=false

# Parse options
while [[ "$#" -gt 0 ]]; do
  case "$1" in
  -b)
    build_flag=true
    shift
    ;;
  -*)
    echo "Invalid option: $1"
    usage
    ;;
  *)
    break
    ;;
  esac
done

# Get required arguments
if [ "$#" -ne 1 ]; then
  echo "Error: Exactly 1 argument required."
  usage
fi
run_name="$1"

# Check environment variables
if [ -z "$NIVA_PROJECT_DATA_ROOT" ]; then
  echo "Error: Environment variable NIVA_PROJECT_DATA_ROOT is not set."
  exit 1
else
  echo "Project data root directory: ${NIVA_PROJECT_DATA_ROOT}"
fi

if [ -z "$DARSHAN_LIBPATH" ]; then
  echo "Error: Environment variable DARSHAN_LIBPATH is not set."
  exit 1
elif [ ! -f "$DARSHAN_LIBPATH" ]; then
  echo "Error: DARSHAN_LIBPATH is not a valid file: $DARSHAN_LIBPATH"
  exit 1
else
  echo "Darshan library found at ${DARSHAN_LIBPATH}"
fi

# Activate Python virtual environment
source "${PYTHON_VENV_PATH}/bin/activate"

# Create dataset if build flag is set
if $build_flag; then
  echo "Build flag is set, creating dataset."
  python3 "${PYTHON_SCRIPT_DIR}/main_preprocessing.py"
  echo "Dataset created."
fi

# Setup output directory
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
