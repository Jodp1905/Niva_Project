#!/bin/bash

# Job parameters
#SBATCH --job-name=training_niva
#SBATCH --time=12:00:00
#SBATCH --output=training_niva-%j.out
#SBATCH --error=training_niva-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Parameters
PYTHON_VENV_PATH="/home/jrisse/venv-niva"
PYTHON_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/scripts" # has to be set for slurm jobs
DARSHAN_LOGS_DIR="/home/jrisse/darshan_logs"
DARSHAN_MODMEM=20000
USE_MPI=false

usage() {
  echo "Usage: $(basename $0) [-b] [-t] [--with_darshan_dir <dir>] <run_name>"
  echo "  -b    Build dataset option"
  echo "  -t    Trace training execution"
  echo "  --with_darshan_dir  Specify the Darshan install directory"
  echo "  <run_name>  Name of the run, used for output files"
  exit 1
}

# Initialize flags
build_flag=false
trace_flag=false
custom_darshan_dir=false

# Custom argument parsing
while [[ "$#" -gt 0 ]]; do
  case "$1" in
  -b)
    build_flag=true
    shift
    ;;
  -t)
    trace_flag=true
    shift
    ;;
  --with_darshan_dir)
    if [[ -n "$2" && "$2" != -* ]]; then
      DARSHAN_DIR="$2"
      custom_darshan_dir=true
      shift 2
    else
      echo "Invalid argument: $2"
      usage
    fi
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

# Check for run name argument
if [ "$#" -ne 1 ]; then
  echo "Error: Exactly one argument is required."
  usage
fi
run_name="$1"

# Output parameters
if [ -z "$NIVA_PROJECT_DATA_ROOT" ]; then
  echo "Error: Environment variable NIVA_PROJECT_DATA_ROOT is not set."
  exit 1
fi
niva_project_data_root_sanitized="${NIVA_PROJECT_DATA_ROOT%/}"
output_dir="${niva_project_data_root_sanitized}/model/${run_name}"
output_dir=$(realpath "${output_dir}")
if [ -d "${output_dir}" ]; then
  echo "Directory already exists: ${output_dir}. Cleaning up."
  rm -r "${output_dir}"
fi
mkdir -p "${output_dir}"
echo "Output directory: ${output_dir}"

# Activate python virtual environment
source "${PYTHON_VENV_PATH}/bin/activate"

# Create dataset if build flag is set
if $build_flag; then
  echo "Build flag is set, creating dataset."
  python3 "${PYTHON_SCRIPT_DIR}/main_preprocessing.py"
  echo "Dataset created."
fi

# Run training
if $trace_flag; then
  # Prepare Darshan environment

  # Custom Darshan directory
  if [ "$custom_darshan_dir" = true ]; then
    echo "Using custom Darshan directory: ${DARSHAN_DIR}"
    darshan_lib_path="${DARSHAN_DIR}/lib/libdarshan.so"

  # Module loaded Darshan library
  else
    echo "Using module loaded Darshan library."
    module load Intel-oneAPI-HPC-Toolkit/mpi/latest
    module load darshan/3.4.4/darshan-runtime
    module load darshan/3.4.4/darshan-util
    darshan_install_path=$(dirname $(dirname $(which darshan-config)))
    darshan_lib_path="${darshan_install_path}/lib/libdarshan.so"
  fi

  # Verify Darshan library path
  if [ ! -f "${darshan_lib_path}" ]; then
    echo "Error: libdarshan.so not found at ${darshan_lib_path}"
    exit 1
  else
    echo "Darshan library path: ${darshan_lib_path}"
  fi

  # Set Darshan environment variables
  export DARSHAN_EXCLUDE_DIRS="${PYTHON_VENV_PATH}"
  export DARSHAN_MODMEM="${DARSHAN_MODMEM}"
  darshan_log_dir="${DARSHAN_LOGS_DIR}/${run_name}"
  mkdir -p "${darshan_log_dir}"
  export DARSHAN_LOGPATH=$darshan_log_dir
  echo "Darshan log directory: ${darshan_log_dir}"
  if [ "$USE_MPI" = "false" ]; then
    export DARSHAN_ENABLE_NONMPI=1
  fi

  # Run training with Darshan tracing
  echo "Trace flag is set, executing training with Darshan tracing for ${run_name}."
  training_path=$(realpath "${PYTHON_SCRIPT_DIR}/training.py")
  echo "Training script path: ${training_path} with preloaded Darshan library."
  env LD_PRELOAD="${darshan_lib_path}" python3 "${training_path}" "${run_name}"

else
  # Run training without Darshan tracing
  echo "Executing training for ${run_name}."
  python3 ${PYTHON_SCRIPT_DIR}/training.py "${run_name}"
fi
