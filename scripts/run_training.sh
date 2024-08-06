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
  echo "Usage: $(basename $0) [-b] [-t] <run_name>"
  echo "  -b    Build dataset option"
  echo "  -t    Trace training execution"
  echo "  <run_name>  Name of the run, used for output files"
  exit 1
}

# Initialize flags
build_flag=false
trace_flag=false

# Parse command line options
while getopts ':bt' opt; do
  case "$opt" in
  b)
    build_flag=true
    ;;
  t)
    trace_flag=true
    ;;
  :)
    echo "Option -$OPTARG requires an argument."
    usage
    ;;
  ?)
    echo "Invalid option: -$OPTARG"
    usage
    ;;
  esac
done
shift "$((OPTIND - 1))"

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
  module load Intel-oneAPI-HPC-Toolkit/mpi/latest
  module load darshan/3.4.4/darshan-runtime
  module load darshan/3.4.4/darshan-util
  echo $LD_LIBRARY_PATH
  darshan_install_path=$(dirname $(dirname $(which darshan-config)))
  darshan_lib_path="${darshan_install_path}/lib/libdarshan.so"
  if [ ! -f "${darshan_lib_path}" ]; then
    echo "Error: Darshan library not found at ${darshan_lib_path}"
    exit 1
  else
    echo "Darshan library path: ${darshan_lib_path}"
  fi
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
  echo "Training script path: ${training_path}"
  env LD_PRELOAD="${darshan_lib_path}" python3 "${training_path}" "${run_name}"

else
  # Run training without Darshan tracing
  echo "Executing training for ${run_name}."
  python3 ${PYTHON_SCRIPT_DIR}/training.py "${run_name}"
fi
