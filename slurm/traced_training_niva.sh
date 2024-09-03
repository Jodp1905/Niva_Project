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
NSIGHT_PATH="/home/jrisse/software/Nsys_2024.5.1/target-linux-x64"
NSIGHT_LOGS_DIR="/home/jrisse/nsight_logs"
LUSTRE_LLITE_DIR="/mnt/lustre-stats/llite"

# Tracing tool choice are both cannot be used at the same time
if [ -z "$1" ]; then
  echo "Usage: $0 <tracing_tool>"
  echo "  tracing_tool: 'darshan' or 'nsight'"
  exit 1
fi
tracing_tool=$1
if [ "$tracing_tool" != "darshan" ] && [ "$tracing_tool" != "nsight" ]; then
  echo "Invalid tracing tool: $tracing_tool"
  echo "  tracing_tool: 'darshan' or 'nsight'"
  exit 1
fi

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
export TZ="Europe/Paris"
date_str=$(date +%m%d%Y-%H%M)
if [ ! -z "$SLURM_JOB_ID" ]; then
  suffix="${SLURM_JOB_ID}_${date_str}"
else
  suffix="${date_str}"
fi

run_name="traced_training_${suffix}"
niva_project_data_root_sanitized="${NIVA_PROJECT_DATA_ROOT%/}"
output_dir="${niva_project_data_root_sanitized}/model/${run_name}"
output_dir=$(realpath "${output_dir}")
mkdir -p "${output_dir}"
echo "Output directory: ${output_dir}"

# Run training with Darshan tracing
training_script_name="training.py"
echo "Executing training with Darshan tracing for ${run_name}."
training_path=$(realpath "${PYTHON_SCRIPT_DIR}/${training_script_name}")
export DARSHAN_LOGFILE="${DARSHAN_LOGS_DIR}/${run_name}.darshan"

# Execute training with Darshan tracing or Nsight profiling
# Using a HERE document to evaluate the command
if [ "$tracing_tool" == "nsight" ]; then
  cmd=$(
    cat <<EOF
${NSIGHT_PATH}/nsys profile \
--enable storage_metrics,\
--lustre-volumes=all,\
--lustre-llite-dir="${LUSTRE_LLITE_DIR}" \
--output="${NSIGHT_LOGS_DIR}/${run_name}" \
python3 \
"${training_path}" \
"${run_name}"
EOF
  )
elif [ "$tracing_tool" == "darshan" ]; then
  cmd=$(
    cat <<EOF
env LD_PRELOAD="${DARSHAN_LIBPATH}" \
python3 \
"${training_path}" \
"${run_name}"
EOF
  )
fi

echo "Executing command with ${tracing_tool}:"
echo "${cmd}"
eval "${cmd}"

# Darshan logdir shenanigans
if [ "$tracing_tool" == "darshan" ]; then
  echo "Darshan logs stored under ${DARSHAN_LOGS_DIR}/${run_name}.darshan"
else
  echo "Nsight logs stored under ${NSIGHT_LOGS_DIR}/${run_name}.nsys-rep"
fi
