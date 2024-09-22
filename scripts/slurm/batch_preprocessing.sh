#!/bin/bash

# Job parameters
#SBATCH --job-name=data_process_niva
#SBATCH --time=24:00:00
#SBATCH --output=data_process_niva%j.out
#SBATCH --error=data_process_niva%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive

PROJECT_DIR="${HOME}"

# Path Parameters
PYTHON_VENV_PATH="${PROJECT_DIR}/venv-niva"
PYTHON_SCRIPT_DIR="${PROJECT_DIR}/niva/Niva_Project/src/training" # Has to be set for slurm jobs
NSIGHT_PATH="${PROJECT_DIR}/software/Nsys_2024.5.1/target-linux-x64"
LUSTRE_LLITE_DIR="/mnt/lustre-stats/llite"
export DARSHAN_LIBPATH="${PROJECT_DIR}/software/darshan-3.4.5/darshan-runtime/install/lib/libdarshan.so"
DARSHAN_LOGDIR="${PROJECT_DIR}/niva/darshan_logs"
NSIGHT_LOGDIR="${PROJECT_DIR}/niva/nsight_logs"

# Tracing parameters
ENABLE_TRACING=1
TRACING_TOOL="darshan" # darshan or nsight

# Tracing tool choice are both cannot be used at the same time
if [ "$ENABLE_TRACING" -eq 1 ]; then
    if [ "$TRACING_TOOL" == "darshan" ]; then
        echo "Tracing enabled: Darshan"
    elif [ "$TRACING_TOOL" == "nsight" ]; then
        echo "Tracing enabled: Nsight"
    else
        echo "Invalid tracing tool: $TRACING_TOOL"
        echo "  tracing_tool: 'darshan' or 'nsight'"
        exit 1
    fi
else
    echo "Tracing disabled"
fi

# Darshan Parameters
export DARSHAN_EXCLUDE_DIRS="${PYTHON_VENV_PATH}" # Exclude Python virtual environment
export DARSHAN_MODMEM=20000                       # 20 GB of memory allowed for Darshan instrumentation
export DARSHAN_ENABLE_NONMPI=1                    # Enable Darshan for non-MPI applications (Python)

# Modules setup (for lab machines)
module load Intel-oneAPI-HPC-Toolkit/mpi/latest || true

# Check environment variables
# niva project data root directory
if [ -z "$NIVA_PROJECT_DATA_ROOT" ]; then
    echo "Error: Environment variable NIVA_PROJECT_DATA_ROOT is not set."
    echo "Please set it to the root directory of ai4boundary project data containing the 'sentinel2' directory."
    exit 1
else
    echo "Project data root directory: ${NIVA_PROJECT_DATA_ROOT}"
fi
# darshan library path
if [ "$ENABLE_TRACING" -eq 1 ] && [ "$TRACING_TOOL" == "darshan" ]; then
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

# Tracing directory
if [ "$ENABLE_TRACING" -eq 1 ]; then
    run_name="preprocessing_traced_${suffix}"
    if [ "$TRACING_TOOL" == "darshan" ]; then
        export DARSHAN_LOGFILE="${DARSHAN_LOGDIR}/${run_name}.darshan"
        echo "Darshan log file: ${DARSHAN_LOGFILE}"
    elif [ "$TRACING_TOOL" == "nsight" ]; then
        export NSIGHT_LOGS_DIR="${NSIGHT_LOGDIR}"
    fi
fi

# Preprocessing execution
preprocessing_script_name="main_preprocessing.py"
preprocessing_path=$(realpath "${PYTHON_SCRIPT_DIR}/${preprocessing_script_name}")
# with tracing
if [ "$ENABLE_TRACING" -eq 1 ]; then
    # nsight
    if [ "$TRACING_TOOL" == "nsight" ]; then
        cmd=$(
            cat <<EOF
${NSIGHT_PATH}/nsys profile \
--enable storage_metrics,\
--lustre-volumes=all,\
--lustre-llite-dir="${LUSTRE_LLITE_DIR}" \
--output="${NSIGHT_LOGS_DIR}/${run_name}" \
--python-sampling=false \
python3 \
"${preprocessing_path}"
EOF
        )
    # darshan
    elif [ "$TRACING_TOOL" == "darshan" ]; then
        cmd=$(
            cat <<EOF
env LD_PRELOAD="${DARSHAN_LIBPATH}" \
python3 \
"${preprocessing_path}"
EOF
        )
    fi
# without tracing
else
    cmd="python3 ${preprocessing_path}"
fi

echo "Executing command :"
echo "${cmd}"
eval "${cmd}"
echo "Preprocessing completed."
