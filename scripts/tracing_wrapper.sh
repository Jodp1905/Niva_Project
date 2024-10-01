#!/bin/bash

# Arguments:
# $1: Tracing tool (either 'darshan' or 'nsight')
# $2: Command to execute (e.g., python or bash script) and its arguments
TRACING_TOOL=$1
shift
EXEC_COMMAND=$@

# Path Parameters
PROJECT_DIR=$HOME
PYTHON_VENV_PATH="${PROJECT_DIR}/venv-niva"
NSIGHT_PATH="${PROJECT_DIR}/software/nsight_2024.6.0/target-linux-x64"
LUSTRE_LLITE_DIR="/mnt/lustre-stats/llite"
DARSHAN_LIBPATH="${PROJECT_DIR}/software/darshan-3.4.5/darshan-runtime/install/lib/libdarshan.so"
DARSHAN_LOGDIR="${PROJECT_DIR}/niva/darshan_logs"
NSIGHT_LOGDIR="${PROJECT_DIR}/niva/nsight_logs"

# Tracing Parameters
ENABLE_TRACING=1

# Darshan DXT (Darshan eXtended Tracing) generates more detailed I/O traces
DARSHAN_DXT=0

# Nsight NVTX (NVIDIA Tools Extension) enables profiling with NVTX annotations
# Sampled batch training steps from the Python script will be captured
# nsight_batch_profiling needs to be enabled in yaml configuration file
NSIGHT_NVTX=1

# Tracing tool choice check
if [ "$TRACING_TOOL" == "darshan" ]; then
    echo "Tracing enabled: Darshan"
elif [ "$TRACING_TOOL" == "nsight" ]; then
    echo "Tracing enabled: Nsight"
else
    echo "Invalid tracing tool: $TRACING_TOOL"
    echo "  tracing_tool: 'darshan' or 'nsight'"
    exit 1
fi

# Darshan Parameters
export DARSHAN_EXCLUDE_DIRS="${PYTHON_VENV_PATH}" # Exclude Python virtual environment
export DARSHAN_MODMEM=20000                       # 20 GB of memory allowed for Darshan instrumentation
export DARSHAN_ENABLE_NONMPI=1                    # Enable Darshan for non-MPI applications (Python)

# Modules setup (for lab machines)
module load Intel-oneAPI-HPC-Toolkit/mpi/latest || true

# Activate Python virtual environment
source "${PYTHON_VENV_PATH}/bin/activate"

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
    if [ ! -f "$DARSHAN_LIBPATH" ]; then
        echo "Error: DARSHAN_LIBPATH is not a valid file: $DARSHAN_LIBPATH"
        exit 1
    else
        echo "Darshan library found at ${DARSHAN_LIBPATH}"
    fi
fi

# Setup output directory
export TZ="Europe/Paris"
date_str=$(date +%m%d%Y-%H%M)
if [ ! -z "$SLURM_JOB_ID" ]; then
    suffix="${SLURM_JOB_ID}_${date_str}"
else
    suffix="${date_str}"
fi
run_name="traced_${suffix}"

# Tracing execution logic
if [ "$ENABLE_TRACING" -eq 1 ]; then
    # nsight
    if [ "$TRACING_TOOL" == "nsight" ]; then
        echo "Nsight log file: ${NSIGHT_LOGDIR}/${run_name}"
        trace_cmd=$(
            cat <<EOF
${NSIGHT_PATH}/nsys profile \
--enable storage_metrics,\
--lustre-volumes=all,\
--lustre-llite-dir="${LUSTRE_LLITE_DIR}" \
--output="${NSIGHT_LOGDIR}/${run_name}" \
--python-sampling=false
EOF
        )
        if [ "$NSIGHT_NVTX" -eq 1 ]; then
            NVTX_RANGE="BATCH"
            trace_cmd="${trace_cmd}\
 --capture-range=nvtx\
 --nvtx-capture=${NVTX_RANGE}\
 --capture-range-end=repeat"
        fi
    # darshan
    elif [ "$TRACING_TOOL" == "darshan" ]; then
        if [ "$DARSHAN_DXT" -eq 1 ]; then
            export DXT_ENABLE_IO_TRACE=1
        fi
        trace_cmd="env LD_PRELOAD=${DARSHAN_LIBPATH}"
    fi
fi

full_cmd="${trace_cmd} ${EXEC_COMMAND}"
echo "Executing command: ${full_cmd}"
eval "${full_cmd}"
echo "Traced execution completed."
