#!/bin/bash

# ======================================= PARAMETERS =======================================

# Project virtual environment path
PYTHON_VENV_PATH="${PROJECT_DIR}/venv-niva"

# Darshan Library Path
DARSHAN_LIBPATH="${HOME}/software/darshan-3.4.5/darshan-runtime/install/lib/libdarshan.so"

# Output Directories
NSIGHT_LOGDIR="${HOME}/niva/nsight_logs"

# DARSHAN PARAMETERS
# Darshan DXT (Darshan eXtended Tracing) generates more detailed I/O traces
DARSHAN_DXT=1

# NSIGHT PARAMETERS
# Nsight NVTX (NVIDIA Tools Extension) enables profiling with NVTX annotations
# Sampled batch training steps from the Python script will be captured
# nsight_batch_profiling needs to be enabled in yaml configuration file
NSIGHT_NVTX=1
# Nsight Storage Metrics enables capturing I/O metrics
NSIGHT_STORAGE_METRICS=1
# Path to Lustre LLITE directory for capturing Lustre I/O metrics
LUSTRE_LLITE_DIR="/mnt/lustre-stats/llite"
# Nsight Python Sampling enables capturing Python function calls (Higher overhead)
NSIGHT_PYTHON_SAMPLING=0

# ======================================== SCRIPT ========================================

# Arguments:
# $1: Tracing tool (either 'darshan' or 'nsight')
# $2: Command to execute (e.g., python or bash script) and its arguments
TRACING_TOOL=$1
shift
EXEC_COMMAND=$@

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

# logdir setup
if [ "$TRACING_TOOL" == "darshan" ]; then
    mkdir -p "${DARSHAN_LOGDIR}"
    echo "Darshan log directory: ${DARSHAN_LOGDIR}"
elif [ "$TRACING_TOOL" == "nsight" ]; then
    mkdir -p "${NSIGHT_LOGDIR}"
    echo "Nsight log directory: ${NSIGHT_LOGDIR}"
fi

# Darshan Parameters
export DARSHAN_EXCLUDE_DIRS="${PYTHON_VENV_PATH}" # Exclude Python virtual environment
export DARSHAN_MODMEM=20000                       # 20 GB of memory allowed for Darshan instrumentation
export DARSHAN_NAMEMEM=1000                       # 1 GB of memory allowed for Darshan record names
export DARSHAN_ENABLE_NONMPI=1                    # Enable Darshan for non-MPI applications (Python)
if [ "$DARSHAN_DXT" -eq 1 ]; then
    export DXT_ENABLE_IO_TRACE=1
    export DARSHAN_MOD_DISABLE="LUSTRE" # Disable Lustre module for Darshan dxt parser
fi

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

# Tracing tool specific checks
# darshan checks
if [ "$TRACING_TOOL" == "darshan" ]; then
    # is darshan library path set
    if [ -z "$DARSHAN_LIBPATH" ]; then
        echo "Error: DARSHAN_LIBPATH is not set."
        echo "Please set it to the path of the Darshan library."
        exit 1
    fi
    # is darshan library path valid
    if [ ! -f "$DARSHAN_LIBPATH" ]; then
        echo "Error: DARSHAN_LIBPATH is not a valid file: $DARSHAN_LIBPATH"
        exit 1
    else
        echo "Darshan library found at ${DARSHAN_LIBPATH}"
    fi
# nsight checks
else
    # is the nsys command available
    if ! command -v nsys &>/dev/null; then
        echo "Error: Nsight command 'nsys' is not available."
        echo "Please make sure the Nsight Systems is installed and available in the PATH."
        exit 1
    fi
    # is nsight lustre llite dir set
    if [ -z "$LUSTR_LLITE_DIR" ]; then
        echo "Warning: LUSTRE_LLITE_DIR is not set."
        echo "Lustre I/O metrics will not be captured."
    fi
    # is nsight lustre llite dir valid
    if [ ! -f "$LUSTRE_LLITE_DIR" ]; then
        echo "Warning: LUSTRE_LLITE_DIR is not a valid directory: $LUSTRE_LLITE_DIR"
        echo "Lustre I/O metrics will not be captured."
    fi
fi

# Setup output directory
date_str=$(date +%m%d%Y-%H%M)
if [ ! -z "$SLURM_JOB_ID" ]; then
    suffix="${SLURM_JOB_ID}_${date_str}"
else
    suffix="${date_str}"
fi
run_name="traced_${suffix}"

# Tracing execution logic
# nsight
if [ "$TRACING_TOOL" == "nsight" ]; then
    echo "Nsight log file: ${NSIGHT_LOGDIR}/${run_name}"
    trace_cmd="nsys profile --output=${NSIGHT_LOGDIR}/${run_name}"
    if [ "$NSIGHT_STORAGE_METRICS" -eq 1 ]; then
        trace_cmd="${trace_cmd}\
 --enable storage_metrics,--lustre-volumes=all,--lustre-llite-dir=${LUSTRE_LLITE_DIR}"
    fi
    if [ "$NSIGHT_PYTHON_SAMPLING" -eq 1 ]; then
        trace_cmd="${trace_cmd}\
 --python-sampling=true"
    fi
    if [ "$NSIGHT_NVTX" -eq 1 ]; then
        NVTX_RANGE="BATCH"
        trace_cmd="${trace_cmd}\
 --capture-range=nvtx\
 --nvtx-capture=${NVTX_RANGE}\
 --capture-range-end=repeat"
    fi
# darshan
elif [ "$TRACING_TOOL" == "darshan" ]; then
    trace_cmd="env LD_PRELOAD=${DARSHAN_LIBPATH}"
fi

full_cmd="${trace_cmd} ${EXEC_COMMAND}"
echo "Executing command: ${full_cmd}"
eval "${full_cmd}"
echo "Traced execution completed."
