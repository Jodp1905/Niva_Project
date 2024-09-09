#!/bin/bash

# Path Parameters
PYTHON_VENV_PATH="/home/jrisse/venv-niva"
PYTHON_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/scripts"
SLURM_SCRIPT_DIR="/home/jrisse/niva/Niva_Project/slurm"
NUM_FOLDS=10

# Python training environment variables
my_index=$SLURM_NODEID
my_hostname=$(hostname)
echo "Node $my_hostname is running training on fold configuration $my_index"

# Training environment variables
export FOLD_LIST="$my_index"
export NUM_EPOCHS=1
export ITERATIONS_PER_EPOCH=20

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

# Modules setup (for lab machines)
module load Intel-oneAPI-HPC-Toolkit/mpi/latest || true

# Activate python virtual environment
source $PYTHON_VENV_PATH/bin/activate

# Run name setup
slurm_jobid=$SLURM_JOB_ID
run_name="training_foldsplit_${slurm_jobid}"
niva_project_data_root_sanitized="${NIVA_PROJECT_DATA_ROOT%/}"
output_dir="${niva_project_data_root_sanitized}/models/${run_name}"
mkdir -p $output_dir

# Tracing directory (one per node)
if [ "$ENABLE_TRACING" -eq 1 ]; then
    tracing_dir="${output_dir}/tracing"
    mkdir -p "${tracing_dir}"
    if [ "$TRACING_TOOL" == "darshan" ]; then
        darshan_filename="${run_name}_${my_index}_${my_hostname}.darshan"
        export DARSHAN_LOGFILE="${tracing_dir}/${darshan_filename}"
        echo "Darshan log file: ${DARSHAN_LOGFILE}"
    elif [ "$TRACING_TOOL" == "nsight" ]; then
        export NSIGHT_LOGS_DIR="${tracing_dir}"
    fi
fi

# Run training
# Training execution
training_script_name="training.py"
training_path=$(realpath "${PYTHON_SCRIPT_DIR}/${training_script_name}")
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
--python-sampling=true \
python3 \
"${training_path}" \
"${run_name}"
EOF
        )
    # darshan
    elif [ "$TRACING_TOOL" == "darshan" ]; then
        cmd=$(
            cat <<EOF
env LD_PRELOAD="${DARSHAN_LIBPATH}" \
python3 \
"${training_path}" \
"${run_name}"
EOF
        )
    fi
# without tracing
else
    cmd="python3 ${training_path} ${run_name}"
fi

echo "Executing command :"
echo "${cmd}"
eval "${cmd}"
echo "Training completed."
