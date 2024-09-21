#!/bin/bash

# Activate virtual environment before running the script
# source /path/to/venv/bin/activate

project_dir=$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")
training_dir=${project_dir}/src/training

datetime_str=$(date +"%Y%m%d_%H%M%S")
training_name="training_${datetime_str}"

python3 ${training_dir}/training.py $training_name
