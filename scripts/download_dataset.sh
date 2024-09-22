#!/bin/bash

# Activate virtual environment before running the script
# source /path/to/venv/bin/activate

project_dir=$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")
training_dir=${project_dir}/src/training

python3 ${training_dir}/download_data.py
