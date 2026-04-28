#!/bin/bash

# Define the path to your virtual environment
VENV_PATH="$HOME/.venvs/pycad/bin/activate"

# 1. Check if the venv exists
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please run ./install.sh first."
    exit 1
fi

# 2. Activate the environment
source "$VENV_PATH"

# 3. Add current directory to PYTHONPATH
# This ensures Python can see your py_cad_modules folder
export PYTHONPATH=$PYTHONPATH:.

# 4. Execute the project
# This passes any arguments you give the script (like test numbers) to run.py
python3 -B benchmarker.py "$@"