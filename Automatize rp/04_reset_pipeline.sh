#!/bin/bash
set -euo pipefail

VENV_PATH="/home/pi/venvs/tflite/bin/activate"
PROJECT_ROOT="/home/pi/IoT_microphone_scripts-main"
SCRIPT_PATH="$PROJECT_ROOT/04_reset_pipeline/reset_pipeline.py"


source "$VENV_PATH"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Launching script at $(date)"
SECONDS=0
python "$SCRIPT_PATH" 
duration=$SECONDS
echo "Finished at $(date)"
echo "Execution time: ${duration} seconds"