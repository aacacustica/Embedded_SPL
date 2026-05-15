#!/bin/bash
set -euo pipefail

VENV_PATH="/home/pi/venvs/tflite/bin/activate"
PROJECT_ROOT="/home/pi/IoT_microphone_scripts-main"
SCRIPT_PATH="$PROJECT_ROOT/01_recording/record_audio.py"
CODEC_PATH="/home/pi/IoT_microphone_scripts-main/activarcodec.sh"

# args en array (mejor que string)
ARGS=(-t 60 -u)

echo "Activating venv"
source "$VENV_PATH"
cd "$PROJECT_ROOT"

echo "Activating audio codec"
bash "$CODEC_PATH"
echo "Launching script"
python3 "$SCRIPT_PATH" "${ARGS[@]}"