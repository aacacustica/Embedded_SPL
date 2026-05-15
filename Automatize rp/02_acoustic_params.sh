#!/bin/bash
set -euo pipefail

VENV_PATH="/home/pi/venvs/tflite/bin/activate"
PROJECT_ROOT="/home/pi/IoT_microphone_scripts-main"
SCRIPT_PATH="$PROJECT_ROOT/02_acoustic_params/acoustic_params.py"

ARGS=(--weighting-yaml "/home/pi/IoT_microphone_scripts-main/weighting_fs16000.yaml" --bank-yaml "/home/pi/IoT_microphone_scripts-main/sos_bank_1_3_fs16000.yaml" -p "/root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/wav_files/" --fs 16000)

source "$VENV_PATH"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Launching script at $(date)"
SECONDS=0
python "$SCRIPT_PATH" "${ARGS[@]}"
duration=$SECONDS
echo "Finished at $(date)"
echo "Execution time: ${duration} seconds"