#!/bin/bash
set -euo pipefail

VENV_PATH="/root/venvs/tflite/bin/activate"
PROJECT_ROOT="/root/IoT_microphone_scripts-main"
SCRIPT_PATH="$PROJECT_ROOT/03_inference/inference_tflite_optimized.py"

ARGS=(-u -w 3 -p "/root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/wav_files" -s 5 -t 0.3)

# Activar entorno virtual
source "$VENV_PATH"

# Ir al root del proyecto
cd "$PROJECT_ROOT"

# Asegurar imports desde el root del repo
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Launching inference script at $(date)"
SECONDS=0

python "$SCRIPT_PATH" "${ARGS[@]}"

duration=$SECONDS
echo "Finished at $(date)"
echo "Execution time: ${duration} seconds"