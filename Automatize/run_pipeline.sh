#!/bin/bash
set -euo pipefail

VENV_PATH="/root/venvs/tflite/bin/activate"
PROJECT_ROOT="/root/IoT_microphone_scripts-main"
AUTOMATIZE_ROOT="$PROJECT_ROOT/Automatize"
BASE="/root"
LOG_DIR="$BASE/logs"

mkdir -p "$LOG_DIR"

scripts=(
    "02_acoustic_params.sh"
    "03_inference.sh"
    "04_reset_pipeline.sh"
    "send_csvs.sh"
)

source "$VENV_PATH"
cd "$PROJECT_ROOT"

LOCKFILE="/tmp/pipeline.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "Pipeline already running. Exiting."; return 1 2>/dev/null || exit 1; }

echo "=== PIPELINE START: $(date) ===" | tee -a "$LOG_DIR/pipeline.log"

for script in "${scripts[@]}"; do
    echo ">>> Running $script at $(date)" | tee -a "$LOG_DIR/$script.log"

    SECONDS=0
    bash "$AUTOMATIZE_ROOT/$script" >> "$LOG_DIR/$script.log" 2>&1
    duration=$SECONDS

    echo ">>> Finished $script at $(date)" | tee -a "$LOG_DIR/$script.log"
    echo "Execution time: ${duration} seconds" | tee -a "$LOG_DIR/$script.log"
done

echo ">>> Deleting *.wav files from wav_files folder ... "
#rm -f /root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/wav_files/*.wav
echo ">>> *.wav files from wav_files folder deleted ... "

echo "=== PIPELINE END: $(date) ===" | tee -a "$LOG_DIR/pipeline.log"