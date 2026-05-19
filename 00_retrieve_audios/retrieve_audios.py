import os
import sys
PROJECT_ROOT = "/root/IoT_microphone_scripts-main"
sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from logging_config import setup_logging

WAV_FILES_FOLDER = "/root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/wav_files"


def main():

    logging = setup_logging(script_name="retrieve_audios")

    if not os.path.isdir(WAV_FILES_FOLDER):
        logging.error(f"No existe la carpeta: {WAV_FILES_FOLDER}")
        return

    try:
        wav_files = sorted([
            f for f in os.listdir(WAV_FILES_FOLDER)
            if f.lower().endswith(".wav")
        ])

        if not wav_files:
            logging.warning("No hay archivos .wav para procesar")
            return

        first_file = wav_files[0]

        # Ejemplo:
        # 20260518_115526.wav -> 20260518_11
        first_hour_tag = Path(first_file).stem[:11]

        audio_files = [
            f for f in wav_files
            if Path(f).stem.startswith(first_hour_tag)
        ]

        for_process_file_list_txt = os.path.join(
            WAV_FILES_FOLDER,
            "processing_files.txt"
        )

        with open(for_process_file_list_txt, mode="w") as f:
            f.writelines(file + "\n" for file in audio_files)

        logging.info(f"Archivos seleccionados para procesar: {len(audio_files)}")
        logging.info(f"Hora/tag procesado: {first_hour_tag}")
        logging.info(f"Lista escrita en: {for_process_file_list_txt}")

    except Exception as e:
        logging.error(f"Error preparando archivos para procesar: {e}")
        return


if __name__ == '__main__':
    main()