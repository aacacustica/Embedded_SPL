
import os

WAV_FOLDER = "/root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/wav_files"
PROCESSED_WAVS_TXT = "/root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/wav_files/processing_files.txt"




INFERENCES_PROCESSED_PATH = "/root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/predictions_litle/processed_predictions.txt"
ACOUSTICS_PROCESSED_PATH = "/root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/acoustic_params/processed_acoustics.txt"


"""
Formato de los archivos .wav : YYYYMMDD_HHMMSS.wav

"""

def remove_wavs():

    with open(PROCESSED_WAVS_TXT) as f:
        lines = f.readlines()
        for file in lines:
            file_path = os.path.join(WAV_FOLDER,file)
            os.remove(file_path)

def reset_wavs_txt():

    os.remove(PROCESSED_WAVS_TXT)


def remove_acoustics_wavs(path_txt, path_wavs):
    if not os.path.isfile(path_txt):
        print(f"TXT not found: {path_txt}")
        return

    with open(path_txt, "r", encoding="utf-8") as f:
        wav_names = [line.strip() for line in f if line.strip()]

    for wav_name in wav_names:
        wav_path = os.path.join(path_wavs, wav_name)
        if os.path.isfile(wav_path):
            os.remove(wav_path)
            print(f"Deleted: {wav_path}")
        else:
            print(f"File not found: {wav_path}")

    open(path_txt, "w", encoding="utf-8").close()


def remove_inferences_wavs(path_txt, path_wavs):
    """
    path_txt : processed_predictions.txt path, which holds the file absolute path
    path_wavs : path to the .wav files in the inferences folder
    path_acoustics : path to the inferences folder

    function:
    - reads processed_predictions.txt
    - extracts the hour block (YYYYMMDD_HH) from the processed wavs
    - deletes ALL wavs in the inferences folder that belong to that hour
    - clears processed_predictions.txt
    """

    if not os.path.isfile(path_txt): return

    try:
        with open(path_txt, "r", encoding="utf-8") as f:
            processed_paths = [line.strip() for line in f if line.strip()]
    except Exception as e:
        return

    if not processed_paths:
        print(f"[remove_inferences_wavs] No processed paths found in: {path_txt}")
        return

    # Obtener la hora de referencia a partir del primer archivo procesado
    # Formato esperado: YYYYMMDD_HHMMSS.wav
    first_file = os.path.basename(processed_paths[4])  # Tomar el quinto archivo procesado como referencia por seguridad

    if not first_file.endswith(".wav") or len(first_file) < 15: return

    hour_prefix = first_file[:11]   # YYYYMMDD_HH

    try:
        for file_name in os.listdir(path_wavs):
            if not file_name.endswith(".wav"):
                continue

            # Borrar todos los wavs de esa misma hora
            if file_name.startswith(hour_prefix):
                file_path = os.path.join(path_wavs, file_name)

                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"[remove_inferences_wavs] Error deleting {file_path}: {e}")

    except Exception as e:
        return

    try:
        open(path_txt, "w", encoding="utf-8").close()
        print(f"[remove_inferences_wavs] Cleared: {path_txt}")
    except Exception as e:
        print(f"[remove_inferences_wavs] Error clearing {path_txt}: {e}")

def main():

    
    remove_wavs()
    reset_wavs_txt()

    remove_acoustics_wavs(
        path_txt=ACOUSTICS_PROCESSED_PATH,
        path_wavs=WAV_FOLDER,
    )

    remove_inferences_wavs(
        path_txt=INFERENCES_PROCESSED_PATH,
        path_wavs=WAV_FOLDER,
    )


if __name__ == "__main__":
    main()