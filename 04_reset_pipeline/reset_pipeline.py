
import os

WAV_FOLDER = "/root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/wav_files"
PROCESSED_WAVS_TXT = "/root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/wav_files/processing_files.txt"


"""
Formato de los archivos .wav : YYYYMMDD_HHMMSS.wav

"""

def remove_wavs():

    with open(PROCESSED_WAVS_TXT) as f:
        lines = f.readlines()
        for file in lines:
            file_path = os.path.join(WAV_FOLDER,file)
            os.remove(file_path)

def main():

    remove_acoustics_wavs(
        path_txt=ACOUSTICS_PROCESSED_PATH,
        path_wavs=ACOUSTIC_WAVS_PATH,
    )

    remove_inferences_wavs(
        path_txt=INFERENCES_PROCESSED_PATH,
        path_wavs=INFERENCES_WAVS_PATH,
    )


if __name__ == "__main__":
    main()