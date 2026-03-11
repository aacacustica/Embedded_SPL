import os
import sys

# Obtener la ruta absoluta de la carpeta raíz
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Carpeta deps
DEPS_DIR = os.path.join(BASE_DIR, 'deps')

# Añadir al path de Python para que busque aquí primero
sys.path.insert(0, DEPS_DIR)




import csv
import datetime
import time
import soxr
import argparse
import time
import leq_levels_oct_weighting_C as m

import tqdm
import audio_metadata

import soundfile as sf
import pandas as pd
import numpy as np

from utils import *
from logging_config import setup_logging



class LeqLevelOct:
    def __init__(
        self,
        fs: int,
        calibration_constant: float,
        window_size: int,
        audio_path: str,
        weighting_yaml_path: str,
        bank_yaml_path: str,
    ):
        self.fs = int(fs)
        self.C = float(calibration_constant)
        self.window_size = int(window_size)
        self.audio_path = audio_path
        
        # A and C weighting filters
        w = load_yaml(weighting_yaml_path)
        if int(w["fs"]) != self.fs:
            raise ValueError(f"Weighting YAML fs={w['fs']} does not match fs={self.fs}")
        
        self.bA = np.asarray(w["A_weighting"]["b"], dtype=np.float32)
        self.aA = np.asarray(w["A_weighting"]["a"], dtype=np.float32)
        self.bC = np.asarray(w["C_weighting"]["b"], dtype=np.float32)
        self.aC = np.asarray(w["C_weighting"]["a"], dtype=np.float32)

        # Load 1/3-oct SOS bank
        b = load_yaml(bank_yaml_path)
        if int(b["fs"]) != self.fs:
            raise ValueError(f"Bank YAML fs={b['fs']} does not match fs={self.fs}")
        
        self.sos_bank = b["sos_bank"]              # list[band] -> list[section] -> 6 floats
        self.center_freqs = b["freq_center"]       # for column labels
        
        
        
        #logging
        self.logging = logging
        logging.info("Initializing LeqLevelOct")
        logging.info(f"with fs={fs}, C={calibration_constant}, window_size={window_size}, audio_path={audio_path}")

    
    def process_audio_files(self, audio_files):
        """
        Returns:
          all_data: list of per-file rows
          col_names: column names
        """
        col_names = ["LA", "LC", "LZ", "LAmax", "LAmin"] + \
                    [f"{f:.2f}Hz" for f in self.center_freqs] + \
                    ["filename", "date"]

        all_data = []

        for audio_file in audio_files:
            db = []

            x, fs_read = sf.read(os.path.join(self.audio_path, audio_file))
            if x.ndim > 1:
                x = x[:, 0]
            x = np.asarray(x, dtype=np.float32).ravel()
            if len(x) < self.window_size:
                logging.warning(f"Skipping {audio_file}: shorter than one window.")
                continue
            
            if fs_read != self.fs:
                logging.warning(f"File {audio_file} has fs={fs_read} but expected {self.fs}. Resampling audio file")
                x = soxr.resample(x,in_rate = fs_read,out_rate=self.fs).astype(np.float32,copy=False)
                logging.info(f"Resampled file {audio_file} into fs={self.fs} ")
                logging.info(
                f"{audio_file}: fs_read={fs_read}, target_fs={self.fs}, window_size={self.window_size}, "
                f"duration_after={len(x)/self.fs:.2f}s, frames={(len(x)-self.window_size)//self.window_size+1}"
                )

            name_split = os.path.splitext(audio_file)[0]
            start_timestamp = datetime.datetime.strptime(name_split, "%Y%m%d_%H%M%S")

            frame_starts = range(0, len(x) - self.window_size + 1, self.window_size)
            timestamps = [
                start_timestamp + datetime.timedelta(seconds=fstart / self.fs)
                for fstart in frame_starts
            ]

            # Streaming states (per file)
            ziA = None
            ziC = None
            ziBands = [None] * len(self.sos_bank)
            #---C++ implementation substitution----
            for fstart, timestamp in zip(frame_starts, timestamps):
                frame = x[fstart:fstart + self.window_size]
                
                # A/C weighting via lfilter_np (b,a)
                #yA, ziA = lfilter_np(self.bA, self.aA, frame, zi=ziA)
                #yC, ziC = lfilter_np(self.bC, self.aC, frame, zi=ziC)

                self.bA = np.ascontiguousarray(self.bA, dtype=np.float32)
                self.aA = np.ascontiguousarray(self.aA, dtype=np.float32)
                frame   = np.ascontiguousarray(frame, dtype=np.float32)

                if ziA is not None:
                    ziA = np.ascontiguousarray(ziA, dtype=np.float32)
                if ziC is not None:
                    ziC = np.ascontiguousarray(ziC, dtype=np.float32)

                yA, ziA  = m.lfilter_np(self.bA, self.aA, frame, ziA)
                #yC, ziC  = m.lfilter_np(self.bC, self.aC, frame, ziC)

                


                #LA = round(get_level_db(yA, self.C), 2)
                #LC = round(get_level_db(yC, self.C), 2)
                #LZ = round(get_level_db(frame, self.C), 2)

                LA = round(float(m.get_level_db(yA, self.C)),2)
                #LC = round(float(m.get_level_db(yC, self.C)),2)
                #LZ = round(float(m.get_level_db(frame, self.C)),2)
                # LAmax/LAmin over FAST subchunks (8 per second if window=fs)

                fast_chunk = self.window_size // 8
                fast_levels = [
                    #get_level_db(yA[i:i + fast_chunk], self.C)
                    float(m.get_level_db(yA[i:i + fast_chunk], self.C))
                    for i in range(0, len(yA) - fast_chunk + 1, fast_chunk)
                ]
                Lmax = round(float(np.max(fast_levels)), 2)
                Lmin = round(float(np.min(fast_levels)), 2)

                # 1/3-oct band levels via SOS bank
                band_levels = []
                
                for i, sos in enumerate(self.sos_bank):
                        
                    sos = np.ascontiguousarray(np.asarray(sos, dtype=np.float32), dtype=np.float32)
                    #yb, ziBands[i] = sosfilt_np(sos, frame, zi=ziBands[i])
                    zi_i = ziBands[i]
                    


                    yb, zi_i = m.sosfilt_np(sos, frame, zi=zi_i)
                    ziBands[i] = np.ascontiguousarray(zi_i, dtype=np.float32)

                    val = float(m.get_level_db(yb, self.C))
                    band_levels.append(round(val, 2))

                #20db fix
                band_levels = twenty_db_fix(band_levels)
                """
                if any((not np.isfinite(v)) or abs(v) > 200 for v in band_levels):
                    logging.warning(
                        f"Crazy band levels in {audio_file} at {timestamp}: "
                        + ",".join(str(v) for v in band_levels[:8])
                    )
                
                def safe_num(v):
                    v = float(v)
                    return v if np.isfinite(v) else ""

                LA = safe_num(m.get_level_db(yA, self.C))
                LC = safe_num(m.get_level_db(yC, self.C))
                LZ = safe_num(m.get_level_db(frame, self.C))
                
                row = [LA, LC, LZ, Lmax, Lmin] + band_levels + [
                    audio_file,
                    timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                ]
                """
                """
                row = [LA, LC, LZ, Lmax, Lmin] +  [
                    audio_file,
                    timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                ]
                """
                row = [LA, Lmax, Lmin] + band_levels + [
                    audio_file,
                    timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                ]
                
                db.append(row)

            if db:
                all_data.append(db)
                logging.info(f"Processed file: {audio_file} (rows={len(db)})")
            else:
                logging.warning(f"Processed file: {audio_file} produced 0 rows (no frames).")

            logging.info(f"Processed file: {audio_file}")

        return all_data, col_names



def load_processed_files(processed_file_path):
    """Load the set of processed filenames from a text file."""
    if os.path.exists(processed_file_path):
        with open(processed_file_path, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return set()



def update_processed_files(processed_file_path, filename):
    """Append a processed filename to the text file."""
    with open(processed_file_path, "a") as f:
        f.write(filename + "\n")



def parse_arguments():
    parser = argparse.ArgumentParser(description='Make prediction with YAMNet model for audio files')
    parser.add_argument('-p', '--path', type=str, required=False,
                        help='Folder containing WAV files to process')
    parser.add_argument('-c', '--calib-const', type=str, required=False, default=0,
                        help='Calibration constant to setup for each audio device.')
    parser.add_argument('-u', '--upload-S3', action='store_true',default=False,
                        help='If provided, upload the final CSV to S3.')
    parser.add_argument("--weighting-yaml", type=str, default=None, help="Path to weighting_fsXXXX.yaml")
    parser.add_argument("--bank-yaml", type=str, default=None, help="Path to sos_bank_1_3_fsXXXX.yaml")
    parser.add_argument("--fs", type=int, default=None, help="If provided, use this fs instead of reading from metadata.")
    return parser.parse_args()



def main():
    try:
        logging = setup_logging(script_name="acoustic_params")
        args = parse_arguments()

        logging.info("Staarting process!!")
        logging.info("")
        
        home_dir = os.getenv("HOME")
        
        try:
            # config
            logging.info("Getting the element form the yamnl file")
            id_micro, location_record, location_place, location_point, \
            audio_sample_rate, audio_window_size, audio_calibration_constant,\
            storage_s3_bucket_name, storage_output_wav_folder, \
            storage_output_acoust_folder,calibration_constants_folder = load_config_acoustic('config.yaml')
            logging.info("Config loaded successfully")
        
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return

        if args.path:
            path = args.path
        else:
            path = os.path.join(home_dir, location_record, location_place, location_point, storage_output_wav_folder)
            # check if it exist
            isdir = os.path.isdir(path)
            if isdir:
                logging.info(f"Path exists --> {path}")
                # continue
            else:
                raise ValueError(f'Path ({path}) doesnt exist.')
                    

         
        if args.calib_const:
            calib = args.calib_const
        else:
            calib = audio_calibration_constant
        
        # upload to bucket S3
        if args.upload_S3:
            upload_s3 = args.upload_S3
        else:
            upload_s3 = None
        
        weighting_yaml = args.weighting_yaml
        bank_yaml = args.bank_yaml

        if not os.path.exists(weighting_yaml):
            raise FileNotFoundError(f"Missing weighting YAML: {weighting_yaml}")
        if not os.path.exists(bank_yaml):
            raise FileNotFoundError(f"Missing bank YAML: {bank_yaml}")
        
        if not os.path.exists(weighting_yaml):
            raise FileNotFoundError(f"Missing weighting YAML: {weighting_yaml}")
        if not os.path.exists(bank_yaml):
            raise FileNotFoundError(f"Missing bank YAML: {bank_yaml}") 

        audiomoth_folders = list(find_audiomoth_folders(path))
        calibration_constants = read_calibration_constants(calibration_constants_folder)

        for subfolder in tqdm.tqdm(audiomoth_folders,desc="Processing folders ... "):
            
            logging.info(f"Processing audio files: {subfolder}...")
            audio_path = os.path.join(subfolder, "AUDIOMOTH", storage_output_wav_folder)

            if not os.path.exists(audio_path):
                logging.warning(f"Skipping {subfolder}, AUDIOMOTH folder not found.")
                continue

            audio_files = get_audiofiles(audio_path)

            if not audio_files:
                logging.warning(f"No audio files found in: {audio_path}")
                continue

            # Read metadata to get sample rates (unless forced)

            sample_rates = []
            valid_audio_files = []

            logging.info("Reading metadata...")

            for file in tqdm.tqdm(audio_files, desc="Reading metadata"):
                try:
                    metadata = audio_metadata.load(os.path.join(audio_path, file))
                    sample_rates.append(metadata.streaminfo.sample_rate)
                    valid_audio_files.append(file)
                except Exception as e:
                    logging.warning(f"Error reading file metadata: {file}, {e}")

            if not valid_audio_files:
                logging.warning(f"No valid audio files to process in {subfolder}")
                continue
            
            if args.fs:
                fs = args.fs
            else:
                fs = int(round(float(np.median(sample_rates)))) if sample_rates else None

            if fs is None:
                logging.warning("Could not determine fs.")
                continue

            logging.info(f"Path: {path}")
            logging.info(f"Upload to bucket S3: {upload_s3}")
            logging.info(f"Calibration constant: {calib}")
            logging.info(f"Using weights stored at: {weighting_yaml}")
            logging.info(f"Using sos bank stored at: {bank_yaml}")
            logging.info(f"Using sample rate:  {fs}")


            calculator = LeqLevelOct(
                fs=fs,
                calibration_constant=-10.16,
                window_size=fs,  # 1 second
                audio_path=audio_path,
                weighting_yaml_path=weighting_yaml,
                bank_yaml_path=bank_yaml,
            )
            
            all_data_subfolder = []
            logging.info(f"Processing {len(valid_audio_files)} files in {subfolder}...")

            processed_files_txt = os.path.join(path, "processed_acoustic.txt")
            processed_files_txt = processed_files_txt.replace("wav_files", "acoustic_params")            
            processed_files = load_processed_files(processed_files_txt)
            valid_audio_files = [f for f in valid_audio_files if f not in processed_files]

            for audio_file in tqdm.tqdm(valid_audio_files, desc="Processing audio files "):
                try:
                    filepath = os.path.join(audio_path, audio_file)
                    metadata = audio_metadata.load(filepath)
                    device_id = get_device_id(metadata)
                    C = calibration_constants.get(device_id, -10.16)
                    calculator.C = C

                    file_data, col_names = calculator.process_audio_files([audio_file])
                    all_data_subfolder.extend(file_data)

                    logging.info(
                        f"Processed file: {audio_file} with device_id={device_id}, C={C}, fs={fs}"
                    )
                except Exception as e:
                    logging.warning(f"Error processing file: {audio_file}, {e}")


            if all_data_subfolder:
                logging.info(f"Saving output for folder {subfolder}...")
                flat_data = [row for file_rows in all_data_subfolder for row in file_rows]

                if not flat_data:
                    logging.warning(f"No rows to save for folder {subfolder} (flat_data is empty). Skipping CSV.")
                    continue

                logging.info(f"Total rows to save: {len(flat_data)}")
                logging.info(f"Column names: {col_names}")
                flat_data_sorted = sorted(flat_data, key=lambda row: parse_dt(row[-1]))  # date es última columna
                df = pd.DataFrame(flat_data_sorted, columns=col_names)
                logging.info("DataFrame created successfully.")
                #df = pd.DataFrame(flat_data_sorted, columns=col_names)
                #df = pd.DataFrame(flat_data, columns=col_names)
                #df = df.sort_values(by="date")

                folder_name = subfolder.split("\\")[-1]
                output_filename = f"leq_oct_P1_TEST_sos_weighting.csv"
                #Testing
                #output_path = os.path.join(audio_path, output_filename)
                output_path = f"/root/NOISEPORT-TENERIFE/C1/3-Medidas/P1/AUDIOMOTH/{output_filename}"
                logging.info(f"Saving data: {output_path}")

                with open(output_path, "w", newline="", encoding="utf-8") as f:

                    w = csv.writer(f)
                    w.writerow(col_names)
                    w.writerows(flat_data_sorted)
                    

                logging.info(f"Output saved to {output_path}")

                #df.to_csv(output_path, index=False)
                logging.info(f"Output saved to {output_path}")
                print(f"Output saved to {output_path}")
            else:
                logging.warning(f"No data to save for folder {subfolder}")

    except KeyboardInterrupt:
        logging.error("Process interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


    #logging end of script
    logging.info("")
    logging.info("Done!")


if __name__ == "__main__":
    main()
