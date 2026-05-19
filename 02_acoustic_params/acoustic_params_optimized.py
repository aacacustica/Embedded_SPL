import time
_SCRIPT_T0 = time.perf_counter()

def boot_timing(label):
    now = time.perf_counter()
    print(f"[BOOT_TIMING] {label}: {now - _SCRIPT_T0:.3f}s", flush=True)


import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Carpeta deps
DEPS_DIR = os.path.join(BASE_DIR, 'deps')

sys.path.insert(0, DEPS_DIR)

import csv
boot_timing("import csv")

import datetime
boot_timing("import datetime")

import argparse
boot_timing("argparse")

import leq_levels_oct_weighting_C as m
boot_timing("import leq_leves")


#import audio_metadata
#boot_timing("audio_metadata")

import numpy as np
boot_timing("import numpy")

from utils import load_yaml,load_config_acoustic,read_calibration_constants,get_valid_audio_files,read_audio,resample_audio,get_device_id,update_processed_files
boot_timing("import util functions")

from logging_config import setup_logging
boot_timing("logging_config")

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

        w = load_yaml(weighting_yaml_path)

        if int(w["fs"]) != self.fs:
            raise ValueError(f"Weighting YAML fs={w['fs']} does not match fs={self.fs}")

        self.bA = np.asarray(w["A_weighting"]["b"], dtype=float)
        self.aA = np.asarray(w["A_weighting"]["a"], dtype=float)
        self.bC = np.asarray(w["C_weighting"]["b"], dtype=float)
        self.aC = np.asarray(w["C_weighting"]["a"], dtype=float)

        b = load_yaml(bank_yaml_path)
        if int(b["fs"]) != self.fs:
            raise ValueError(f"Bank YAML fs={b['fs']} does not match fs={self.fs}")

        self.sos_bank = np.asarray(b["sos_bank"], dtype=float)
        self.center_freqs = b["freq_center"]

        self.processor = m.AcousticProcessor(
            np.ascontiguousarray(self.bA, dtype=float),
            np.ascontiguousarray(self.aA, dtype=float),
            np.ascontiguousarray(self.bC, dtype=float),
            np.ascontiguousarray(self.aC, dtype=float),
            np.ascontiguousarray(self.sos_bank, dtype=float),
        )

    def process_audio_files(self,x, audio_file, mode="no_bands"):

        """
        Returns:
        all_data: list of per-file rows
        col_names: column names
        """

        if mode == "bands":

            col_names = (
                ["LA", "LC", "LZ", "LAmax", "LAmin"]
                + [f"{f:.2f}Hz" for f in self.center_freqs]
                + ["filename", "date"]
            )
            compute_bands = True
            
        elif mode == "no_bands":

            col_names = ["LA", "LC", "LZ", "LAmax", "LAmin", "filename", "date"]
            compute_bands = False

        else:

            raise ValueError(f"Unsupported mode: {mode}")
        if len(x) < self.window_size:
            logging.warning(f"Skipping {audio_file}: shorter than one window.")
            return [], col_names

        levels = self.processor.process(
            x,
            self.window_size,
            float(self.C),
            compute_bands,
        )

        filename = os.path.basename(audio_file)
        name_split = os.path.splitext(filename)[0]

        start_timestamp = datetime.datetime.strptime(
            name_split,
            "%Y%m%d_%H%M%S"
        )

        rows = []

        for frame_idx, values in enumerate(levels):
            timestamp = start_timestamp + datetime.timedelta(
                seconds=(frame_idx * self.window_size) / self.fs
            )

            numeric_values = [
                round(float(v), 2) if np.isfinite(v) else ""
                for v in values
            ]

            row = numeric_values + [
                audio_file,
                timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            ]

            rows.append(row)

        return [rows], col_names








def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Make prediction with YAMNet model for audio files'
    )
    parser.add_argument(
        '-p', '--path',
        type=str,
        required=False,
        help='Folder containing WAV files to process'
    )
    parser.add_argument(
        '-c', '--calib-const',
        type=str,
        required=False,
        default=0,
        help='Calibration constant to setup for each audio device.'
    )
    parser.add_argument(
        "--weighting-yaml",
        type=str,
        default=None,
        help="Path to weighting_fsXXXX.yaml"
    )
    parser.add_argument(
        "--bank-yaml",
        type=str,
        default=None,
        help="Path to sos_bank_1_3_fsXXXX.yaml"
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=None,
        help="If provided, use this fs instead of reading from metadata."
    )
    parser.add_argument(
        "-b", "--bands",
        action="store_true",
        help="Calcula bandas de tercios de octava."
    )
    parser.add_argument(
        "-d","--debug",
        action="store_true",
        help="Activa el modo debug"
    )
    return parser.parse_args()


def main():
    try:
        logging = setup_logging(script_name="acoustic_params")
        args = parse_arguments()
        home_dir = os.getenv("HOME")

        # ---------------------------------------
        # Load config info
        # ---------------------------------------
        try:
            t0_config = time.perf_counter()

            
            name_device,id_micro, location_record, location_place, location_point, \
            audio_sample_rate, audio_window_size, audio_calibration_constant, \
            storage_s3_bucket_name, storage_output_wav_folder, \
            storage_output_acoust_folder, calibration_constants_folder = load_config_acoustic('config.yaml')
          

            t1_config = time.perf_counter()
        
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return

        
        
        

        # ---------------------------------------
        # Parseo de argumentos
        # ---------------------------------------
        if args.calib_const: calib = args.calib_const
        else: calib = audio_calibration_constant
            
        if args.fs: fs = args.fs
        else: fs = audio_sample_rate
            
        if args.weighting_yaml:  weighting_yaml = args.weighting_yaml
        else: raise FileNotFoundError(f"Missing weighting YAML: {weighting_yaml}")
            
        if args.bank_yaml: bank_yaml = args.bank_yaml 
        else: raise FileNotFoundError(f"Missing bank YAML: {bank_yaml}")
        
        if args.bands: mode= "bands"
        else: mode = "no_bands"

        if args.debug: debug = True
        else: debug = False


        # ---------------------------------------
        # Load wav dir path 
        # ---------------------------------------
        if args.path:
            audio_path = args.path.strip()
            if not os.path.isdir(audio_path): raise ValueError(f"Audio path does not exist: {audio_path}")
            audio_path = os.path.abspath(os.path.normpath(audio_path))
        else:
            audio_path = os.path.join(
                home_dir.strip(),
                location_record.strip(),
                location_place.strip(),
                location_point.strip(),
                "AUDIOMOTH",
                storage_output_wav_folder.strip()
            )
            if not os.path.isdir(audio_path): raise ValueError(f"Audio path does not exist: {audio_path}")
            audio_path = os.path.abspath(os.path.normpath(audio_path))

        # ---------------------------------------
        # Setup output dir,trackeo de archivos procesados, carga de constantes de calibracion
        # ---------------------------------------
        output_dir = os.path.join(
            home_dir.strip(),
            location_record.strip(),
            location_place.strip(),
            location_point.strip(),
            "AUDIOMOTH",
            storage_output_acoust_folder.strip()
        )

        output_dir = os.path.abspath(os.path.normpath(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        processed_files_txt = os.path.join(output_dir, "processed_acoustic.txt")

        t0_calibration = time.perf_counter()
        calibration_constants = read_calibration_constants(calibration_constants_folder)
        t1_calibration = time.perf_counter()

        t0_valid_files = time.perf_counter()
        valid_audio_files = get_valid_audio_files(audio_path, processed_files_txt)
        t1_valid_files = time.perf_counter()



        # ---------------------------------------
        # Creacion de objeto calculadora
        # ---------------------------------------
        t0_calculator = time.perf_counter()
        calculator = LeqLevelOct(
            fs=fs,
            calibration_constant=float(calib),
            window_size=fs,  # 1 second
            audio_path=audio_path,
            weighting_yaml_path=weighting_yaml,
            bank_yaml_path=bank_yaml,
        )
        t1_calculator = time.perf_counter()

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"leq_oct_P1_TEST_sos_weighting_{timestamp_str}.csv"
        output_path = os.path.join(output_dir, output_filename)


        rows_written = 0
        csv_initialized = False

        if debug : 
            logging.info(f"HOME raw: {repr(home_dir)}")
            logging.info(f"Location record: {repr(location_record)}")
            logging.info(f"Location place: {repr(location_place)}")
            logging.info(f"Location point: {repr(location_point)}")
            logging.info(f"Storage output wav folder: {repr(storage_output_wav_folder)}")
            logging.info(f"Audio path: {repr(audio_path)}")

            logging.info(f"Audio path: {audio_path}")
            logging.info(f"Output dir: {output_dir}")
            logging.info(f"Processed file list path: {processed_files_txt}")
            logging.info(f"Processed file list exists: {os.path.exists(processed_files_txt)}")
            logging.info(f"Calibration constant: {calib}")
            logging.info(f"Using weighting YAML: {weighting_yaml}")
            logging.info(f"Using bank YAML: {bank_yaml}")

            logging.info(f"Using sample rate: {fs}")
            logging.info(f"Saving data to: {output_path}")
            logging.info(f"Processing {len(valid_audio_files)} new audio files...")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for audio_file in valid_audio_files:
                try:

                    t0_audio_process = time.perf_counter()

                    #---------------------------
                    # Check valid format
                    #---------------------------

                    if not audio_file.lower().endswith(".wav"):
                        logging.warning(f"Skipping non-wav file: {audio_file}")
                        continue

                    filepath = os.path.join(audio_path, audio_file)
                    
                    t0_metadata = time.perf_counter()
                    #metadata = audio_metadata.load(filepath)
                    t1_metadata = time.perf_counter()
                    
                    t0_device = time.perf_counter()
                    #device_id = get_device_id(metadata)
                    device_id = "songmeter"
                    logging.info(f"Device id: {device_id}")
                    t1_device = time.perf_counter()
                    
                    t0_calibration_read = time.perf_counter()
                    C = calibration_constants.get(device_id, float(calib))
                    t1_calibration_read = time.perf_counter()
                    calculator.C = C

                    #---------------------------
                    # Read audio file
                    #---------------------------

                    try:
                        t0_read = time.perf_counter()
                        x, fs_read = read_audio(filepath)
                        t1_read = time.perf_counter()
                        if x is None:
                            continue  
                    except Exception as e:
                        logging.warning(f"Error reading {audio_file}: {e}")
                        continue

                    #---------------------------
                    # Resample if needed
                    #---------------------------

                    if fs_read != calculator.fs:
                        try:

                            t0_resample = time.perf_counter()
                            x,fs_read = resample_audio(calculator, audio_file, fs_read, x)
                            t1_resample = time.perf_counter()

                        except Exception as e:
                            logging.warning(f"Error resampling {audio_file}: {e}")
                            continue

                    #---------------------------
                    # Call processor and calculate
                    #---------------------------
                    t0_process = time.perf_counter()
                    file_data, col_names = calculator.process_audio_files(
                        x,
                        audio_file,
                        mode=mode,
                    )
                    t1_process = time.perf_counter()

                    if not csv_initialized:
                        writer.writerow(col_names)
                        csv_initialized = True

                    t0_writing = time.perf_counter()
                    file_rows_written = 0
                    for file_rows in file_data:
                        writer.writerows(file_rows)
                        file_rows_written += len(file_rows)
                    t1_writing = time.perf_counter()

                    if file_rows_written > 0:
                        rows_written += file_rows_written
                        update_processed_files(processed_files_txt, audio_file)
                        if debug : logging.info(f"Marked as processed: {audio_file}")
                    else:
                        logging.warning(
                            f"No usable data produced for {audio_file}; "
                            "not marking as processed."
                        )

                    logging.info(
                        f"Processed file: {audio_file} "
                        f"with device_id={device_id}, C={C}, fs={fs}, "
                        f"rows={file_rows_written}"
                    )

                    t1_audio_process = time.perf_counter()
                    
                    if debug : logging.info(f"TIMING read_audio = {t1_read - t0_read}")
                    if debug : logging.info(f"TIMING resample = {t1_resample - t0_resample}")
                    if debug : logging.info(f"TIMING cpp_process = {t1_process - t0_process}")
                    if debug : logging.info(f"TIMING csv_write = {t1_writing - t0_writing}")
                    if debug : logging.info(f"TIMING reading metadata = {t1_metadata - t0_metadata}")
                    if debug : logging.info(f"TIMING reading device = {t1_device - t0_device}")
                    if debug : logging.info(f"TIMING reading calibration = {t1_calibration_read - t0_calibration_read}")
                    if debug : logging.info(f"")
                    if debug : logging.info(f"TIMING per audio processing : {t1_audio_process - t0_audio_process}")

                except Exception as e:
                    logging.warning(f"Error processing file: {audio_file}, {e}")



        if debug : logging.info(f"TIMING reading config = {t1_config - t0_config}")
        if debug : logging.info(f"TIMING making calibration = {t1_calibration - t0_calibration}")
        if debug : logging.info(f"TIMING reading valid files = {t1_valid_files - t0_valid_files}")
        if debug : logging.info(f"TIMING creatin calculator = {t1_calculator - t0_calculator}")

        if rows_written == 0:
            logging.warning("No data to save.")
            try:
                os.remove(output_path)
                logging.info(f"Removed empty output file: {output_path}")
            except OSError:
                pass
            return

        logging.info(f"Total rows written: {rows_written}")
        logging.info(f"Output saved to {output_path}")
        print(f"Output saved to {output_path}")

    except KeyboardInterrupt:
        logging.error("Process interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


boot_timing("before main")
if __name__ == "__main__":
    main()
boot_timing("after main")