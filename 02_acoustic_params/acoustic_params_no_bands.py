import os
import sys

# Obtener la ruta absoluta de la carpeta raiz
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Carpeta deps
DEPS_DIR = os.path.join(BASE_DIR, 'deps')

sys.path.insert(0, DEPS_DIR)

import csv
import datetime
import soxr
import argparse
import leq_levels_oct_weighting_C as m

import tqdm
import audio_metadata

import soundfile as sf
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

        # Load 1/3-oct SOS bank (se mantiene la carga por compatibilidad,
        # pero ya no se usan las bandas en la salida)
        b = load_yaml(bank_yaml_path)
        if int(b["fs"]) != self.fs:
            raise ValueError(f"Bank YAML fs={b['fs']} does not match fs={self.fs}")

        self.sos_bank = b["sos_bank"]
        self.center_freqs = b["freq_center"]

        self.logging = logging
        logging.info("Initializing LeqLevelOct")
        logging.info(
            f"with fs={fs}, C={calibration_constant}, "
            f"window_size={window_size}, audio_path={audio_path}"
        )

    def process_audio_files(self, audio_files):
        """
        Returns:
          all_data: list of per-file rows
          col_names: column names
        """
        col_names = ["LA", "LC", "LZ", "LAmax", "LAmin", "filename", "date"]
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
                logging.warning(
                    f"File {audio_file} has fs={fs_read} but expected {self.fs}. "
                    "Resampling audio file"
                )
                x = soxr.resample(
                    x,
                    in_rate=fs_read,
                    out_rate=self.fs
                ).astype(np.float32, copy=False)
                logging.info(f"Resampled file {audio_file} into fs={self.fs}")
                logging.info(
                    f"{audio_file}: fs_read={fs_read}, target_fs={self.fs}, "
                    f"window_size={self.window_size}, "
                    f"duration_after={len(x)/self.fs:.2f}s, "
                    f"frames={(len(x)-self.window_size)//self.window_size+1}"
                )

            name_split = os.path.splitext(audio_file)[0]
            start_timestamp = datetime.datetime.strptime(name_split, "%Y%m%d_%H%M%S")

            frame_starts = range(0, len(x) - self.window_size + 1, self.window_size)
            timestamps = [
                start_timestamp + datetime.timedelta(seconds=fstart / self.fs)
                for fstart in frame_starts
            ]

            ziA = None
            ziC = None

            for fstart, timestamp in zip(frame_starts, timestamps):
                frame = x[fstart:fstart + self.window_size]

                self.bA = np.ascontiguousarray(self.bA, dtype=np.float32)
                self.aA = np.ascontiguousarray(self.aA, dtype=np.float32)
                self.bC = np.ascontiguousarray(self.bC, dtype=np.float32)
                self.aC = np.ascontiguousarray(self.aC, dtype=np.float32)
                frame = np.ascontiguousarray(frame, dtype=np.float32)

                if ziA is not None:
                    ziA = np.ascontiguousarray(ziA, dtype=np.float32)
                if ziC is not None:
                    ziC = np.ascontiguousarray(ziC, dtype=np.float32)

                yA, ziA = m.lfilter_np(self.bA, self.aA, frame, ziA)
                yC, ziC = m.lfilter_np(self.bC, self.aC, frame, ziC)

                fast_chunk = self.window_size // 8
                fast_levels = [
                    float(m.get_level_db(yA[i:i + fast_chunk], self.C))
                    for i in range(0, len(yA) - fast_chunk + 1, fast_chunk)
                ]
                Lmax = round(float(np.max(fast_levels)), 2)
                Lmin = round(float(np.min(fast_levels)), 2)

                def safe_num(v):
                    v = float(v)
                    return v if np.isfinite(v) else ""

                LA = safe_num(m.get_level_db(yA, self.C))
                LC = safe_num(m.get_level_db(yC, self.C))
                LZ = safe_num(m.get_level_db(frame, self.C))

                row = [
                    LA,
                    LC,
                    LZ,
                    Lmax,
                    Lmin,
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
        '-u', '--upload-S3',
        action='store_true',
        default=False,
        help='If provided, upload the final CSV to S3.'
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
    return parser.parse_args()


def main():
    try:
        logging = setup_logging(script_name="acoustic_params")
        args = parse_arguments()

        logging.info("Starting process")
        home_dir = os.getenv("HOME")

        try:
            logging.info("Loading config.yaml")
            id_micro, location_record, location_place, location_point, \
            audio_sample_rate, audio_window_size, audio_calibration_constant, \
            storage_s3_bucket_name, storage_output_wav_folder, \
            storage_output_acoust_folder, calibration_constants_folder = load_config_acoustic('config.yaml')
            logging.info("Config loaded successfully")
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return

        if args.path:
            audio_path = args.path.strip()
        else:
            audio_path = os.path.join(
                home_dir.strip(),
                location_record.strip(),
                location_place.strip(),
                location_point.strip(),
                "AUDIOMOTH",
                storage_output_wav_folder.strip()
            )

        audio_path = os.path.abspath(os.path.normpath(audio_path))

        logging.info(f"HOME raw: {repr(home_dir)}")
        logging.info(f"Location record: {repr(location_record)}")
        logging.info(f"Location place: {repr(location_place)}")
        logging.info(f"Location point: {repr(location_point)}")
        logging.info(f"Storage output wav folder: {repr(storage_output_wav_folder)}")
        logging.info(f"Audio path: {repr(audio_path)}")

        if not os.path.isdir(audio_path):
            raise ValueError(f"Audio path does not exist: {audio_path}")

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
        processed_files = load_processed_files(processed_files_txt)

        if args.calib_const:
            calib = args.calib_const
        else:
            calib = audio_calibration_constant

        upload_s3 = args.upload_S3 if args.upload_S3 else None

        weighting_yaml = args.weighting_yaml
        bank_yaml = args.bank_yaml

        if not weighting_yaml or not os.path.exists(weighting_yaml):
            raise FileNotFoundError(f"Missing weighting YAML: {weighting_yaml}")
        if not bank_yaml or not os.path.exists(bank_yaml):
            raise FileNotFoundError(f"Missing bank YAML: {bank_yaml}")

        calibration_constants = read_calibration_constants(calibration_constants_folder)

        logging.info(f"Audio path: {audio_path}")
        logging.info(f"Output dir: {output_dir}")
        logging.info(f"Processed file list path: {processed_files_txt}")
        logging.info(f"Processed file list exists: {os.path.exists(processed_files_txt)}")
        logging.info(f"Loaded {len(processed_files)} processed filenames")
        logging.info(f"Upload to bucket S3: {upload_s3}")
        logging.info(f"Calibration constant: {calib}")
        logging.info(f"Using weighting YAML: {weighting_yaml}")
        logging.info(f"Using bank YAML: {bank_yaml}")

        audio_files = get_audiofiles(audio_path)
        if not audio_files:
            logging.warning(f"No audio files found in: {audio_path}")
            return

        sample_rates = []
        valid_audio_files = []
        
        logging.info("Reading metadata...")
        for file in tqdm.tqdm(audio_files, desc="Reading metadata"):
            try:
                #metadata = audio_metadata.load(os.path.join(audio_path, file))
                #sample_rates.append(metadata.streaminfo.sample_rate)
                valid_audio_files.append(file)
            except Exception as e:
                logging.warning(f"Error reading file metadata: {file}, {e}")
        #- Se omite la lectura de metadata para acelerar el proceso, se asume que todos los archivos tienen el mismo fs o se especifica con --fs """
        if not valid_audio_files:
            logging.warning("No valid audio files to process.")
            return

        logging.info(f"Valid audio files before filtering: {len(valid_audio_files)}")
        valid_audio_files = [f for f in valid_audio_files if f not in processed_files]
        valid_audio_files = sorted(valid_audio_files)
        logging.info(f"Valid audio files after filtering: {len(valid_audio_files)}")

        if not valid_audio_files:
            logging.info("No new audio files to process.")
            return

        if args.fs:
            fs = args.fs
        else:
            fs = int(round(float(np.median(sample_rates)))) if sample_rates else None

        if fs is None:
            logging.warning("Could not determine fs.")
            return

        logging.info(f"Using sample rate: {fs}")

        calculator = LeqLevelOct(
            fs=fs,
            calibration_constant=float(calib),
            window_size=fs,  # 1 second
            audio_path=audio_path,
            weighting_yaml_path=weighting_yaml,
            bank_yaml_path=bank_yaml,
        )

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"leq_oct_P1_TEST_sos_weighting_{timestamp_str}.csv"
        output_path = os.path.join(output_dir, output_filename)

        logging.info(f"Saving data to: {output_path}")
        logging.info(f"Processing {len(valid_audio_files)} new audio files...")

        rows_written = 0
        csv_initialized = False

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for audio_file in tqdm.tqdm(valid_audio_files, desc="Processing audio files"):
                try:
                    filepath = os.path.join(audio_path, audio_file)
                    metadata = audio_metadata.load(filepath)
                    device_id = get_device_id(metadata)
                    C = calibration_constants.get(device_id, float(calib))
                    calculator.C = C

                    file_data, col_names = calculator.process_audio_files([audio_file])

                    if not csv_initialized:
                        writer.writerow(col_names)
                        csv_initialized = True

                    file_rows_written = 0
                    for file_rows in file_data:
                        writer.writerows(file_rows)
                        file_rows_written += len(file_rows)

                    if file_rows_written > 0:
                        rows_written += file_rows_written
                        update_processed_files(processed_files_txt, audio_file)
                        logging.info(f"Marked as processed: {audio_file}")
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
                except Exception as e:
                    logging.warning(f"Error processing file: {audio_file}, {e}")

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

    logging.info("Done!")


if __name__ == "__main__":
    main()