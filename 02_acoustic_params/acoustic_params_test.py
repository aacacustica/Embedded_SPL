import os
import csv
import pandas as pd
import datetime
import time
import numpy as np
import pandas as pd
import argparse
import time

import soundfile as sf
from pyfilterbank.splweighting import a_weighting_coeffs_design, c_weighting_coeffs_design
from utils import *
from scipy.signal import lfilter

from logging_config import setup_logging
from PyOctaveBand import *



class LeqLevelOct:
    def __init__(self, id_micro, fs, calibration_constant, window_size, audio_path, wav_files, acoustic_params, s3_bucket_name, upload_s3, logging):
        """
        Set up the LeqLevelOct object with the necessary parameters
        :param fs:
            Sample rate of the audio
        :param calibration_constant:
            Calibration constant for the microphone
        :param window_size:
            Size of the window for calculating SPL levels
        :param audio_path:
            Path to the audio files
        """
        self.id_micro = id_micro
        self.fs = fs
        self.C = calibration_constant
        self.window_size = window_size
        self.audio_path = audio_path
        self.acoustic_params = acoustic_params
        self.wav_files = wav_files
        self.s3_bucket_name = s3_bucket_name
        self.upload_s3 = upload_s3
        
        # A and C weighting filters
        self.bA, self.aA = a_weighting_coeffs_design(fs)
        self.bC, self.aC = c_weighting_coeffs_design(fs)
        
        # 1/3- and octave filter banks
        # self.third_oct, self.octave = filterbanks(fs)
        
        #logging
        self.logging = logging
        logging.info("Initializing LeqLevelOct")
        logging.info(f"with fs={fs}, C={calibration_constant}, window_size={window_size}, audio_path={audio_path}, wav_files={wav_files}, acoustic_params={acoustic_params}")
    
    

    def get_oct_levels(self, frame):
        """
        Calculate 1/3-octave levels for a frame of audio data.
        Returns a list of 1/3-octave levels.
        """
        y_oct, _ = self.third_oct.filter(frame)
        oct_level_temp = [get_db_level(y_band, self.C) for y_band in y_oct.T]
        return oct_level_temp

    
    def process_audio_files(self, path):
        """
        Process each WAV file in audio_files, compute SPL metrics,
        and write a CSV per file with frame-by-frame data.
        
        :param audio_files: List of .wav filenames in self.audio_path
        :return: all_data, a list of lists (one sub-list per file).
        """
        # ---------------------------
        # INIZIALATIN PROCESSING FILE
        # ---------------------------
        
        processed_files_txt = os.path.join(path, "processed_acoustic.txt")
        processed_files_txt = processed_files_txt.replace("wav_files", "acoustic_params")
        self.logging.info(f"Saving the processed file txt here --> {processed_files_txt}")
        
        processed_files = load_processed_files(processed_files_txt)


        # ----------------------------------
        # COLLECTING AUDIO FILES TO PROCESS
        # ----------------------------------
        try:
            audio_files = [f for f in os.listdir(path) if f.lower().endswith('.wav')]
            self.logging.info(f"Found {len(audio_files)} audio files: {audio_files}")
            full_paths = [os.path.join(path, file) for file in audio_files]
        
        except Exception as e:
            self.logging.error(f"Errorgetting the audio files: {e}")
        
        self.logging.info("")
        self.logging.info(f"Full path: \n{full_paths}")



        # ----------------------------------
        #HEADERS
        # ----------------------------------
        col_names = ['id_micro', 'Filename', 'Timestamp', 'Unixtimestamp', 'LA', 'LC', 'LZ', 'LAmax', 'LAmin']
        # 1/3-oct band names 
        # band_names = [f"{freq:.2f}Hz" for freq in self.third_oct.center_frequencies]
        # col_names.extend(band_names)#1/3-oct columns
        
        freq_labels = None
        
        
        # ----------------------------------
        # PROCESSING
        # ----------------------------------
        all_data = []
        for audio_file in full_paths:
            try:
                file_start_time = time.time()


                if audio_file in processed_files:
                    self.logging.info(f"Skipping {audio_file}, already processed.")
                    continue


                self.logging.info(f"Processing audio file: {audio_file}")
                db = []
                # reading audio data
                x, _ = sf.read(audio_file)
                
                #naming 
                name_split = audio_file.split(".")[0]  # '20250107_130000'
                name_split = name_split.split("/")[-1]  #'20250107_130000'
                self.logging.info(f"Name split: {name_split}")
                
                #CET
                local_tz = datetime.datetime.now().astimezone().tzinfo
                start_timestamp = datetime.datetime.strptime(name_split, '%Y%m%d_%H%M%S')
                start_timestamp = start_timestamp.replace(tzinfo=local_tz)
                self.logging.info(start_timestamp)
                

                # build a list of frame start timestamps
                # each frame has length self.window_size samples, so:
                # fstart is 0, window_size, 2*window_size, ...
                # timestamps will be used in each row
                timestamps = [
                    start_timestamp + datetime.timedelta(seconds=fstart/self.fs) 
                    for fstart in range(0, len(x) - self.window_size + 1, self.window_size)
                ]
                
                self.logging.info("A and C weighting filters to the signal")
                # A and C weighting filters to the signal
                y_A_weighted = lfilter(self.bA, self.aA, x)
                y_C_weighted = lfilter(self.bC, self.aC, x)
                

                
                self.logging.info("Processing frame!!!")
                #process frame
                for fstart, timestamp in zip(
                    range(0, len(x) - self.window_size + 1, self.window_size),
                    timestamps):
                    # getting and weighting the frame
                    frame = x[fstart:fstart + self.window_size]
                    yA = y_A_weighted[fstart:fstart + self.window_size]
                    yC = y_C_weighted[fstart:fstart + self.window_size]

                    #weighted SPL levels
                    LA = round(get_db_level(yA, self.C), 2)
                    LC = round(get_db_level(yC, self.C), 2)
                    LZ = round(get_db_level(frame, self.C), 2)

                    # LAmax and LAmin (over "fast" sub-intervals)
                    # like splitting the frame into 8 smaller chunks
                    # for a "Fast" time weighting approach
                    fast_chunk_size = self.window_size // 8
                    fast_levels = [
                        get_db_level(yA[i:i + fast_chunk_size], self.C)
                        for i in range(0, len(frame) - fast_chunk_size + 1, fast_chunk_size)
                    ]
                    Lmax = round(np.max(fast_levels), 2)
                    Lmin = round(np.min(fast_levels), 2)
                    

                    # 1/3-octave band SPL
                    # oct_level_temp = self.get_oct_levels(frame)
                    # oct_level_temp_rounded = [round(level, 2) for level in oct_level_temp]
                    #----------------------------
                    # CALCULATE 1/3 OCTAVE LEVELS
                    #----------------------------
                    levels, freqs =third_octave_filter(frame, self.fs, order=6, limits=[12, 20000], show=0, sigbands=0)
                    # eound the levels to 2 decimal places
                    levels = [round(level, 2) for level in levels]
                    if freq_labels is None:
                        freq_labels = [f"{round(freq, 1)}Hz" for freq in freqs]
                        col_names.extend(freq_labels)  # 1/3-oct columns
                    
                    
                    #unixtimestamp
                    unix_ts = int(timestamp.timestamp())

                    #building a single row
                    level_temp = [
                        self.id_micro,
                        audio_file,
                        timestamp,
                        unix_ts,
                        LA,
                        LC,
                        LZ,
                        Lmax,
                        Lmin,
                        # *oct_level_temp_rounded # expanding the list, not a pointer
                        *levels # expanding the list, not a pointer
                    ]
                    db.append(level_temp)
                # append to all_data
                all_data.append(db)


                # --------------------------------------------------
                # csv for !THIS! audio file
                # --------------------------------------------------
                self.logging.info("")
                csv_filename = audio_file.replace(".wav", ".csv")
                self.logging.info(f"/wav_file/ replaced: {csv_filename}")

                # change thie wav folder for the acoustic one
                csv_acoustic_path = csv_filename.replace(self.wav_files, self.acoustic_params)
                self.logging.info(f"csv_acoustic_path: {csv_acoustic_path}")

                # remove the last element
                csv_full_path = os.path.dirname(csv_acoustic_path)
                self.logging.info(f"csv_full_path: {csv_full_path}")
                os.makedirs(csv_full_path, exist_ok=True)


                # saving results
                self.logging.info("Save the csv file!!")
                with open(csv_acoustic_path, mode='w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    #headers
                    writer.writerow(col_names)
                    # rows
                    for row in db:
                        writer.writerow(row)

                self.logging.info(f"Processed and wrote CSV for file: {audio_file}")


                #debugging
                df = pd.read_csv(csv_acoustic_path)
                print(df)
                exit()


                # --------------------------------------------------
                #UPLOAD TO BUCKET S3
                # --------------------------------------------------
                if self.upload_s3 is not None:
                    try:
                        self.logging.info("Uploading the csv file to bucket S3")
                        upload_file_to_s3(csv_acoustic_path, self.s3_bucket_name, self.logging)
                    except Exception as e:
                        self.logging.error(f"Failed to upload {csv_acoustic_path} to S3: {e}")
                else:
                    self.logging.info("Not Uploading the csv file to bucket S3")
            

                # ----------------------------
                # MARKING FILE AS PROICESSED
                # ----------------------------
                update_processed_files(processed_files_txt, audio_file)
                update_processed_files(processed_files_txt, csv_acoustic_path)
                processed_files.add(audio_file)
                processed_files.add(csv_acoustic_path)
                logging.info(f"Final CSV file added to the processed file. {audio_file}")

                file_end_time = time.time()
                elapsed_time = file_end_time - file_start_time
                self.logging.info(f"Processing of {audio_file} took {elapsed_time:.2f} seconds")
                print(f"Processing of {audio_file} took {elapsed_time:.2f} seconds")
                # exit()
            # -------------
            # END
            # ---------------
            except Exception as e:
                self.logging.error(f"Error processing file {audio_file}: {e}")
                continue

        return all_data



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
            storage_output_acoust_folder = load_config_acoustic('config.yaml')
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

        logging.info(f"Path: {path}")
        logging.info(f"Upload to bucket S3: {upload_s3}")
        logging.info(f"Calibration constant: {calib}")

        
        logging.info("Creating the leq_processor")
        leq_processor = LeqLevelOct(
                audio_path=path,

                id_micro=id_micro,
                fs=audio_sample_rate,
                window_size=audio_window_size,
                calibration_constant=calib,
                
                acoustic_params=storage_output_acoust_folder,
                wav_files=storage_output_wav_folder,
                s3_bucket_name=storage_s3_bucket_name,
                
                upload_s3=upload_s3,
                logging=logging
            )
            
        logging.info("Processing audio files...")
        leq_processor.process_audio_files(path)


    except KeyboardInterrupt:
        logging.error("Recording interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


    #logging end of script
    logging.info("")
    logging.info("Done!")


if __name__ == "__main__":
    main()