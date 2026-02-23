import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
DEPS_DIR = os.path.join(BASE_DIR,'deps')
sys.path.insert(0,DEPS_DIR)

import argparse
import numpy as np
import datetime
import time
import csv
import soxr
import soundfile as sf
import tflite_runtime.interpreter as tflite

import warnings

from utils import *
from logging_config import setup_logging


#removing 
warnings.filterwarnings("ignore", 
                        message="FNV hashing is not implemented in Numba",
                        category=UserWarning
                        )




def inference(path,id_micro,file_list, model_path, sample_rate, chunk_size, window_size, threshold, upload_s3, logging, output_wav_folder, output_predict_lt_folder, s3_bucket_name, cwd, yamnet_class_map_csv):
    """Perform inference on one or more audio files.

    Args:
        file_list (list[str]): List of file paths to process.
        window_size (float, optional): Window size in seconds. If None, process the entire file at once.
        threshold (float, optional): Threshold for classification.
    """
    logging.info("")
    logging.info("Making inference")

    # ---------------------------
    # INIZIALATIN PROCESSING FILE
    # ---------------------------
    processed_files_txt = os.path.join(path, "processed_predictions.txt")
    processed_files_txt = processed_files_txt.replace("wav_files", "predictions_litle")
    logging.info(f"Saving the processed file txt here --> {processed_files_txt}")
    
    processed_files = load_processed_files(processed_files_txt)
    
    # --------------------------------------------------------
    # 1) create the TFLite interpreter
    # --------------------------------------------------------
    logging.info("Setting the TF Model and loading the classes")
    
    if model_path is not None:
        interpreter = tflite.Interpreter(model_path=model_path)
        logging.info(f"Model path --> {model_path}")
    else:
        raise Exception('Model Path doesnt exist.')
    
    yamnet_classes_csv = os.path.join(cwd, yamnet_class_map_csv)
    yamnet_classes = class_names_csv(yamnet_classes_csv)
    logging.info("Classes map loaded")


    # --------------------
    # Processing audio files
    # --------------------
    for audio_file in file_list:
        try:
            logging.info("")
            logging.info(f"Processing --> {audio_file}")

            if audio_file in processed_files:
                logging.info(f"Skipping {audio_file}, already processed.")
                continue
            
            file_start_time = time.time()

            # -----------------------------------------------------------
            # csv file name and folder
            # -----------------------------------------------------------
            wav_filename = os.path.basename(audio_file)  # e.g. 20250108_142606.wav
            logging.info(f"WAV file name --> {wav_filename}")

            # name wave file
            wav_file_raw = os.path.splitext(wav_filename)[0]

            # setting time
            local_tz = datetime.datetime.now().astimezone().tzinfo
            start_timestamp = datetime.datetime.strptime(wav_file_raw, '%Y%m%d_%H%M%S')
            start_timestamp = start_timestamp.replace(tzinfo=local_tz)
            logging.info(f"Start_timestamp --> {start_timestamp}")
            

            if window_size is None:
                csv_filename = wav_filename.replace(".wav", "_tflt.csv")  # e.g. 20250108_142606.csv
            else:
                csv_filename = wav_filename.replace(".wav", f"_tflt_w_{window_size}.csv")  # e.g. 20250108_142606.csv
            logging.info(f"CSV filename --> {csv_filename}")



            prediction_folder = os.path.dirname(audio_file).replace(output_wav_folder, output_predict_lt_folder)
            os.makedirs(prediction_folder, exist_ok=True)
            logging.info(f"Making litRT prediction folder --> {prediction_folder}")

            csv_full_path = os.path.join(prediction_folder, csv_filename)
            logging.info(f"CSV FULL PATH --> {csv_full_path}")


            # --------------------------------------------------------
            # 2 get input/output details
            # --------------------------------------------------------
            logging.info("")
            logging.info("INTERPRETER --> Get input/output details")
            input_details = interpreter.get_input_details()
            logging.info(f"Input details --> {input_details}")
            output_details = interpreter.get_output_details()
            waveform_input_index = input_details[0]['index']
            scores_output_index = output_details[0]['index']


            # --------------------------------------------------------
            # 3 prepare waveform input (0.975s @ 16kHz => 15600 samples)
            # Decode the WAV file
            # -----------------------------------------------------------
            logging.info("")
            logging.info("Decoding WAV file")
            wav_data, sr = sf.read(audio_file, dtype=np.int16)
            assert wav_data.dtype == np.int16, f'Bad sample type: {wav_data.dtype}'

            waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
            waveform = waveform.astype('float32')

            # convert to mono and the sample rate expected by YAMNet
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
                logging.info("Audio file converted to mono")
            if sr != sample_rate:
                #waveform = resampy.resample(waveform, sr, sample_rate)
                waveform = soxr.resample(waveform, sr, sample_rate)
                logging.info("Audio file resampled to 16KHz")


            # -----------------------------------------------------------
            # create a fresh CSV data list for this file
            # -----------------------------------------------------------
            csv_data = [["id_micro", "Filename", "Timestamp", "Unixtimestamp", "class", "probability"]]
            
            if window_size is None:
                logging.info("")
                logging.info("Processing the whole audio file")
                # --------------------------------------------------------
                # 4 resize input tensor and allocate
                # --------------------------------------------------------
                interpreter.resize_tensor_input(waveform_input_index, [waveform.size], strict=False)
                interpreter.allocate_tensors()


                # --------------------------------------------------------
                # 5set input tensor and run inference
                # --------------------------------------------------------
                interpreter.set_tensor(waveform_input_index, waveform)
                interpreter.invoke()
                scores = interpreter.get_tensor(scores_output_index)  # shape (1, 521)

    
                # ---------------------------------------------------------
                # predcition
                # ---------------------------------------------------------
                prediction = np.mean(scores, axis=0)
                # top 3
                top3_i = np.argsort(prediction)[::-1][:3]
                top3_classes = [str(yamnet_classes[i]) for i in top3_i]
                top3_probs = [f"{prediction[i]:.4f}" for i in top3_i]
                logging.info(f"top 3 prediction --> {top3_classes} \t{top3_probs}")

                #unixtimestamp
                unix_ts = int(start_timestamp.timestamp())

                csv_data.append([
                    id_micro,
                    audio_file,
                    str(start_timestamp),
                    unix_ts,
                    str(top3_classes),
                    str(top3_probs)
                ])
                logging.info("Adding the result to the CSV file")
                logging.info("")


            # -------------------------------
            # WINDOWED
            # -------------------------------
            else:
                logging.info("")
                logging.info("Processing windowed audio file")

                window_size_samples = int(window_size * sample_rate)
                logging.info(f"Window size --> {window_size_samples}")
                
                
                start_idx = 0
                target_len = 15600
                hop = target_len
                interpreter.allocate_tensors()
                
                while start_idx < len(waveform):
                    #end_idx = min(start_idx + window_size_samples, len(waveform))
                    end_idx = min(start_idx + target_len, len(waveform))

                    #waveform_window = waveform[start_idx:end_idx]
                    waveform_window = waveform[start_idx:end_idx].astype(np.float32,copy=False)

                    # [NEW] if the last window is shorter than target_len, pad with zeros -> 5: set input tensor and run inference
                    if waveform_window.shape[0] < target_len:
                        waveform_window = np.pad(waveform_window, (0,target_len - waveform_window.shape[0]))
                    waveform_window = np.ascontiguousarray(waveform_window)

                    logging.info(f"waveform_window --> {waveform_window}")
                    logging.info(f"Making prediction ... ")
                    interpreter.set_tensor(waveform_input_index, waveform_window)
                    interpreter.invoke()
                    scores = interpreter.get_tensor(scores_output_index)  # shape (1, 521)
                    logging.info(f"Scores --> {scores}")
                    logging.info(f"Scores shape --> {scores.shape}")

                    #interpreter.resize_tensor_input(waveform_input_index, [waveform_window.size], strict=False)
                    #interpreter.allocate_tensors()

                    # --------------------------------------------------------
                    # 5set input tensor and run inference
                    # --------------------------------------------------------
                    #logging.info("Making prediction")
                    #interpreter.set_tensor(waveform_input_index, waveform_window)
                    #interpreter.invoke()
                    #scores = interpreter.get_tensor(scores_output_index)  # shape (1, 521)


                    # ---------------------------------------------------------
                    # predcition
                    # ---------------------------------------------------------
                    prediction = np.mean(scores, axis=0)
                    # top 3
                    top3_i = np.argsort(prediction)[::-1][:3]
                    top3_classes = [str(yamnet_classes[i]) for i in top3_i]
                    top3_probs = [f"{prediction[i]:.4f}" for i in top3_i]
                    logging.info(f"top 3 prediction --> {top3_classes} \t{top3_probs}")

                    # timestamp for this window
                    start_time_s = start_idx / sample_rate
                    window_timestamp_actual = start_timestamp + datetime.timedelta(seconds=int(start_time_s))
                    unix_ts = int(window_timestamp_actual.timestamp())
                    
                    csv_data.append([
                        id_micro,
                        audio_file,
                        window_timestamp_actual,
                        unix_ts,
                        str(top3_classes),
                        str(top3_probs)
                    ])

                    start_idx = end_idx
                    logging.info("Adding the result to the CSV file")
                    logging.info("")


            # -----------------------------------------------------------
            # save csv
            # -----------------------------------------------------------
            with open(csv_full_path, mode="w", newline="") as final_csv:
                writer = csv.writer(final_csv)
                writer.writerows(csv_data)
            logging.info(f"Final CSV file saved at {csv_full_path}")

            # df = pd.read_csv(csv_full_path)


            # -------------------
            # UPLOAD TO BUCKET S3
            # -------------------
            if upload_s3 is not None:
                try:
                    upload_file_to_s3(csv_full_path, s3_bucket_name, logging)
                except Exception as e:
                    logging.error(f"Failed to upload {csv_full_path} to S3: {e}")
            else:
                logging.warning("The final CVS final will not be update to the bucket S3")

            
            # ----------------------------
            # MARKING FILE AS PROICESSED
            # ----------------------------
            update_processed_files(processed_files_txt, csv_full_path)
            processed_files.add(audio_file)
            logging.info(f"Final CSV file added to the processed file. {audio_file}")
            logging.info(f"Final CSV file added to the processed file. {csv_full_path}")

            
            file_end_time = time.time()
            elapsed_time = file_end_time - file_start_time
            logging.info(f"Processing of {audio_file} took {elapsed_time:.2f} seconds")
            print(f"Processing of {audio_file} took {elapsed_time:.2f} seconds")
            # exit()


        # -------------
        # END
        # ---------------
        except Exception as e:
                logging.error(f"Error processing file {audio_file}: {e}")
                continue



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
    parser.add_argument('-p', '--path', type=str, required=False,help='Folder containing WAV files to process')
    
    parser.add_argument('-w', '--window-size', type=float, default=None,
                        help='Window size in seconds for processing audio files. '
                             'Default is None for processing the entire audio.')

    parser.add_argument('-t', '--threshold', type=float, default=None, help='Classification threshold for predictions.')
    parser.add_argument('-m', '--model-path', type=str, default=None, help='Insert the model path to make predictions.')
    parser.add_argument('-u', '--upload-S3', action='store_true',default=False, help='If provided, upload the final CSV to S3.')
    return parser.parse_args()




def main():
    try:
        logging = setup_logging(script_name="inference_tflite")
        args = parse_arguments()
        
        logging.info("Staarting process!!")
        logging.info("")
        
        cwd = os.path.dirname(os.path.realpath(__file__))
        home_dir = os.getenv("HOME")
        logging.info(f"Current working dir --> {cwd}")
        logging.info(f"Home dir --> {home_dir}")
        
        
        logging.info("Getting the element form the yamnl file")
        try: 
            id_micro, location_record, location_place, location_point, storage_s3_bucket_name, \
            storage_output_wav_folder, storage_output_acoust_folder, storage_output_predict_folder, \
            storage_output_predict_lt_folder, prediction_yamnet_class_map_csv, prediction_sample_rate, \
            prediction_chunk_size, _, prediction_model_tflt= load_config_inference('config.yaml',cwd)
            logging.info("Config loaded successfully")
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return


        # ----------------------------
        # PARSE ARGUMENTS & CONFIG
        # ----------------------------
        #WAV PÀTH
        if args.path:
            path = args.path
        else:
            path = os.path.join(home_dir, location_record, location_place, location_point, storage_output_wav_folder)
            if os.path.exists(path):
                logging.info(f"Path exists --> {path}")
            else:
                raise Exception('Path doesnt exist.')
        
        # DEEP LEARNING MODEL PATH
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = prediction_model_tflt

        # WINDOW
        if args.window_size:
            window_size = args.window_size
        else:
            window_size = None

        # THRESHOLD
        if args.threshold:
            threshold = args.threshold
        else:
            threshold = None

        # UPLOAD BUCKET S3
        if args.upload_S3:
            upload_s3 = args.upload_S3
        else:
            upload_s3 = None

    except Exception as e:
        logging.error(f"Errorgetting the config info: {e}")


    logging.info(f"Path: {path}")
    logging.info(f"ID Micro: {id_micro}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Window size: {window_size}")
    logging.info(f"Probability treshold: {threshold}")
    logging.info(f"Upload to bucket S3: {upload_s3}")



    # -----------------------
    # GETTING AUDIO FILES
    # -----------------------
    try:
        audio_files = [f for f in os.listdir(path) if f.lower().endswith('.wav')]
        full_paths = [os.path.join(path, file) for file in audio_files]
    except Exception as e:
        logging.error(f"Errorgetting the audio files: {e}")

    logging.info(f"Found {len(audio_files)} audio files: {audio_files}")



    # ----------
    # INFERENCE
    # ----------
    try:
        inference(
            path=path,
            file_list=full_paths,
            
            id_micro=id_micro,
            model_path=model_path,
            yamnet_class_map_csv=prediction_yamnet_class_map_csv,
            
            sample_rate=prediction_sample_rate,
            chunk_size=prediction_chunk_size,
            window_size=window_size,
            threshold=threshold,
            
            upload_s3=upload_s3,
            
            output_wav_folder=storage_output_wav_folder,
            output_predict_lt_folder=storage_output_predict_lt_folder,
            s3_bucket_name=storage_s3_bucket_name,
            
            cwd=cwd,
            
            logging=logging
        )
        logging.info("Inference finished.")
    
    except Exception as e:
        logging.error(f"Error making inference: {e}")



if __name__ == '__main__':
    main()
