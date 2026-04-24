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

from utils import load_config_inference, class_names_csv, upload_file_to_s3
from logging_config import setup_logging


#removing 
warnings.filterwarnings("ignore", 
                        message="FNV hashing is not implemented in Numba",
                        category=UserWarning
                        )


def filter_predictions(predictions,threshold):
    top_i = int(np.argmax(predictions))
    top_pred = float(predictions[top_i])
    if top_pred > threshold:
        return top_i, top_pred
    return None, None

def inference(path,id_micro,file_list, model_path, sample_rate,window_size, threshold, upload_s3, logging, output_wav_folder, output_predict_lt_folder, s3_bucket_name, cwd, yamnet_class_map_csv   , num_threads):
    
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

        interpreter = tflite.Interpreter(
                                        model_path     =       model_path,
                                        num_threads    =       num_threads
                                        )
        logging.info(f"Model path --> {model_path}")
    else:
        raise Exception('Model Path doesnt exist.')
    
    yamnet_classes_csv = os.path.join(cwd, yamnet_class_map_csv)
    yamnet_classes = class_names_csv(yamnet_classes_csv)
    logging.info("Classes map loaded")


    # --------------------
    # Processing audio files
    # --------------------
    logging.info("INTERPRETER --> Get input/output details")
    input_details = interpreter.get_input_details()
    logging.info(f"Input details --> {input_details}")
    output_details = interpreter.get_output_details()
    waveform_input_index = input_details[0]['index']
    scores_output_index = output_details[0]['index']

    # El modelo ya viene con shape fijo [15600], así que reservamos una vez
    interpreter.allocate_tensors()


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

            # name wave file
            wav_file_raw = os.path.splitext(wav_filename)[0]

            # setting time
            local_tz = datetime.datetime.now().astimezone().tzinfo
            start_timestamp = datetime.datetime.strptime(wav_file_raw, '%Y%m%d_%H%M%S')
            start_timestamp = start_timestamp.replace(tzinfo=local_tz)
            

            if window_size is None:
                csv_filename = wav_filename.replace(".wav", "_tflt.csv")  # e.g. 20250108_142606.csv
            else:
                csv_filename = wav_filename.replace(".wav", f"_tflt_w_{window_size}.csv")  # e.g. 20250108_142606.csv



            prediction_folder = os.path.dirname(audio_file).replace(output_wav_folder, output_predict_lt_folder)
            os.makedirs(prediction_folder, exist_ok=True)

            csv_full_path = os.path.join(prediction_folder, csv_filename)


            # --------------------------------------------------------
            # 2 get input/output details --> Removed for optimization reasons
            # --------------------------------------------------------
            logging.info("")

            # --------------------------------------------------------
            # 3 prepare waveform input (0.975s @ 16kHz => 15600 samples)
            # Decode the WAV file
            # -----------------------------------------------------------
            t0 = time.perf_counter()
            wav_data, sr = sf.read(audio_file, dtype=np.int16)
            t1 = time.perf_counter()
            assert wav_data.dtype == np.int16, f'Bad sample type: {wav_data.dtype}'


            waveform = wav_data.astype(np.float32) / 32768.0 # Convert to [-1.0, +1.0]

            # convert to mono and the sample rate expected by YAMNet
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
                logging.info("Audio file converted to mono")
            if sr != sample_rate:
                #waveform = resampy.resample(waveform, sr, sample_rate)
                waveform = soxr.resample(waveform, sr, sample_rate)
                logging.info("Audio file resampled to 16KHz")
            t2 = time.perf_counter()

            # -----------------------------------------------------------
            # create a fresh CSV data list for this file
            # -----------------------------------------------------------
            with open(csv_full_path, mode="w", newline="") as final_csv:
                writer = csv.writer(final_csv)
                writer.writerow(["id_micro", "Filename", "Timestamp", "Unixtimestamp", "class", "probability"])


                if window_size is None:
                    logging.info("")
                    logging.info(f"Processing the whole audio file: {audio_file}")
                    # --------------------------------------------------------
                    # 4 resize input tensor and allocate
                    # --------------------------------------------------------
                    t_inf_start = time.perf_counter()
                    input_len = int(waveform.shape[0])
                    interpreter.resize_tensor_input(waveform_input_index, [input_len], strict=False)
                    interpreter.allocate_tensors()


                    # --------------------------------------------------------
                    # 5set input tensor and run inference
                    # --------------------------------------------------------
                    interpreter.set_tensor(waveform_input_index, waveform)
                    interpreter.invoke()
                    scores = interpreter.get_tensor(scores_output_index)  
                    t_inf_end = time.perf_counter()
        
                    # ---------------------------------------------------------
                    # predcition
                    # ---------------------------------------------------------
                    prediction = np.mean(scores, axis=0)

                    top_i,top_prediction = filter_predictions(prediction,threshold)
                    
                    if top_i is not None:
                        top_class = yamnet_classes[top_i]
                        top_prediction = f"{top_prediction:.4f}"
                    else:
                        top_class = []
                        top_prediction = []
                    

                    #unixtimestamp
                    unix_ts = int(start_timestamp.timestamp())

                    """
                    csv_data.append([
                        id_micro,
                        audio_file,
                        str(start_timestamp),
                        unix_ts,
                        str(top_class),
                        str(top_prediction)
                    ])
                    """
                    writer.writerow([
                        id_micro,
                        audio_file,
                        str(start_timestamp),
                        unix_ts,
                        str(top_class),
                        str(top_prediction)
                    ])
                    

                # -------------------------------
                # WINDOWED
                # -------------------------------
                else:
                    logging.info(f"Processing windowed audio file: {audio_file}")

                    target_len = 15600
                    input_buffer = np.zeros((target_len,), dtype=np.float32)

                    t_inf_start = time.perf_counter()

                    for start_idx in range(0, len(waveform), target_len):

                        end_idx = min(start_idx + target_len, len(waveform))
                        valid_len = end_idx - start_idx

                        input_buffer.fill(0.0)
                        input_buffer[:valid_len] = waveform[start_idx:end_idx]

                        interpreter.set_tensor(waveform_input_index, input_buffer)
                        interpreter.invoke()
                        scores = interpreter.get_tensor(scores_output_index)

                        prediction = np.mean(scores, axis=0)

                        top_i, top_prediction = filter_predictions(prediction, threshold)
                        if top_i is not None:
                            top_class = yamnet_classes[top_i]
                            top_prediction = f"{top_prediction:.4f}"
                        else:
                            top_class = []
                            top_prediction = []

                        start_time_s = start_idx / sample_rate
                        window_timestamp_actual = start_timestamp + datetime.timedelta(seconds=int(start_time_s))
                        unix_ts = int(window_timestamp_actual.timestamp())

                        writer.writerow([
                            id_micro,
                            audio_file,
                            window_timestamp_actual,
                            unix_ts,
                            str(top_class),
                            str(top_prediction)
                        ])

                    t_inf_end = time.perf_counter()


                t3 = time.perf_counter()
                logging.info(
                    "Timings %s | read=%.3f s | preprocess=%.3f s | inference=%.3f s | total=%.3f s",
                    audio_file,
                    t1 - t0,
                    t2 - t1,
                    t_inf_end - t_inf_start,
                    t3 - t0
                )
            # -----------------------------------------------------------
            # save csv
            # -----------------------------------------------------------

            logging.info(f"Final CSV file saved at {csv_full_path}")

            
            # ----------------------------
            # MARKING FILE AS PROCESSED
            # ----------------------------
            update_processed_files(processed_files_txt, audio_file)
            processed_files.add(audio_file)
            
            file_end_time = time.time()
            elapsed_time = file_end_time - file_start_time
            logging.info(f"Processing of {audio_file} took {elapsed_time:.2f} seconds")

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
    parser.add_argument('--num-threads', type=int, default=2,
                    help='Number of TFLite threads')
    parser.add_argument('-s','--step', type=int, default=5 , help='Step of files to process, default is 5 (process every 5 files)')
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

        if args.num_threads:
            num_threads = args.num_threads
        if args.path:
            path = args.path
        else:
            path = os.path.join(
            home_dir,
            location_record,
            location_place,
            location_point,
            "AUDIOMOTH",
            storage_output_wav_folder
            )
            if os.path.exists(path):
                logging.info(f"Path exists --> {path}")
            else:
                raise Exception('Path doesnt exist.')
        
        # DEEP LEARNING MODEL PATH
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = "/root/IoT_microphone_scripts-main/03_inference/yamnet.tflite"
        
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
        
        if args.step:
            step = args.step
        else:
            step = 5

    except Exception as e:
        logging.error(f"Error getting the config info: {e}")

    
    logging.info(f"Path: {path}")
    logging.info(f"ID Micro: {id_micro}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Window size: {window_size}")
    logging.info(f"Probability treshold: {threshold}")
    logging.info(f"Upload to bucket S3: {upload_s3}")


    try:

        audio_files = sorted([f for f in os.listdir(path) if f.lower().endswith('.wav')])
        audio_files = audio_files[::step]  # Process every nth file based on the specified range given in the arguments
        full_paths = [os.path.join(path, file) for file in audio_files]

    except Exception as e:
        logging.error(f"Error getting the audio files: {e}")
        return

    logging.info(f"Found {len(audio_files)} audio files: {audio_files}")


    try:
        inference(
            path=path,
            file_list=full_paths,
            id_micro=id_micro,
            model_path=model_path,
            yamnet_class_map_csv=prediction_yamnet_class_map_csv,
            sample_rate=prediction_sample_rate,
            window_size=window_size,
            threshold=threshold,
            upload_s3=upload_s3,    
            output_wav_folder=storage_output_wav_folder,
            output_predict_lt_folder=storage_output_predict_lt_folder,
            s3_bucket_name=storage_s3_bucket_name,
            cwd=cwd,
            num_threads = num_threads,    
            logging=logging
        )

        logging.info("Inference finished.")
    
    except Exception as e:
        logging.error(f"Error making inference: {e}")



if __name__ == '__main__':
    main()
