from __future__ import division, print_function

import os
import argparse
import numpy as np
import datetime
import csv

import resampy
import soundfile as sf
import tensorflow as tf

from . import params as yamnet_params
from . import yamnet as yamnet_model

import warnings

from utils import *
from logging_config import setup_logging


warnings.filterwarnings("ignore", 
                        message="FNV hashing is not implemented in Numba",
                        category=UserWarning)





def inference(file_list, model_path, sample_rate, chunk_size, window_size, threshold, upload_s3, logging, output_wav_folder, output_predict_folder, s3_bucket_name, cwd, yamnet_class_map_csv):
    """Perform inference on one or more audio files.
    Args:
        file_list (list[str]): List of file paths to process.
        window_size (float, optional): Window size in seconds. If None, process the entire file at once.
        threshold (float, optional): Threshold for classification.
    """
    logging.info("")
    logging.info("Making inference")

    # ----------------------------
    # 1) Create the TF object
    # ----------------------------
    logging.info("Setting the TF Model and loading the classes")
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights(model_path)
    yamnet_classes = yamnet_model.class_names(yamnet_class_map_csv)

    logging.info("Model and Classes map loaded")


    # --------------------
    # Processing audio files
    # --------------------
    for audio_file in file_list:
        logging.info("")
        logging.info(f"Processing --> {audio_file}")
        
        # -----------------------------------------------------------
        # CSV file name and folder
        # -----------------------------------------------------------
        wav_filename = os.path.basename(audio_file)  # e.g. 20250108_142606.wav
        logging.info(f"WAV file name --> {wav_filename}")

        wav_file_raw = os.path.splitext(wav_filename)[0]
        start_timestamp = datetime.datetime.strptime(wav_file_raw, '%Y%m%d_%H%M%S')
        if window_size is None:
            csv_filename = wav_filename.replace(".wav", "_tf.csv")  # e.g. 20250108_142606.csv
        else:
            csv_filename = wav_filename.replace(".wav", f"_tf_w_{window_size}.csv")  # e.g. 20250108_142606.csv
            
        logging.info(f"CSV filename --> {csv_filename}")

        prediction_folder = os.path.dirname(audio_file).replace(output_wav_folder, output_predict_folder)
        os.makedirs(prediction_folder, exist_ok=True)
        logging.info(f"Making TF prediction folder --> {prediction_folder}")

        csv_full_path = os.path.join(prediction_folder, csv_filename)
        logging.info(f"CSV FULL PATH --> {csv_full_path}")



        # -----------------------------------------------------------
        # Decode the WAV file
        # -----------------------------------------------------------
        logging.info("")
        logging.info("Decoding WAV file")
        wav_data, sr = sf.read(audio_file, dtype=np.int16)
        assert wav_data.dtype == np.int16, f'Bad sample type: {wav_data.dtype}'

        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        waveform = waveform.astype('float32')

        # Convert to mono and the sample rate expected by YAMNet
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
            logging.info("Audio file converted to mono")
        if sr != params.sample_rate:
            waveform = resampy.resample(waveform, sr, params.sample_rate)
            logging.info("Audio file resampled to 16KHz")


        # -----------------------------------------------------------
        # Perform inference
        # -----------------------------------------------------------
        csv_data = [["filename", "date", "class", "probability"]]

        if window_size is None:
            #single chunk inference
            scores, embeddings, spectrogram = yamnet(waveform)
            prediction = np.mean(scores, axis=0)

            # top 3
            top3_i = np.argsort(prediction)[::-1][:3]
            top3_classes = [str(yamnet_classes[i]) for i in top3_i]
            top3_probs = [f"{prediction[i]:.4f}" for i in top3_i]

            csv_data.append([
                wav_filename,
                str(start_timestamp),
                str(top3_classes),
                str(top3_probs)
            ])


        # -----------------------------------------------------------
        # Windowed inference
        # -----------------------------------------------------------
        else:
            window_size_samples = int(window_size * params.sample_rate)
            start_idx = 0
            while start_idx < len(waveform):
                end_idx = min(start_idx + window_size_samples, len(waveform))
                waveform_window = waveform[start_idx:end_idx]

                scores, embeddings, spectrogram = yamnet(waveform_window)
                prediction = np.mean(scores, axis=0)

                # top 3
                top3_i = np.argsort(prediction)[::-1][:3]
                top3_classes = [str(yamnet_classes[i]) for i in top3_i]
                top3_probs = [f"{prediction[i]:.4f}" for i in top3_i]

                # timestamp for this window
                start_time_s = start_idx / params.sample_rate
                window_timestamp_actual = start_timestamp + datetime.timedelta(seconds=int(start_time_s))
                formatted_time = window_timestamp_actual.strftime("%H:%M:%S")

                csv_data.append([
                    wav_filename,
                    formatted_time,
                    str(top3_classes),
                    str(top3_probs)
                ])

                start_idx = end_idx


        # -----------------------------------------------------------
        # Save CSV
        # -----------------------------------------------------------
        with open(csv_full_path, mode="w", newline="") as final_csv:
            writer = csv.writer(final_csv)
            writer.writerows(csv_data)
        logging.info(f"Final CSV file saved at {csv_full_path}")


        # -------------
        # UPLOAD TO BUCKET S3
        # ---------------
        if upload_s3 is not None:
            try:
                upload_file_to_s3(csv_full_path, s3_bucket_name, logging)
            except Exception as e:
                logging.error(f"Failed to upload {csv_full_path} to S3: {e}")
        else:
            logging.warning("The final CVS final will not be update to the bucket S3")





def parse_arguments():
    parser = argparse.ArgumentParser(description='Make prediction with YAMNet model for audio files')
    parser.add_argument('-p', '--path', type=str, required=False,
                        help='Folder containing WAV files to process')

    parser.add_argument('-w', '--window-size', type=float, default=None,
                        help='Window size in seconds for processing audio files. '
                             'Default is None for processing the entire audio.')

    parser.add_argument('-t', '--threshold', type=float, default=None,
                        help='Classification threshold for predictions.')

    parser.add_argument('-m', '--model-path', type=str, default=None,
                        help='Insert the model path to make predictions.')

    parser.add_argument('-u', '--upload-S3', action='store_true',default=False,
                        help='If provided, upload the final CSV to S3.')

    return parser.parse_args()




def main():
    try:
        logging = setup_logging(script_name="inference")
        args = parse_arguments()
        cwd = os.path.dirname(os.path.realpath(__file__))
        home_dir = os.getenv("HOME")

        s3_bucket_name, place, point, output_parent_folder, output_wav_folder, output_acoust_folder, output_predict_folder, _, sample_rate, chunk_size, model_tf, _, yamnet_class_map_csv = load_config_inference('config.yaml',cwd)
        
        # ----------------------------
        # PARSE ARGUMENTS & CONFIG
        # ----------------------------
        #WAV PÀTH
        if args.path:
            path = args.path
        else:
            path = os.path.join(home_dir, place, point, output_wav_folder)
            if os.path.exists(path):
                logging.info(f"Path exists --> {path}")
            else:
                raise Exception('Path doesnt exist.')

        # DEEP LEARNING MODEL PATH
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = model_tf

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



    # -----------------------
    # INFERENCE
    # -----------------------
    try:
        inference(
            file_list=full_paths,
            model_path=model_path,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            window_size=window_size,
            threshold=threshold,
            upload_s3=upload_s3,
            logging=logging,
            output_wav_folder=output_wav_folder,
            output_predict_folder=output_predict_folder,
            s3_bucket_name=s3_bucket_name,
            cwd=cwd,
            yamnet_class_map_csv=yamnet_class_map_csv
        )
        logging.info("Inference finished.")
    
    except Exception as e:
        logging.error(f"Error making inference: {e}")



if __name__ == '__main__':
    main()
