import os
import subprocess
import logging
import yaml
import csv
import boto3
import configparser
import datetime

import leq_levels_oct_weighting_C as m
# Constantes de inicializacion
T = 1
# C = -54.70




# -----------------------------------
# REGULAR FUNCTIONS
# -----------------------------------



def upload_parts(s3, f, PART_SIZE, parts_count, bucket, object_key, idx, uploaded_parts, logger):
    while True:
        chunk = f.read(PART_SIZE)
        if not chunk:
            break

        part_key = f"{object_key}.part{idx:04d}"
        logger.info(f"Uploading part {idx+1}/{parts_count} to {part_key} ...")

        s3.put_object(
            Bucket=bucket,
            Key=part_key,
            Body=chunk,
            ContentLength=len(chunk),
            ContentType="application/octet-stream",
        )

        logger.info(f"Uploaded part {idx+1}/{parts_count} to {part_key}")
        uploaded_parts += 1
        idx += 1

    return uploaded_parts, idx

def load_processed_files(processed_file_path):
    """Load the set of processed filenames from a text file."""
    if os.path.exists(processed_file_path):
        with open(processed_file_path, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return set()

def upload_manifest(s3, file_path, bucket, object_key, PART_SIZE, uploaded_parts, logger):
    manifest_body = (
        f"version=1\n"
        f"bucket={bucket}\n"
        f"original_key={object_key}\n"
        f"part_size={PART_SIZE}\n"
        f"parts={uploaded_parts}\n"
    )

    s3.put_object(
        Bucket=bucket,
        Key=f"{object_key}.manifest",
        Body=manifest_body.encode("utf-8"),
        ContentType="text/plain",
    )
    logger.info(f"Uploaded manifest: {object_key}.manifest")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def twenty_db_fix(levels):
    levels_fix = []
    """"
    for row in levels:
        row_fix = []
        for octave_level in row:
            row_fix.append(octave_level + 20)
        levels_fix.append(row_fix)
    """
    for level in levels:
        levels_fix.append(level + 20)

    return levels_fix

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_config_record(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

        location_record = config["location"]["record"]
        location_place = config['location']['place']
        location_point = config['location']['point']

        audio_format = config['audio']['format']
        audio_channels = config['audio']['channels']
        audio_sample_rate = config['audio']['sample_rate']
        audio_chunk_size = config['audio']['chunk_size']

        storage_s3_bucket_name = config['storage']['s3_bucket_name']
        storage_output_wav_folder = config['storage']['output_wav_folder']
        file_part_size = config['storage']['file_part_size']

    return location_record, location_place, location_point, audio_format, audio_channels, audio_sample_rate, audio_chunk_size, storage_s3_bucket_name, storage_output_wav_folder,file_part_size

def load_config_acoustic(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

        id_micro = config["device"]["id_micro"]
        name_device = config["device_type"]["name"]

        location_record = config["location"]["record"]
        location_place = config['location']['place']
        location_point = config['location']['point']

        audio_sample_rate = config['audio']['sample_rate']
        audio_window_size = config['audio']['window_size']
        audio_calibration_constant = config['audio']['calibration_constant']
        calibration_constants_folder = config['audio']['calibration_constants_folder']

        storage_s3_bucket_name = config['storage']['s3_bucket_name']
        storage_output_wav_folder = config['storage']['output_wav_folder']
        storage_output_acoust_folder = config['storage']['output_acoust_folder']
        
    return name_device,id_micro, location_record, location_place, location_point, audio_sample_rate, audio_window_size, audio_calibration_constant, storage_s3_bucket_name, storage_output_wav_folder, storage_output_acoust_folder,calibration_constants_folder

def load_config_inference(yaml_path: str, cwd: str) -> dict:
    
    with open(yaml_path, 'r') as file:

        config = yaml.safe_load(file)

        id_micro = config["device"]["id_micro"]
        name_device = config["device_type"]["name"]

        location_record = config["location"]["record"]
        location_place = config['location']['place']
        location_point = config['location']['point']

        storage_s3_bucket_name = config['storage']['s3_bucket_name']
        storage_output_wav_folder = config['storage']['output_wav_folder']
        storage_output_acoust_folder = config['storage']['output_acoust_folder']
        storage_output_predict_folder = config['storage']['output_predict_folder']
        storage_output_predict_lt_folder = config['storage']['output_predict_lt_folder']

        prediction_yamnet_class_map_csv = config['prediction']['yamnet_class_map_csv']
        prediction_sample_rate = config['prediction']['sample_rate']
        prediction_chunk_size = config['prediction']['chunk_size']
        prediction_model_tf = os.path.join(cwd, config['prediction']['model_tf'])
        prediction_model_tflt = os.path.join(cwd, config['prediction']['model_tflt'])

    return name_device,id_micro, location_record, location_place, location_point, storage_s3_bucket_name, storage_output_wav_folder, storage_output_acoust_folder, storage_output_predict_folder, storage_output_predict_lt_folder, prediction_yamnet_class_map_csv, prediction_sample_rate, prediction_chunk_size, prediction_model_tf, prediction_model_tflt

def class_names_csv(class_map_csv):
    import numpy as np
    with open(class_map_csv) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)   # Skip header
        return np.array([display_name for (_, _, display_name) in reader])

def upload_file_to_s3(file_path, bucket_name, logging):
    s3 = boto3.client('s3')
    s3_path = "/".join(file_path.split("/")[3:])
    logging.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_path}")
    try:
        s3.upload_file(file_path, bucket_name, s3_path)
        logging.info("Upload successful!")
    except Exception as e:
        logging.error(f"Failed to upload to S3: {e}")


def get_audiofiles(path):
    """
    Args:
        path (str): The path to the directory containing the audio files.
    Returns:
        list: A list containing the full paths to all '.wav' files in the specified directory.
    """
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return []
    
    audio_files = [file for file in os.listdir(path) if file.lower().endswith('.wav')]
    
    if len(audio_files) == 0:
        print(f"Found 0 audio files in {path}, trying wav_files folder")
        audio_files = [file for file in os.listdir(os.path.join(path,'wav_files')) if file.lower().endswith('.wav')]
    
    print(f"Audio files: {audio_files}")
    return audio_files


def find_audiomoth_folders(base_path: str):
    """Recursively find all subdirectories containing an 'AUDIOMOTH' folder."""
    for root, dirs, _files in os.walk(base_path):
        if "AUDIOMOTH" in dirs:
            yield root

def get_valid_audio_files(audio_path, processed_txt_path):
    processed_files = load_processed_files(processed_txt_path)
    audio_files = get_audiofiles(audio_path)

    if not audio_files:
        logging.error(f"No audio files found in: {audio_path}")
        raise FileNotFoundError(f"No audio files found in: {audio_path}")
    
    valid_audio_files = []
    valid_audio_files = [f for f in audio_files if f not in processed_files]

    if not valid_audio_files:
        logging.error("Already processed all files in folder, nothing to do.")
        raise ValueError("Already processed all files in folder, nothing to do.")
    
    valid_audio_files = sorted(valid_audio_files)

    return valid_audio_files

def update_processed_files(processed_file_path, filename):
    """Append a processed filename to the text file."""
    with open(processed_file_path, "a") as f:
        f.write(filename + "\n")

def get_device_id(metadata) -> str:
    artist_tags = metadata.tags.get("artist", ["songmeter"])
    if not artist_tags or len(artist_tags[0].split(" ")) < 2:
        return "songmeter"
    device_id = artist_tags[0].split(" ")[1].lower()
    logging.info(f"Device ID: {device_id}")
    return device_id


def read_calibration_constants(ini_file: str) -> dict:
    cfg = configparser.ConfigParser()
    read_files = cfg.read(ini_file)

    print("Read files:", read_files)
    print("Sections:", cfg.sections())

    return {k: float(v) for k, v in cfg["CalibrationConstants"].items()}

def parse_dt(s):
    # "YYYY-MM-DD HH:MM:SS.mmm"
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")