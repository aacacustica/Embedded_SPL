import os
import argparse
import datetime
import pyaudio
import wave
import boto3
import threading
import time  


from pathlib import Path
from utils import *
from logging_config import setup_logging
from queue import Queue
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

import os
import time
import boto3
from botocore.config import Config

upload_queue = Queue(maxsize=10)  # Limitar a 10 archivos en cola para evitar saturación [TODO: ajustar]
last_successful_upload_time = time.time()





def read_credentials():
    credentials_path = os.path.join(os.getcwd(),'credentials_aws')
    config_path = os.path.join(os.getcwd(),"config_aws")
    
    with open(credentials_path, 'r', encoding='UTF-8') as file:
        while line := file.readline():

            if 'aws_access_key_id' in line:
                aws_access_key_id = line.replace('aws_access_key_id = ','')
                aws_access_key_id = aws_access_key_id.replace("\n", "")
            if 'aws_secret_access_key' in line:
                aws_secret_access_key = line.replace('aws_secret_access_key = ','')
                aws_secret_access_key = aws_secret_access_key.replace("\n", "")
    return aws_access_key_id,aws_secret_access_key



def get_device_index(logging, target_name="stm32max98088"):
    """Automatically find the input device index by name."""
    p = pyaudio.PyAudio()
    device_index = None

    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        logging.info(f"Device {i}: {device_info['name']}")
        if target_name.lower() in device_info['name'].lower() and device_info['maxInputChannels'] > 0:
            device_index = i
            logging.info(f"Found target device: {device_info['name']} (Index: {device_index})")
            break

    p.terminate()

    if device_index is None:
        raise ValueError(f"Target audio device '{target_name}' not found.")
    return device_index



def make_s3_client():
    AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY = read_credentials()

    cfg = Config(
        retries={"max_attempts": 5, "mode": "standard"},
        connect_timeout=5,
        read_timeout=60,
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )

    return boto3.client(
        "s3",
        region_name="eu-west-1",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=cfg,
    )


def upload_worker(
    s3,
    bucket_name,
    location_record,
    location_place,
    location_point,
    storage_output_wav_folder,
    file_part_size,
    logger,
):
    global last_successful_upload_time
    PART_SIZE = file_part_size

    while True:
        file_path = upload_queue.get()
        try:
            if file_path is None:
                break # No return , ya que queremos que vaya por el finally :)

            object_key = (
                f"{location_record}/{location_place}/{location_point}/"
                f"{storage_output_wav_folder}/{os.path.basename(file_path)}"
            )

            if not os.path.exists(file_path):
                logger.warning(f"File no longer exists, skipping: {file_path}")
                continue

            size = os.path.getsize(file_path)
            parts_count = (size + PART_SIZE - 1) // PART_SIZE

            logger.info(
                f"Uploading {os.path.basename(file_path)} ({size/1024/1024:.2f} MB) "
                f"as {parts_count} chunks to s3://{bucket_name}/{object_key}"
            )

            uploaded_parts = 0
            idx = 0

            with open(file_path, "rb") as f:
                uploaded_parts, idx = upload_parts(
                    s3, f, PART_SIZE, parts_count,
                    bucket_name, object_key, idx, uploaded_parts,
                    logger
                )

            upload_manifest(
                s3, file_path, bucket_name, object_key, PART_SIZE, uploaded_parts, logger
            )

            last_successful_upload_time = time.time()
            logger.info(f"Uploaded and queued for assembly: {object_key}")

            try:
                os.remove(file_path)
                logger.info(f"Removed local file: {file_path}")
            except OSError as e:
                logger.warning(f"Uploaded but failed to remove local file {file_path}: {e}")

        except Exception as e:
            logger.exception(f"Failed to upload {file_path}: {e}")

            # Limpieza best-effort
            try:
                s3.delete_object(Bucket=bucket_name, Key=f"{object_key}.manifest")
            except Exception:
                pass
            try:
                prefix = f"{object_key}.part"
                resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                if "Contents" in resp:
                    to_delete = [{"Key": obj["Key"]} for obj in resp["Contents"]]
                    for i in range(0, len(to_delete), 1000):
                        s3.delete_objects(Bucket=bucket_name, Delete={"Objects": to_delete[i:i+1000]})
            except Exception:
                pass

        finally:
            upload_queue.task_done()



def upload_file_immediately(
    file_path,
    storage_s3_bucket_name,
    location_record,
    location_place,
    location_point,
    storage_output_wav_folder,
    file_part_size,
    logging
):
    AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY = read_credentials()
    PART_SIZE = file_part_size

    cfg = Config(
        retries={"max_attempts": 5, "mode": "standard"},
        connect_timeout=5,
        read_timeout=60,
    )

    s3 = boto3.client(
        "s3",
        region_name="eu-west-1",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=cfg
    )

    object_key = (
        f"{location_record}/{location_place}/{location_point}/"
        f"{storage_output_wav_folder}/{os.path.basename(file_path)}"
    )

    size = os.path.getsize(file_path)
    parts_count = (size + PART_SIZE - 1) // PART_SIZE

    logging.info(f"Uploading immediately: {file_path} ({parts_count} parts)")

    uploaded_parts = 0
    idx = 0

    with open(file_path, "rb") as f:
        uploaded_parts, idx = upload_parts(
            s3, f, PART_SIZE, parts_count,
            storage_s3_bucket_name,
            object_key,
            idx,
            uploaded_parts
        )

    upload_manifest(
        s3,
        file_path,
        storage_s3_bucket_name,
        object_key,
        PART_SIZE,
        uploaded_parts
    )

    logging.info(f"Upload finished: {object_key}")

    os.remove(file_path)
    logging.info(f"Removed local file: {file_path}")


def record_segment(stream, p, record_seconds, location_record, location_place, location_point, audio_format, audio_channels, audio_sample_rate, audio_chunk_size, storage_s3_bucket_name, storage_output_wav_folder, logging):
    
    """
    
    Record `record_seconds` of audio data from the stream,
    and save it to a .wav file named with the current datetime.
    Returns the full path to the saved file.
    
    """
    
    home_dir = str(Path.home())
    frames = []

    # number of chunks for record_seconds
    num_chunks = int(audio_sample_rate / audio_chunk_size * record_seconds)
    logging.info(f"Chunk number: {num_chunks}")
    
    for _ in range(num_chunks):
        data = stream.read(audio_chunk_size, exception_on_overflow=False)
        frames.append(data)



    # output folder
    output_folder = os.path.join(home_dir, location_record, location_place, location_point, storage_output_wav_folder)
    logging.info(f"This is the output folder: {output_folder}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Making folder: {output_folder}")



    # filename with current time
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp_str}.wav"
    full_path = os.path.join(output_folder, filename)
    logging.info(f"This is the final full path: {full_path}")

    # save
    wf = wave.open(full_path, 'wb')
    wf.setnchannels(audio_channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(audio_sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    logging.info(f"Saved {record_seconds}-second recording to {full_path}")
    return full_path




def record_audio_continuous(device_index, location_record, location_place, location_point, audio_format, audio_channels, audio_sample_rate, audio_chunk_size, storage_s3_bucket_name, storage_output_wav_folder, logging, upload_s3,file_part_size, record_seconds=60):
    # recording setup
    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format,
                    channels=audio_channels,
                    rate=audio_sample_rate,
                    frames_per_buffer=audio_chunk_size,
                    input=True,
                    input_device_index=device_index)

    # start background thread for uploading
    uploader = None
    if upload_s3:
        s3 = make_s3_client()
        uploader = threading.Thread(
            target=upload_worker,
            args=(
                s3,
                storage_s3_bucket_name,
                location_record,
                location_place,
                location_point,
                storage_output_wav_folder,
                file_part_size,
                logging,
            ),
            daemon=True,  # daemon True: si el proceso muere, no se queda colgado
            name="uploader",
        )
        uploader.start()
        logging.info(f"Uploader thread started, alive={uploader.is_alive()}")
    else:
        logging.warning("Not uploading wav files")



    try:
        while True:

            try:
                print(f"\nRecording continuous {record_seconds}-second segments... (Press Ctrl+C to stop)\n")
                logging.info(f"Recording continuous {record_seconds}-second segments...")
                file_path = record_segment(
                    stream, p, record_seconds, location_record, location_place, location_point,
                    audio_format, audio_channels, audio_sample_rate, audio_chunk_size,
                    storage_s3_bucket_name, storage_output_wav_folder, logging
                )
                logging.info(f"Enqueuing {file_path} for upload...")

                if upload_s3:
                    try:
                        upload_queue.put(file_path, block=True, timeout=5)
                        logging.info(f"Queued for upload: {file_path} (queue size={upload_queue.qsize()})")
                    except Exception:
                        logging.warning("Upload queue full; keeping file locally for later.")
                else:
                    logging.info("Upload disabled; keeping file locally.")

            except Exception as segment_error:
                error_message = f"Error during segment processing: {segment_error}. Continuing to next segment."
                logging.error(error_message)
                time.sleep(1)
                continue


    except KeyboardInterrupt:
        logging.error("Recording stopped by user.")



    finally:
        if upload_s3:
            upload_queue.put(None)
            upload_queue.join()  



def check_uploads(logging, check_interval=60, threshold=70):
    while True:
        time.sleep(check_interval)
        elapsed = time.time() - last_successful_upload_time
        if last_successful_upload_time == 0:
            logging.warning("No uploads yet since startup.")
            continue
        
        
        if elapsed > threshold:
            error_message = f"No upload in the last {elapsed:.0f} seconds."
            logging.error(error_message)
            
        else:
            logging.info(f"Upload check passed. Last upload was {elapsed:.0f} seconds ago.")




def arg_parser():
    """
    Parse command-line arguments.
    Use --time to set how many seconds each continuous segment should be.
    Defaults to 60 seconds if not specified.
    """
    parser = argparse.ArgumentParser(description='Audio recording script')
    parser.add_argument('-t', '--time', type=int, default=60,
                        help='Length (in seconds) of each continuous recording. Default is 60.')
    parser.add_argument('-u', '--upload-S3', action='store_true', default=False,
                        help='If provided, upload the final CSV to S3.')

    return parser.parse_args()




def main():
        
        logging = setup_logging(script_name="record_audio")
        args = arg_parser()

        logging.info("Starting process!!")
        logging.info("")

        upload_s3 = args.upload_S3 if args.upload_S3 else None
        record_seconds = args.time if args.time else 60

        logging.info(f"Upload to bucket S3: {upload_s3}")
        logging.info(f"Recording {record_seconds} seconds")

        # device index
        try:
            device_index = get_device_index(logging)
            logging.info(f"Using device index: {device_index}")
        except Exception as e:
            logging.error(f"Error getting the device index: {e}")
            return



        # configuration
        try:
            location_record, location_place, location_point, audio_format, \
            audio_channels, audio_sample_rate, audio_chunk_size, storage_s3_bucket_name, \
            storage_output_wav_folder,file_part_size = load_config_record('config.yaml')

            

            if audio_format == "pyaudio.paInt16":
                audio_format = pyaudio.paInt16

        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return

        #checkout
        if upload_s3:
            threading.Thread(
                target=check_uploads,
                args=(logging,),
                daemon=True
            ).start()
        logging.info("Started upload checkout thread.")



        logging.info("")
        logging.info("Entering recording audio workflow!")

        record_audio_continuous(
            device_index =                      device_index,
            location_record =                   location_record,
            location_place =                    location_place,
            location_point =                    location_point,
            audio_format =                      audio_format,
            audio_channels =                    audio_channels,
            audio_sample_rate =                 audio_sample_rate,
            audio_chunk_size =                  audio_chunk_size,
            storage_s3_bucket_name =            storage_s3_bucket_name,
            storage_output_wav_folder =         storage_output_wav_folder,
            logging =                           logging,
            upload_s3=                          upload_s3,
            file_part_size=                     file_part_size,
            record_seconds=                     record_seconds
        )


            


if __name__ == "__main__":
    main()
