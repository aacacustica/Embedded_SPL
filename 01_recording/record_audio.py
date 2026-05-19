
import sys



import os
import argparse
import datetime
import pyaudio
import wave
import time  
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from logging_config import setup_logging
from utils import load_config_record
import os
import time

strf_time_daystamp = "%Y%m%d_%H%M%S"
strf_time = "%H:%M:%S"


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






def record_segment(
    stream,
    p,
    record_seconds,
    location_record,
    location_place,
    location_point,
    audio_format,
    audio_channels,
    audio_sample_rate,
    audio_chunk_size,
    storage_s3_bucket_name,
    storage_output_wav_folder,
    logging
):
    """
    Record `record_seconds` of audio data from the stream in real elapsed time,
    and save it to a .wav file named with the segment start datetime.
    Returns the full path to the saved file.
    """

    home_dir = str(Path.home())
    frames = []

    segment_start_dt = datetime.datetime.now()
    segment_start_monotonic = time.monotonic()

    logging.info(f"Segment start: {segment_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Configured sample rate: {audio_sample_rate}")
    logging.info(f"Configured chunk size: {audio_chunk_size}")

    while time.monotonic() - segment_start_monotonic < record_seconds:
        data = stream.read(audio_chunk_size, exception_on_overflow=False)
        frames.append(data)

    segment_end_dt = datetime.datetime.now()
    elapsed = time.monotonic() - segment_start_monotonic
    logging.info(f"Segment end: {segment_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Real elapsed seconds: {elapsed:.2f}")

    output_folder = os.path.join(
        home_dir,
        location_record,
        location_place,
        location_point,
        'AUDIOMOTH',
        storage_output_wav_folder
    )
    logging.info(f"This is the output folder: {output_folder}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Making folder: {output_folder}")

    # Nombre con el instante de inicio real del segmento
    timestamp_str = segment_start_dt.strftime(strf_time_daystamp)
    filename = f"{timestamp_str}.wav"
    full_path = os.path.join(output_folder, filename)

    logging.info(f"Saved file: {full_path}")
    logging.info(f"This is the final full path: {full_path}")

    wf = wave.open(full_path, 'wb')
    wf.setnchannels(audio_channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(audio_sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    logging.info(f"Saved {record_seconds}-second recording to {full_path}")
    return full_path



def record_audio_continuous(device_index, location_record, location_place, location_point, audio_format, audio_channels, audio_sample_rate, audio_chunk_size, storage_s3_bucket_name, storage_output_wav_folder, logging, file_part_size, record_seconds=60):
    
    p = pyaudio.PyAudio()

    stream = p.open(format=audio_format,
                    channels=audio_channels,
                    rate=audio_sample_rate,
                    frames_per_buffer=audio_chunk_size,
                    input=True,
                    input_device_index=device_index)

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


            except Exception as segment_error:
                error_message = f"Error during segment processing: {segment_error}. Continuing to next segment."
                logging.error(error_message)
                time.sleep(1)
                continue


    except KeyboardInterrupt:
        logging.error("Recording stopped by user.")



    finally:
        


        stream.stop_stream()
        stream.close()
        p.terminate()
        logging.info("Audio stream closed and PyAudio terminated.")  







def arg_parser():
    """
    Parse command-line arguments.
    Use --time to set how many seconds each continuous segment should be.
    Defaults to 60 seconds if not specified.
    """
    parser = argparse.ArgumentParser(description='Audio recording script')
    parser.add_argument('-t', '--time', type=int, default=60,
                        help='Length (in seconds) of each continuous recording. Default is 60.')


    return parser.parse_args()




def main():
        
        logging = setup_logging(script_name="record_audio")
        args = arg_parser()

        logging.info("Starting process!!")
        logging.info("")

        record_seconds = args.time if args.time else 60

        logging.info(f"Recording {record_seconds} seconds")

        # device index
        try:
            device_index = get_device_index(logging)
            logging.info(f"Using device index: {device_index}")
        except Exception as e:
            logging.error(f"Error getting the device index: {e}")
            sys.exit(1)



        # configuration
        try:
            location_record, location_place, location_point, audio_format, \
            audio_channels, audio_sample_rate, audio_chunk_size, storage_s3_bucket_name, \
            storage_output_wav_folder,file_part_size = load_config_record('config.yaml')

            

            if audio_format == "pyaudio.paInt16":
                audio_format = pyaudio.paInt16

        except Exception as e:
            logging.error(f"Error loading config: {e}")
            sys.exit(1)



        logging.info("")
        logging.info("Entering recording audio workflow!")

        logging.info(f"PID: {os.getpid()}")
        logging.info(f"Recording {record_seconds} seconds")

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
            file_part_size=                     file_part_size,
            record_seconds=                     record_seconds
        )


            


if __name__ == "__main__":
    main()
