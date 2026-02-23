import logging
import os
from datetime import datetime


def setup_logging(script_name, level=logging.INFO):
    home_dir = os.getenv("HOME")
    log_dir = "log"
    full_path_log_dir = os.path.join(home_dir, log_dir)
    os.makedirs(full_path_log_dir, exist_ok=True)

    #log file 
    log_file = os.path.join(full_path_log_dir, f"{script_name}.log")
    

    logger = logging.getLogger(script_name)
    logger.setLevel(level)

    # file handler overwrites the log file each time
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)


    # preventing duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()


    logger.addHandler(file_handler)
    return logger