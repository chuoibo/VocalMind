import os
from utils.api_logger import logging

def audio_stream(audio_path):
    with open(audio_path, "rb") as audio_file:
        yield from audio_file

def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted temporary file: {file_path}")
    except Exception as e:
        logging.error(f"Failed to delete temporary file {file_path}: {str(e)}")
