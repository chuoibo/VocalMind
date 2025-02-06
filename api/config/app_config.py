import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATABASE_API_URL = os.getenv('DATABASE_API_URL')
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY =os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    BUCKET_NAME = os.getenv("BUCKET_NAME")


class RecordingConfig:
    RATE = 16000
    CHANNELS = 1
    FRAME_DURATION = 30 
    SILENCE_THRESHOLD = 2
    VAD_MODE = 3
    DEVICE_NAME = 'default'
    SAVE_DIR = '/speech/common_dir/recordings'
    SAVE_PATH = '/speech/common_dir/recordings/recorded_file.wav'
