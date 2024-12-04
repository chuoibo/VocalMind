import os
from dotenv import load_dotenv

from src.utils.common import read_yaml_file

load_dotenv()


class Config:
    DEBUG = False
    TESTING = False

    # RabbitMQ Configuration
    RMQ_USER = os.getenv('RMQ_USER', 'guest')
    RMQ_PWD = os.getenv('RMQ_PWD', 'guest')
    MQ_URL = os.getenv('MQ_URL', f'amqp://{RMQ_USER}:{RMQ_PWD}@rabbitmq:5672/')
    
    # Redis Configuration
    REDIS_PWD = os.getenv('REDIS_PWD', '')
    REDIS_URL = os.getenv('REDIS_URL', f'redis://:{REDIS_PWD}@redis:6379/0')
    
    # Additional configurations
    LOG_DIR = "/asr/logs"
    LOG_FILEPATH = os.path.join(LOG_DIR,"celery.log")
    COMMON_DIR = "/common_dir"

    SPEECH_CONFIG_FILEPATH = '/asr/src/config/speech_cfg.yaml'

    @classmethod
    def load_config(cls):
        return read_yaml_file(cls.SPEECH_CONFIG_FILEPATH)


class Speech2TxtConfig(Config):
    config = Config.load_config()
    speech2txt_cfg = config['speech2txt']

    #For asr
    output_file_path = speech2txt_cfg['output_file_path']
    model_name = speech2txt_cfg['model_name']
    model_cache= speech2txt_cfg['model_cache']
    post_processing_task = speech2txt_cfg['post_processing_task']
    post_processing_model_cache = speech2txt_cfg['post_processing_model_cache']
    max_len_chunk = speech2txt_cfg['max_len_chunk']
    pre_processing_multiplier = speech2txt_cfg['pre_processing_multiplier']
    sampling_rate = speech2txt_cfg['sampling_rate']

    #For recording
    vad_mode = speech2txt_cfg['vad_mode']
    silence_limit_seconds = speech2txt_cfg['silence_limit_seconds']
    rate = speech2txt_cfg['rate']
    frame_duration = speech2txt_cfg['frame_duration']
    min_pause = speech2txt_cfg['min_pause']
    max_pause = speech2txt_cfg['max_pause']
    

class EmotionAnalysisConfig(Config):
    config = Config.load_config()
    emotion_analysis_cfg = config['emotion_analysis']

    model_name = emotion_analysis_cfg['model_name']
    model_cache = emotion_analysis_cfg['model_cache']
    padding = emotion_analysis_cfg['padding']
    truncation = emotion_analysis_cfg['truncation']
    return_tensors = emotion_analysis_cfg['return_tensors']
    max_length = emotion_analysis_cfg['max_length']


class Txt2SpeechConfig(Config):
    config = Config.load_config()
    txt2speech_cfg = config['txt2speech']

    output_file_path = txt2speech_cfg['output_file_path']