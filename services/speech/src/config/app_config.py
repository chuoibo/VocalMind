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
    LOG_DIR = "/speech/logs"
    LOG_FILEPATH = os.path.join(LOG_DIR,"celery.log")
    COMMON_DIR = "/common_dir"

    SPEECH_CONFIG_FILEPATH = '/speech/src/config/speech_cfg.yaml'


    @classmethod
    def load_config(cls):
        return read_yaml_file(cls.SPEECH_CONFIG_FILEPATH)


class Speech2TxtConfig(Config):
    config = Config.load_config()
    speech2txt_cfg = config['speech2txt']

    #For asr
    model_name = speech2txt_cfg['model_name']
    model_cache= speech2txt_cfg['model_cache']
    post_processing_task = speech2txt_cfg['post_processing_task']
    post_processing_model_cache = speech2txt_cfg['post_processing_model_cache']
    device_name = speech2txt_cfg['device_name']
    sampling_rate = speech2txt_cfg['sampling_rate']
    return_tensors = speech2txt_cfg['return_tensors']
    padding = speech2txt_cfg['padding']
    max_len_post_processing = speech2txt_cfg['max_len_post_processing']

    #For recording
    vad_mode = speech2txt_cfg['vad_mode']
    silence_limit_seconds = speech2txt_cfg['silence_limit_seconds']
    rate = speech2txt_cfg['rate']
    frame_duration = speech2txt_cfg['frame_duration']
    channels = speech2txt_cfg['channels']
    

class EmotionAnalysisConfig(Config):
    config = Config.load_config()
    emotion_analysis_cfg = config['emotion_analysis']

    model_name = emotion_analysis_cfg['model_name']
    model_cache = emotion_analysis_cfg['model_cache']
    padding = emotion_analysis_cfg['padding']
    truncation = emotion_analysis_cfg['truncation']
    return_tensors = emotion_analysis_cfg['return_tensors']
    max_length = emotion_analysis_cfg['max_length']


class LLMConfig(Config):
    config = Config.load_config()
    llm_cfg = config['llm']

    model_cache = llm_cfg['model_cache']
    return_tensors = llm_cfg['return_tensors']
    max_new_tokens = llm_cfg['max_new_tokens']
    temperature = llm_cfg['temperature']
    repetition_penalty = llm_cfg['repetition_penalty']


class Txt2SpeechConfig(Config):
    config = Config.load_config()
    txt2speech_cfg = config['txt2speech']
    tts_ref_cfg = config['tts_ref']

    model_cache = txt2speech_cfg['model_cache']
    model_config = txt2speech_cfg['model_config']
    output_file_path = txt2speech_cfg['output_file_path']
    language = txt2speech_cfg['language']
 
    #Config for tts_ref
    ref_audio_neutral = tts_ref_cfg['ref_audio_neutral']
    ref_audio_sad = tts_ref_cfg['ref_audio_sad']
    ref_audio_happy = tts_ref_cfg['ref_audio_happy']
    ref_audio_sympathy = tts_ref_cfg['ref_audio_sympathy']
    ref_audio_surprise = tts_ref_cfg['ref_audio_surprise']
    ref_audio_anger = tts_ref_cfg['ref_audio_anger']


