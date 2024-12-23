import logging
import torch
import time

from TTS.api import TTS

from src.config.app_config import Txt2SpeechConfig as tc

class Txt2Speech:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model_cache = tc.model_cache
        self.model_config = tc.model_config
        self.output_file_path = tc.output_file_path
        self.language = tc.language

        self.model = TTS(model_path=self.model_cache, config_path=self.model_config).to(self.device)
        logging.info('Loading Text to Speech model ...')

        logging.info('Initialize Text to Speech parameters ...')


    def mapping_emotion_analysis(self, emotion):
        logging.info('Start mapping emotion with reference audio')
        if emotion == 'neutral':
            ref_audio = tc.ref_audio_neutral
        elif emotion in ['sad', 'fear', 'anger']:
            ref_audio = tc.ref_audio_sympathy
        elif emotion == 'joy':
            ref_audio = tc.ref_audio_happy
        elif emotion == 'surprise':
            ref_audio = tc.ref_audio_surprise
        return ref_audio
    

    def run(self, input_text, emotion):
        logging.info('Start inferencing text to speech ...')

        ref_audio = self.mapping_emotion_analysis(emotion)

        start_time = time.time()
        self.model.tts_to_file(text=input_text, 
                               speaker_wav=ref_audio, 
                               language=self.language, 
                               file_path=self.output_file_path)
        end_time = time.time()

        logging.info(f'Finish Text to Speech process saved in {self.output_file_path} in {end_time-start_time}s')

        return self.output_file_path
