import torch
import logging
import numpy as np
import onnxruntime as ort

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.utils.common import *
from src.config.app_config import Speech2TxtConfig as sc

WAV2_VEC2_MODEL = None

class Wav2vec2Inference:
    def __init__(self):
        global WAV2_VEC2_MODEL

        self.model_name = sc.model_name
        self.model_cache = sc.model_cache
        self.sampling_rate = sc.sampling_rate
        self.return_tensors = sc.return_tensors
        self.padding = sc.padding

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(sc.model_cache):
            make_directory(sc.model_cache)

        if WAV2_VEC2_MODEL is None:
            onnx_file = find_files(directory_path=sc.model_cache, type_file='onnx')
            if onnx_file:
                self.model_type = 'onnx'
                WAV2_VEC2_MODEL = ort.InferenceSession(onnx_file[0], providers=['CPUExecutionProvider'])
                logging.info('Loaded ONNX model for the first time.')
            else:
                self.model_type = 'hf'
                WAV2_VEC2_MODEL = Wav2Vec2ForCTC.from_pretrained(
                    pretrained_model_name_or_path=self.model_name,
                    cache_dir=self.model_cache
                ).to(self.device)
                logging.info('Loaded pretrained Hugging Face model for the first time.')

        self.model = WAV2_VEC2_MODEL
        logging.info('Initialized pretrained model.')

        if self.model_type == 'hf':
            warm_up_model(model=self.model, device=self.device)
            logging.info('Warming up model ...')

        self.processor = Wav2Vec2Processor.from_pretrained(
            pretrained_model_name_or_path=sc.model_name, 
            cache_dir=sc.model_cache)

        logging.info('Loading pretrained processor ...')


    def speech_recognition(self, audio_buffer):
        if self.model_type == 'hf':
            if len(audio_buffer) == 0:
                return ""

            inputs = self.processor(torch.tensor(audio_buffer), 
                                    sampling_rate=self.sampling_rate, 
                                    return_tensors=self.return_tensors, 
                                    padding=self.padding).input_values

            with torch.no_grad():
                logits = self.model(inputs.to(self.device)).logits            

            predicted_ids = torch.argmax(logits, dim=-1)
            generated_text = self.processor.batch_decode(predicted_ids)[0]

        
        elif self.model_type == 'onnx':
            input_values = self.processor(audio_buffer, sampling_rate=self.sampling_rate, return_tensors='np').input_values
            onnx_inputs = {"input_values": input_values}
            logging.info('Loading and pre-processing onnx input values ...')

            logits = self.model.run(None, onnx_inputs)[0]
            predicted_ids = np.argmax(logits, axis=-1)
            generated_text = self.processor.batch_decode(predicted_ids)[0]

        return generated_text
