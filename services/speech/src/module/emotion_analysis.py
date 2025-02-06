import logging
import time
import torch
import numpy as np
import onnxruntime as ort

from transformers import pipeline, AutoTokenizer

from src.utils.common import *
from src.config.app_config import EmotionAnalysisConfig as ec


class EmotionAnalysis:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_name = ec.model_name
        self.model_cache = ec.model_cache
        self.padding = ec.padding
        self.truncation = ec.truncation
        self.return_tensors = ec.return_tensors
        self.max_length = ec.max_length
        self.class_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

        onnx_file = find_files(directory_path=ec.model_cache, type_file='onnx')
        if onnx_file:
            self.model_type = 'onnx'
            self.model = ort.InferenceSession(onnx_file[0], providers=['CUDAExecutionProvider'])
            logging.info('Loading ONNX model for the first time.')
        else:
            self.model_type = 'hf'
            self.model = pipeline(self.model_name, model=self.model_cache, device=self.device)
            logging.info('Loading pretrained Hugging Face model for the first time.')
            
        logging.info('Initialized pretrained model.')

        self.processor = AutoTokenizer.from_pretrained(self.model_cache)
        logging.info('Initialized emotion analysis module ...')
    

    def post_processing_result(self, result):
        logging.info('Post processing the emotion analysis result')
        if result in ['disgust', 'sadness']:
            result = 'sad'
        return result
    

    def run(self, input_text):
        if self.model_type == 'hf':
            start_time = time.time()
            result = self.model(input_text)[0]['label']
            end_time = time.time()

        elif self.model_type == 'onnx':
            inputs = self.processor(
                input_text,
                return_tensors=self.return_tensors,  
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length
                )
            
            input_names = [input.name for input in self.model.get_inputs()]
            output_names = [output.name for output in self.model.get_outputs()]

            onnx_inputs = {input_names[0]: inputs["input_ids"], input_names[1]: inputs["attention_mask"]}

            start_time = time.time()

            outputs = self.model.run(output_names, onnx_inputs)
            logits = outputs[0]
            probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
            predicted_class_index = np.argmax(probabilities, axis=1)[0]
            result = self.class_labels[predicted_class_index]
            
            end_time = time.time()

        final_result = self.post_processing_result(result=result)
        logging.info(f'Finish inference module emotion analysis in {end_time-start_time}')

        return final_result
    