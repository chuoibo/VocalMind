import torch
import logging

from transformers import pipeline

from src.config.app_config import Speech2TxtConfig as sc

class TextProcessing:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len_post_processing = sc.max_len_post_processing

        self.model = pipeline(
            sc.post_processing_task, 
            model=sc.post_processing_model_cache, 
            device=self.device)
        
        logging.info('Initialize Speech Post Processing Module ...')

    
    def text_post_processing(self, text):
        return self.model(text, max_length=self.max_len_post_processing)[0]['generated_text']

