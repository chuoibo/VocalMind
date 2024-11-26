import torch
import time
import re
import logging

from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

from config.app_config import Speech2TxtConfig as sc

POST_PROCESSING_GLOBAL_MODEL = None

class TextProcessing:
    def __init__(self):
        global POST_PROCESSING_GLOBAL_MODEL

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if POST_PROCESSING_GLOBAL_MODEL is None:
            logging.info('The model is downloading if it is the first time ...')
            POST_PROCESSING_GLOBAL_MODEL = pipeline(sc.post_processing_task, 
                                                    model=sc.post_processing_model_cache, 
                                                    device=self.device)

        self.model = POST_PROCESSING_GLOBAL_MODEL
        logging.info('Loading post processing model ...')
        
        logging.info('Initialize Speech Post Processing Module ...')

    
    def split_text_by_punctuation(self, text):
        chunks = re.split(r'(?<=[.!?])\s+', text)
        logging.info('Splitting text ...')
        return chunks
    

    def process_chunk(self, chunk):
        return self.model(chunk, max_length=sc.max_len_chunk)[0]['generated_text']


    def text_post_processing(self, text):
        chunks = self.split_text_by_punctuation(text)

        start_time = time.time()
        with ThreadPoolExecutor() as executor:
            corrected_chunks = list(executor.map(self.process_chunk, chunks))
        end_time = time.time()

        logging.info(f'Inference time for post processing: {end_time-start_time}s')

        corrected_text = '. '.join(corrected_chunks) + '.'

        logging.info('Finish post processing for speech module')
        
        return corrected_text

