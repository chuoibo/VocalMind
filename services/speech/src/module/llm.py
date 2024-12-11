import torch
import logging
import re
import time

from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from src.config.app_config import LLMConfig as lc


class TextGeneration:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(lc.model_cache)
        logging.info('Initialize text generation tokenizer')


        self.model = LlamaForCausalLM.from_pretrained(lc.model_cache)
        
        logging.info('Initialize text generation model')

        self.temperature = lc.temperature
        self.return_tensors = lc.return_tensors
        self.max_new_tokens = lc.max_new_tokens
        self.repetition_penalty = lc.repetition_penalty

        logging.info('Initialize text generation module hyperparameters')
    

    def postprocess_text(self, text):
        logging.info('Post-process the text generation')
        answer = re.sub(r"<\|begin_of_text\|>.*?Response:\s*", "", text, flags=re.DOTALL)
        answer = re.sub(r"<\|end_of_text\|>", "", answer)
        answer = answer.strip()
        
        last_newline_pos = answer.rfind("\n")
        if last_newline_pos == -1: 
            final_text = answer  
        else:
            after_newline = answer[last_newline_pos + 1:]
            last_full_stop_pos = after_newline.rfind(".")
            if last_full_stop_pos != -1: 
                final_text = answer[:last_newline_pos + 1] + after_newline[:last_full_stop_pos + 1]
            else:  
                final_text = answer[:last_newline_pos]

        final_text = final_text.replace("\n", " ")
        return final_text.strip()


    def run(self, input_text, emotion):
        logging.info('Starting text generation inference process')

        self.model.to(self.device)

        prompt = f"Please respond to the following input in a concise manner, considering the user's emotion:\nEmotion: {emotion}\nInput: {input_text}\nResponse:"


        inputs = self.tokenizer(prompt, return_tensors=self.return_tensors).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            output = self.model.generate(**inputs, 
                                         max_new_tokens=self.max_new_tokens, 
                                         temperature=self.temperature, 
                                         repetition_penalty=self.repetition_penalty)

        answer = self.tokenizer.decode(output[0])
        end_time = time.time()

        processed_answer = self.postprocess_text(answer)
        logging.info(f'Finish LLM module in {end_time - start_time}s')
        return processed_answer
