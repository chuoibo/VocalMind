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

        logging.info('Loading text genearation if this is the first time ...')
        self.model = LlamaForCausalLM.from_pretrained(lc.model_cache)
        
        logging.info('Initialize text generation model')

        self.temperature = lc.temperature
        self.return_tensors = lc.return_tensors
        self.max_new_tokens = lc.max_new_tokens
        self.repetition_penalty = lc.repetition_penalty

        logging.info('Initialize text generation module hyperparameters')
    

    def postprocess_text(self, text):
        match = re.search(r"Response:\s*(.*)", text, flags=re.DOTALL)
        if match:
            answer = match.group(1)  
        else:
            logging.warning("Response pattern not found in text.")
            answer = text  

        answer = re.sub(r"<\|end_of_text\|>", "", answer).strip()

        answer = answer.replace("\n", " ").strip()
        
        last_full_stop_pos = answer.rfind(".")
        
        if last_full_stop_pos != -1:
            return answer[:last_full_stop_pos + 1].strip()
        else:
            return answer.strip()


    def run(self, input_text, emotion):
        logging.info('Starting text generation inference process')

        self.model.to(self.device)

        if emotion == 'sad':
            prompt = (
                f"The current emotional is sad. Please respond the below input in a manner that aligns with their current emotional and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'joy':
            prompt = (
                f"The current emotional is happy. Please respond the below input in a manner that aligns with their current emotional and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'fear':
            prompt = (
                f"The current emotional is fear. Please respond the below input in a manner that aligns with their current emotional and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'anger':
            prompt = (
                f"The current emotional is angry. Please respond the below input in a manner that aligns with their current emotional and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'surprise':
            prompt = (
                f"The current emotional is surprised. Please respond the below input in a manner that aligns with their current emotional and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'neutral':
            prompt = (
                f"The current emotional is neutral. Please respond the below input in a manner that aligns with their current emotional and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )


        inputs = self.tokenizer(prompt, return_tensors=self.return_tensors).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            output = self.model.generate(**inputs, 
                                         max_new_tokens=self.max_new_tokens, 
                                         temperature=self.temperature, 
                                         repetition_penalty=self.repetition_penalty)

        answer = self.tokenizer.decode(output[0])
        end_time = time.time()

        logging.info(f'Text before preprocessing: {answer}')

        processed_answer = self.postprocess_text(answer)
        
        logging.info(f'Final generated text: {processed_answer}')

        logging.info(f'Finish LLM module in {end_time - start_time}s')
        return processed_answer
