import torch
import logging
import re
import time

from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from src.config.app_config import LLMConfig as lc

TEXT_GENERATION_MODEL = None

class TextGeneration:
    def __init__(self):
        global TEXT_GENERATION_MODEL
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(lc.model_cache)
        logging.info('Initialize text generation tokenizer')

        if TEXT_GENERATION_MODEL is None:
            logging.info('Loading text genearation if this is the first time ...')
            TEXT_GENERATION_MODEL = LlamaForCausalLM.from_pretrained(lc.model_cache)
        
        self.model = TEXT_GENERATION_MODEL
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

        if emotion == 'sad':
            prompt = (
                f"The user is feeling sad. Respond in a compassionate and comforting way to uplift their spirits."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'joy':
            prompt = (
                f"The user is feeling happy. Respond in an enthusiastic and celebratory manner to encourage their positivity."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'fear':
            prompt = (
                f"The user is feeling happy. Respond in an enthusiastic and celebratory manner to encourage their positivity."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'anger':
            prompt = (
                f"The user is feeling angry. Respond in a calm and understanding manner, acknowledging their feelings and helping them process their emotions constructively."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'surprise':
            prompt = (
                f"The user is feeling surprised. Respond in a curious and engaging manner to match their surprise and encourage them to share more about their experience."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'neutral':
            prompt = (
                f"Respond in an informative and balanced manner, maintaining a neutral and respectful tone"
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

        processed_answer = self.postprocess_text(answer)
        logging.info(f'Finish LLM module in {end_time - start_time}s')
        return processed_answer
