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
                f"The user is feeling sad. Please respond the below input in a manner that aligns with their current emotional state and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'joy':
            prompt = (
                f"The user is feeling happy. Respond the below input in an enthusiastic and celebratory manner to encourage their positivity and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'fear':
            prompt = (
                f"The user is feeling happy. Respond the below input in an enthusiastic and celebratory manner to encourage their positivity and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'anger':
            prompt = (
                f"The user is feeling angry. Respond the below input in a calm and understanding manner, acknowledging their feelings and helping them process their emotions constructively and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'surprise':
            prompt = (
                f"The user is feeling surprised. Respond the below input in a curious and engaging manner to match their surprise and encourage them to share more about their experience and as short as possible."
                f"\n\nInput: {input_text}\nResponse:"
            )
        
        elif emotion == 'neutral':
            prompt = (
                f"Respond the below input in an informative and balanced manner, maintaining a neutral and respectful tone and as short as possible"
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
        
        logging.info(f'Final generated text: {processed_answer}')

        logging.info(f'Finish LLM module in {end_time - start_time}s')
        return processed_answer
