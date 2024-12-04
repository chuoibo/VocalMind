import torch
import time
import logging
import numpy as np
import onnxruntime as ort

from itertools import groupby
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.utils.common import *
from src.config.app_config import Speech2TxtConfig as sc

from src.module.record.record_speech import Record
from src.module.processing.text_processing import TextProcessing
from src.module.processing.record_processing import RecordProcessing


from src.schema.speech2txt_schema import (InputSpeech2TxtModel, 
                                          OutputSpeech2TxtModel,
                                          ResultSpeech2TxtModel, 
                                          StatusEnum, 
                                          StatusModel)

SPEECH_2_TXT_MODEL = None
SPEECH_2_TXT_MODEL_FLAG = None
SPEECH_2_TXT_PROCESSOR = None

class Speech2TxtRecognition:
    def __init__(self, inp: InputSpeech2TxtModel):
        global SPEECH_2_TXT_MODEL, SPEECH_2_TXT_PROCESSOR, SPEECH_2_TXT_MODEL_FLAG

        self.live_record = inp.live_record

        self.model_name = sc.model_name
        self.model_cache = sc.model_cache

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        make_directory(sc.model_cache)

        if SPEECH_2_TXT_MODEL is None:
            onnx_file = find_files(directory_path=sc.model_cache, type_file='onnx')
            if onnx_file:
                SPEECH_2_TXT_MODEL_FLAG = 'onnx'
                SPEECH_2_TXT_MODEL = ort.InferenceSession(onnx_file[0], providers=['CUDAExecutionProvider'])
                logging.info('Loaded ONNX model for the first time.')
            else:
                SPEECH_2_TXT_MODEL_FLAG = 'hf'
                SPEECH_2_TXT_MODEL = Wav2Vec2ForCTC.from_pretrained(
                    pretrained_model_name_or_path=self.model_name,
                    cache_dir=self.model_cache
                ).to(self.device)
                logging.info('Loaded pretrained Hugging Face model for the first time.')

        self.model = SPEECH_2_TXT_MODEL
        self.model_type = SPEECH_2_TXT_MODEL_FLAG
        logging.info('Initialized pretrained model.')

        if self.model_type == 'hf':
            warm_up_model(model=self.model, device=self.device)
            logging.info('Warming up model ...')

        if SPEECH_2_TXT_PROCESSOR is None:
            SPEECH_2_TXT_PROCESSOR = Wav2Vec2Processor.from_pretrained(pretrained_model_name_or_path=sc.model_name, 
                                                                       cache_dir=sc.model_cache)

        self.processor = SPEECH_2_TXT_PROCESSOR
        logging.info('Loading pretrained processor ...')

        self.sampling_rate = sc.sampling_rate
        self.output_file_path = sc.output_file_path
        
        logging.info('Initialized speech recognition module ...')
    

    def word_level_forced_align(self, predicted_ids: torch.Tensor, input_values: np.ndarray, output_generated_speech: str):
        logging.info('Starting to get the word-level start and end timestamps.')
        words = [w for w in output_generated_speech.split(' ') if len(w) > 0]
        predicted_ids = predicted_ids[0].tolist()
        duration_sec = input_values.shape[1] / 16_000

        ids_w_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]
        ids_w_time = [i for i in ids_w_time if i[1] != self.processor.tokenizer.pad_token_id]
        split_ids_w_time = [list(group) for k, group
                            in groupby(ids_w_time, lambda x: x[1] == self.processor.tokenizer.word_delimiter_token_id)
                            if not k]

        assert len(split_ids_w_time) == len(words)  
        word_start_times = []
        word_end_times = []
        for cur_ids_w_time, _ in zip(split_ids_w_time, words):
            _times = [_time for _time, _id in cur_ids_w_time]
            word_start_times.append(min(_times))
            word_end_times.append(max(_times))
        
        logging.info(words)

        word_timings = {
            i: {
                'word': word,
                'start_time': round(word_start_time, 4),
                'end_time': round(word_end_time, 4)
            }
            for i, (word, word_start_time, word_end_time) in enumerate(zip(words, word_start_times, word_end_times))
        }
        logging.info('Finish getting the word-level start and end timestamps.')
        
        return word_timings


    def run(self) -> OutputSpeech2TxtModel:
        if self.live_record:
            record_voice, pause_markers = Record().record_audio()
    
        record_process = RecordProcessing(pause_markers=pause_markers)
        classified_pauses = record_process.process_pause_markers()
        
        if self.model_type == 'hf':
            input_values = self.processor(record_voice, sampling_rate=sc.sampling_rate, return_tensors='pt').input_values
            logging.info('Loading and pre-processing input values ...')

            start_time = time.time()

            with torch.no_grad():  
                logits = self.model(input_values.to(self.device)).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            generated_text = self.processor.batch_decode(predicted_ids)

            end_time = time.time()

            inference_time = end_time - start_time
        
        elif self.model_type == 'onnx':
            input_values = self.processor(record_voice, sampling_rate=sc.sampling_rate, return_tensors='np').input_values
            onnx_inputs = {"input_values": input_values}
            logging.info('Loading and pre-processing onnx input values ...')

            start_time = time.time()
            logits = self.model.run(None, onnx_inputs)[0]
            predicted_ids = np.argmax(logits, axis=-1)
            generated_text = self.processor.batch_decode(predicted_ids)
            end_time = time.time()

        logging.info(f"Inference time : {inference_time:.4f} seconds")

        output_generated_speech = str(generated_text[0]).lower()

        word_timings = self.word_level_forced_align(input_values=input_values, 
                                                    predicted_ids=predicted_ids,
                                                    output_generated_speech=output_generated_speech)
        
        generated_text_with_punctuations = record_process.mapping_punctuations(word_timings=word_timings,
                                                                               pause_markers=classified_pauses)

        logging.info('Starting post processing generated text ...')
        postprocessing = TextProcessing()
        processed_generated_speech = postprocessing.text_post_processing(generated_text_with_punctuations)

        generated_text_file = write_to_txt_file(text=processed_generated_speech, file_path=self.output_file_path)

        result = ResultSpeech2TxtModel(
            generated_text_file=generated_text_file
            )
        
        status = StatusModel(status=StatusEnum.success, message="Processing completed successfully")
        logging.info('Finish speech to text recognition module ...')

        return OutputSpeech2TxtModel(
            result=result,
            status=status
        )
        