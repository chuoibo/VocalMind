import logging
import os
import re
from pathlib import Path
import numpy as np
import soundfile as sf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav
    )
from f5_tts.model import DiT

from src.config.app_config import Txt2SpeechConfig as tc
from src.utils.common import load_text_file, make_directory

F5_VOCODER = None
F5_TTS_MODEL = None

class Txt2SpeechRecognition:
    def __init__(self):
        global F5_VOCODER, F5_TTS_MODEL

        make_directory(path=tc.output_dir)
        self.wave_path = Path(tc.output_dir) / tc.output_file
        self.model_name = tc.model_name
        self.ckpt_file = tc.ckpt_file
        self.vocab_file = tc.vocab_file
        self.speed = tc.speed
        self.vocoder_name = tc.vocoder_name
        self.ref_audio_neutral = tc.ref_audio_neutral
        self.ref_text_neutral = tc.ref_text_neutral
        self.remove_silence = tc.remove_silence
        self.ref_audio = None
        self.ref_text = None
        logging.info('Initialize F5 TTS parameters')

        if F5_VOCODER is None:
            F5_VOCODER = load_vocoder(vocoder_name=tc.vocoder_name,
                                      is_local=tc.load_vocoder_from_local,
                                      local_path=tc.vocoder_local_path)
            logging.info('Loading F5 TTS vocoder for the first time')
        
        self.vocoder = F5_VOCODER
        logging.info('Initialized pretrained vocoder.')

        model_cls = DiT
        model_cfg = dict(dim=tc.dim, 
                         depth=tc.depth, 
                         heads=tc.heads, 
                         ff_mult=tc.ff_mult, 
                         text_dim=tc.text_dim, 
                         conv_layers=tc.conv_layers)
        
        if F5_TTS_MODEL is None:
            F5_TTS_MODEL = load_model(model_cls=model_cls,
                                      model_cfg=model_cfg,
                                      ckpt_path=self.ckpt_file,
                                      mel_spec_type=self.vocoder_name,
                                      vocab_file=self.vocab_file)
            
            logging.info('Loading F5 TTS model for the first time')
        
        self.model = F5_TTS_MODEL
        logging.info('Initialized pretrained model F5 TTS.')

        logging.info('Initialize text to speech module ...')


    def mapping_emotion_analysis(self, emotion):
        if emotion == 'neutral':
            self.ref_audio = tc.ref_audio_neutral
            self.ref_text = tc.ref_text_neutral
        elif emotion == 'sad':
            self.ref_audio = tc.ref_audio_sad
            self.ref_text = tc.ref_text_sad
        return self.ref_audio, self.ref_text
    

    def run(self, text_gen, emotion):
        self.ref_audio, self.ref_text = self.mapping_emotion_analysis(emotion=emotion)
        
        main_voice = {"ref_audio": self.ref_audio, "ref_text": self.ref_text}
        voices = {"main": main_voice}

        for voice in voices:
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
        
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, text_gen)
        reg2 = r"\[(\w+)\]"
        for text in chunks:
            if not text.strip():
                continue
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                logging.info("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                logging.info(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            gen_text = text.strip()
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]
            logging.info(f"Voice: {voice}")

            audio, final_sample_rate, _ = infer_process(
                ref_audio=ref_audio, 
                ref_text=ref_text, 
                gen_text=gen_text, 
                model_obj=self.model, 
                vocoder=self.vocoder, 
                mel_spec_type=self.vocoder_name, 
                speed=self.speed
            )
            generated_audio_segments.append(audio)
        
        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
            
            with open(self.wave_path, "wb") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                if self.remove_silence:
                    remove_silence_for_generated_wav(f.name)

        return self.wave_path
