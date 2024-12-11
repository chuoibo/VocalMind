import logging
import os
import torch
import re
from pathlib import Path
import numpy as np
import soundfile as sf

from src.config.app_config import Txt2SpeechConfig as tc
from src.utils.common import  make_directory

class Txt2Speech:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(tc.output_dir):
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
        self.model = None
        self.vocoder = None
        self.ref_audio = None
        self.ref_text = None
        logging.info('Initialize F5 TTS parameters')
        
        logging.info('Initialize text to speech module ...')
    
    def load_models(self):
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import (load_model,
                                              load_vocoder)

        if not self.vocoder:
            logging.info('Initialized pretrained vocoder.')
            self.vocoder = load_vocoder(vocoder_name=tc.vocoder_name,
                                        is_local=tc.load_vocoder_from_local,
                                        local_path=tc.vocoder_local_path)
        
        if not self.model:
            self.model_cls = DiT
            self.model_cfg = dict(dim=tc.dim, 
                            depth=tc.depth, 
                            heads=tc.heads, 
                            ff_mult=tc.ff_mult, 
                            text_dim=tc.text_dim, 
                            conv_layers=tc.conv_layers)
            
            logging.info('Initialized pretrained model F5 TTS.')
            self.model = load_model(model_cls=self.model_cls,
                                    model_cfg=self.model_cfg,
                                    ckpt_path=self.ckpt_file,
                                    mel_spec_type=self.vocoder_name,
                                    vocab_file=self.vocab_file)
        
        return self.vocoder, self.model


    def mapping_emotion_analysis(self, emotion):
        if emotion == 'neutral':
            self.ref_audio = tc.ref_audio_neutral
            self.ref_text = tc.ref_text_neutral
        # elif emotion == 'sad':

        #     self.ref_audio = tc.ref_audio_sad
        #     self.ref_text = tc.ref_text_sad
        return self.ref_audio, self.ref_text
    

    def run(self, text_gen, emotion):
        
        from f5_tts.infer.utils_infer import (
            infer_process,
            preprocess_ref_audio_text,
            remove_silence_for_generated_wav
            )

        self.vocoder, self.model = self.load_models()
        
        # self.ref_audio, self.ref_text = self.mapping_emotion_analysis(emotion=emotion)
        self.ref_audio = tc.ref_audio_neutral
        self.ref_text = tc.ref_text_neutral
        
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
                speed=self.speed,
                device=self.device
            )
            generated_audio_segments.append(audio)
        
        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
            
            with open(self.wave_path, "wb") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                if self.remove_silence:
                    remove_silence_for_generated_wav(f.name)

        return self.wave_path
