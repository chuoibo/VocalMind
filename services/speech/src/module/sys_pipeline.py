import logging

from concurrent.futures import ThreadPoolExecutor

from src.module.llm import TextGeneration
from src.module.emotion_analysis import EmotionAnalysis
from src.module.speech2txt import Speech2Txt
from src.module.txt2speech import Txt2Speech
from src.schema.speech_system_schema import (InputSpeechSystemModel,
                                             ResultSpeechSystemModel,
                                             OutputSpeechSystemModel,
                                             StatusModel,
                                             StatusEnum)

class SpeechSystem:
    def __init__(self, inp: InputSpeechSystemModel):
        self.live_record = inp.live_record
        self.text_generation = None
        self.emotion_analysis = None
        self.speech_to_text = None
        self.text_to_speech = None

    def load_text_gen(self):
        if not self.text_generation:
            self.text_generation = TextGeneration()
        return self.text_generation

    def load_emotion_analysis(self):
        if not self.emotion_analysis:
            self.emotion_analysis = EmotionAnalysis()
        return self.emotion_analysis

    def load_speech2txt(self):
        if not self.speech_to_text:
            self.speech_to_text = Speech2Txt()
        return self.speech_to_text

    def load_txt2speech(self):
        if not self.text_to_speech:
            self.text_to_speech = Txt2Speech()
        return self.text_to_speech

    
    def run(self) -> OutputSpeechSystemModel:
        speech_recognition = self.load_speech2txt().run(self.live_record)

        with ThreadPoolExecutor() as executor:
            emotion_future = executor.submit(self.load_emotion_analysis().run, speech_recognition)
            text_generation_future = executor.submit(self.load_text_gen().run, speech_recognition)

            emotion_analysis = emotion_future.result()
            text_generation = text_generation_future.result()


        generated_speech = self.load_txt2speech().run(text_generation, emotion_analysis)

        result = ResultSpeechSystemModel(
            generated_audio_file=generated_speech
        )

        status = StatusModel(status=StatusEnum.success, message="Processing completed successfully")
        logging.info('Finish speech module ...')

        return OutputSpeechSystemModel(
            result=result,
            status=status
        )
