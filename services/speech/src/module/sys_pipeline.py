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

        self.text_generation = TextGeneration()
        self.emotion_analysis = EmotionAnalysis()
        self.speech_to_text = Speech2Txt()
        self.text_to_speech = Txt2Speech()

        logging.info('Initialize speech pipeline ...')

    
    def run(self) -> OutputSpeechSystemModel:
        speech_recognition = self.speech_to_text.run(self.live_record)

        with ThreadPoolExecutor() as executor:
            emotion_future = executor.submit(self.emotion_analysis.run, speech_recognition)
            text_generation_future = executor.submit(self.text_generation.run, speech_recognition)

            emotion_analysis = emotion_future.result()
            text_generation = text_generation_future.result()


        generated_speech = self.text_to_speech.run(text_generation, emotion_analysis)

        result = ResultSpeechSystemModel(
            generated_audio_file=generated_speech
        )

        status = StatusModel(status=StatusEnum.success, message="Processing completed successfully")
        logging.info('Finish speech module ...')

        return OutputSpeechSystemModel(
            result=result,
            status=status
        )

