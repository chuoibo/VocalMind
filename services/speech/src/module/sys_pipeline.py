import logging

from concurrent.futures import ThreadPoolExecutor

from src.module.llm import TextGeneration
from src.module.emotion_analysis import EmotionAnalysis
from src.module.speech2txt import Speech2Txt
# from src.module.txt2speech import Txt2Speech
from src.schema.speech_system_schema import (InputSpeechSystemModel,
                                             ResultSpeechSystemModel,
                                             OutputSpeechSystemModel,
                                             StatusModel,
                                             StatusEnum)

TEXT_GENERATION_INIT = None
EMOTION_ANALYSIS_INIT = None

class SpeechSystem:
    def __init__(self, inp: InputSpeechSystemModel):
        global TEXT_GENERATION_INIT, EMOTION_ANALYSIS_INIT

        self.live_record = inp.live_record
        self.input_audio_file_path = inp.input_audio_file_path

        with ThreadPoolExecutor() as executor:
            future_text_generation = executor.submit(
                lambda: TEXT_GENERATION_INIT or TextGeneration()
            )

            future_emotion_analysis = executor.submit(
                lambda: EMOTION_ANALYSIS_INIT or EmotionAnalysis()
            )

            future_speech_to_text = executor.submit(
                Speech2Txt, self.live_record, self.input_audio_file_path
            )

            TEXT_GENERATION_INIT = future_text_generation.result()
            EMOTION_ANALYSIS_INIT = future_emotion_analysis.result()

            self.speech_to_text = future_speech_to_text.result()

        self.emotion_analysis = EMOTION_ANALYSIS_INIT
        self.text_generation = TEXT_GENERATION_INIT



    def run(self) -> OutputSpeechSystemModel:
        speech_recognition = self.speech_to_text.run()

        if speech_recognition == '':
            raise ValueError('Cannot do automatic speech recognition ...')

        emotion = self.emotion_analysis.run(speech_recognition)

        generated_text = self.text_generation.run(speech_recognition, emotion)

        # generated_speech = self.load_txt2speech().run(text_generation, emotion_analysis)

        result = ResultSpeechSystemModel(
            generated_audio_file=generated_text
        )

        status = StatusModel(status=StatusEnum.success, message="Processing completed successfully")
        logging.info('Finish speech module ...')

        return OutputSpeechSystemModel(
            result=result,
            status=status
        )
