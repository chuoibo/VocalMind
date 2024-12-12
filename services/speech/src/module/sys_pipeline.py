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

class SpeechSystem:
    def __init__(self, inp: InputSpeechSystemModel):
        self.live_record = inp.live_record
        self.input_audio_file_path = inp.input_audio_file_path
        
        with ThreadPoolExecutor() as executor:
            future_text_speech_to_text = executor.submit(Speech2Txt, self.live_record, self.input_audio_file_path)
            future_text_generation = executor.submit(TextGeneration)
            future_emotion_analysis = executor.submit(EmotionAnalysis)

            self.text_generation = future_text_generation.result()
            self.emotion_analysis = future_emotion_analysis.result()
            self.speech_to_text = future_text_speech_to_text.result()

    
    def run(self) -> OutputSpeechSystemModel:
        # speech_recognition = self.speech_to_text.run()

        # if speech_recognition == '':
        #     raise ValueError('Cannot do automatic speech recognition ...')

        speech_recognition = 'I dont know what to say, but know i am feeling like really lost'

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
