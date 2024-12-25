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

TEXT_GENERATION_INIT = None
EMOTION_ANALYSIS_INIT = None
TXT_TO_SPEECH_INIT = None

class SpeechSystem:
    def __init__(self, inp: InputSpeechSystemModel):
        global TEXT_GENERATION_INIT, EMOTION_ANALYSIS_INIT, TXT_TO_SPEECH_INIT

        self.live_record = inp.live_record
        self.input_audio_file_path = inp.input_audio_file_path

        if self.live_record:
            self.system_input = 'live_recording'
        else:
            self.system_input = self.input_audio_file_path

        with ThreadPoolExecutor() as executor:
            future_text_generation = executor.submit(
                lambda: TEXT_GENERATION_INIT or TextGeneration()
            )

            future_emotion_analysis = executor.submit(
                lambda: EMOTION_ANALYSIS_INIT or EmotionAnalysis()
            )

            future_text_to_speech = executor.submit(
                lambda: TXT_TO_SPEECH_INIT or Txt2Speech()
            )

            future_speech_to_text = executor.submit(
                Speech2Txt, self.live_record, self.input_audio_file_path
            )

            TEXT_GENERATION_INIT = future_text_generation.result()
            EMOTION_ANALYSIS_INIT = future_emotion_analysis.result()
            TXT_TO_SPEECH_INIT = future_text_to_speech.result()

            self.speech_to_text = future_speech_to_text.result()

        self.emotion_analysis = EMOTION_ANALYSIS_INIT
        self.text_generation = TEXT_GENERATION_INIT
        self.text_to_speech = TXT_TO_SPEECH_INIT



    def run(self) -> OutputSpeechSystemModel:
        try:
            speech_recognition = self.speech_to_text.run()
            if speech_recognition == '':
                raise ValueError('Cannot do automatic speech recognition ...')
            
            emotion = self.emotion_analysis.run(speech_recognition)
            if not emotion:
                raise ValueError('Failed to analyze emotion.')
            
            generated_text = self.text_generation.run(speech_recognition, emotion)
            if not generated_text:
                raise ValueError('Failed to generate text.')
            logging.info(f'Final generated text: {generated_text}')
            
            generated_speech = self.text_to_speech.run(generated_text, emotion)
            if not generated_speech:
                raise ValueError('Failed to generate speech audio.')
            
            result = ResultSpeechSystemModel(generated_audio_file=generated_speech)
            status = StatusModel(status=StatusEnum.success, message="Processing completed successfully")
            logging.info('Finish speech module ...')

        except Exception as e:
            logging.error(f'Error in speech module: {str(e)}')
            result = None
            status = StatusModel(status=StatusEnum.failure, message=str(e))
        
        return OutputSpeechSystemModel(
            input=self.system_input,
            result=result,
            status=status
        )
