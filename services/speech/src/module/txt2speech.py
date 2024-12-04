import logging

from RealtimeTTS import TextToAudioStream, GTTSEngine

from src.config.app_config import Txt2SpeechConfig as tc
from src.utils.common import load_text_file
from src.module.processing.text_processing import TextProcessing
from src.schema.text2speech_schema import (InputTxt2SpeechModel,
                                           OutputTxt2SpeechModel,
                                           ResultTxt2SpeechModel,
                                           StatusEnum,
                                           StatusModel)


class Txt2SpeechRecognition:
    def __init__(self, inp: InputTxt2SpeechModel):
        self.input_file_path = inp.input_file_path
        self.output_file_path = tc.output_file_path
        self.pre_process = TextProcessing()
        self.engine = GTTSEngine() 
        logging.info('Initialize text to speech recognition ...')


    def run(self) -> OutputTxt2SpeechModel:
        input_text = load_text_file(self.input_file_path)
        logging.info('Loading input file path')

        pre_process_text = self.pre_process.text_post_processing(input_text)
        logging.info('Finish preprocessing the raw text')

        stream = TextToAudioStream(self.engine)
        stream.feed(pre_process_text)
        logging.info('Finish streaming for text to speech recognition module ...')

        stream.play(output_wavfile=self.output_file_path)

        result = ResultTxt2SpeechModel(
            output_file_path=self.output_file_path
        )

        status = StatusModel(status=StatusEnum.success, message="Processing completed successfully")
        logging.info('Finish text to speech recognition module ...')

        return OutputTxt2SpeechModel(
            result=result,
            status=status
        )