import logging

from schema.speech2txt_schema import InputSpeech2TxtModel
from schema.text2speech_schema import InputTxt2SpeechModel

from module.speech2txt import Speech2TxtRecognition
from module.txt2speech import Txt2SpeechRecognition

from celery_tasks.app_worker import app


@app.task(name='speech2txt')
def speech2txt(input_data):
    logging.info('Celery starts to send speech2txt task')
    live_record = input_data.get("live_record")
    
    inp = InputSpeech2TxtModel(
        live_record=live_record,
    )
    
    result = Speech2TxtRecognition(inp=inp).run()
    logging.info('Implementing task speech recognition ...')
    return result.dict()


@app.task(name='txt2speech')
def txt_2_speech_recognition(input_data):
    input_file_path = input_data.get("input_file_path")
    
    inp = InputTxt2SpeechModel(
        input_file_path=input_file_path,
    )
    
    result = Txt2SpeechRecognition(inp=inp).run()
    logging.info('Implementing task speech recogntion ...')
    return result.dict()
