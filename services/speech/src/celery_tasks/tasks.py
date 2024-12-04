import logging

from schema.speech_system_schema import InputSpeechSystemModel
from module.sys_pipeline import SpeechSystem
from celery_tasks.app_worker import app


@app.task(name='speech_ai')
def speech_ai(input_data):
    logging.info('Celery starts to send speech2txt task')
    live_record = input_data.get("live_record")
    
    inp = InputSpeechSystemModel(
        live_record=live_record,
    )
    
    result = SpeechSystem(inp=inp).run()
    logging.info('Implementing speech task...')
    return result.dict()

