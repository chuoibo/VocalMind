import os
import logging
import requests
import uuid

from src.schema.speech_system_schema import InputSpeechSystemModel
from src.module.sys_pipeline import SpeechSystem
from src.celery_tasks.app_worker import app


@app.task(name='speech_ai')
def speech_ai(input_data):
    live_record = input_data.get("live_record")
    input_audio_file_path = input_data.get("input_audio_file_path")
    

    file_url = input_audio_file_path
    try:
        logging.info(f"Downloading audio file from {file_url}...")
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download file from {file_url}")

        local_file_path = f"./common/input/{uuid.uuid4()}_input_audio.wav"
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        with open(local_file_path, "wb") as f:
            f.write(response.content)
        
        logging.info(f"Audio file saved to {local_file_path}")

        inp = InputSpeechSystemModel(
            live_record=live_record,
            input_audio_file_path=local_file_path
        )

    except Exception as e:
        logging.error(f"Error downloading or saving file: {e}")
        raise e
        
    result = SpeechSystem(inp=inp).run()
    logging.info('Implementing speech task...')
    return result.dict()

