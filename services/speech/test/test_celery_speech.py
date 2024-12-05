from celery import Celery
from celery.result import AsyncResult
import time
import os
from dotenv import load_dotenv

load_dotenv() 

RMQ_USER = os.getenv('RMQ_USER')
RMQ_PWD = os.getenv('RMQ_PWD')
REDIS_PWD = os.getenv('REDIS_PWD')

BROKER_URI = f'pyamqp://{RMQ_USER}:{RMQ_PWD}@localhost//'
BACKEND_URI = f'redis://:{REDIS_PWD}@localhost:6379/0'

app = Celery(
    'speech',
    broker=BROKER_URI,
    backend=BACKEND_URI,
)

def test_celery_task(input_data):
    result = app.send_task(name='speech_ai', queue='speech_ai_queue', args=[input_data])
    task_result = AsyncResult(result.id)

    while not task_result.ready():
        print("Waiting for the task result...")
        time.sleep(15)

    if task_result.successful():
        print(f"Task result: {task_result.result}")
    else:
        print(f"Task failed with state: {task_result.state}")
        if task_result.traceback:
            print(f"Task traceback: {task_result.traceback}")

if __name__ == "__main__":
    input_model = {
        "live_record": True
        }
    test_celery_task(input_model)