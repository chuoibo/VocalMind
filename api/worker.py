import os

from dotenv import load_dotenv
from celery import Celery

load_dotenv()

BROKER_URI = os.getenv('MQ_URL')
BACKEND_URI = os.getenv('REDIS_URL')

celery_client = Celery(
    'llm',
    broker=BROKER_URI,
    backend=BROKER_URI,
)

celery_client.conf.task_routes = {
    'llm_processing': {'queue': 'llm_processing_queue'}
}

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_hijack_root_logger=False
)