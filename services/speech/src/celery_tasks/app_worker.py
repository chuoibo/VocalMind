from celery import Celery

from src.config.app_config import Config


app = Celery(
    '__name__',
    broker=Config.MQ_URL,
    backend=Config.REDIS_URL,
)

app.conf.task_routes = {
    'speech_ai': {'queue': 'record_speech_queue'}
}

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_hijack_root_logger=False
)