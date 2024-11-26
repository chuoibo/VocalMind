from celery import Celery

from config.app_config import Config

app = Celery(
    'speech',
    broker=Config.MQ_URL,
    backend=Config.REDIS_URL,
)

app.conf.task_routes = {
    'txt2speech': {'queue': 'txt2speech_queue'},
    'speech2txt': {'queue': 'speech2txt_queue'}
}

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_hijack_root_logger=False
)