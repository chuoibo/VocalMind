import os

from dotenv import load_dotenv
from celery import Celery

load_dotenv()

BROKER_URI = os.getenv('MQ_URL')
BACKEND_URI = os.getenv('REDIS_URL')

celery_client = Celery(
    '__name__',
    broker=BROKER_URI,
    backend=BACKEND_URI,
)
