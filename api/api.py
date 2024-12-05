from fastapi import FastAPI
from celery.result import AsyncResult, GroupResult
from celery import group, signature
from worker import celery_client

app = FastAPI()