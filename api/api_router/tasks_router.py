import requests

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from utils.api_logger import logging
from config.app_config import Config as cfg

router = APIRouter(prefix="/metadata", tags=["Metadata"])

@router.get("/get_by_user/{user_name}")
async def get_tasks(user_name):
    
    logging.info(f'Start getting tasks from {user_name} ...')

    if not user_name:
        return JSONResponse({"error": "User_name must be provided."}, status_code=400)


    save_payload = {"task_id": None, "user_id": user_name}
    response = requests.get(f"{cfg.DATABASE_API_URL}/task/get_by_user", json=save_payload)

    if response.status_code != 200:
        logging.error(f"Error fetching tasks: {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    result = response.json()

    logging.info('Finish getting tasks ...')

    return result['result']


@router.get("/get_task/{task_id}")
async def get_tasks_metadata(task_id: str):
    
    logging.info(f'Start getting tasks {task_id} ...')

    if not task_id:
        return JSONResponse({"error": "task_id must be provided."}, status_code=400)


    save_payload = {"task_id": task_id}
    response = requests.get(f"{cfg.DATABASE_API_URL}/task/get_task", json=save_payload)

    if response.status_code != 200:
        logging.error(f"Error fetching tasks: {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    result = response.json()

    logging.info('Finish getting tasks ...')

    return result['result']
