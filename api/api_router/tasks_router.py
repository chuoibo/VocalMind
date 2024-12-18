import requests

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from utils.api_logger import logging
from utils import DATABASE_API_URL

router = APIRouter(prefix="/metadata", tags=["Metadata"])

@router.get("/get_by_user/{user_name}")
async def get_tasks(user_name):
    
    logging.info(f'Start getting tasks from {user_name} ...')

    if not user_name:
        return JSONResponse({"error": "User_name must be provided."}, status_code=400)


    save_payload = {"task_id": None, "user_id": user_name}
    response = requests.get(f"{DATABASE_API_URL}/task/get_by_user", json=save_payload)

    logging.info('Finish getting tasks ...')

    return {"result": response['result']}


@router.get("/get_task/{user_name}/{task_id}")
async def get_tasks(user_name: str,
                    task_id: str):
    
    logging.info(f'Start getting tasks from {user_name} ...')

    if not user_name or not task_id:
        return JSONResponse({"error": "User_name and task_id must be provided."}, status_code=400)


    save_payload = {"task_id": task_id, "user_id": user_name}
    response = requests.get(f"{DATABASE_API_URL}/task/get_specific_task", json=save_payload)

    logging.info('Finish getting tasks ...')

    return {"result": response['result']}
