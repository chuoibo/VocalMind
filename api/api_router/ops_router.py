import os
import requests
import asyncio
import uuid

from fastapi import APIRouter, UploadFile, HTTPException, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTasks
from celery.result import AsyncResult
from datetime import datetime

from utils import DATABASE_API_URL
from aws_client import upload_to_s3
from worker import celery_client
from utils.api_logger import logging
from utils.common import audio_stream, delete_file

router = APIRouter(prefix="/ops", tags=["Tasks Operations"])

@router.post("/add")
async def add_task(
    user_name: str = Form(None),
    live_record: bool = Form(...),
    input_audio_file_path: UploadFile = File(None)):
    
    logging.info('Start adding speech task ...')

    if not user_name:
        return JSONResponse({"error": "User_name must be provided."}, status_code=400)


    if live_record:
        input_model = {
            "live_record": True,
            "input_audio_file_path": None,  
        }
    else:
        if not input_audio_file_path:
            return JSONResponse(
                {"error": "File upload is required when not using live record function."},
                status_code=400,
            )
        
        try:
            file_url = upload_to_s3(input_audio_file_path.file, input_audio_file_path.filename)
            input_model = {
                "live_record": False,
                "input_audio_file_path": file_url,  
            }

        except Exception as e:
            return JSONResponse(
                {"error": f"Failed to upload file to S3: {str(e)}"}, status_code=500
            )
        
    task = celery_client.send_task(
        "speech_ai", args=[input_model], queue="speech_ai_queue"
    )

    save_payload = {
        "user_id": user_name,
        "task_id": task.id, 
        "input_path_remote": file_url, 
        "time_sent": datetime.now().isoformat()}
    
    requests.post(f"{DATABASE_API_URL}/task/save", json=save_payload)

    logging.info('Finish adding speech task ...')

    return {"task_id": task.id}


@router.get("/stream-input")
def stream_audio(user_id: str, task_id: str, background_tasks: BackgroundTasks):
    try:
        logging.info('Start streaming input audio file')

        save_payload = {
            "user_id": user_id, 
            "task_id": task_id
        }
        response = requests.get(f"{DATABASE_API_URL}/task/get_specific_task", json=save_payload)
        
        if response.status_code != 200:
            logging.error(f"Error fetching tasks: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        result = response.json()
        result = result['result']
        input_path_remote = result['input_path_remote']

        file_response = requests.get(input_path_remote)

        if file_response.status_code != 200:
            raise Exception(f"Failed to download file from {input_path_remote}")

        local_file_path = f"./tmp/{uuid.uuid4()}_input_audio.wav"
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        with open(local_file_path, "wb") as temp_file:
            temp_file.write(file_response.content)

        if not local_file_path or not os.path.exists(local_file_path):
            raise HTTPException(status_code=404, detail="Audio file not found.")
        
        background_tasks.add_task(delete_file, local_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing file: {str(e)}")

    return StreamingResponse(
        audio_stream(local_file_path),
        media_type="audio/wav",  
        background=background_tasks
    )


@router.get("/stream-output/{task_id}")
async def stream_task_result(task_id: str):
    logging.info(f"Processing request for task_id: {task_id}")

    task_result = AsyncResult(task_id, app=celery_client)

    await asyncio.sleep(3)

    while not task_result.ready():
        await asyncio.sleep(0.5) 

    result = task_result.get()
    result_data = result["result"]
    output_path = result_data.get("generated_audio_file", None)

    local_input_data = result['input']

    status_data = result['status']
    status = status_data.get("status", None)

    update_payload = {"task_id": task_id, "status": status, "input_path_local": local_input_data,  "output_path": output_path}
    requests.post(f"{DATABASE_API_URL}/task/update", json=update_payload)

    if output_path and os.path.exists(output_path):
        logging.info(f"Streaming audio file: {output_path}")
        return StreamingResponse(
            audio_stream(output_path),
            media_type="audio/wav",
        )

    return {
        "task_id": task_id,
        "status": 'Failed',
        "result": None,
    }