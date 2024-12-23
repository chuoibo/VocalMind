import os
import requests
import asyncio

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from celery.result import AsyncResult
from datetime import datetime

from utils import DATABASE_API_URL
from aws_client import upload_to_s3
from worker import celery_client
from utils.api_logger import logging
from utils.common import audio_stream

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

    save_payload = {"task_id": task.id, "input_path_remote": file_url, "time_sent": datetime.now().isoformat()}
    requests.post(f"{DATABASE_API_URL}/task/save", json=save_payload)

    logging.info('Finish adding speech task ...')

    return {"task_id": task.id}


@router.get("/stream-input")
def stream_audio(file_path: str):
    try:
        with open(file_path, "rb") as _:
            pass
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing file: {str(e)}")

    return StreamingResponse(
        audio_stream(file_path),
        media_type="audio/mpeg",  # Adjust based on file type
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
    inp = local_input_data.get("input", None)

    status_data = result['status']
    status = status_data.get("status", None)

    update_payload = {"task_id": task_id, "status": status, "input_path_local": inp, "output_path": output_path}
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