import os
import requests

from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from celery.result import AsyncResult
from datetime import datetime

from utils import DATABASE_API_URL
from aws_client import upload_to_s3
from worker import celery_client
from utils.api_logger import logging
from utils.common import audio_stream

app = FastAPI()

@app.post("/speech")
async def add_task(live_record: bool = Form(...),
                   input_audio_file_path: UploadFile = File(None)):
    
    logging.info('Start adding speech task ...')

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

    save_payload = {"task_id": task.id, "input_path": file_url, "time_sent": datetime.now().isoformat()}
    requests.post(f"{DATABASE_API_URL}/task/save", json=save_payload)

    logging.info('Finish adding speech task ...')

    return {"task_id": task.id}


@app.get("/check/{task_id}")
async def get_task_result(task_id: str):
    task_result = AsyncResult(task_id, app=celery_client)
    if task_result.ready():
        result = task_result.get()
        result_data = result["result"]
        output_path = result_data.get("generated_audio_file", None)

        status_data = result['status']
        status = status_data.get("status", None)
        
        update_payload = {"task_id": task_id, "status": status, "output_path": output_path}
        requests.post(f"{DATABASE_API_URL}/task/update", json=update_payload)

        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.get(),
        }
    else:
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": "Not ready",
        }


def audio_stream(audio_path):
    with open(audio_path, "rb") as audio_file:
        yield from audio_file

    
@app.get("/stream/{task_id}")
async def stream_task_result(task_id: str):
    logging.info('Start streaming task result ...')

    task_result = AsyncResult(task_id, app=celery_client)
    if task_result.ready():
        result = task_result.get()
        result = result['result']
        if "generated_audio_file" in result:
            audio_path = result["generated_audio_file"]
            logging.info(f'Result file path: {audio_path}')

            if os.path.exists(audio_path):
                return StreamingResponse(
                    audio_stream(audio_path),
                    media_type="audio/wav",
                )
            else:
                raise HTTPException(
                    status_code=404, detail="Audio file not found on the server."
                )
        else:
            raise HTTPException(
                status_code=400, detail="Task result does not contain an audio file."
            )
    else:
        raise HTTPException(
            status_code=202, detail="Task result is not ready yet. Please try again later."
        )