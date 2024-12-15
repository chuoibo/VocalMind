import os
import uuid

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from worker import celery_client

app = FastAPI()

@app.post("/speech")
async def add_task(live_record: bool = Form(...),
                   input_audio_file_path: UploadFile = File(None)):
    
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
        
        os.makedirs("./common_dir/input/", exist_ok=True)

        
        file_location = f"./common_dir/input/{uuid.uuid4()}_{input_audio_file_path.filename}"

        with open(file_location, "wb") as buffer:
            buffer.write(await input_audio_file_path.read())

        input_model = {
            "live_record": False,
            "input_audio_file_path": file_location,
        }

    task = celery_client.send_task(
        "speech_ai", args=[input_model], queue="speech_ai_queue"
    )

    return {"task_id": task.id}


@app.get("/check/{task_id}")
async def get_task_result(task_id: str):
    task_result = AsyncResult(task_id, app=celery_client)
    if task_result.ready():
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