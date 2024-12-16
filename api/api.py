import os

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from aws_client import upload_to_s3
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
    
    
@app.get("/stream/{task_id}")
async def stream_task_result(task_id: str):
    task_result = AsyncResult(task_id, app=celery_client)
    if task_result.ready():
        result = task_result.get()
        if "output_audio_file_path" in result:
            audio_path = result["output_audio_file_path"]
            if os.path.exists(audio_path):
                return FileResponse(
                    path=audio_path,
                    media_type="audio/wav",
                    filename=os.path.basename(audio_path),
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