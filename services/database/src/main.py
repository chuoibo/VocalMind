from fastapi import FastAPI, HTTPException

from src.crud.speech_base_crud import SpeechCRUD
from src.schema.task_schema import TaskMetadata, TaskUpdate
from src.utils.api_logger import logging

app = FastAPI()
speech_crud = SpeechCRUD()


@app.post("/task/save")
def save_task(data: TaskMetadata):
    logging.info('Endpoint to save a new task to MongoDB.')
    try:
        speech_crud.save_task_metadata(
            task_id=data.task_id,
            input_path=data.input_path,
            time_sent=data.time_sent,
        )
        return {"message": "Task metadata saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/task/update")
def update_task(data: TaskUpdate):
    logging.info('Endpoint to update an existing task in MongoDB.')
    try:
        speech_crud.update_task_metadata(
            task_id=data.task_id,
            status=data.status,
            output_path=data.output_path,
        )
        return {"message": "Task metadata updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
