from fastapi import FastAPI, HTTPException

from src.crud.speech_crud import SpeechCRUD
from src.schema.task_schema import TaskMetadata, TaskUpdate, TaskGet
from src.utils.api_logger import logging

app = FastAPI()
speech_crud = SpeechCRUD()


@app.post("/task/save")
def save_task(data: TaskMetadata):
    logging.info('Endpoint to save a new task to MongoDB.')
    try:
        speech_crud.save_task_metadata(
            user_id=data.user_id,
            task_id=data.task_id,
            input_path_remote=data.input_path_remote,
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
            input_path_local=data.input_path_local
        )
        return {"message": "Task metadata updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/get_by_user")
def get_tasks(data: TaskGet):
    logging.info('Endpoint to get existing tasks by specific user in MongoDB.')
    try:
        result = speech_crud.get_tasks_for_user(
            user_id=data.user_id
            )
        return {"message": f"Tasks of {data.user_id} get succesfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/get_task")
def get_specific_task(data: TaskGet):
    logging.info('Endpoint to get specific task in MongoDB.')
    try:
        result = speech_crud.get_task_metadata(
            task_id=data.task_id
        )
        return {"message": f"Metadata of task {data.task_id} for {data.user_id} get succesfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))