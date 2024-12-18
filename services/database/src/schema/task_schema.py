from pydantic import BaseModel

class TaskMetadata(BaseModel):
    task_id: str
    input_path: str
    time_sent: str

class TaskUpdate(BaseModel):
    task_id: str
    status: str
    output_path: str = None
