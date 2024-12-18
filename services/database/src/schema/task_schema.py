from pydantic import BaseModel
from typing import Optional

class TaskMetadata(BaseModel):
    user_id: str
    task_id: str
    input_path: str
    time_sent: str

class TaskUpdate(BaseModel):
    task_id: str
    status: str
    output_path: str = None

class TaskGet(BaseModel):
    task_id: Optional[str]
    user_id: Optional[str]
