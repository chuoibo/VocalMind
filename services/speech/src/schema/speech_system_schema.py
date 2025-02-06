from pydantic import BaseModel, StrictStr
from typing import Optional
from enum import Enum

class StatusEnum(str, Enum):
    success = "success"
    failure = "failure"

class InputSpeechSystemModel(BaseModel):
    input_audio_file_path: Optional[StrictStr]

class ResultSpeechSystemModel(BaseModel):
    generated_audio_file: Optional[StrictStr]

class StatusModel(BaseModel):
    status: StatusEnum
    message: Optional[str] = None

class OutputSpeechSystemModel(BaseModel):
    input: Optional[StrictStr]
    result: ResultSpeechSystemModel
    status: StatusModel