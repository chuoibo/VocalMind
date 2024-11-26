from pydantic import BaseModel, StrictStr
from typing import Optional
from enum import Enum

class StatusEnum(str, Enum):
    success = "success"
    failure = "failure"

class InputSpeech2TxtModel(BaseModel):
    live_record: bool

class ResultSpeech2TxtModel(BaseModel):
    generated_text_file: StrictStr

class StatusModel(BaseModel):
    status: StatusEnum
    message: Optional[str] = None

class OutputSpeech2TxtModel(BaseModel):
    result: ResultSpeech2TxtModel
    status: StatusModel