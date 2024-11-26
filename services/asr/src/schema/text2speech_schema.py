from pydantic import BaseModel, StrictStr
from typing import List, Optional
from enum import Enum

class StatusEnum(str, Enum):
    success = "success"
    failure = "failure"

class InputTxt2SpeechModel(BaseModel):
    input_file_path: StrictStr

class ResultTxt2SpeechModel(BaseModel):
    output_file_path: str

class StatusModel(BaseModel):
    status: StatusEnum
    message: Optional[str] = None

class OutputTxt2SpeechModel(BaseModel):
    result: ResultTxt2SpeechModel
    status: StatusModel