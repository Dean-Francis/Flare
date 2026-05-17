from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    body: str

class PredictResponse(BaseModel):
    legitimate: float
    phishing: float
    predicted: str
    confidence: float

class FlagRequest(BaseModel):
    user_id: str
    body: str
    label: int
    confidence: Optional[float] = None

class FlagResponse(BaseModel):
    message: str
    count: int
    training_triggered: bool

class FlaggedEmailRecord(BaseModel):
    id: int
    body: str
    label: int
    timestamp: str
    confidence: Optional[float] = None

class FlaggedEmailsResponse(BaseModel):
    emails: list[FlaggedEmailRecord]
