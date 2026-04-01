from pydantic import BaseModel


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

class FlagResponse(BaseModel):
    message: str
    count: int
    training_triggered: bool
