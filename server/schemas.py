from pydantic import BaseModel


class UpdateRequest(BaseModel):
    user_id: str
    weights: str  # base64-encoded pickled weight-delta state dict
    num_samples: int


class UpdateResponse(BaseModel):
    message: str
    round_id: int


class RoundResponse(BaseModel):
    round_id: int
