from pydantic import BaseModel, Field


class UpdateRequest(BaseModel):
    user_id: str = Field(min_length=1)
    weights: str = Field(min_length=1)  # base64-encoded pickled weight-delta state dict
    num_samples: int = Field(gt=0)


class UpdateResponse(BaseModel):
    message: str
    round_id: int


class RoundResponse(BaseModel):
    round_id: int
