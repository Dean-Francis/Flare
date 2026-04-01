import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI

from config import FLAG_THRESHOLD, SAVED_MODEL_DIR, CLIENT_PORT
from database import init_db, insert_flagged_email, count_flagged_emails
from schemas import PredictRequest, PredictResponse, FlagRequest, FlagResponse
from trainer import run_local_training
from model.flare import Flare


detector: Flare = None
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    init_db()
    detector = Flare(model_name=str(SAVED_MODEL_DIR))
    yield


app = FastAPI(lifespan=lifespan)


def reload_model():
    global detector
    new_detector = Flare(model_name=str(SAVED_MODEL_DIR))
    with model_lock:
        detector = new_detector


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    with model_lock:
        result = detector.predict(request.body)
    return result


@app.post("/flag", response_model=FlagResponse)
def flag(request: FlagRequest):
    insert_flagged_email(request.user_id, request.body, request.label)
    count = count_flagged_emails()

    training_triggered = count % FLAG_THRESHOLD == 0
    if training_triggered:
        thread = threading.Thread(target=run_local_training, args=(reload_model,), daemon=True)
        thread.start()

    return FlagResponse(message="Email flagged successfully", count=count, training_triggered=training_triggered)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=CLIENT_PORT)
