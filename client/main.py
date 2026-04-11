import threading
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config import FLAG_THRESHOLD, CLIENT_MODEL_DIR, CLIENT_PORT, BASE_DIR
from .database import (
    init_db,
    insert_flagged_email,
    count_flagged_emails,
    get_flagged_emails_by_user,
    get_all_flagged_emails,
    get_distinct_user_ids,
    get_flag_stats,
)
from .schemas import (
    PredictRequest,
    PredictResponse,
    FlagRequest,
    FlagResponse,
    FlaggedEmailRecord,
    FlaggedEmailsResponse,
)
from .trainer import run_local_training
from model.flare import Flare


detector: Flare = None
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    init_db()
    detector = Flare(model_name=str(CLIENT_MODEL_DIR))
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def reload_model():
    global detector
    new_detector = Flare(model_name=str(CLIENT_MODEL_DIR))
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
        thread = threading.Thread(
            target=run_local_training, args=(reload_model,), daemon=True
        )
        thread.start()

    return FlagResponse(
        message="Email flagged successfully",
        count=count,
        training_triggered=training_triggered,
    )


@app.get("/flagged", response_model=FlaggedEmailsResponse)
def flagged(user_id: str):
    rows = get_flagged_emails_by_user(user_id)
    return FlaggedEmailsResponse(
        emails=[
            FlaggedEmailRecord(
                id=r.id, body=r.body, label=r.label, timestamp=r.timestamp
            )
            for r in rows
        ]
    )


@app.get("/admin/users")
def admin_users():
    return get_distinct_user_ids()


@app.get("/admin/flagged")
def admin_flagged(user_id: Optional[str] = None):
    rows = get_all_flagged_emails(user_id or None)
    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "body": r.body,
            "label": r.label,
            "timestamp": r.timestamp,
        }
        for r in rows
    ]


@app.get("/admin/stats")
def admin_stats(
    user_id: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    return get_flag_stats(user_id or None, start or None, end or None)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html_path = BASE_DIR / "client" / "admin_dashboard.html"
    return html_path.read_text()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=CLIENT_PORT)
