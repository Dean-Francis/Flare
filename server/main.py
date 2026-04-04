import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import aggregator
from config import AGGREGATION_THRESHOLD, ROUND_TIMEOUT, SAVED_MODEL_DIR
from .database import (
    count_updates_for_round, create_round, get_open_round,
    has_user_submitted, init_db, insert_weight_update,
)
from .schemas import RoundResponse, UpdateRequest, UpdateResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    loop = asyncio.get_running_loop()
    aggregator.init(loop)

    open_round = get_open_round()
    if open_round:
        deadline = open_round.deadline
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        seconds_remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        aggregator.current_round_id = open_round.id
        if seconds_remaining <= 0:
            loop.run_in_executor(None, aggregator.aggregate, open_round.id)
        else:
            aggregator.schedule_deadline(open_round.id, seconds_remaining)
    else:
        new_round = create_round()
        aggregator.current_round_id = new_round.id
        aggregator.schedule_deadline(new_round.id, ROUND_TIMEOUT)

    yield

    aggregator.cancel_timer()


app = FastAPI(lifespan=lifespan)


@app.post("/update", response_model=UpdateResponse)
async def submit_update(req: UpdateRequest, background_tasks: BackgroundTasks):
    round_id = aggregator.current_round_id

    if has_user_submitted(round_id, req.user_id):
        raise HTTPException(status_code=409, detail="Already submitted for this round")

    weights_bytes = base64.b64decode(req.weights)
    insert_weight_update(round_id, req.user_id, weights_bytes, req.num_samples)

    count = count_updates_for_round(round_id)
    if count >= AGGREGATION_THRESHOLD:
        aggregator.cancel_timer()
        background_tasks.add_task(aggregator.aggregate, round_id)

    return UpdateResponse(message="Update received", round_id=round_id)


@app.get("/round", response_model=RoundResponse)
async def get_round():
    return RoundResponse(round_id=aggregator.current_round_id)


@app.get("/model")
async def get_model():
    event = aggregator.get_model_ready_event()
    await event.wait()

    model_path = Path(SAVED_MODEL_DIR) / "model.safetensors"
    if not model_path.exists():
        model_path = Path(SAVED_MODEL_DIR) / "pytorch_model.bin"

    return FileResponse(
        str(model_path),
        media_type="application/octet-stream",
        filename=model_path.name,
    )
