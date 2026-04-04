import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import pickle
import threading

from transformers import AutoModelForSequenceClassification

from config import SAVED_MODEL_DIR, MIN_CLIENTS, ROUND_TIMEOUT
from .database import (
    close_round, create_round, count_updates_for_round,
    delete_updates_for_round, get_updates_for_round,
)

# ── Global state ──────────────────────────────────────────────────────────────
_loop: asyncio.AbstractEventLoop = None
_model_ready_event: asyncio.Event = None
_model_lock = threading.Lock()
_timer: threading.Timer | None = None
current_round_id: int = None


# ── Init ──────────────────────────────────────────────────────────────────────
def init(loop: asyncio.AbstractEventLoop):
    global _loop, _model_ready_event
    _loop = loop
    _model_ready_event = asyncio.Event()


def get_model_ready_event() -> asyncio.Event:
    return _model_ready_event


# ── Timer management ──────────────────────────────────────────────────────────
def schedule_deadline(round_id: int, seconds: float):
    global _timer
    _timer = threading.Timer(seconds, _on_deadline, args=[round_id])
    _timer.daemon = True
    _timer.start()


def cancel_timer():
    global _timer
    if _timer is not None:
        _timer.cancel()
        _timer = None


# ── Deadline callback (runs in timer thread) ──────────────────────────────────
def _on_deadline(round_id: int):
    count = count_updates_for_round(round_id)
    if count >= MIN_CLIENTS:
        aggregate(round_id)
    else:
        _skip_round(round_id)


# ── Skip round — too few clients, no aggregation ──────────────────────────────
def _skip_round(round_id: int):
    delete_updates_for_round(round_id)
    close_round(round_id)
    new_round = create_round()
    _loop.call_soon_threadsafe(_set_current_round, new_round.id, ROUND_TIMEOUT)


def _set_current_round(round_id: int, timeout: float):
    global current_round_id
    current_round_id = round_id
    schedule_deadline(round_id, timeout)


# ── Aggregation (runs in a worker thread) ─────────────────────────────────────
def aggregate(round_id: int):
    acquired = _model_lock.acquire(blocking=False)
    if not acquired:
        return

    try:
        updates = get_updates_for_round(round_id)
        model = _load_model()
        model = _fedavg(updates, model)
        _save_model(model)
        delete_updates_for_round(round_id)
        close_round(round_id)
        new_round = create_round()
        _loop.call_soon_threadsafe(_on_round_complete, new_round.id)
    finally:
        _model_lock.release()


def _fedavg(updates, model) -> AutoModelForSequenceClassification:
    total_samples = sum(u.num_samples for u in updates)
    deltas = [pickle.loads(u.weights) for u in updates]
    state = model.state_dict()

    for key in state:
        weighted_sum = sum(
            delta[key] * (u.num_samples / total_samples)
            for delta, u in zip(deltas, updates)
        )
        state[key] = state[key] + weighted_sum

    model.load_state_dict(state)
    return model


def _load_model() -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(str(SAVED_MODEL_DIR))


def _save_model(model: AutoModelForSequenceClassification):
    model.save_pretrained(str(SAVED_MODEL_DIR))


# ── Post-aggregation round transition (runs on event loop thread) ─────────────
def _on_round_complete(new_round_id: int):
    global current_round_id, _model_ready_event
    current_round_id = new_round_id
    schedule_deadline(new_round_id, ROUND_TIMEOUT)

    old_event = _model_ready_event
    _model_ready_event = asyncio.Event()
    old_event.set()
