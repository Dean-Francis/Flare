import base64
import copy
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DataCollatorWithPadding,
)

from config import (
    CLIENT_MODEL_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MAX_LENGTH,
    SERVER_HOST,
    CENTRAL_PORT,
    ROUND_TIMEOUT,
    CLIENT_ID,
)
from .database import get_untrained_emails, mark_emails_trained
from model.dataset import PhishingDataset

logger = logging.getLogger(__name__)

SERVER_URL = f"http://{SERVER_HOST}:{CENTRAL_PORT}"


def run_local_training(callback: Optional[Callable] = None):
    try:
        _run_local_training(callback)
    except Exception:
        logger.exception("Local training failed")


def _run_local_training(callback: Optional[Callable] = None):
    emails = get_untrained_emails()
    if not emails:
        return

    num_samples = len(emails)
    email_ids = [e.id for e in emails]
    df = pd.DataFrame([{"body": e.body, "label": e.label} for e in emails])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(str(CLIENT_MODEL_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(CLIENT_MODEL_DIR))
    model = model.to(device)

    original_state = copy.deepcopy(model.state_dict())

    dataset = PhishingDataset(df, tokenizer, max_length=MAX_LENGTH)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(LOCAL_EPOCHS):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            outputs.loss.backward()
            optimizer.step()

    trained_state = model.state_dict()
    delta = {k: trained_state[k] - original_state[k] for k in original_state}
    weights_b64 = base64.b64encode(pickle.dumps(delta)).decode()

    response = requests.post(
        f"{SERVER_URL}/update",
        json={"user_id": CLIENT_ID, "weights": weights_b64, "num_samples": num_samples},
        timeout=30,
    )
    response.raise_for_status()

    global_model_response = requests.get(
        f"{SERVER_URL}/model",
        stream=True,
        timeout=ROUND_TIMEOUT + 60,
    )
    global_model_response.raise_for_status()

    model_path = CLIENT_MODEL_DIR / "model.safetensors"
    with tempfile.NamedTemporaryFile(
        delete=False, dir=CLIENT_MODEL_DIR, suffix=".tmp"
    ) as tmp:
        tmp_path = tmp.name
        for chunk in global_model_response.iter_content(chunk_size=8192):
            tmp.write(chunk)
    os.replace(tmp_path, model_path)

    mark_emails_trained(email_ids)

    if callback:
        callback()
