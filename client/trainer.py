import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DataCollatorWithPadding
from typing import Callable, Optional

from config import SAVED_MODEL_DIR, BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS, MAX_LENGTH
from database import SessionLocal, FlaggedEmail
from model.dataset import PhishingDataset


def run_local_training(callback: Optional[Callable] = None):
    with SessionLocal() as session:
        emails = session.query(FlaggedEmail).all()

    df = pd.DataFrame([{"body": e.body, "label": e.label} for e in emails])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(str(SAVED_MODEL_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(SAVED_MODEL_DIR))
    model = model.to(device)

    dataset = PhishingDataset(df, tokenizer, max_length=MAX_LENGTH)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(LOCAL_EPOCHS):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()

    model.save_pretrained(str(SAVED_MODEL_DIR))
    tokenizer.save_pretrained(str(SAVED_MODEL_DIR))

    if callback:
        callback()
