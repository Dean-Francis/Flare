import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import DistilBertTokenizer, DataCollatorWithPadding, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from dataset import PhishingDataset
from extract import Extract
from config import BATCH_SIZE, MODEL_NAME
from config import EPOCHS, LEARNING_RATE
from config import SAVED_MODEL_DIR
import torch
from torch.optim import AdamW
from typing import Tuple
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, DistilBertTokenizer]:
    extractor = Extract()
    train_df, val_df, test_df = extractor.get_splits()

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = PhishingDataset(train_df, tokenizer)
    val_dataset = PhishingDataset(val_df, tokenizer)
    test_dataset = PhishingDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

    return train_loader, val_loader, test_loader, tokenizer

def build_model(model_name: str = MODEL_NAME) -> Tuple[DistilBertForSequenceClassification, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = 2,
        id2label = {0: "BENIGN", 1: "PHISHING"},
        label2id = {"BENIGN": 0, "PHISHING": 1}
    )

    model = model.to(device)
    return model, device

def compute_metrics(labels, predictions) -> dict:
    return {
        'accuracy':         accuracy_score(labels, predictions),
        'precision':        precision_score(labels, predictions),
        'recall':           recall_score(labels, predictions),
        'f1':               f1_score(labels, predictions),
        'confusion_matrix': confusion_matrix(labels, predictions)
    }

def collect_predictions(model, data_loader, device) -> Tuple[float, list, list]:
    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            predictions = torch.argmax(outputs.logits, dim = -1)

            total_loss += outputs.loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(data_loader), all_predictions, all_labels

def validate(model, val_loader, device):
    model.eval()
    return collect_predictions(model, val_loader, device)

def save_model(model, tokenizer):
    SAVED_MODEL_DIR.mkdir(parents = True, exist_ok = True)
    model.save_pretrained(SAVED_MODEL_DIR)
    tokenizer.save_pretrained(SAVED_MODEL_DIR)

def train(model, train_loader, val_loader, device, tokenizer):
    optimizer = AdamW(model.parameters(), lr = LEARNING_RATE)
    best_recall = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc = f"Epoch {epoch + 1}/{EPOCHS}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        avg_val_loss, val_predictions, val_labels = validate(model, val_loader, device)
        metrics = compute_metrics(val_labels, val_predictions)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {average_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

        if metrics['recall'] > best_recall:
            best_recall = metrics['recall']
            save_model(model, tokenizer)
            print(f"New best recall: {best_recall:.4f} — model saved")
       

if __name__ == "__main__":
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders()
    model, device = build_model()
    train(model, train_loader, val_loader, device, tokenizer)
