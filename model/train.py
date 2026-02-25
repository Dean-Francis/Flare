from transformers import DistilBertTokenizer, DataCollatorWithPadding, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from dataset import PhishingDataset
from extract import Extract
from config import BATCH_SIZE, MODEL_NAME
from config import EPOCHS, LEARNING_RATE
from pathlib import Path
from config import SAVED_MODEL_DIR
import torch
from torch.optim import AdamW
from typing import Tuple
from tqdm import tqdm
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

def train(model, train_loader, val_loader, device, tokenizer):
    optimizer = AdamW(model.parameters(), lr = LEARNING_RATE) 
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
        avg_val_loss = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {average_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        SAVED_MODEL_DIR.mkdir(parents = True, exist_ok = True)
        model.save_pretrained(SAVED_MODEL_DIR)
        tokenizer.save_pretrained(SAVED_MODEL_DIR)
        
def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
    
    return total_val_loss / len(val_loader)

if __name__ == "__main__":
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders()
    model, device = build_model()
    train(model, train_loader, val_loader, device, tokenizer)
