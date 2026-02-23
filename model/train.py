from transformers import DistilBertTokenizer, DataCollatorWithPadding, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from dataset import PhishingDataset
from extract import Extract
from config import BATCH_SIZE, MODEL_NAME
import torch

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
