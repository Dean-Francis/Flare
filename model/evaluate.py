import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader

from .extract import Extract
from .dataset import PhishingDataset
from .train import collect_predictions, compute_metrics
from config import SAVED_MODEL_DIR, BATCH_SIZE


def evaluate():
    extractor = Extract()
    _, _, test_df = extractor.get_splits()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(str(SAVED_MODEL_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(SAVED_MODEL_DIR))
    model = model.to(device)
    model.eval()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        PhishingDataset(test_df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    loss, predictions, labels = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(labels, predictions)

    print(f"Test Loss:  {loss:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1:         {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")


if __name__ == "__main__":
    evaluate()
