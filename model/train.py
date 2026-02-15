from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


dataset = PhishingDataset(clean_dataset, tokenizer)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    dataset,
    batch_size = 32,
    shuffle = True,
    collate_fn = data_collator
)

