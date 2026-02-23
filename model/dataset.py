import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
import pandas as pd
from typing import Dict

class PhishingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: DistilBertTokenizer, max_length: int = 512) -> None:
        # Just resets index after preprocessing
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Returns the number of rows
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = self.df.iloc[index]["body"]
        label = self.df.iloc[index]["label"]

        #Tokenize this one sample on-the-fly
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    from extract import Extract
    from transformers import DataCollatorWithPadding
    from torch.utils.data import DataLoader

    extractor = Extract()
    clean_df = extractor.get_clean_data()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    dataset = PhishingDataset(clean_df, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size = 32,
        shuffle = True,
        collate_fn = data_collator
    )

    
    # 6. Test it - get one batch
    print(f"Dataset size: {len(dataset)}")
    print(f"\nTesting DataLoader with dynamic padding:")
    
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")  # [batch_size, dynamic_length]
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")  # Note: 'labels' not 'label' in batch
    
    # Show that padding is dynamic (different batch sizes)
    batch2 = next(iter(dataloader))
    print(f"\nSecond batch Input IDs shape: {batch2['input_ids'].shape}")
    print("Notice: sequence length may differ between batches (dynamic padding!)")

