import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, VALIDATE_SIZE, RANDOM_STATE
from config import BASE_DIR, DATA_DIR, MODEL_DIR, DATA_FILE


class Extract:
    def __init__(self) -> None:
        self.df = pd.read_csv(DATA_DIR / DATA_FILE)
    
    # Prints missing values as percentage
    def get_missing_values(self) -> pd.Series: 
        return round((self.df.isnull().sum() / self.df.shape[0]) * 100, 2)

    def clean_data(self) -> pd.DataFrame:
        self.df = self.df.dropna(subset=["body"])
        self.df["label"] = self.df["label"].astype(int)
        self.df = self.df.drop_duplicates(subset=["body"])
        return self.df

    def get_clean_data(self) -> pd.DataFrame:
        self.clean_data()
        return self.df[["body", "label"]]

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        clean_df = self.get_clean_data()
        train_df, temp_df = train_test_split(clean_df, test_size = TEST_SIZE, random_state = RANDOM_STATE, stratify=clean_df['label'])
        val_df, test_df = train_test_split(temp_df, test_size = VALIDATE_SIZE, random_state = RANDOM_STATE, stratify=temp_df['label'])
        return train_df, val_df, test_df

if __name__ == "__main__":
    extractor = Extract()
    print(extractor.get_clean_data().head())
