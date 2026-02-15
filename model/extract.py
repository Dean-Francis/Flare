import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

class Extract:
    def __init__(self) -> None:
        self.df = pd.read_csv(DATA_DIR / "data.csv")
    
    # Prints missing values as percentage
    def get_missing_values(self) -> pd.Series: 
        return round((self.df.isnull().sum() / self.df.shape[0]) * 100, 2)

    def clean_data(self) -> pd.DataFrame:
        self.df = self.df.dropna(subset=["message_body"])
        self.df["sender"] = self.df["sender"].fillna("")
        self.df["url"] = self.df["url"].fillna("")
        self.df["label"] = self.df["label"].astype(int)
        self.df = self.df.drop_duplicates(subset=["message_body"])
        return self.df

    def get_clean_data(self) -> pd.DataFrame:
        self.clean_data()
        return self.df[["message_body", "label"]]

if __name__ == "__main__":
    extractor = Extract()
    print(extractor.get_clean_data().head())
