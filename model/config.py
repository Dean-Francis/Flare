from pathlib import Path
# Data
TEST_SIZE = 0.2
VALIDATE_SIZE = 0.5
RANDOM_STATE = 42

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DATA_FILE = "raw_data.csv"
