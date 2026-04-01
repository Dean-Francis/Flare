from pathlib import Path
# Data
TEST_SIZE = 0.2
VALIDATE_SIZE = 0.5
RANDOM_STATE = 42

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DATA_FILE = "raw_data.csv"
SAVED_MODEL_DIR = BASE_DIR / "saved_model"
DATABASE_URL = "sqlite:///" + str(BASE_DIR / "client" / "flare_client.db")

# MODEL
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16

# Train 
EPOCHS = 5
LEARNING_RATE = 2e-5

# Test 
MAX_LENGTH = 128

# Client
FLAG_THRESHOLD = 50
LOCAL_EPOCHS = 3
CLIENT_PORT = 8000
CENTRAL_PORT = 8001

