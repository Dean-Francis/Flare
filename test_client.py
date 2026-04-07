import pandas as pd
import requests

from config import FLAG_THRESHOLD, CLIENT_PORT, DATA_DIR

CLIENT_URL = f"http://localhost:{CLIENT_PORT}"
BACKUP_DATA = DATA_DIR / "backup" / "data.csv"


df = pd.read_csv(BACKUP_DATA)
benign = df[df["label"] == 0].sample(FLAG_THRESHOLD // 2, random_state=42)
phishing = df[df["label"] == 1].sample(FLAG_THRESHOLD // 2, random_state=42)
sample = pd.concat([benign, phishing]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Sending {len(sample)} emails to client /flag ({FLAG_THRESHOLD // 2} benign, {FLAG_THRESHOLD // 2} phishing)\n")

for i, row in sample.iterrows():
    r = requests.post(f"{CLIENT_URL}/flag", json={
        "user_id": "test_user",
        "body": row["message_body"],
        "label": int(row["label"]),
    })
    body = r.json()
    status = f"[{i+1}/{len(sample)}] label={row['label']} → {r.status_code}"
    if body.get("training_triggered"):
        status += " *** TRAINING TRIGGERED ***"
    print(status)

print("\nDone. Watch the client and server terminals for training and aggregation progress.")
print("Once complete, run: python -m model.evaluate")
