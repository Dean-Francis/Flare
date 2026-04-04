# Flare — Federated Learning Phishing Defense System

## Project Context

Flare is a graduation project. It is a hybrid federated learning system for phishing email detection. The AI team is responsible for all model and ML infrastructure components. A separate software engineering team owns the browser extension UI. This document is a precise handoff for any AI assistant continuing work on this project.

---

## System Architecture

```
[Central Server]
 - Hosts global DistilBERT model (pre-trained on separate 10K dataset)
 - Exposes REST API for: model download, receiving weight updates
 - Runs FedAvg aggregation on a weekly schedule
 - Distributes updated global model to connecting clients

        ↕  weight updates (no raw email data ever leaves local)

[Local Server — Desktop App]
 - Installed by the user (individual or enterprise IT)
 - On first install: pulls latest global model from central server
 - Exposes REST API on localhost for the browser extension
 - Runs inference on emails sent from the extension
 - Stores user-flagged emails locally
 - Triggers local training once N emails have been flagged
 - Sends weight updates to central server on a weekly schedule

        ↕  email text / prediction result

[Browser Extension]
 - Sends email text to local server → displays phishing/benign result
 - Allows user to flag any email as phishing or benign
 - Built by the software engineering team (not AI team scope)
```

### Enterprise vs Individual — Same Flow
Both individual users and enterprise deployments use identical architecture. The only difference is that an enterprise IT team installs the desktop app on a company server and deploys the extension to all employees, who connect to the IT server instead of a personal local server. The central server interaction is identical.

### Data Privacy Guarantee
Raw email data never leaves the user's device. Only model weight updates are sent to the central server.

---

## Current Codebase State

### What Exists
- DistilBERT fine-tuned for binary phishing/benign email classification
- Centralized training pipeline (no FL yet)
- 309K labeled email dataset (columns: `body`, `label`) — used for current development/testing
- Inference class (`Flare`)

### What Does Not Exist Yet
- Federated learning infrastructure (local server, central server, aggregation)
- REST APIs (local or central)
- Weekly round scheduler
- Local data storage and training trigger
- Full evaluation metrics (only loss is tracked currently)

---

## File Structure

```
Flare/
├── data/
│   └── raw_data.csv              # 309K emails: columns = body (str), label (int: 0=benign, 1=phishing)
├── model/
│   ├── config.py                 # All hyperparameters and paths — single source of truth
│   ├── extract.py                # Loads, cleans, splits data (Extract class)
│   ├── dataset.py                # PyTorch Dataset with on-the-fly tokenization (PhishingDataset)
│   ├── train.py                  # Training loop, validation, model saving
│   ├── flare.py                  # Inference class (Flare) — has known bugs (see below)
│   └── Test.ipynb                # Exploratory notebook
├── saved_model/                  # Model checkpoints saved here after each epoch
├── CLAUDE.md                     # Guidance for Claude Code
├── requirements.txt
└── README.md
```

---

## Key Configuration (`model/config.py`)

| Parameter | Value | Notes |
|---|---|---|
| `MODEL_NAME` | `distilbert-base-uncased` | HuggingFace model |
| `BATCH_SIZE` | 16 | |
| `EPOCHS` | 5 | |
| `LEARNING_RATE` | 2e-5 | Standard fine-tuning LR |
| `MAX_LENGTH` | 128 | Tokenizer max length for training |
| `TEST_SIZE` | 0.2 | 20% held out first |
| `VALIDATE_SIZE` | 0.5 | 50% of that → results in 80/10/10 split |
| `SAVED_MODEL_DIR` | `<root>/saved_model/` | |

---

## Label Convention

- `0` = BENIGN
- `1` = PHISHING

---

## Known Bugs (Fix Before Building FL)

### `model/flare.py`
1. **Line 53** — typo: `predicted_clas` should be `predicted_class`. This causes a `NameError` when the email is classified as legitimate.
2. **Line 66** — syntax error: `detector.predict(test):` has a trailing colon, should be `detector.predict(test)`.
3. **Missing imports** — `Dict` and `Any` are used in type hints but never imported from `typing`.

### `model/extract.py`
4. **Line 26** — `Tuple` is used in the return type hint of `get_splits()` but never imported from `typing`.

---

## Implementation Roadmap

### Phase 1 — Fix & Validate Base Model (current priority)
- Fix all 4 bugs listed above
- Add evaluation metrics: accuracy, F1, precision, recall, confusion matrix
- Run full evaluation on test set
- Confirm model performance is solid before adding FL

### Phase 2 — Local Server
- FastAPI server running on localhost
- `POST /predict` — receives email text, returns classification + confidence
- Local SQLite or flat-file storage for flagged emails
- Local training pipeline triggered once N flagged emails are accumulated
- Pluggable N threshold (configurable)

### Phase 3 — Central Server
- FastAPI server
- `GET /model` — returns latest global model weights
- `POST /weights` — receives weight update from a local server
- FedAvg aggregation logic — designed to be swappable with FedProx
- Weekly round scheduler

### Phase 4 — Federated Loop
- Local server trains on flagged data → serializes weight delta → POST to central
- Central aggregates all received weights → updates global model
- Local server pulls updated global model on weekly sync
- End-to-end round validated

---

## Aggregation Strategy

- **Current target**: FedAvg (McMahan et al.)
- **Future**: FedProx — the aggregation logic must be written as a pluggable/swappable component so switching requires minimal changes
- **Privacy**: None for now (raw weights transmitted). Differential privacy and secure aggregation are planned for later iterations.

---

## Data

- **Current dataset**: 309K emails in `data/raw_data.csv`. Named entities replaced with bracketed placeholders (`[NAME]`, `[DATE]`, `[ADDRESS]`). Nearly balanced: ~50.6% phishing, ~49.4% benign.
- **Initial global model dataset**: 10K samples drawn from `data/raw_data.csv` will be used to pre-train the first version of the global model that gets distributed to users on install.
- **Local training data**: User-flagged emails stored on-device. Distribution will be non-IID across clients (realistic — different users encounter different email patterns).

---

## Running the Code

All scripts must be run from inside `model/` due to relative imports:

```bash
cd model
python train.py      # full training pipeline
python flare.py      # single inference test
python extract.py    # inspect data pipeline
```

---

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- DistilBERT (`distilbert-base-uncased`) — 67M parameters, 6-layer transformer
- FastAPI — planned for both local and central servers
- CUDA supported (torch+cu126 in requirements.txt)
