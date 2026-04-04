# Flare — Federated Learning Phishing Defense System

## Project Context

Flare is a graduation project. It is a hybrid federated learning system for phishing email detection. The AI team is responsible for all model and ML infrastructure components. A separate software engineering team owns the browser extension UI. This document is a precise handoff for any AI assistant continuing work on this project.

---

## System Architecture

```
[Central Server]
 - Hosts global DistilBERT model (pre-trained on separate 10K dataset)
 - Exposes REST API for: model download, receiving weight updates
 - Runs FedAvg aggregation on a hybrid threshold/timeout schedule
 - Distributes updated global model to connecting clients

        ↕  weight updates (no raw email data ever leaves local)

[Local Server — Desktop App]
 - Installed by the user (individual or enterprise IT)
 - On first install: pulls latest global model from central server
 - Exposes REST API on localhost for the browser extension
 - Runs inference on emails sent from the extension
 - Stores user-flagged emails locally in SQLite
 - Triggers local training once N emails have been flagged
 - Sends weight updates to central server after local training

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
- Centralized training pipeline
- 309K labeled email dataset (columns: `body`, `label`) — used for initial model training
- Inference class (`Flare`) — bugs fixed
- **Central server** (FastAPI) — fully built with FedAvg aggregation, hybrid threshold/timeout round management, and SQLite-backed persistence
- **Local client server** (FastAPI) — fully built with inference, email flagging, local training pipeline, and model hot-reloading

### What Does Not Exist Yet
- End-to-end federated loop: client does not yet send weight deltas to the central server after local training, and does not pull the updated global model back from central
- Browser extension (SE team scope)
- Full evaluation metrics (only loss tracked in training loop)

---

## File Structure

```
Flare/
├── data/
│   └── raw_data.csv              # 309K emails: columns = body (str), label (int: 0=benign, 1=phishing)
├── model/
│   ├── extract.py                # Loads, cleans, splits data (Extract class)
│   ├── dataset.py                # PyTorch Dataset with on-the-fly tokenization (PhishingDataset)
│   ├── train.py                  # Training loop, validation, model saving
│   └── flare.py                  # Inference class (Flare)
├── server/
│   ├── main.py                   # Central server FastAPI app (POST /update, GET /round, GET /model)
│   ├── aggregator.py             # FedAvg aggregation, round lifecycle, threading.Timer scheduler
│   ├── database.py               # SQLAlchemy models: Round, WeightUpdate
│   └── schemas.py                # Pydantic schemas: UpdateRequest, UpdateResponse, RoundResponse
├── client/
│   ├── main.py                   # Local client FastAPI app (POST /predict, POST /flag)
│   ├── trainer.py                # Local training pipeline — fine-tunes on flagged emails
│   ├── database.py               # SQLAlchemy model: FlaggedEmail
│   └── schemas.py                # Pydantic schemas: PredictRequest/Response, FlagRequest/Response
├── saved_model/                  # Model checkpoints saved here after training
├── config.py                     # All hyperparameters and paths — single source of truth
├── CLAUDE.md                     # Guidance for Claude Code
├── requirements.txt
└── README.md
```

---

## Key Configuration (`config.py`)

| Parameter | Value | Notes |
|---|---|---|
| `MODEL_NAME` | `distilbert-base-uncased` | HuggingFace model |
| `BATCH_SIZE` | 16 | |
| `EPOCHS` | 5 | Central training epochs |
| `LOCAL_EPOCHS` | 3 | Local client fine-tuning epochs |
| `LEARNING_RATE` | 2e-5 | Standard fine-tuning LR |
| `MAX_LENGTH` | 128 | Tokenizer max length |
| `TEST_SIZE` | 0.2 | 20% held out first |
| `VALIDATE_SIZE` | 0.5 | 50% of that → results in 80/10/10 split |
| `SAVED_MODEL_DIR` | `<root>/saved_model/` | |
| `FLAG_THRESHOLD` | 50 | Flagged emails needed to trigger local training |
| `AGGREGATION_THRESHOLD` | 5 | Weight updates needed to trigger early aggregation |
| `MIN_CLIENTS` | 2 | Minimum updates needed at deadline to aggregate (else skip round) |
| `ROUND_TIMEOUT` | 86400 | Round deadline in seconds (24 hours) |
| `CLIENT_PORT` | 8000 | Local client server port |
| `CENTRAL_PORT` | 8001 | Central server port |

---

## Label Convention

- `0` = BENIGN
- `1` = PHISHING

---

## Central Server API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/update` | Receive a weight update from a client. Body: `{user_id, weights (base64), num_samples}`. Returns `409` if user already submitted this round. |
| `GET` | `/round` | Returns the current open round ID. |
| `GET` | `/model` | Long-polls until the model is ready (post-aggregation), then streams the model file. |

### Round Lifecycle
A round starts on server boot (or after the previous round closes). It closes when either:
- `AGGREGATION_THRESHOLD` updates are received (early trigger), or
- `ROUND_TIMEOUT` seconds elapse and at least `MIN_CLIENTS` updates exist

If the deadline passes with fewer than `MIN_CLIENTS` updates, the round is skipped and a new one opens.

---

## Local Client API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Run inference on an email. Body: `{body}`. Returns probabilities + predicted class. |
| `POST` | `/flag` | Store a user-flagged email. Body: `{user_id, body, label}`. Triggers local training when flagged count hits `FLAG_THRESHOLD`. |

---

## Implementation Roadmap

### Phase 1 — Fix & Validate Base Model ✓
- All 4 bugs fixed in `flare.py` and `extract.py`
- Remaining: full evaluation metrics (accuracy, F1, precision, recall, confusion matrix)

### Phase 2 — Local Server ✓
- `POST /predict` — inference on email text
- `POST /flag` — stores flagged email, triggers local training at threshold
- Local SQLite storage for flagged emails
- Local training pipeline (`trainer.py`) with model hot-reload after training

### Phase 3 — Central Server ✓
- `POST /update`, `GET /round`, `GET /model` endpoints
- FedAvg aggregation
- Hybrid threshold/timeout round trigger
- SQLite-backed round and weight update persistence
- `threading.Timer` deadline scheduling with graceful restart on server boot

### Phase 4 — Federated Loop (current priority)
- After local training, serialize weight delta (local weights − global weights) and `POST /update` to central server
- After aggregation completes, client pulls updated global model from `GET /model` and hot-reloads
- End-to-end round validated: flag → train → send delta → aggregate → pull → hot-reload

---

## Aggregation Strategy

- **Current**: FedAvg (McMahan et al.) — weighted average of weight deltas by `num_samples`
- **Future**: FedProx — aggregation logic in `server/aggregator.py` should remain a swappable component
- **Privacy**: None for now (raw weight deltas transmitted). Differential privacy and secure aggregation planned for later iterations.

---

## Data

- **Current dataset**: `data/raw_data.csv` contains ~10K emails. Named entities replaced with bracketed placeholders (`[NAME]`, `[DATE]`, `[ADDRESS]`). Nearly balanced classes.
- **Initial global model**: This 10K dataset is used to pre-train the first version of the global model distributed to users on install.
- **Local training data**: User-flagged emails stored on-device. Distribution will be non-IID across clients (realistic — different users encounter different email patterns).

---

## Running the Code

All model scripts must be run from inside `model/` due to relative imports:

```bash
cd model
python train.py      # full training pipeline
python flare.py      # single inference test
python extract.py    # inspect data pipeline
```

Run the central server from the project root:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8001
```

Run the local client server from the `client/` directory:

```bash
cd client
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- DistilBERT (`distilbert-base-uncased`) — 67M parameters, 6-layer transformer
- FastAPI + Uvicorn — both central and local servers
- SQLAlchemy + SQLite — persistence for rounds, weight updates, and flagged emails
- CUDA supported (torch+cu126 in requirements.txt)
