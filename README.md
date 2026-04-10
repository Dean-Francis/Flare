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
 - Serves an admin dashboard (GET /dashboard) for monitoring flags and stats

        ↕  email text / prediction result

[Browser Extension — Chrome MV3]
 - Gmail content script auto-detects opened emails and sends text to local server
 - Displays phishing warning popup on Gmail page when phishing is detected
 - Extension popup shows status circle (idle/legitimate/phishing) with confidence
 - Manual email check: collapsible text input to paste email content for assessment
 - Flag correction: user can override a result (Mark as Phishing / Mark as Legitimate)
 - Caches flag overrides locally so rescanning returns the corrected result
 - Extension dashboard with flagged emails list and user settings (user ID)
 - Persists last scan result and user preferences via chrome.storage.local
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
- ~10K labeled email dataset (columns: `body`, `label`) — used for initial model training
- Inference class (`Flare`) — bugs fixed
- **Central server** (FastAPI) — fully built with FedAvg aggregation, hybrid threshold/timeout round management, and SQLite-backed persistence
- **Local client server** (FastAPI) — fully built with inference, email flagging, local training pipeline, model hot-reloading, admin dashboard, and admin API endpoints
- **Browser extension** (Chrome MV3) — Gmail content script with auto-detection, popup with status indicator and manual check, flag correction workflow, extension dashboard with settings

---

## File Structure

```
Flare/
├── data/
│   ├── raw_data.csv              # ~10K emails: columns = body (str), label (int: 0=benign, 1=phishing)
│   ├── AI_Model_Research/        # Research documents
│   ├── Documentation_02_27_2026.odt
│   ├── Test.ipynb
│   └── backup/
├── docs/                         # Additional project documentation
├── model/
│   ├── extract.py                # Loads, cleans, splits data (Extract class)
│   ├── dataset.py                # PyTorch Dataset with on-the-fly tokenization (PhishingDataset)
│   ├── train.py                  # Training loop, validation, model saving
│   ├── flare.py                  # Inference class (Flare)
│   └── evaluate.py               # Standalone evaluation script — loads saved model, runs test set, prints metrics
├── server/
│   ├── main.py                   # Central server FastAPI app (POST /update, GET /round, GET /model)
│   ├── aggregator.py             # FedAvg aggregation, round lifecycle, threading.Timer scheduler
│   ├── database.py               # SQLAlchemy models: Round, WeightUpdate
│   └── schemas.py                # Pydantic schemas: UpdateRequest, UpdateResponse, RoundResponse
├── client/
│   ├── main.py                   # Local client FastAPI app (POST /predict, POST /flag, GET /flagged, GET /dashboard, admin API)
│   ├── trainer.py                # Local training pipeline — fine-tunes on flagged emails
│   ├── database.py               # SQLAlchemy model: FlaggedEmail + query helpers (per-user, all-users, stats)
│   ├── schemas.py                # Pydantic schemas: PredictRequest/Response, FlagRequest/Response, FlaggedEmailRecord/Response
│   └── admin_dashboard.html      # Admin dashboard served at GET /dashboard (overview stats, emails, settings)
├── extention/
│   ├── manifest.json             # Chrome MV3 manifest (popup, content script, background service worker)
│   ├── popup.html                # Extension popup — status circle, flag button, manual check, dashboard link
│   ├── popup.js                  # Popup logic — restore state from storage, assess, flag, live updates
│   ├── background.js             # Service worker — handles predict/flag API calls, caches overrides
│   ├── gmail_content.js          # Content script — detects opened emails, shows phishing warning popup
│   ├── dashboard.html            # Extension dashboard — flagged emails table, settings (user ID)
│   ├── dashboard.js              # Dashboard logic — fetch flagged emails, sidebar nav, user ID persistence
│   ├── alert_popup.html          # Placeholder (unused)
│   ├── hello_extensions.png      # Extension icon
│   └── example.png               # Reference design for Gmail warning popup
├── saved_model/                  # Model checkpoints saved here after training
├── config.py                     # All hyperparameters and paths — single source of truth
├── test_client.py                # Manual test script for the client server
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
| `CLIENT_MODEL_DIR` | `<root>/client/saved_model/` | Client's local copy of the model |
| `FLAG_THRESHOLD` | 50 | Flagged emails needed to trigger local training |
| `AGGREGATION_THRESHOLD` | 1 | Weight updates needed to trigger early aggregation |
| `MIN_CLIENTS` | 1 | Minimum updates needed at deadline to aggregate (else skip round) |
| `ROUND_TIMEOUT` | 86400 | Round deadline in seconds (24 hours) |
| `CLIENT_PORT` | 8000 | Local client server port |
| `CENTRAL_PORT` | 8001 | Central server port |
| `CLIENT_ID` | `test` | Default client identifier |

---

## Label Convention

- `0` = BENIGN / LEGITIMATE
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
| `POST` | `/predict` | Run inference on an email. Body: `{body}`. Returns `{legitimate, phishing, predicted, confidence}`. |
| `POST` | `/flag` | Store a user-flagged email. Body: `{user_id, body, label}`. Triggers local training when flagged count hits `FLAG_THRESHOLD`. |
| `GET` | `/flagged` | Get flagged emails for a user. Query: `?user_id=<id>`. Returns `{emails: [...]}`. |
| `GET` | `/dashboard` | Serves the admin dashboard HTML page. |
| `GET` | `/admin/users` | Returns a list of distinct user IDs that have flagged emails. |
| `GET` | `/admin/flagged` | Get all flagged emails, optionally filtered. Query: `?user_id=<id>`. |
| `GET` | `/admin/stats` | Time-series false positive/negative counts. Query: `?user_id=<id>&start=<date>&end=<date>`. |

---

## Browser Extension

### Architecture
- **Manifest V3** Chrome extension
- **Service worker** (`background.js`) — handles all API communication with the local client server, caches flag overrides in `chrome.storage.local`
- **Content script** (`gmail_content.js`) — injected on `mail.google.com`, detects email opens via URL hash + MutationObserver, sends email text to background, shows phishing warning popup on the Gmail page
- **Popup** (`popup.html/js`) — status indicator circle (grey=idle, green=legitimate, red=phishing), flag correction button, collapsible manual email check input, dashboard link
- **Dashboard** (`dashboard.html/js`) — sidebar navigation with Emails (flagged emails table with visual override dropdown) and Settings (user ID configuration)

### Extension Permissions
- `storage` — persists last scan result, user ID, and flag override cache

---

## Admin Dashboard

Served by the client server at `GET /dashboard`. Three sections:

- **Overview** — stat cards (total flags, false positives, false negatives) + Chart.js line graph of FP/FN over time. User dropdown (all users or specific) and time range selection (presets: 7d/30d/90d/All + custom date pickers).
- **Emails** — table of all flagged emails with user, body preview, label badge, timestamp, and visual override dropdown. User dropdown filter.
- **Settings** — LLM fallback configuration (provider dropdown, API key, confidence range for triggering LLM). UI only for now, not yet functional.

---

## Implementation Roadmap

### Phase 1 — Fix & Validate Base Model ✓
- All bugs fixed in `flare.py` and `extract.py`
- Full evaluation metrics implemented (`model/evaluate.py`): accuracy, F1, precision, recall, confusion matrix
- Pre-FL baseline (test set): Loss 0.1491 | Accuracy 0.9632 | Recall 0.9535 | F1 0.9633

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

### Phase 4 — Federated Loop ✓
- After local training, weight delta (local weights − global weights) is serialized and `POST /update` to central server
- After aggregation completes, client pulls updated global model from `GET /model` and hot-reloads
- End-to-end round: flag → train → send delta → aggregate → pull → hot-reload

### Phase 5 — Browser Extension ✓
- Gmail content script auto-detects opened emails and sends to local server
- Phishing warning popup injected into Gmail page
- Extension popup with status circle (idle/legitimate/phishing), confidence display
- Manual email check via collapsible input
- Flag correction (Mark as Phishing / Mark as Legitimate) with local caching
- Extension dashboard with flagged emails and user ID settings
- Admin dashboard on client server with overview stats, email browser, and settings

### Phase 6 — Planned
- Functional LLM fallback for low-confidence predictions
- Automatic user ID assignment from client server to extensions
- Admin dashboard override actions persisted to server
- Central server admin dashboard for cross-client monitoring

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

All scripts are run from the **project root** using `python -m`:

```bash
# Full training pipeline
python -m model.train

# Evaluate saved model on test set
python -m model.evaluate

# Single inference test
python -m model.flare

# Inspect data pipeline
python -m model.extract
python -m model.dataset

# Central aggregation server (port 8001)
python -m uvicorn server.main:app --port 8001

# Local client server (port 8000)
python -m uvicorn client.main:app --port 8000

# Admin dashboard available at http://localhost:8000/dashboard

# Manual test for the client server
python test_client.py
```

---

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- DistilBERT (`distilbert-base-uncased`) — 67M parameters, 6-layer transformer
- FastAPI + Uvicorn — both central and local servers
- SQLAlchemy + SQLite — persistence for rounds, weight updates, and flagged emails
- Chart.js — admin dashboard charting (loaded from CDN)
- Chrome Extension Manifest V3 — browser integration
- CUDA supported (torch+cu126 in requirements.txt)
