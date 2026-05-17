import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def temp_client_db(monkeypatch):
    """Replace the client database engine/SessionLocal with a fresh in-memory SQLite."""
    from client import database

    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_session = sessionmaker(bind=test_engine)

    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(database, "SessionLocal", test_session)

    database.Base.metadata.create_all(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def fake_flare(monkeypatch):
    """Stub the Flare detector so tests never load the real DistilBERT model."""
    class FakeFlare:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, body: str):
            if "phish" in body.lower() or "click here" in body.lower():
                return {"legitimate": 0.1, "phishing": 0.9, "predicted": "phishing", "confidence": 0.9}
            return {"legitimate": 0.85, "phishing": 0.15, "predicted": "legitimate", "confidence": 0.85}

    monkeypatch.setattr("client.main.Flare", FakeFlare)
    return FakeFlare


@pytest.fixture
def api_client(temp_client_db, fake_flare, monkeypatch):
    """FastAPI TestClient with patched DB, stubbed detector, and no-op training trigger."""
    from fastapi.testclient import TestClient
    import client.main

    # Prevent the real federated training pipeline from ever firing in tests
    monkeypatch.setattr(client.main, "run_local_training", lambda *a, **k: None)

    with TestClient(client.main.app) as c:
        yield c
