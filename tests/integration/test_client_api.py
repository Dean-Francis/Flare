import pytest


def test_predict_returns_expected_shape(api_client):
    resp = api_client.post("/predict", json={"body": "click here to claim your prize"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"legitimate", "phishing", "predicted", "confidence"}
    assert body["predicted"] in {"legitimate", "phishing"}


def test_flag_persists_and_appears_in_flagged_with_confidence(api_client):
    resp = api_client.post("/flag", json={
        "user_id": "alice", "body": "manual flag", "label": 1, "confidence": 0.87,
    })
    assert resp.status_code == 200
    assert resp.json()["count"] == 1

    listing = api_client.get("/flagged", params={"user_id": "alice"}).json()
    assert len(listing["emails"]) == 1
    assert listing["emails"][0]["confidence"] == 0.87
    assert listing["emails"][0]["label"] == 1


def test_flag_without_confidence_stores_null(api_client):
    api_client.post("/flag", json={"user_id": "alice", "body": "no conf", "label": 0})
    rows = api_client.get("/flagged", params={"user_id": "alice"}).json()["emails"]
    assert rows[0]["confidence"] is None


def test_flagged_is_scoped_per_user(api_client):
    api_client.post("/flag", json={"user_id": "alice", "body": "a", "label": 0})
    api_client.post("/flag", json={"user_id": "bob",   "body": "b", "label": 1})

    alice = api_client.get("/flagged", params={"user_id": "alice"}).json()["emails"]
    bob   = api_client.get("/flagged", params={"user_id": "bob"}).json()["emails"]
    assert [e["body"] for e in alice] == ["a"]
    assert [e["body"] for e in bob]   == ["b"]


def test_admin_users_returns_distinct(api_client):
    for uid in ("alice", "alice", "bob", "carol"):
        api_client.post("/flag", json={"user_id": uid, "body": f"body-{uid}", "label": 0})

    users = api_client.get("/admin/users").json()
    assert set(users) == {"alice", "bob", "carol"}


def test_admin_flagged_returns_all_or_filtered(api_client):
    api_client.post("/flag", json={"user_id": "alice", "body": "a", "label": 0, "confidence": 0.6})
    api_client.post("/flag", json={"user_id": "bob",   "body": "b", "label": 1, "confidence": 0.7})

    all_rows = api_client.get("/admin/flagged").json()
    assert len(all_rows) == 2
    assert {r["user_id"] for r in all_rows} == {"alice", "bob"}

    filtered = api_client.get("/admin/flagged", params={"user_id": "bob"}).json()
    assert len(filtered) == 1
    assert filtered[0]["confidence"] == 0.7


def test_admin_stats_counts_fp_and_fn(api_client):
    api_client.post("/flag", json={"user_id": "alice", "body": "a", "label": 0})  # FP
    api_client.post("/flag", json={"user_id": "alice", "body": "b", "label": 1})  # FN
    api_client.post("/flag", json={"user_id": "alice", "body": "c", "label": 1})  # FN

    stats = api_client.get("/admin/stats").json()
    assert sum(d["false_positives"] for d in stats) == 1
    assert sum(d["false_negatives"] for d in stats) == 2


def test_flag_triggers_training_at_threshold(api_client, monkeypatch):
    import client.main
    # Lower the threshold so the test doesn't need 50 flags
    monkeypatch.setattr(client.main, "FLAG_THRESHOLD", 2)

    first  = api_client.post("/flag", json={"user_id": "alice", "body": "1", "label": 0}).json()
    second = api_client.post("/flag", json={"user_id": "alice", "body": "2", "label": 1}).json()

    assert first["training_triggered"] is False
    assert second["training_triggered"] is True


def test_dashboard_endpoint_serves_html(api_client):
    resp = api_client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "FLARE" in resp.text or "Flare" in resp.text


@pytest.mark.parametrize("payload", [
    {"body": "only body"},                                  # missing user_id, label
    {"user_id": "alice", "body": "x"},                       # missing label
    {"user_id": "alice", "label": 1},                        # missing body
])
def test_flag_rejects_invalid_payload(api_client, payload):
    resp = api_client.post("/flag", json=payload)
    assert resp.status_code == 422
