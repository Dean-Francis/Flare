import pandas as pd
import pytest


@pytest.fixture
def stub_csv(monkeypatch):
    """Patch pd.read_csv inside model.extract so Extract() uses synthetic data."""
    def _install(df: pd.DataFrame):
        monkeypatch.setattr("model.extract.pd.read_csv", lambda *a, **k: df.copy())
    return _install


def _balanced_df(n_per_class: int) -> pd.DataFrame:
    bodies = [f"benign_{i}" for i in range(n_per_class)] + [f"phish_{i}" for i in range(n_per_class)]
    labels = [0] * n_per_class + [1] * n_per_class
    return pd.DataFrame({"body": bodies, "label": labels})


def test_clean_drops_null_bodies(stub_csv):
    df = pd.DataFrame({"body": ["a", None, "b"], "label": [0, 1, 1]})
    stub_csv(df)

    from model.extract import Extract
    cleaned = Extract().get_clean_data()
    assert list(cleaned["body"]) == ["a", "b"]


def test_clean_drops_duplicate_bodies(stub_csv):
    df = pd.DataFrame({"body": ["a", "a", "b"], "label": [0, 0, 1]})
    stub_csv(df)

    from model.extract import Extract
    cleaned = Extract().get_clean_data()
    assert sorted(cleaned["body"]) == ["a", "b"]


def test_clean_coerces_label_to_int(stub_csv):
    df = pd.DataFrame({"body": ["a", "b"], "label": ["0", "1"]})
    stub_csv(df)

    from model.extract import Extract
    cleaned = Extract().get_clean_data()
    assert cleaned["label"].dtype.kind == "i"
    assert list(cleaned["label"]) == [0, 1]


def test_get_splits_returns_80_10_10(stub_csv):
    df = _balanced_df(n_per_class=50)  # 100 rows total
    stub_csv(df)

    from model.extract import Extract
    train_df, val_df, test_df = Extract().get_splits()

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == 100
    assert len(train_df) == 80
    assert len(val_df) == 10
    assert len(test_df) == 10


def test_splits_preserve_class_stratification(stub_csv):
    df = _balanced_df(n_per_class=50)
    stub_csv(df)

    from model.extract import Extract
    train_df, val_df, test_df = Extract().get_splits()

    # Each split should keep the original 50/50 class ratio
    for split in (train_df, val_df, test_df):
        counts = split["label"].value_counts()
        assert counts.get(0, 0) == counts.get(1, 0)


def test_splits_are_disjoint(stub_csv):
    df = _balanced_df(n_per_class=50)
    stub_csv(df)

    from model.extract import Extract
    train_df, val_df, test_df = Extract().get_splits()

    train_bodies = set(train_df["body"])
    val_bodies   = set(val_df["body"])
    test_bodies  = set(test_df["body"])
    assert train_bodies.isdisjoint(val_bodies)
    assert train_bodies.isdisjoint(test_bodies)
    assert val_bodies.isdisjoint(test_bodies)
