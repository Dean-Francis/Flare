from client.database import (
    insert_flagged_email,
    count_flagged_emails,
    get_untrained_emails,
    get_flagged_emails_by_user,
    get_all_flagged_emails,
    get_distinct_user_ids,
    get_flag_stats,
    mark_emails_trained,
)


def test_insert_and_get_by_user(temp_client_db):
    insert_flagged_email("alice", "hello", label=0, confidence=0.72)
    insert_flagged_email("bob",   "spam", label=1, confidence=0.91)
    insert_flagged_email("alice", "more", label=1, confidence=0.55)

    alice = get_flagged_emails_by_user("alice")
    assert len(alice) == 2
    assert {e.body for e in alice} == {"hello", "more"}
    assert all(e.user_id == "alice" for e in alice)


def test_confidence_round_trips_and_is_optional(temp_client_db):
    insert_flagged_email("alice", "with conf", label=1, confidence=0.83)
    insert_flagged_email("alice", "without",   label=0)  # confidence omitted

    rows = get_flagged_emails_by_user("alice")
    by_body = {r.body: r for r in rows}
    assert by_body["with conf"].confidence == 0.83
    assert by_body["without"].confidence is None


def test_get_all_flagged_with_and_without_filter(temp_client_db):
    insert_flagged_email("alice", "a", label=0)
    insert_flagged_email("bob",   "b", label=1)
    insert_flagged_email("carol", "c", label=0)

    assert len(get_all_flagged_emails()) == 3
    assert len(get_all_flagged_emails(user_id="bob")) == 1
    assert get_all_flagged_emails(user_id="nobody") == []


def test_get_distinct_user_ids(temp_client_db):
    insert_flagged_email("alice", "x", label=0)
    insert_flagged_email("alice", "y", label=1)
    insert_flagged_email("bob",   "z", label=0)

    users = set(get_distinct_user_ids())
    assert users == {"alice", "bob"}


def test_count_only_includes_untrained(temp_client_db):
    insert_flagged_email("alice", "1", label=0)
    insert_flagged_email("alice", "2", label=1)
    assert count_flagged_emails() == 2

    untrained = get_untrained_emails()
    mark_emails_trained([untrained[0].id])
    assert count_flagged_emails() == 1


def test_mark_trained_excludes_from_untrained_query(temp_client_db):
    insert_flagged_email("alice", "1", label=0)
    insert_flagged_email("alice", "2", label=1)
    rows = get_untrained_emails()
    assert len(rows) == 2

    mark_emails_trained([rows[0].id, rows[1].id])
    assert get_untrained_emails() == []


def test_flag_stats_groups_fp_and_fn_per_date(temp_client_db):
    # label=0 → false positive (model said phishing, user said legitimate)
    # label=1 → false negative (model said legitimate, user said phishing)
    insert_flagged_email("alice", "a", label=0)
    insert_flagged_email("alice", "b", label=1)
    insert_flagged_email("alice", "c", label=1)

    stats = get_flag_stats()
    assert len(stats) == 1
    day = stats[0]
    assert day["false_positives"] == 1
    assert day["false_negatives"] == 2


def test_flag_stats_filters_by_user(temp_client_db):
    insert_flagged_email("alice", "a", label=0)
    insert_flagged_email("bob",   "b", label=1)

    assert sum(d["false_positives"] for d in get_flag_stats(user_id="alice")) == 1
    assert sum(d["false_negatives"] for d in get_flag_stats(user_id="alice")) == 0
    assert sum(d["false_negatives"] for d in get_flag_stats(user_id="bob")) == 1
