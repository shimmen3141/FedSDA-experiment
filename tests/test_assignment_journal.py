import torch
import pytest

from federated_drift_experiment.clients.assignment_journal import (
    RecentAssignmentJournal,
)


def _add_stat(stats, value):
    stats["n"] += 1
    delta = value - stats["mean"]
    stats["mean"] += delta / stats["n"]
    stats["M2"] += delta * (value - stats["mean"])


def test_journal_selects_only_entries_at_or_after_estimated_start():
    journal = RecentAssignmentJournal(capacity=100)
    data = [(torch.tensor([[float(i)]]), torch.zeros((1, 1))) for i in range(3)]
    for sample_idx, item in enumerate(data):
        journal.record(sample_idx, item, loss=sample_idx / 10, class_id=0)

    assert journal.preview_since(1) == data[1:]
    assert journal.count_since(1) == 2


def test_journal_reassignment_removes_store_data_and_welford_stats():
    journal = RecentAssignmentJournal(capacity=100)
    data = [(torch.tensor([[float(i)]]), torch.zeros((1, 1))) for i in range(3)]
    losses = [0.1, 0.2, 0.3]
    train_data_store = {0: list(data)}
    model_stats = {
        0: {
            "n": 0,
            "mean": 0.0,
            "M2": 0.0,
            "class_stats": {0: {"n": 0, "mean": 0.0, "M2": 0.0}},
        }
    }
    for sample_idx, (item, loss) in enumerate(zip(data, losses)):
        journal.record(sample_idx, item, loss=loss, class_id=0)
        _add_stat(model_stats[0], loss)
        _add_stat(model_stats[0]["class_stats"][0], loss)

    reassigned = journal.reassign_since(1, train_data_store, model_stats)

    assert reassigned == data[1:]
    assert train_data_store[0] == data[:1]
    assert model_stats[0]["n"] == 1
    assert model_stats[0]["mean"] == pytest.approx(losses[0])
    assert model_stats[0]["class_stats"][0]["n"] == 1


def test_zero_capacity_disables_journal():
    journal = RecentAssignmentJournal(capacity=0)
    item = (torch.zeros((1, 1)), torch.zeros((1, 1)))
    journal.record(0, item, loss=0.1, class_id=0)

    assert journal.preview_since(0) == []
    assert journal.count_since(0) == 0
