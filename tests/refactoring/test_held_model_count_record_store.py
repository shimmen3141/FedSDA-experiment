"""標本ごとの保有モデル数のowner: 足した順の保持、読取りの不変、拒否。"""

import pytest

from federated_learning_experiments.evaluation.held_model_count_record_store import (
    HeldModelCountRecordStore,
)


def test_store_keeps_held_model_counts_in_append_order():
    count_store = HeldModelCountRecordStore()
    assert count_store.snapshot_held_model_counts() == ()
    for held_model_count in (1, 3, 2, 2):
        assert count_store.append_held_model_count(held_model_count=held_model_count) is None
    snapshot = count_store.snapshot_held_model_counts()
    assert snapshot == (1, 3, 2, 2)
    assert type(snapshot) is tuple
    # 読取りは、後からの追加で変わらない。別のownerとは、列を共有しない。
    count_store.append_held_model_count(held_model_count=5)
    assert snapshot == (1, 3, 2, 2)
    assert count_store.snapshot_held_model_counts() == (1, 3, 2, 2, 5)
    assert HeldModelCountRecordStore().snapshot_held_model_counts() == ()


@pytest.mark.parametrize(
    ("invalid_count", "expected_exception_type"),
    [
        (True, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_store_rejects_invalid_counts_without_changing_state(
    invalid_count, expected_exception_type
):
    count_store = HeldModelCountRecordStore()
    count_store.append_held_model_count(held_model_count=2)
    with pytest.raises(expected_exception_type):
        count_store.append_held_model_count(held_model_count=invalid_count)
    assert count_store.snapshot_held_model_counts() == (2,)


def test_append_requires_keyword_argument():
    count_store = HeldModelCountRecordStore()
    with pytest.raises(TypeError):
        count_store.append_held_model_count(2)
    assert count_store.snapshot_held_model_counts() == ()
