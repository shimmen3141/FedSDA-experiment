"""クロス評価の診断の記録のowner: 追加の順、記録の検査、不正な入力で変わらないこと。"""

import math
from dataclasses import replace

import pytest
from test_held_candidate_validation_progress import make_subclass_copy

from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    ClientCrossEvaluationRecord,
    CrossEvaluationRecordStore,
)

VALID_RECORD_FIELDS = dict(
    round_index=3,
    client_id=1,
    candidate_model_id=2,
    target_model_id=0,
    evaluated_sample_count=7,
    bounded_loss_sum=2.5,
    squared_bounded_loss_sum=1.25,
    correctness_counts=(1, 2, 3, 1),
    class_correctness_counts=((0, 4, 1, 0, 2, 1), (2, 3, 0, 2, 1, 0)),
)


def make_cross_evaluation_record(**replaced_fields):
    return ClientCrossEvaluationRecord(**VALID_RECORD_FIELDS | replaced_fields)


def test_record_store_keeps_records_in_appended_order():
    record_store = CrossEvaluationRecordStore()
    assert record_store.snapshot_client_cross_evaluation_records() == ()
    records = (
        make_cross_evaluation_record(),
        # 正誤を比べなかった評価（評価する側と対象が同じモデル）。
        make_cross_evaluation_record(
            candidate_model_id=0, correctness_counts=None, class_correctness_counts=()
        ),
        # 件数0の評価。
        make_cross_evaluation_record(
            evaluated_sample_count=0,
            bounded_loss_sum=0.0,
            squared_bounded_loss_sum=0.0,
            correctness_counts=None,
            class_correctness_counts=(),
        ),
        # クラス別を持たない、正誤の比較。
        make_cross_evaluation_record(class_correctness_counts=()),
    )
    for record in records:
        record_store.append_client_cross_evaluation_record(client_cross_evaluation_record=record)
    snapshot = record_store.snapshot_client_cross_evaluation_records()
    assert snapshot == records
    # 写しは、その後の追加の影響を受けない。
    record_store.append_client_cross_evaluation_record(client_cross_evaluation_record=records[0])
    assert len(snapshot) == 4
    assert len(record_store.snapshot_client_cross_evaluation_records()) == 5


INVALID_RECORD_FIELD_CASES = [
    (dict(round_index=True), TypeError),
    (dict(round_index=-1), ValueError),
    (dict(client_id=1.0), TypeError),
    (dict(client_id=-1), ValueError),
    (dict(candidate_model_id="2"), TypeError),
    (dict(candidate_model_id=-101), ValueError),
    (dict(target_model_id=None), TypeError),
    (dict(target_model_id=-101), ValueError),
    (dict(evaluated_sample_count=7.0), TypeError),
    (dict(evaluated_sample_count=-1), ValueError),
    (dict(bounded_loss_sum=2), TypeError),
    (dict(bounded_loss_sum=math.nan), ValueError),
    (dict(squared_bounded_loss_sum=None), TypeError),
    (dict(squared_bounded_loss_sum=math.inf), ValueError),
    (dict(correctness_counts=[1, 2, 3, 1]), TypeError),
    (dict(correctness_counts=(1, 2, 4)), ValueError),
    (dict(correctness_counts=(1, 2, 3, True)), TypeError),
    (dict(correctness_counts=(1, 2, 3, -1)), ValueError),
    # 4つの数の和が、件数と違う。
    (dict(correctness_counts=(1, 2, 3, 2)), ValueError),
    # 正誤の比較がないのに、クラス別がある。
    (dict(correctness_counts=None), ValueError),
    (dict(class_correctness_counts=[(0, 7, 1, 2, 3, 1)]), TypeError),
    (dict(class_correctness_counts=([0, 7, 1, 2, 3, 1],)), TypeError),
    (dict(class_correctness_counts=((0, 7, 1, 2, 3),)), ValueError),
    (dict(class_correctness_counts=((0.0, 7, 1, 2, 3, 1),)), TypeError),
    (dict(class_correctness_counts=((-1, 7, 1, 2, 3, 1),)), ValueError),
    # クラスの昇順でない、重複する。
    (dict(class_correctness_counts=((2, 3, 0, 2, 1, 0), (0, 4, 1, 0, 2, 1))), ValueError),
    (dict(class_correctness_counts=((0, 4, 1, 0, 2, 1), (0, 3, 0, 2, 1, 0))), ValueError),
    # クラスの中の4つの数の和が、クラスの件数と違う。
    (dict(class_correctness_counts=((0, 4, 1, 0, 2, 0), (2, 3, 0, 2, 1, 0))), ValueError),
    # クラスの件数の和が、全体の件数と違う。
    (dict(class_correctness_counts=((0, 4, 1, 0, 2, 1),)), ValueError),
]


@pytest.mark.parametrize("invalid_fields,expected_exception", INVALID_RECORD_FIELD_CASES)
def test_record_rejects_invalid_fields(invalid_fields, expected_exception):
    with pytest.raises(expected_exception):
        make_cross_evaluation_record(**invalid_fields)


def test_record_store_rejects_invalid_records_without_change():
    record_store = CrossEvaluationRecordStore()
    valid_record = make_cross_evaluation_record()
    record_store.append_client_cross_evaluation_record(client_cross_evaluation_record=valid_record)
    # frozenを回避して書き換えた記録も、保存の前に拒否する。
    mutated_record = replace(valid_record)
    object.__setattr__(mutated_record, "evaluated_sample_count", 8)
    for invalid_record in (
        None,
        VALID_RECORD_FIELDS,
        make_subclass_copy(valid_record),
        mutated_record,
    ):
        with pytest.raises((TypeError, ValueError)):
            record_store.append_client_cross_evaluation_record(
                client_cross_evaluation_record=invalid_record
            )
        assert record_store.snapshot_client_cross_evaluation_records() == (valid_record,)
