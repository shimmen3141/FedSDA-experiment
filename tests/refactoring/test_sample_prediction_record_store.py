"""標本ごとの予測の記録のowner: 観測順の保持、写し、不正な記録の拒否。"""

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from federated_learning_experiments.evaluation.sample_prediction_record_store import (
    SamplePredictionRecord,
    SamplePredictionRecordStore,
)


def make_sample_prediction_record(**replaced_fields):
    """正しい記録（位置5、保有モデル2つで重みが0.75と0.25のとき）。指定したfieldだけ置き換える。"""
    return replace(
        SamplePredictionRecord(
            sample_index=5,
            observed_concept_id=2,
            observed_class_id=1,
            combined_prediction_is_correct=True,
            maximum_weight_model_id=-3,
            maximum_prediction_weight=0.75,
            effective_model_count=1.6,
            any_model_or_combined_prediction_is_correct=True,
            maximum_weight_model_prediction_is_correct=False,
            global_diagnostic_prediction_is_correct=False,
            true_concept_diagnostic_prediction_is_correct=True,
            highest_confidence_model_prediction_is_correct=False,
        ),
        **replaced_fields,
    )


class RecordSubclass(SamplePredictionRecord):
    pass


def test_record_store_keeps_records_in_observation_order():
    sample_prediction_record_store = SamplePredictionRecordStore()
    assert sample_prediction_record_store.snapshot_sample_prediction_records() == ()
    assert sample_prediction_record_store.last_recorded_sample_index is None
    first_record = make_sample_prediction_record()
    # 概念IDのない標本は、概念別の診断の項目もなし。保有モデルが1つなら、重み1・実効モデル数1。
    second_record = make_sample_prediction_record(
        sample_index=6,
        observed_concept_id=None,
        true_concept_diagnostic_prediction_is_correct=None,
        maximum_prediction_weight=1.0,
        effective_model_count=1.0,
    )
    sample_prediction_record_store.append_sample_prediction_record(
        sample_prediction_record=first_record
    )
    recorded_snapshot = sample_prediction_record_store.snapshot_sample_prediction_records()
    assert sample_prediction_record_store.last_recorded_sample_index == 5
    sample_prediction_record_store.append_sample_prediction_record(
        sample_prediction_record=second_record
    )
    assert sample_prediction_record_store.last_recorded_sample_index == 6
    latest_snapshot = sample_prediction_record_store.snapshot_sample_prediction_records()
    assert type(latest_snapshot) is tuple
    assert len(latest_snapshot) == 2
    assert latest_snapshot[0] is first_record
    assert latest_snapshot[1] is second_record
    # 写しは、その後の追加の影響を受けない。記録は後から変えられない。
    assert recorded_snapshot == (first_record,)
    with pytest.raises(FrozenInstanceError):
        first_record.sample_index = 9


def test_record_store_accepts_any_first_sample_index_and_rounding_of_effective_model_count():
    sample_prediction_record_store = SamplePredictionRecordStore()
    # 最初の記録の位置は0でなくてよい。実効モデル数は、丸めで1をわずかに下回ることがある。
    sample_prediction_record_store.append_sample_prediction_record(
        sample_prediction_record=make_sample_prediction_record(
            sample_index=0, observed_class_id=0, effective_model_count=1.0 - 1e-12
        )
    )
    assert sample_prediction_record_store.last_recorded_sample_index == 0


@pytest.mark.parametrize(
    "invalid_fields,expected_exception",
    [
        (dict(sample_index=True), TypeError),
        (dict(sample_index=6.0), TypeError),
        (dict(sample_index=np.int64(6)), TypeError),
        (dict(observed_concept_id=True), TypeError),
        (dict(observed_concept_id=2.0), TypeError),
        (dict(observed_class_id=None), TypeError),
        (dict(observed_class_id=True), TypeError),
        (dict(maximum_weight_model_id=False), TypeError),
        (dict(maximum_weight_model_id="4"), TypeError),
        (dict(combined_prediction_is_correct=1), TypeError),
        (dict(combined_prediction_is_correct=1.0), TypeError),
        (dict(any_model_or_combined_prediction_is_correct=None), TypeError),
        (dict(maximum_weight_model_prediction_is_correct=0), TypeError),
        (dict(global_diagnostic_prediction_is_correct=np.bool_(True)), TypeError),
        (dict(true_concept_diagnostic_prediction_is_correct=1), TypeError),
        (dict(highest_confidence_model_prediction_is_correct="True"), TypeError),
        (dict(maximum_prediction_weight=1), TypeError),
        (dict(maximum_prediction_weight=np.float64(0.5)), TypeError),
        (dict(effective_model_count=2), TypeError),
        (dict(effective_model_count=None), TypeError),
        (dict(observed_class_id=-1), ValueError),
        (dict(maximum_prediction_weight=0.0), ValueError),
        (dict(maximum_prediction_weight=1.0000001), ValueError),
        (dict(maximum_prediction_weight=math.nan), ValueError),
        (dict(effective_model_count=0.999), ValueError),
        (dict(effective_model_count=math.inf), ValueError),
        (dict(effective_model_count=math.nan), ValueError),
        # 概念IDと、概念別の診断の項目の不対応。
        (dict(observed_concept_id=None), ValueError),
        (dict(true_concept_diagnostic_prediction_is_correct=None), ValueError),
        # 位置が、最後の記録（5）の次でない。
        (dict(sample_index=5), ValueError),
        (dict(sample_index=4), ValueError),
        (dict(sample_index=7), ValueError),
    ],
)
def test_record_store_rejects_invalid_record_without_change(invalid_fields, expected_exception):
    sample_prediction_record_store = SamplePredictionRecordStore()
    sample_prediction_record_store.append_sample_prediction_record(
        sample_prediction_record=make_sample_prediction_record()
    )
    recorded_snapshot = sample_prediction_record_store.snapshot_sample_prediction_records()
    with pytest.raises(expected_exception):
        sample_prediction_record_store.append_sample_prediction_record(
            sample_prediction_record=make_sample_prediction_record(
                **dict(sample_index=6) | invalid_fields
            )
        )
    assert sample_prediction_record_store.snapshot_sample_prediction_records() == recorded_snapshot
    assert sample_prediction_record_store.last_recorded_sample_index == 5
    # 拒否の後に、正しい記録を足せる。
    sample_prediction_record_store.append_sample_prediction_record(
        sample_prediction_record=make_sample_prediction_record(sample_index=6)
    )
    assert sample_prediction_record_store.last_recorded_sample_index == 6


def test_record_store_rejects_negative_first_sample_index():
    sample_prediction_record_store = SamplePredictionRecordStore()
    with pytest.raises(ValueError):
        sample_prediction_record_store.append_sample_prediction_record(
            sample_prediction_record=make_sample_prediction_record(sample_index=-1)
        )
    assert sample_prediction_record_store.snapshot_sample_prediction_records() == ()


@pytest.mark.parametrize(
    "invalid_record",
    [
        None,
        dict(sample_index=5),
        RecordSubclass(**vars(make_sample_prediction_record())),
    ],
)
def test_record_store_rejects_non_record(invalid_record):
    sample_prediction_record_store = SamplePredictionRecordStore()
    with pytest.raises(TypeError):
        sample_prediction_record_store.append_sample_prediction_record(
            sample_prediction_record=invalid_record
        )
    assert sample_prediction_record_store.snapshot_sample_prediction_records() == ()
