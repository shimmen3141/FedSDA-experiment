"""指標（精度、Kappa系、変化点の後の精度、概念の同定）と系列の入力の検査を確かめる（riverを使わない）。"""

import json
import math

import numpy as np
import pytest

from single_stream_hard_assignment_soft_prediction.single_stream_input import (
    compute_stream_array_digest,
    load_single_stream_input,
)
from single_stream_hard_assignment_soft_prediction.single_stream_metrics import (
    classify_concept_identification,
    compute_prequential_metrics,
)


def test_accuracy_and_kappa_statistics_match_hand_computation():
    observed = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    predicted = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    concepts = np.zeros(8, dtype=int)
    m = compute_prequential_metrics(
        predicted_class_ids=predicted,
        observed_class_ids=observed,
        concept_ids=concepts,
        class_count=2,
    )
    p0 = 5 / 8
    # 周辺分布: 予測 0:4 1:4、観測 0:3 1:5。
    pc = (4 / 8) * (3 / 8) + (4 / 8) * (5 / 8)
    assert m["prequential_accuracy"] == pytest.approx(p0)
    assert m["kappa"] == pytest.approx((p0 - pc) / (1 - pc))
    # それまでの最多のクラス（無し・同点は小さい番号）: 観測の累計 {}→0, {0:1}→0, {0:1,1:1}→0（同点）, {0:1,1:2}→1,
    # {0:2,1:2}→0（同点）, {0:2,1:3}→1, {0:2,1:4}→1, {0:2,1:5}→1。
    majority_predictions = [0, 0, 0, 1, 0, 1, 1, 1]
    pm = sum(int(a == b) for a, b in zip(majority_predictions, observed)) / 8
    assert m["kappa_m"] == pytest.approx((p0 - pm) / (1 - pm))
    # 直前のラベル（最初はクラス0）: 0,0,1,1,0,1,1,1。
    no_change = [0, 0, 1, 1, 0, 1, 1, 1]
    pt = sum(int(a == b) for a, b in zip(no_change, observed)) / 8
    assert m["kappa_temporal"] == pytest.approx((p0 - pt) / (1 - pt))


def test_kappa_is_nan_when_the_denominator_is_zero():
    observed = np.zeros(5, dtype=int)
    m = compute_prequential_metrics(
        predicted_class_ids=observed.copy(),
        observed_class_ids=observed,
        concept_ids=np.zeros(5, dtype=int),
        class_count=2,
    )
    assert math.isnan(m["kappa"]) and math.isnan(m["kappa_m"]) and math.isnan(m["kappa_temporal"])


def test_accuracy_after_true_changes_pools_samples_over_change_points():
    concepts = np.array([0] * 10 + [1] * 4 + [0] * 40)
    correct = np.zeros(54, dtype=bool)
    correct[10:12] = True  # 1つ目の変化点（10）の後の区間（4件）で2件正解
    correct[14 : 14 + 20] = True  # 2つ目の変化点（14）の後の最初の20件が正解
    observed = np.zeros(54, dtype=int)
    predicted = np.where(correct, 0, 1)
    m = compute_prequential_metrics(
        predicted_class_ids=predicted,
        observed_class_ids=observed,
        concept_ids=concepts,
        class_count=2,
    )
    # 最初の30件: 1つ目は区間の全部4件（2正解）、2つ目は30件（20正解）→ 22/34。
    assert m["accuracy_first_30_after_change"] == pytest.approx(22 / 34)
    assert m["accuracy_first_100_after_change"] == pytest.approx((2 + 20) / (4 + 40))
    assert m["true_change_count"] == 2


def test_concept_identification_classification():
    # 標本0〜9はコンセプト0でモデル0、10〜19はコンセプト1でモデル1、20〜29はコンセプト0でモデル0、
    # 30〜39はコンセプト1でモデル2（再来なのに新規作成）。
    concepts = np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10)
    assigned = np.array([0] * 10 + [1] * 10 + [0] * 10 + [2] * 10)
    alarms = [
        dict(
            sample_index=12,
            adaptation_outcome="new",
            destination_model_id=1,
            tail_concept_ids=(1, 1, 1),
        ),
        dict(
            sample_index=22,
            adaptation_outcome="reuse",
            destination_model_id=0,
            tail_concept_ids=(0, 0, 0),
        ),
        dict(
            sample_index=32,
            adaptation_outcome="new",
            destination_model_id=2,
            tail_concept_ids=(1, 1, 1),
        ),
        dict(
            sample_index=35,
            adaptation_outcome="reuse",
            destination_model_id=0,
            tail_concept_ids=(1, 1, 0),
        ),
        dict(
            sample_index=36,
            adaptation_outcome="stay",
            destination_model_id=2,
            tail_concept_ids=(1,),
        ),
    ]
    created = {0: -1, 1: 12, 2: 32}
    counts = classify_concept_identification(
        alarms=alarms,
        assigned_model_ids=assigned,
        concept_ids=concepts,
        model_created_sample_index_by_model_id=created,
    )
    assert counts == dict(
        correct_reuse=1,
        incorrect_reuse=1,
        new_model_for_recurring_concept=1,
        new_model_for_first_seen_concept=1,
        stay=1,
    )


def write_stream(directory, *, labels=None, dataset_name="sine2", digest=None, **extra_metadata):
    arrays = dict(
        features=np.arange(12, dtype=np.float32).reshape(6, 2),
        labels=np.array([0, 1, 0, 1, 1, 0] if labels is None else labels),
        concept_ids=np.array([0, 0, 0, 1, 1, 1]),
        pretraining_features=np.zeros((2, 2), dtype=np.float32),
        pretraining_labels=np.array([0, 1]),
    )
    path = directory / "stream.npz"
    np.savez_compressed(path, **arrays)
    metadata = dict(
        dataset_name=dataset_name,
        random_seed=3,
        array_sha256=digest or compute_stream_array_digest(arrays),
        **extra_metadata,
    )
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return path, arrays


def test_stream_input_is_loaded_and_checked(tmp_path):
    path, arrays = write_stream(tmp_path)
    stream = load_single_stream_input(stream_path=path)
    assert stream.class_count == 2
    assert stream.dataset_name == "sine2" and stream.random_seed == 3
    np.testing.assert_array_equal(stream.labels, arrays["labels"])


@pytest.mark.parametrize(
    "kwargs",
    [dict(labels=[0, 1, 0, 2, 1, 0]), dict(dataset_name="unknown_dataset"), dict(digest="0" * 64)],
)
def test_invalid_stream_input_is_rejected(tmp_path, kwargs):
    path, _ = write_stream(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        load_single_stream_input(stream_path=path)


def test_digest_matches_the_stream_export_tool():
    pytest.importorskip("torch")
    from tools.research_single_client_stream_baselines.stream_export import array_digest

    arrays = dict(features=np.ones((3, 2), dtype=np.float32), labels=np.array([0, 1, 1]))
    assert compute_stream_array_digest(arrays) == array_digest(arrays)


def test_class_count_in_the_metadata_is_used_for_a_dataset_outside_the_product(tmp_path):
    path, _ = write_stream(
        tmp_path, dataset_name="stagger3", labels=[0, 1, 2, 1, 1, 0], class_count=3
    )
    stream = load_single_stream_input(stream_path=path)
    assert stream.class_count == 3 and stream.dataset_name == "stagger3"


def test_class_count_in_the_metadata_must_match_the_product_definition(tmp_path):
    path, _ = write_stream(tmp_path, dataset_name="sine2", class_count=2)
    assert load_single_stream_input(stream_path=path).class_count == 2
    path, _ = write_stream(tmp_path, dataset_name="sine2", class_count=3)
    with pytest.raises(ValueError, match="class_count"):
        load_single_stream_input(stream_path=path)


@pytest.mark.parametrize("class_count", [1, 0, True, "2", 2.0, None])
def test_invalid_class_count_in_the_metadata_is_rejected(tmp_path, class_count):
    path, _ = write_stream(tmp_path, dataset_name="stagger3", class_count=class_count)
    with pytest.raises(ValueError, match="class_count"):
        load_single_stream_input(stream_path=path)


def test_labels_outside_the_metadata_class_count_are_rejected(tmp_path):
    path, _ = write_stream(
        tmp_path, dataset_name="stagger3", labels=[0, 1, 2, 1, 1, 0], class_count=2
    )
    with pytest.raises(ValueError, match="ラベル"):
        load_single_stream_input(stream_path=path)
