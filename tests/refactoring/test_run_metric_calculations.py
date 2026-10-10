"""指標の計算の部品: 実旧の関数（変更位置の抽出、対応づけ、定常精度、指標の計算）との対照と、拒否。"""

import random
from dataclasses import FrozenInstanceError
from math import isnan
from types import SimpleNamespace

import pytest

from federated_drift_experiment.data.schedules import extract_true_drift_events
from federated_drift_experiment.metrics import _stable_accuracy, compute_metrics, match_events
from federated_learning_experiments.evaluation.run_metric_calculations import (
    DetectionMetrics,
    EventMatchCounts,
    RunMetricSettings,
    calculate_detection_metrics,
    calculate_prediction_accuracy,
    calculate_stable_period_prediction_accuracy,
    count_events_matched_to_concept_changes,
    extract_concept_change_sample_indices,
)


def make_random_concept_ids(*, random_generator, sample_count, change_probability):
    concept_ids = []
    concept_id = 0
    for _ in range(sample_count):
        if concept_ids and random_generator.random() < change_probability:
            concept_id = 1 - concept_id
        concept_ids.append(concept_id)
    return tuple(concept_ids)


CONCEPT_ID_CASES = [
    (),
    (0,),
    (1,),
    (0, 0, 0),
    (0, 1),
    (1, 0, 1, 0),
    (0, 0, 1, 1, 1, 0),
    *[
        make_random_concept_ids(
            random_generator=random.Random(case_seed),
            sample_count=sample_count,
            change_probability=change_probability,
        )
        for case_seed, (sample_count, change_probability) in enumerate(
            [(40, 0.1), (200, 0.02), (200, 0.5), (300, 0.0), (64, 1.0)]
        )
    ],
]


@pytest.mark.parametrize("concept_ids", CONCEPT_ID_CASES)
def test_concept_change_indices_match_real_legacy_true_drift_events(concept_ids):
    """概念が変わった標本位置が、実旧の真のドリフト位置の抽出と一致する。"""
    concept_change_sample_indices = extract_concept_change_sample_indices(
        concept_ids_by_sample_index=concept_ids
    )
    assert type(concept_change_sample_indices) is tuple
    assert list(concept_change_sample_indices) == extract_true_drift_events([list(concept_ids)])[0]
    assert all(type(sample_index) is int for sample_index in concept_change_sample_indices)


def make_random_positions(*, random_generator, position_count, maximum_position, sort_positions):
    positions = [random_generator.randrange(maximum_position + 1) for _ in range(position_count)]
    return tuple(sorted(positions) if sort_positions else positions)


# (変更位置, イベントの位置, 許容遅延)。境界: 空、許容遅延0、同じ位置、範囲の両端、順が昇順でない列。
EVENT_MATCHING_CASES = [
    ((), (), 5),
    ((10,), (), 5),
    ((), (10,), 5),
    ((10,), (10,), 0),
    ((10,), (9,), 0),
    ((10,), (11,), 0),
    ((10,), (15,), 5),
    ((10,), (16,), 5),
    ((10, 20), (12, 12), 5),
    ((10, 12), (13,), 5),
    ((10, 12), (13, 13), 5),
    ((10, 30), (31, 11), 5),
    ((30, 10), (11, 31), 5),
    ((10, 10), (10, 10, 10), 0),
    *[
        (
            make_random_positions(
                random_generator=random.Random(100 + case_seed),
                position_count=change_count,
                maximum_position=120,
                sort_positions=sort_positions,
            ),
            make_random_positions(
                random_generator=random.Random(200 + case_seed),
                position_count=event_count,
                maximum_position=140,
                sort_positions=sort_positions,
            ),
            maximum_delay,
        )
        for case_seed, (change_count, event_count, maximum_delay, sort_positions) in enumerate(
            [
                (5, 5, 10, True),
                (8, 20, 3, True),
                (20, 8, 30, True),
                (12, 12, 0, True),
                (6, 30, 200, True),
                (10, 10, 10, False),
                (15, 25, 4, False),
            ]
        )
    ],
]


@pytest.mark.parametrize(
    ("concept_change_sample_indices", "event_sample_indices", "maximum_delay"),
    EVENT_MATCHING_CASES,
)
def test_event_match_counts_match_real_legacy_greedy_matching(
    concept_change_sample_indices, event_sample_indices, maximum_delay
):
    """対応したイベントの数と、対応した変更位置の数が、実旧の対応づけと一致する。"""
    event_match_counts = count_events_matched_to_concept_changes(
        concept_change_sample_indices=concept_change_sample_indices,
        event_sample_indices=event_sample_indices,
        maximum_delay_sample_count=maximum_delay,
    )
    legacy_used_events, legacy_matched_changes, _ = match_events(
        list(concept_change_sample_indices), list(event_sample_indices), maximum_delay
    )
    assert type(event_match_counts) is EventMatchCounts
    assert event_match_counts == EventMatchCounts(
        matched_event_count=len(legacy_used_events),
        matched_concept_change_count=len(legacy_matched_changes),
    )
    assert type(event_match_counts.matched_event_count) is int


def test_event_matching_cases_cover_matches_and_misses():
    """上の対照の入力が、対応あり・対応なし・一部だけ対応、のすべてを含む。"""
    matched_fractions = set()
    for concept_change_sample_indices, event_sample_indices, maximum_delay in EVENT_MATCHING_CASES:
        used_events, _, _ = match_events(
            list(concept_change_sample_indices), list(event_sample_indices), maximum_delay
        )
        if not event_sample_indices:
            continue
        matched_fractions.add(
            "all"
            if len(used_events) == len(event_sample_indices)
            else ("none" if not used_events else "some")
        )
    assert matched_fractions == {"all", "none", "some"}


def make_run_case(*, case_seed, client_count, sample_count, change_probability, switch_count):
    """clientごとの、正誤の列・概念列・検出位置（昇順でない）を、決まったseedで作る。"""
    random_generator = random.Random(case_seed)
    prediction_correctness_by_client = tuple(
        tuple(random_generator.random() < 0.7 for _ in range(sample_count))
        for _ in range(client_count)
    )
    concept_ids_by_client = tuple(
        make_random_concept_ids(
            random_generator=random_generator,
            sample_count=sample_count,
            change_probability=change_probability,
        )
        for _ in range(client_count)
    )
    detection_sample_indices_by_client = tuple(
        tuple(random_generator.randrange(max(1, sample_count)) for _ in range(switch_count))
        for _ in range(client_count)
    )
    return (
        prediction_correctness_by_client,
        concept_ids_by_client,
        detection_sample_indices_by_client,
    )


RUN_CASES = [
    make_run_case(
        case_seed=case_seed,
        client_count=client_count,
        sample_count=sample_count,
        change_probability=change_probability,
        switch_count=switch_count,
    )
    for case_seed, (client_count, sample_count, change_probability, switch_count) in enumerate(
        [
            (1, 60, 0.05, 3),
            (3, 200, 0.02, 6),
            (3, 200, 0.0, 4),
            (2, 150, 0.03, 0),
            (4, 120, 0.3, 30),
            (2, 0, 0.0, 0),
            (3, 80, 0.0, 0),
        ]
    )
]


def make_legacy_metric_inputs(run_case):
    """実旧の指標の計算へ渡す、clientの代わり（必要な属性だけを持つ）と、真のドリフト位置。"""
    prediction_correctness_by_client, concept_ids_by_client, detection_sample_indices_by_client = (
        run_case
    )
    legacy_clients = [
        SimpleNamespace(
            history_accuracy=[float(is_correct) for is_correct in prediction_correctness],
            local_switch_positions=list(detection_sample_indices),
        )
        for prediction_correctness, detection_sample_indices in zip(
            prediction_correctness_by_client, detection_sample_indices_by_client, strict=True
        )
    ]
    legacy_true_drift_events = extract_true_drift_events(
        [list(concept_ids) for concept_ids in concept_ids_by_client]
    )
    return legacy_clients, legacy_true_drift_events


@pytest.mark.parametrize("run_case", RUN_CASES)
@pytest.mark.parametrize("maximum_delay", [0, 5, 100])
@pytest.mark.parametrize("recovery_window", [0, 1, 50, 1000])
def test_accuracy_and_detection_metrics_match_real_legacy_compute_metrics(
    run_case, maximum_delay, recovery_window
):
    """精度・定常精度・検出の指標が、実旧の指標の計算と、完全に一致する。"""
    prediction_correctness_by_client, concept_ids_by_client, detection_sample_indices_by_client = (
        run_case
    )
    legacy_clients, legacy_true_drift_events = make_legacy_metric_inputs(run_case)
    legacy_metrics = compute_metrics(
        legacy_clients,
        legacy_true_drift_events,
        delay_tolerance=maximum_delay,
        stable_window=recovery_window,
    )
    concept_change_sample_indices_by_client = tuple(
        extract_concept_change_sample_indices(concept_ids_by_sample_index=concept_ids)
        for concept_ids in concept_ids_by_client
    )
    assert (
        calculate_prediction_accuracy(
            prediction_correctness_by_client=prediction_correctness_by_client
        )
        == legacy_metrics["accuracy"]
    )
    stable_period_prediction_accuracy = calculate_stable_period_prediction_accuracy(
        prediction_correctness_by_client=prediction_correctness_by_client,
        concept_change_sample_indices_by_client=concept_change_sample_indices_by_client,
        recovery_window_sample_count=recovery_window,
    )
    legacy_stable_accuracy = _stable_accuracy(
        legacy_clients, legacy_true_drift_events, recovery_window
    )
    assert legacy_metrics["stable_accuracy"] == legacy_stable_accuracy or (
        isnan(legacy_metrics["stable_accuracy"]) and isnan(legacy_stable_accuracy)
    )
    if isnan(legacy_stable_accuracy):
        assert isnan(stable_period_prediction_accuracy)
    else:
        assert stable_period_prediction_accuracy == legacy_stable_accuracy
    detection_metrics = calculate_detection_metrics(
        concept_change_sample_indices_by_client=concept_change_sample_indices_by_client,
        detection_sample_indices_by_client=detection_sample_indices_by_client,
        maximum_delay_sample_count=maximum_delay,
    )
    assert type(detection_metrics) is DetectionMetrics
    assert detection_metrics == DetectionMetrics(
        detection_precision=legacy_metrics["precision"],
        detection_recall=legacy_metrics["recall"],
        detection_f1=legacy_metrics["f1"],
        detection_count=legacy_metrics["total_detect"],
        concept_change_count=legacy_metrics["total_true"],
        matched_detection_count=legacy_metrics["tp"],
    )
    for metric_value in (
        detection_metrics.detection_precision,
        detection_metrics.detection_recall,
        detection_metrics.detection_f1,
    ):
        assert type(metric_value) is float
        assert 0.0 <= metric_value <= 1.0


def test_run_cases_cover_zero_denominators_and_stable_period_without_samples():
    """上の対照の入力が、検出なし、変更なし、標本なし、定常の標本なし（NaN）、を含む。"""
    observed_cases = set()
    for run_case in RUN_CASES:
        legacy_clients, legacy_true_drift_events = make_legacy_metric_inputs(run_case)
        for recovery_window in (0, 1000):
            legacy_metrics = compute_metrics(
                legacy_clients,
                legacy_true_drift_events,
                delay_tolerance=5,
                stable_window=recovery_window,
            )
            if legacy_metrics["total_detect"] == 0:
                observed_cases.add("no_detection")
            if legacy_metrics["total_true"] == 0:
                observed_cases.add("no_concept_change")
            if not any(legacy_client.history_accuracy for legacy_client in legacy_clients):
                observed_cases.add("no_sample")
            if isnan(legacy_metrics["stable_accuracy"]):
                observed_cases.add("stable_accuracy_nan")
            if 0 < legacy_metrics["tp"] < legacy_metrics["total_detect"]:
                observed_cases.add("partially_matched")
            if legacy_metrics["stable_accuracy"] not in (legacy_metrics["accuracy"],) and not isnan(
                legacy_metrics["stable_accuracy"]
            ):
                observed_cases.add("stable_accuracy_differs_from_accuracy")
    assert observed_cases == {
        "no_detection",
        "no_concept_change",
        "no_sample",
        "stable_accuracy_nan",
        "partially_matched",
        "stable_accuracy_differs_from_accuracy",
    }


def test_stable_period_keeps_samples_before_first_change_and_drops_recovery_window():
    """最初の変更より前の標本は含め、各変更の直後の窓の件数だけを除く（手計算の例）。"""
    # 変更位置は3と6。窓2: 位置3・4と6・7を除く。残りは位置0・1・2・5・8・9。
    prediction_correctness = (True, False, True, False, False, True, False, False, True, True)
    assert (
        calculate_stable_period_prediction_accuracy(
            prediction_correctness_by_client=(prediction_correctness,),
            concept_change_sample_indices_by_client=((3, 6),),
            recovery_window_sample_count=2,
        )
        == (1 + 0 + 1 + 1 + 1 + 1) / 6
    )
    # 変更位置が昇順でなくても、同じ結果になる。
    assert (
        calculate_stable_period_prediction_accuracy(
            prediction_correctness_by_client=(prediction_correctness,),
            concept_change_sample_indices_by_client=((6, 3),),
            recovery_window_sample_count=2,
        )
        == (1 + 0 + 1 + 1 + 1 + 1) / 6
    )


def test_run_metric_settings_hold_validated_immutable_values():
    run_metric_settings = RunMetricSettings(
        maximum_detection_delay_sample_count=100, post_change_recovery_window_sample_count=0
    )
    assert run_metric_settings.maximum_detection_delay_sample_count == 100
    assert run_metric_settings.post_change_recovery_window_sample_count == 0
    with pytest.raises(FrozenInstanceError):
        run_metric_settings.maximum_detection_delay_sample_count = 1
    with pytest.raises(TypeError):
        RunMetricSettings(100, 50)


@pytest.mark.parametrize(
    "field_name",
    ["maximum_detection_delay_sample_count", "post_change_recovery_window_sample_count"],
)
@pytest.mark.parametrize(
    ("invalid_value", "expected_exception_type"),
    [(True, TypeError), (1.0, TypeError), ("1", TypeError), (None, TypeError), (-1, ValueError)],
)
def test_run_metric_settings_reject_invalid_values(
    field_name, invalid_value, expected_exception_type
):
    valid_values = dict(
        maximum_detection_delay_sample_count=100, post_change_recovery_window_sample_count=50
    )
    with pytest.raises(expected_exception_type):
        RunMetricSettings(**valid_values | {field_name: invalid_value})


INVALID_POSITION_SEQUENCES = [
    ([1, 2], TypeError),
    ((1, True), TypeError),
    ((1, 2.0), TypeError),
    ((1, "2"), TypeError),
    ((1, -1), ValueError),
    (None, TypeError),
]


@pytest.mark.parametrize(
    ("invalid_sequence", "expected_exception_type"), INVALID_POSITION_SEQUENCES
)
def test_calculations_reject_invalid_position_sequences(invalid_sequence, expected_exception_type):
    """位置の列は、非負のbuiltin intのexact tuple。"""
    with pytest.raises(expected_exception_type):
        extract_concept_change_sample_indices(concept_ids_by_sample_index=invalid_sequence)
    for argument_name in ("concept_change_sample_indices", "event_sample_indices"):
        with pytest.raises(expected_exception_type):
            count_events_matched_to_concept_changes(
                **dict(
                    concept_change_sample_indices=(1,),
                    event_sample_indices=(1,),
                    maximum_delay_sample_count=1,
                )
                | {argument_name: invalid_sequence}
            )
    with pytest.raises(expected_exception_type):
        calculate_detection_metrics(
            concept_change_sample_indices_by_client=(invalid_sequence,),
            detection_sample_indices_by_client=((1,),),
            maximum_delay_sample_count=1,
        )
    with pytest.raises(expected_exception_type):
        calculate_detection_metrics(
            concept_change_sample_indices_by_client=((1,),),
            detection_sample_indices_by_client=(invalid_sequence,),
            maximum_delay_sample_count=1,
        )
    with pytest.raises(expected_exception_type):
        calculate_stable_period_prediction_accuracy(
            prediction_correctness_by_client=((True,),),
            concept_change_sample_indices_by_client=(invalid_sequence,),
            recovery_window_sample_count=1,
        )


@pytest.mark.parametrize(
    ("invalid_count", "expected_exception_type"),
    [(True, TypeError), (1.0, TypeError), (None, TypeError), (-1, ValueError)],
)
def test_calculations_reject_invalid_counts(invalid_count, expected_exception_type):
    with pytest.raises(expected_exception_type):
        count_events_matched_to_concept_changes(
            concept_change_sample_indices=(1,),
            event_sample_indices=(1,),
            maximum_delay_sample_count=invalid_count,
        )
    with pytest.raises(expected_exception_type):
        calculate_detection_metrics(
            concept_change_sample_indices_by_client=((1,),),
            detection_sample_indices_by_client=((1,),),
            maximum_delay_sample_count=invalid_count,
        )
    with pytest.raises(expected_exception_type):
        calculate_stable_period_prediction_accuracy(
            prediction_correctness_by_client=((True,),),
            concept_change_sample_indices_by_client=((1,),),
            recovery_window_sample_count=invalid_count,
        )


@pytest.mark.parametrize(
    ("invalid_correctness_by_client", "expected_exception_type"),
    [
        ([(True,)], TypeError),
        (([True],), TypeError),
        (((1,),), TypeError),
        (((1.0,),), TypeError),
        (None, TypeError),
    ],
)
def test_calculations_reject_invalid_prediction_correctness(
    invalid_correctness_by_client, expected_exception_type
):
    """正誤の列は、boolのexact tupleの、exact tuple。"""
    with pytest.raises(expected_exception_type):
        calculate_prediction_accuracy(
            prediction_correctness_by_client=invalid_correctness_by_client
        )
    with pytest.raises(expected_exception_type):
        calculate_stable_period_prediction_accuracy(
            prediction_correctness_by_client=invalid_correctness_by_client,
            concept_change_sample_indices_by_client=((),),
            recovery_window_sample_count=1,
        )


def test_calculations_reject_client_count_mismatch_and_positional_arguments():
    with pytest.raises(ValueError):
        calculate_detection_metrics(
            concept_change_sample_indices_by_client=((1,), (2,)),
            detection_sample_indices_by_client=((1,),),
            maximum_delay_sample_count=1,
        )
    with pytest.raises(ValueError):
        calculate_stable_period_prediction_accuracy(
            prediction_correctness_by_client=((True,),),
            concept_change_sample_indices_by_client=((1,), (2,)),
            recovery_window_sample_count=1,
        )
    with pytest.raises(TypeError):
        calculate_detection_metrics(((1,),), ((1,),), 1)
    with pytest.raises(TypeError):
        extract_concept_change_sample_indices((0, 1))
