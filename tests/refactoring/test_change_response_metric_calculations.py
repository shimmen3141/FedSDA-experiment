"""変更位置とイベントの対応づけ、対応づかなかったイベントの分類、変化点の推定の誤差を、実旧の関数と照合する。"""

import random
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from federated_drift_experiment import metrics as legacy_metrics
from federated_learning_experiments.evaluation.change_response_metric_calculations import (
    ConceptChangeEventMatching,
    UnmatchedEventCounts,
    calculate_change_point_estimation_errors,
    count_unmatched_events_by_timing,
    match_events_to_concept_changes,
)
from federated_learning_experiments.evaluation.run_metric_calculations import (
    count_events_matched_to_concept_changes,
)


def make_random_positions(random_generator, *, maximum_count, maximum_position, sort_positions):
    positions = [
        random_generator.randrange(maximum_position)
        for _ in range(random_generator.randrange(maximum_count + 1))
    ]
    return tuple(sorted(positions) if sort_positions else positions)


RANDOM_CASES = [
    (random_seed, maximum_delay_sample_count, sort_events)
    for random_seed in range(12)
    for maximum_delay_sample_count in (0, 3, 20)
    for sort_events in (True, False)
]


@pytest.mark.parametrize(("random_seed", "maximum_delay_sample_count", "sort_events"), RANDOM_CASES)
def test_matching_and_unmatched_classification_match_legacy_functions(
    random_seed, maximum_delay_sample_count, sort_events
):
    """乱数で作った列で、対応づけ（位置と遅れ）と、対応づかなかったイベントの分類が、実旧と一致する。"""
    random_generator = random.Random(random_seed)
    concept_change_sample_indices = make_random_positions(
        random_generator, maximum_count=8, maximum_position=120, sort_positions=True
    )
    # 同じ位置のイベントが重なる場合を含む。
    event_sample_indices = make_random_positions(
        random_generator, maximum_count=14, maximum_position=130, sort_positions=sort_events
    )
    legacy_used, legacy_matched_true, legacy_delays = legacy_metrics.match_events(
        list(concept_change_sample_indices),
        list(event_sample_indices),
        maximum_delay_sample_count,
    )
    matching = match_events_to_concept_changes(
        concept_change_sample_indices=concept_change_sample_indices,
        event_sample_indices=event_sample_indices,
        maximum_delay_sample_count=maximum_delay_sample_count,
    )
    assert type(matching) is ConceptChangeEventMatching
    assert set(matching.matched_event_positions) == legacy_used
    assert len(set(matching.matched_event_positions)) == len(matching.matched_event_positions)
    assert set(matching.matched_concept_change_positions) == legacy_matched_true
    assert list(matching.matched_delay_sample_counts) == legacy_delays
    # 対応づいた順（変更位置の順）で、遅れは、イベントの位置−変更位置。
    assert list(matching.matched_concept_change_positions) == sorted(
        matching.matched_concept_change_positions
    )
    for event_position, concept_change_position, delay_sample_count in zip(
        matching.matched_event_positions,
        matching.matched_concept_change_positions,
        matching.matched_delay_sample_counts,
        strict=True,
    ):
        assert delay_sample_count == (
            event_sample_indices[event_position]
            - concept_change_sample_indices[concept_change_position]
        )
        assert 0 <= delay_sample_count <= maximum_delay_sample_count
    # 既存の、数だけを返す計算と、同じ対応づけ。
    event_match_counts = count_events_matched_to_concept_changes(
        concept_change_sample_indices=concept_change_sample_indices,
        event_sample_indices=event_sample_indices,
        maximum_delay_sample_count=maximum_delay_sample_count,
    )
    assert event_match_counts.matched_event_count == len(matching.matched_event_positions)
    assert event_match_counts.matched_concept_change_count == len(
        matching.matched_concept_change_positions
    )
    # 対応づかなかったイベントの分類。
    legacy_unmatched_counts = legacy_metrics._classify_unmatched_events(
        list(concept_change_sample_indices),
        list(event_sample_indices),
        legacy_used,
        maximum_delay_sample_count,
    )
    unmatched_event_counts = count_unmatched_events_by_timing(
        concept_change_sample_indices=concept_change_sample_indices,
        event_sample_indices=event_sample_indices,
        matched_event_positions=matching.matched_event_positions,
        maximum_delay_sample_count=maximum_delay_sample_count,
    )
    assert type(unmatched_event_counts) is UnmatchedEventCounts
    assert unmatched_event_counts == UnmatchedEventCounts(
        duplicate_event_count=legacy_unmatched_counts["duplicate"],
        early_event_count=legacy_unmatched_counts["early"],
        late_event_count=legacy_unmatched_counts["late"],
        isolated_event_count=legacy_unmatched_counts["isolated"],
    )
    assert (
        unmatched_event_counts.duplicate_event_count
        + unmatched_event_counts.early_event_count
        + unmatched_event_counts.late_event_count
        + unmatched_event_counts.isolated_event_count
    ) == len(event_sample_indices) - len(matching.matched_event_positions)


def test_random_cases_cover_every_unmatched_timing():
    """上の乱数の条件が、4つの区分と、対応づけを、実際に通っている。"""
    total_counts = dict(duplicate=0, early=0, late=0, isolated=0, matched=0)
    for random_seed, maximum_delay_sample_count, sort_events in RANDOM_CASES:
        random_generator = random.Random(random_seed)
        concept_changes = make_random_positions(
            random_generator, maximum_count=8, maximum_position=120, sort_positions=True
        )
        events = make_random_positions(
            random_generator, maximum_count=14, maximum_position=130, sort_positions=sort_events
        )
        matching = match_events_to_concept_changes(
            concept_change_sample_indices=concept_changes,
            event_sample_indices=events,
            maximum_delay_sample_count=maximum_delay_sample_count,
        )
        unmatched = count_unmatched_events_by_timing(
            concept_change_sample_indices=concept_changes,
            event_sample_indices=events,
            matched_event_positions=matching.matched_event_positions,
            maximum_delay_sample_count=maximum_delay_sample_count,
        )
        total_counts["matched"] += len(matching.matched_event_positions)
        total_counts["duplicate"] += unmatched.duplicate_event_count
        total_counts["early"] += unmatched.early_event_count
        total_counts["late"] += unmatched.late_event_count
        total_counts["isolated"] += unmatched.isolated_event_count
    assert all(count > 0 for count in total_counts.values()), total_counts


def test_unmatched_events_are_classified_by_first_matching_timing():
    """区分は、重複→早期→遅延→単独の順で、最初に当てはまるもの（境界を含む）。許容遅延は10。"""
    concept_changes = (100, 200)
    # 100に対応づくのは、最初の100。
    events = (100, 110, 111, 120, 121, 90, 89, 199, 150, 0)
    matching = match_events_to_concept_changes(
        concept_change_sample_indices=concept_changes,
        event_sample_indices=events,
        maximum_delay_sample_count=10,
    )
    assert matching == ConceptChangeEventMatching(
        matched_event_positions=(0,),
        matched_concept_change_positions=(0,),
        matched_delay_sample_counts=(0,),
    )
    assert count_unmatched_events_by_timing(
        concept_change_sample_indices=concept_changes,
        event_sample_indices=events,
        matched_event_positions=matching.matched_event_positions,
        maximum_delay_sample_count=10,
    ) == UnmatchedEventCounts(
        # 110（100＋10以下）。
        duplicate_event_count=1,
        # 90（100−10以上）と、199（200より前）。
        early_event_count=2,
        # 111と120（100＋10より後、100＋20以下）。
        late_event_count=2,
        # 121、89、150、0。
        isolated_event_count=4,
    )


@pytest.mark.parametrize("random_seed", range(20))
@pytest.mark.parametrize("maximum_delay_sample_count", [0, 5, 30])
def test_change_point_estimation_errors_match_legacy_function(
    random_seed, maximum_delay_sample_count
):
    """変更に対応した警報の、推定の誤差（推定−変更位置）が、実旧の`_change_point_errors`と一致する。"""
    random_generator = random.Random(1000 + random_seed)
    # 実旧は、変更位置を昇順にして見る。与える順は、昇順でなくてもよい。
    concept_change_sample_indices = make_random_positions(
        random_generator, maximum_count=7, maximum_position=150, sort_positions=False
    )
    alarm_sample_indices = make_random_positions(
        random_generator, maximum_count=12, maximum_position=170, sort_positions=True
    )
    estimated_change_point_sample_indices = tuple(
        max(0, alarm_sample_index - random_generator.randrange(25))
        for alarm_sample_index in alarm_sample_indices
    )
    legacy_errors = legacy_metrics._change_point_errors(
        [
            SimpleNamespace(
                detected_event_positions=list(alarm_sample_indices),
                estimated_drift_start_positions=list(estimated_change_point_sample_indices),
            )
        ],
        [list(concept_change_sample_indices)],
        maximum_delay_sample_count,
    )
    errors = calculate_change_point_estimation_errors(
        concept_change_sample_indices=concept_change_sample_indices,
        alarm_sample_indices=alarm_sample_indices,
        estimated_change_point_sample_indices=estimated_change_point_sample_indices,
        maximum_delay_sample_count=maximum_delay_sample_count,
    )
    assert type(errors) is tuple
    assert list(errors) == legacy_errors
    assert all(type(error) is int for error in errors)


def test_change_point_estimation_error_is_estimate_minus_change_position():
    assert calculate_change_point_estimation_errors(
        concept_change_sample_indices=(50, 100),
        alarm_sample_indices=(55, 60, 104, 300),
        estimated_change_point_sample_indices=(48, 59, 103, 290),
        maximum_delay_sample_count=10,
    ) == (-2, 3)


VALID_MATCHING_ARGUMENTS = dict(
    concept_change_sample_indices=(10, 50),
    event_sample_indices=(12, 60),
    maximum_delay_sample_count=5,
)


@pytest.mark.parametrize(
    ("argument_name", "invalid_value"),
    [
        ("concept_change_sample_indices", [10, 50]),
        ("concept_change_sample_indices", (10, -1)),
        ("concept_change_sample_indices", (10, 1.0)),
        ("concept_change_sample_indices", (True,)),
        ("event_sample_indices", None),
        ("event_sample_indices", (-3,)),
        ("event_sample_indices", ("12",)),
        ("maximum_delay_sample_count", -1),
        ("maximum_delay_sample_count", 2.0),
        ("maximum_delay_sample_count", True),
    ],
)
def test_calculations_reject_invalid_arguments(argument_name, invalid_value):
    """位置の列は、0以上のbuiltin intのtuple。許容遅延は、0以上のbuiltin int。"""
    invalid_arguments = VALID_MATCHING_ARGUMENTS | {argument_name: invalid_value}
    with pytest.raises((TypeError, ValueError), match=argument_name):
        match_events_to_concept_changes(**invalid_arguments)
    with pytest.raises((TypeError, ValueError), match=argument_name):
        count_unmatched_events_by_timing(matched_event_positions=(), **invalid_arguments)
    if argument_name != "event_sample_indices":
        with pytest.raises((TypeError, ValueError), match=argument_name):
            calculate_change_point_estimation_errors(
                alarm_sample_indices=(12, 60),
                estimated_change_point_sample_indices=(11, 55),
                **{
                    name: value
                    for name, value in invalid_arguments.items()
                    if name != "event_sample_indices"
                },
            )


@pytest.mark.parametrize(
    "invalid_matched_event_positions", [[0], (2,), (-1,), (0, 0), (True,), (0.0,), None]
)
def test_unmatched_counting_rejects_invalid_matched_positions(invalid_matched_event_positions):
    """対応づいたイベントの位置は、イベントの列の中の、重ならない位置のtuple。"""
    with pytest.raises((TypeError, ValueError), match="matched_event_positions"):
        count_unmatched_events_by_timing(
            matched_event_positions=invalid_matched_event_positions, **VALID_MATCHING_ARGUMENTS
        )


@pytest.mark.parametrize(
    ("alarm_sample_indices", "estimated_change_point_sample_indices"),
    [((12, 60), (11,)), ((12,), (11, 55)), ((12, 60), [11, 55]), ((12, 60), (11, -1))],
)
def test_change_point_errors_reject_mismatched_alarm_and_estimate_sequences(
    alarm_sample_indices, estimated_change_point_sample_indices
):
    """警報の位置と、推定した変化点は、同じ長さのtuple（実旧は、長さが違うclientを黙って飛ばす）。"""
    with pytest.raises((TypeError, ValueError)):
        calculate_change_point_estimation_errors(
            concept_change_sample_indices=(10,),
            alarm_sample_indices=alarm_sample_indices,
            estimated_change_point_sample_indices=estimated_change_point_sample_indices,
            maximum_delay_sample_count=5,
        )


def test_results_are_frozen_and_functions_are_keyword_only():
    matching = match_events_to_concept_changes(**VALID_MATCHING_ARGUMENTS)
    with pytest.raises(FrozenInstanceError):
        matching.matched_event_positions = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        match_events_to_concept_changes((10,), (12,), 5)  # type: ignore[misc]
    with pytest.raises(TypeError):
        count_unmatched_events_by_timing((10,), (12,), (), 5)  # type: ignore[misc]
    with pytest.raises(TypeError):
        calculate_change_point_estimation_errors((10,), (12,), (11,), 5)  # type: ignore[misc]
    with pytest.raises(TypeError):
        UnmatchedEventCounts(0, 0, 0, 0)  # type: ignore[misc]
