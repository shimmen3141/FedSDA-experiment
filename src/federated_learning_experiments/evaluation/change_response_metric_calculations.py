"""手法に依存しない、概念の変更に対するイベントの、対応づけ・時機の分類・変化点の推定の誤差の計算。

位置は、client 1つの、標本位置（0以上のbuiltin int）のtuple。対応づけの規則は、
`run_metric_calculations.count_events_matched_to_concept_changes`と同じ（変更位置を与えられた順に見て、
変更位置以上・変更位置＋許容遅延以下にある、まだ使っていない最初のイベントを、与えられた順に取る）。
"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ConceptChangeEventMatching:
    """変更位置とイベントの、1対1の対応づけの結果（対応づいた順＝変更位置を見た順）。"""

    # イベントの列の中の位置。
    matched_event_positions: tuple[int, ...]
    # 変更位置の列の中の位置。
    matched_concept_change_positions: tuple[int, ...]
    # イベントの標本位置−変更の標本位置。
    matched_delay_sample_counts: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class UnmatchedEventCounts:
    """変更に対応づかなかったイベントの、時機ごとの数（どれか1つの区分に入る）。

    区分は、上から順に、最初に当てはまるもの: 重複（どれかの変更位置以上・変更位置＋許容遅延以下）、
    早期（どれかの変更位置−許容遅延以上・変更位置未満）、遅延（どれかの変更位置＋許容遅延より後・
    変更位置＋2×許容遅延以下）、単独（どれでもない）。
    """

    duplicate_event_count: int
    early_event_count: int
    late_event_count: int
    isolated_event_count: int


def _validate_nonnegative_integer(*, specified_value: object, parameter_name: str) -> None:
    if type(specified_value) is not int:
        raise TypeError(f"{parameter_name} must be builtin int")
    if specified_value < 0:
        raise ValueError(f"{parameter_name} must be nonnegative")


def _validate_sample_indices(*, sample_indices: object, parameter_name: str) -> None:
    if type(sample_indices) is not tuple:
        raise TypeError(f"{parameter_name} must be exact tuple")
    for sample_index in sample_indices:
        _validate_nonnegative_integer(specified_value=sample_index, parameter_name=parameter_name)


def _match_validated_events(
    *,
    concept_change_sample_indices: tuple[int, ...],
    concept_change_positions: tuple[int, ...],
    event_sample_indices: tuple[int, ...],
    maximum_delay_sample_count: int,
) -> ConceptChangeEventMatching:
    """検査済みの入力を対応づける。変更位置は、`concept_change_positions`の順に見る。"""
    used_event_positions: set[int] = set()
    matched_event_positions: list[int] = []
    matched_concept_change_positions: list[int] = []
    matched_delay_sample_counts: list[int] = []
    for concept_change_position in concept_change_positions:
        concept_change_sample_index = concept_change_sample_indices[concept_change_position]
        for event_position, event_sample_index in enumerate(event_sample_indices):
            if event_position in used_event_positions:
                continue
            if (
                concept_change_sample_index
                <= event_sample_index
                <= concept_change_sample_index + maximum_delay_sample_count
            ):
                used_event_positions.add(event_position)
                matched_event_positions.append(event_position)
                matched_concept_change_positions.append(concept_change_position)
                matched_delay_sample_counts.append(event_sample_index - concept_change_sample_index)
                break
    return ConceptChangeEventMatching(
        matched_event_positions=tuple(matched_event_positions),
        matched_concept_change_positions=tuple(matched_concept_change_positions),
        matched_delay_sample_counts=tuple(matched_delay_sample_counts),
    )


def match_events_to_concept_changes(
    *,
    concept_change_sample_indices: tuple[int, ...],
    event_sample_indices: tuple[int, ...],
    maximum_delay_sample_count: int,
) -> ConceptChangeEventMatching:
    """変更位置とイベントを、与えられた順に、1対1で対応づけて、対応づいた位置と遅れを返す。"""
    _validate_sample_indices(
        sample_indices=concept_change_sample_indices,
        parameter_name="concept_change_sample_indices",
    )
    _validate_sample_indices(
        sample_indices=event_sample_indices, parameter_name="event_sample_indices"
    )
    _validate_nonnegative_integer(
        specified_value=maximum_delay_sample_count, parameter_name="maximum_delay_sample_count"
    )
    return _match_validated_events(
        concept_change_sample_indices=concept_change_sample_indices,
        concept_change_positions=tuple(range(len(concept_change_sample_indices))),
        event_sample_indices=event_sample_indices,
        maximum_delay_sample_count=maximum_delay_sample_count,
    )


def count_unmatched_events_by_timing(
    *,
    concept_change_sample_indices: tuple[int, ...],
    event_sample_indices: tuple[int, ...],
    matched_event_positions: tuple[int, ...],
    maximum_delay_sample_count: int,
) -> UnmatchedEventCounts:
    """対応づかなかったイベント（`matched_event_positions`以外）を、変更位置に対する時機で分類して数える。"""
    _validate_sample_indices(
        sample_indices=concept_change_sample_indices,
        parameter_name="concept_change_sample_indices",
    )
    _validate_sample_indices(
        sample_indices=event_sample_indices, parameter_name="event_sample_indices"
    )
    _validate_nonnegative_integer(
        specified_value=maximum_delay_sample_count, parameter_name="maximum_delay_sample_count"
    )
    _validate_sample_indices(
        sample_indices=matched_event_positions, parameter_name="matched_event_positions"
    )
    if len(set(matched_event_positions)) != len(matched_event_positions) or any(
        matched_event_position >= len(event_sample_indices)
        for matched_event_position in matched_event_positions
    ):
        raise ValueError("matched_event_positions must be distinct positions of the events")
    matched_event_position_set = set(matched_event_positions)
    duplicate_event_count = 0
    early_event_count = 0
    late_event_count = 0
    isolated_event_count = 0
    for event_position, event_sample_index in enumerate(event_sample_indices):
        if event_position in matched_event_position_set:
            continue
        if any(
            concept_change_sample_index
            <= event_sample_index
            <= concept_change_sample_index + maximum_delay_sample_count
            for concept_change_sample_index in concept_change_sample_indices
        ):
            duplicate_event_count += 1
        elif any(
            concept_change_sample_index - maximum_delay_sample_count
            <= event_sample_index
            < concept_change_sample_index
            for concept_change_sample_index in concept_change_sample_indices
        ):
            early_event_count += 1
        elif any(
            concept_change_sample_index + maximum_delay_sample_count
            < event_sample_index
            <= concept_change_sample_index + 2 * maximum_delay_sample_count
            for concept_change_sample_index in concept_change_sample_indices
        ):
            late_event_count += 1
        else:
            isolated_event_count += 1
    return UnmatchedEventCounts(
        duplicate_event_count=duplicate_event_count,
        early_event_count=early_event_count,
        late_event_count=late_event_count,
        isolated_event_count=isolated_event_count,
    )


def calculate_change_point_estimation_errors(
    *,
    concept_change_sample_indices: tuple[int, ...],
    alarm_sample_indices: tuple[int, ...],
    estimated_change_point_sample_indices: tuple[int, ...],
    maximum_delay_sample_count: int,
) -> tuple[int, ...]:
    """変更に対応した警報の、変化点の推定の誤差（推定した位置−変更位置）を、対応づいた順に返す。

    変更位置は、昇順に見る。警報は、与えられた順（起きた順）に取る。警報ごとに、推定した変化点が1つある。
    """
    _validate_sample_indices(
        sample_indices=concept_change_sample_indices,
        parameter_name="concept_change_sample_indices",
    )
    _validate_sample_indices(
        sample_indices=alarm_sample_indices, parameter_name="alarm_sample_indices"
    )
    _validate_sample_indices(
        sample_indices=estimated_change_point_sample_indices,
        parameter_name="estimated_change_point_sample_indices",
    )
    _validate_nonnegative_integer(
        specified_value=maximum_delay_sample_count, parameter_name="maximum_delay_sample_count"
    )
    if len(alarm_sample_indices) != len(estimated_change_point_sample_indices):
        raise ValueError(
            "alarm_sample_indices and estimated_change_point_sample_indices must have the same length"
        )
    matching = _match_validated_events(
        concept_change_sample_indices=concept_change_sample_indices,
        # 同じ位置の変更は、与えられた順のまま（安定な整列）。
        concept_change_positions=tuple(
            sorted(
                range(len(concept_change_sample_indices)),
                key=lambda position: concept_change_sample_indices[position],
            )
        ),
        event_sample_indices=alarm_sample_indices,
        maximum_delay_sample_count=maximum_delay_sample_count,
    )
    return tuple(
        estimated_change_point_sample_indices[matched_alarm_position]
        - concept_change_sample_indices[matched_concept_change_position]
        for matched_alarm_position, matched_concept_change_position in zip(
            matching.matched_event_positions,
            matching.matched_concept_change_positions,
            strict=True,
        )
    )
