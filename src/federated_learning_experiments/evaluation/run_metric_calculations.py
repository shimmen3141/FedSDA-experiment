"""手法に依存しない、run 1回の指標の計算（概念の変更位置、イベントの対応づけ、精度、検出の指標）。"""

from bisect import bisect_right
from dataclasses import dataclass


def _validate_nonnegative_count(*, specified_value: int, parameter_name: str) -> None:
    if type(specified_value) is not int:
        raise TypeError(f"{parameter_name} must be builtin int")
    if specified_value < 0:
        raise ValueError(f"{parameter_name} must be nonnegative")


def _validate_nonnegative_integer_sequence(
    *, integer_sequence: tuple[int, ...], parameter_name: str
) -> None:
    if type(integer_sequence) is not tuple:
        raise TypeError(f"{parameter_name} must be exact tuple")
    for specified_value in integer_sequence:
        _validate_nonnegative_count(specified_value=specified_value, parameter_name=parameter_name)


def _validate_sequences_by_client(
    *, sequences_by_client: tuple[tuple[int, ...], ...], parameter_name: str
) -> None:
    if type(sequences_by_client) is not tuple:
        raise TypeError(f"{parameter_name} must be exact tuple")
    for integer_sequence in sequences_by_client:
        _validate_nonnegative_integer_sequence(
            integer_sequence=integer_sequence, parameter_name=parameter_name
        )


def _validate_prediction_correctness_by_client(
    *, prediction_correctness_by_client: tuple[tuple[bool, ...], ...]
) -> None:
    if type(prediction_correctness_by_client) is not tuple:
        raise TypeError("prediction_correctness_by_client must be exact tuple")
    for prediction_correctness in prediction_correctness_by_client:
        if type(prediction_correctness) is not tuple:
            raise TypeError("prediction_correctness_by_client must contain exact tuples")
        for prediction_is_correct in prediction_correctness:
            if type(prediction_is_correct) is not bool:
                raise TypeError("prediction correctness must be builtin bool")


@dataclass(frozen=True, kw_only=True)
class RunMetricSettings:
    """指標の計算の設定。

    `maximum_detection_delay_sample_count`: 概念の変更位置から、この件数以内にある検出を、
    その変更の検出として数える。
    `post_change_recovery_window_sample_count`: 定常精度から除く、各変更位置の直後の件数。
    """

    maximum_detection_delay_sample_count: int
    post_change_recovery_window_sample_count: int

    def __post_init__(self) -> None:
        _validate_nonnegative_count(
            specified_value=self.maximum_detection_delay_sample_count,
            parameter_name="maximum_detection_delay_sample_count",
        )
        _validate_nonnegative_count(
            specified_value=self.post_change_recovery_window_sample_count,
            parameter_name="post_change_recovery_window_sample_count",
        )


@dataclass(frozen=True, kw_only=True)
class EventMatchCounts:
    """変更位置とイベントの、1対1の対応づけの結果。"""

    matched_event_count: int
    matched_concept_change_count: int


@dataclass(frozen=True, kw_only=True)
class DetectionMetrics:
    """全clientの合計の、検出の指標と、その計数。分母が0の比は0。"""

    detection_precision: float
    detection_recall: float
    detection_f1: float
    detection_count: int
    concept_change_count: int
    matched_detection_count: int


def extract_concept_change_sample_indices(
    *, concept_ids_by_sample_index: tuple[int, ...]
) -> tuple[int, ...]:
    """前の標本と概念が違う標本位置を、昇順で返す（先頭の標本は、変更位置にしない）。"""
    _validate_nonnegative_integer_sequence(
        integer_sequence=concept_ids_by_sample_index, parameter_name="concept_ids_by_sample_index"
    )
    return tuple(
        sample_index
        for sample_index in range(1, len(concept_ids_by_sample_index))
        if concept_ids_by_sample_index[sample_index]
        != concept_ids_by_sample_index[sample_index - 1]
    )


def _count_matched_events(
    *,
    concept_change_sample_indices: tuple[int, ...],
    event_sample_indices: tuple[int, ...],
    maximum_delay_sample_count: int,
) -> EventMatchCounts:
    """検査済みの入力を対応づける。変更位置を与えられた順に見て、未使用の最初のイベントを取る。"""
    matched_event_positions: set[int] = set()
    matched_concept_change_count = 0
    for concept_change_sample_index in concept_change_sample_indices:
        for event_position, event_sample_index in enumerate(event_sample_indices):
            if event_position in matched_event_positions:
                continue
            if (
                concept_change_sample_index
                <= event_sample_index
                <= concept_change_sample_index + maximum_delay_sample_count
            ):
                matched_event_positions.add(event_position)
                matched_concept_change_count += 1
                break
    return EventMatchCounts(
        matched_event_count=len(matched_event_positions),
        matched_concept_change_count=matched_concept_change_count,
    )


def count_events_matched_to_concept_changes(
    *,
    concept_change_sample_indices: tuple[int, ...],
    event_sample_indices: tuple[int, ...],
    maximum_delay_sample_count: int,
) -> EventMatchCounts:
    """変更位置とイベントを、順に1対1で対応づけて、対応した数を返す。

    変更位置を与えられた順に見て、それぞれに、変更位置以上・変更位置＋許容遅延以下にある、
    まだ使っていない最初のイベント（与えられた順）を対応づける。
    """
    _validate_nonnegative_integer_sequence(
        integer_sequence=concept_change_sample_indices,
        parameter_name="concept_change_sample_indices",
    )
    _validate_nonnegative_integer_sequence(
        integer_sequence=event_sample_indices, parameter_name="event_sample_indices"
    )
    _validate_nonnegative_count(
        specified_value=maximum_delay_sample_count, parameter_name="maximum_delay_sample_count"
    )
    return _count_matched_events(
        concept_change_sample_indices=concept_change_sample_indices,
        event_sample_indices=event_sample_indices,
        maximum_delay_sample_count=maximum_delay_sample_count,
    )


def calculate_prediction_accuracy(
    *, prediction_correctness_by_client: tuple[tuple[bool, ...], ...]
) -> float:
    """全clientの全標本の、正解の割合（標本がなければ0）。"""
    _validate_prediction_correctness_by_client(
        prediction_correctness_by_client=prediction_correctness_by_client
    )
    sample_count = sum(
        len(prediction_correctness) for prediction_correctness in prediction_correctness_by_client
    )
    if sample_count == 0:
        return 0.0
    correct_sample_count = sum(
        sum(prediction_correctness) for prediction_correctness in prediction_correctness_by_client
    )
    return correct_sample_count / sample_count


def calculate_stable_period_prediction_accuracy(
    *,
    prediction_correctness_by_client: tuple[tuple[bool, ...], ...],
    concept_change_sample_indices_by_client: tuple[tuple[int, ...], ...],
    recovery_window_sample_count: int,
) -> float:
    """各変更位置の直後の回復の窓を除いた、正解の割合（残る標本がなければNaN）。

    標本位置以下の最も近い変更位置から、窓の件数より手前にある標本を除く。
    最初の変更位置より前の標本は含める。
    """
    _validate_prediction_correctness_by_client(
        prediction_correctness_by_client=prediction_correctness_by_client
    )
    _validate_sequences_by_client(
        sequences_by_client=concept_change_sample_indices_by_client,
        parameter_name="concept_change_sample_indices_by_client",
    )
    _validate_nonnegative_count(
        specified_value=recovery_window_sample_count,
        parameter_name="recovery_window_sample_count",
    )
    if len(prediction_correctness_by_client) != len(concept_change_sample_indices_by_client):
        raise ValueError(
            "prediction correctness and concept change indices must have the same client count"
        )
    stable_sample_count = 0
    correct_stable_sample_count = 0
    for prediction_correctness, concept_change_sample_indices in zip(
        prediction_correctness_by_client, concept_change_sample_indices_by_client, strict=True
    ):
        ordered_concept_change_sample_indices = sorted(concept_change_sample_indices)
        for sample_index, prediction_is_correct in enumerate(prediction_correctness):
            latest_change_position = (
                bisect_right(ordered_concept_change_sample_indices, sample_index) - 1
            )
            if (
                latest_change_position >= 0
                and sample_index
                < ordered_concept_change_sample_indices[latest_change_position]
                + recovery_window_sample_count
            ):
                continue
            stable_sample_count += 1
            correct_stable_sample_count += prediction_is_correct
    if stable_sample_count == 0:
        return float("nan")
    return correct_stable_sample_count / stable_sample_count


def calculate_detection_metrics(
    *,
    concept_change_sample_indices_by_client: tuple[tuple[int, ...], ...],
    detection_sample_indices_by_client: tuple[tuple[int, ...], ...],
    maximum_delay_sample_count: int,
) -> DetectionMetrics:
    """clientごとに、検出位置を昇順にして変更位置と対応づけ、全clientの合計で指標を計算する。"""
    _validate_sequences_by_client(
        sequences_by_client=concept_change_sample_indices_by_client,
        parameter_name="concept_change_sample_indices_by_client",
    )
    _validate_sequences_by_client(
        sequences_by_client=detection_sample_indices_by_client,
        parameter_name="detection_sample_indices_by_client",
    )
    _validate_nonnegative_count(
        specified_value=maximum_delay_sample_count, parameter_name="maximum_delay_sample_count"
    )
    if len(concept_change_sample_indices_by_client) != len(detection_sample_indices_by_client):
        raise ValueError(
            "concept change indices and detection indices must have the same client count"
        )
    detection_count = 0
    concept_change_count = 0
    matched_detection_count = 0
    for concept_change_sample_indices, detection_sample_indices in zip(
        concept_change_sample_indices_by_client, detection_sample_indices_by_client, strict=True
    ):
        event_match_counts = _count_matched_events(
            concept_change_sample_indices=concept_change_sample_indices,
            event_sample_indices=tuple(sorted(detection_sample_indices)),
            maximum_delay_sample_count=maximum_delay_sample_count,
        )
        detection_count += len(detection_sample_indices)
        concept_change_count += len(concept_change_sample_indices)
        matched_detection_count += event_match_counts.matched_event_count
    detection_precision = matched_detection_count / detection_count if detection_count else 0.0
    detection_recall = (
        matched_detection_count / concept_change_count if concept_change_count else 0.0
    )
    detection_f1 = (
        2 * detection_precision * detection_recall / (detection_precision + detection_recall)
        if detection_precision + detection_recall > 0
        else 0.0
    )
    return DetectionMetrics(
        detection_precision=detection_precision,
        detection_recall=detection_recall,
        detection_f1=detection_f1,
        detection_count=detection_count,
        concept_change_count=concept_change_count,
        matched_detection_count=matched_detection_count,
    )
