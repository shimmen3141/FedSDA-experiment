"""手法に依存しない、予測の診断の計算（区間ごとの正解の割合、クラスごとの正解の割合の、クラスの間のまとめ）。"""

from dataclasses import dataclass, fields

from numpy import mean, std


@dataclass(frozen=True, kw_only=True)
class StableAndRecoveryPeriodAccuracies:
    """概念の変更の後の回復の区間と、それ以外の定常の区間の、正解の割合。その区間に標本がなければ0。"""

    stable_period_accuracy: float
    recovery_period_accuracy: float


@dataclass(frozen=True, kw_only=True)
class PredictionCorrectCounts:
    """予測した標本数と、予測の種類ごとの、正解の数。

    「実際の予測」は、予測重みで全保有モデルの出力を結合した予測。「どれかのモデルか実際の予測」は、
    保有モデルのどれか、または、実際の予測が正解した標本の数（ほかの3つの正解は、これに含まれる）。
    """

    predicted_sample_count: int
    combined_prediction_correct_count: int
    any_model_or_combined_prediction_correct_count: int
    maximum_weight_model_prediction_correct_count: int
    highest_confidence_model_prediction_correct_count: int

    def __post_init__(self) -> None:
        for count_field in fields(self):
            count = getattr(self, count_field.name)
            if type(count) is not int:
                raise TypeError(f"{count_field.name} must be builtin int")
            if count < 0:
                raise ValueError(f"{count_field.name} must be nonnegative")
        if self.any_model_or_combined_prediction_correct_count > self.predicted_sample_count:
            raise ValueError(
                "any_model_or_combined_prediction_correct_count must not exceed predicted_sample_count"
            )
        for included_count_name in (
            "combined_prediction_correct_count",
            "maximum_weight_model_prediction_correct_count",
            "highest_confidence_model_prediction_correct_count",
        ):
            if (
                getattr(self, included_count_name)
                > self.any_model_or_combined_prediction_correct_count
            ):
                raise ValueError(
                    f"{included_count_name} must not exceed "
                    "any_model_or_combined_prediction_correct_count"
                )


@dataclass(frozen=True, kw_only=True)
class ClassMacroPredictionMetrics:
    """クラスごとの正解の割合の、クラスの間のまとめ（標本のあるクラスだけ。標本数で重みづけしない）。

    差（gap）は、クラスごとの、「どれかのモデルか実際の予測」の正解の割合−実際の予測の正解の割合。
    比（ratio）は、クラスごとの、実際の予測の正解の数÷「どれかのモデルか実際の予測」の正解の数
    （後者が0のクラスは、入れない）。値がなければ0。標準偏差は、母標準偏差。
    """

    macro_combined_prediction_accuracy: float
    macro_any_model_or_combined_prediction_accuracy: float
    macro_maximum_weight_model_prediction_accuracy: float
    macro_highest_confidence_model_prediction_accuracy: float
    mean_any_model_or_combined_accuracy_gap: float
    any_model_or_combined_accuracy_gap_standard_deviation: float
    mean_combined_to_any_model_or_combined_correct_ratio: float
    minimum_combined_to_any_model_or_combined_correct_ratio: float


def _validate_flags_by_client(*, flags_by_client: object, parameter_name: str) -> None:
    if type(flags_by_client) is not tuple:
        raise TypeError(f"{parameter_name} must be exact tuple")
    for flags in flags_by_client:
        if type(flags) is not tuple:
            raise TypeError(f"{parameter_name} must hold exact tuple")
        for flag in flags:
            if type(flag) is not bool:
                raise TypeError(f"{parameter_name} must hold builtin bool")


def calculate_stable_and_recovery_period_accuracies(
    *,
    prediction_correctness_by_client: tuple[tuple[bool, ...], ...],
    concept_change_sample_indices_by_client: tuple[tuple[int, ...], ...],
    recovery_window_sample_count: int,
    sample_inclusion_by_client: tuple[tuple[bool, ...], ...] | None = None,
) -> StableAndRecoveryPeriodAccuracies:
    """回復の区間（各変更位置から、窓の件数の標本）と、それ以外の定常の区間の、正解の割合を返す。

    変更位置は、標本の列より後にあってもよい（その区間は、標本の列の範囲に切り詰める）。
    `sample_inclusion_by_client`を与えたときは、真の標本だけを数える。
    """
    _validate_flags_by_client(
        flags_by_client=prediction_correctness_by_client,
        parameter_name="prediction_correctness_by_client",
    )
    if type(concept_change_sample_indices_by_client) is not tuple:
        raise TypeError("concept_change_sample_indices_by_client must be exact tuple")
    for concept_change_sample_indices in concept_change_sample_indices_by_client:
        if type(concept_change_sample_indices) is not tuple:
            raise TypeError("concept_change_sample_indices_by_client must hold exact tuple")
        for concept_change_sample_index in concept_change_sample_indices:
            if type(concept_change_sample_index) is not int:
                raise TypeError("concept change sample indices must be builtin int")
            if concept_change_sample_index < 0:
                raise ValueError("concept change sample indices must be nonnegative")
    if type(recovery_window_sample_count) is not int:
        raise TypeError("recovery_window_sample_count must be builtin int")
    if recovery_window_sample_count < 0:
        raise ValueError("recovery_window_sample_count must be nonnegative")
    if len(prediction_correctness_by_client) != len(concept_change_sample_indices_by_client):
        raise ValueError(
            "prediction correctness and concept change indices must have the same client count"
        )
    if sample_inclusion_by_client is not None:
        _validate_flags_by_client(
            flags_by_client=sample_inclusion_by_client,
            parameter_name="sample_inclusion_by_client",
        )
        if len(sample_inclusion_by_client) != len(prediction_correctness_by_client):
            raise ValueError(
                "sample inclusion and prediction correctness must have the same client count"
            )
        for sample_inclusion, prediction_correctness in zip(
            sample_inclusion_by_client, prediction_correctness_by_client, strict=True
        ):
            if len(sample_inclusion) != len(prediction_correctness):
                raise ValueError(
                    "sample inclusion and prediction correctness must have the same sample count"
                )
    stable_sample_count = 0
    stable_correct_count = 0
    recovery_sample_count = 0
    recovery_correct_count = 0
    for client_position, prediction_correctness in enumerate(prediction_correctness_by_client):
        sample_is_in_recovery_period = [False] * len(prediction_correctness)
        for concept_change_sample_index in concept_change_sample_indices_by_client[client_position]:
            for sample_index in range(
                concept_change_sample_index,
                min(
                    len(prediction_correctness),
                    concept_change_sample_index + recovery_window_sample_count,
                ),
            ):
                sample_is_in_recovery_period[sample_index] = True
        for sample_index, prediction_is_correct in enumerate(prediction_correctness):
            if (
                sample_inclusion_by_client is not None
                and not sample_inclusion_by_client[client_position][sample_index]
            ):
                continue
            if sample_is_in_recovery_period[sample_index]:
                recovery_sample_count += 1
                recovery_correct_count += int(prediction_is_correct)
            else:
                stable_sample_count += 1
                stable_correct_count += int(prediction_is_correct)
    return StableAndRecoveryPeriodAccuracies(
        stable_period_accuracy=(
            stable_correct_count / stable_sample_count if stable_sample_count else 0.0
        ),
        recovery_period_accuracy=(
            recovery_correct_count / recovery_sample_count if recovery_sample_count else 0.0
        ),
    )


def _mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def summarize_class_prediction_metrics(
    *, class_prediction_correct_counts: tuple[PredictionCorrectCounts, ...]
) -> ClassMacroPredictionMetrics:
    """クラスごとの数（与えられた順）から、クラスの間のまとめを作る。標本のないクラスは、数えない。"""
    if type(class_prediction_correct_counts) is not tuple:
        raise TypeError("class_prediction_correct_counts must be exact tuple")
    combined_accuracies: list[float] = []
    any_model_or_combined_accuracies: list[float] = []
    maximum_weight_model_accuracies: list[float] = []
    highest_confidence_model_accuracies: list[float] = []
    accuracy_gaps: list[float] = []
    correct_ratios: list[float] = []
    for prediction_correct_counts in class_prediction_correct_counts:
        if type(prediction_correct_counts) is not PredictionCorrectCounts:
            raise TypeError(
                "class_prediction_correct_counts must hold exact PredictionCorrectCounts"
            )
        # 手で壊した数も、値を使う前に検査し直す。
        prediction_correct_counts.__post_init__()
        sample_count = prediction_correct_counts.predicted_sample_count
        if not sample_count:
            continue
        any_model_or_combined_correct_count = (
            prediction_correct_counts.any_model_or_combined_prediction_correct_count
        )
        any_model_or_combined_accuracy = any_model_or_combined_correct_count / sample_count
        combined_accuracy = (
            prediction_correct_counts.combined_prediction_correct_count / sample_count
        )
        combined_accuracies.append(combined_accuracy)
        any_model_or_combined_accuracies.append(any_model_or_combined_accuracy)
        maximum_weight_model_accuracies.append(
            prediction_correct_counts.maximum_weight_model_prediction_correct_count / sample_count
        )
        highest_confidence_model_accuracies.append(
            prediction_correct_counts.highest_confidence_model_prediction_correct_count
            / sample_count
        )
        accuracy_gaps.append(any_model_or_combined_accuracy - combined_accuracy)
        if any_model_or_combined_correct_count:
            correct_ratios.append(
                prediction_correct_counts.combined_prediction_correct_count
                / any_model_or_combined_correct_count
            )
    return ClassMacroPredictionMetrics(
        macro_combined_prediction_accuracy=_mean_or_zero(combined_accuracies),
        macro_any_model_or_combined_prediction_accuracy=_mean_or_zero(
            any_model_or_combined_accuracies
        ),
        macro_maximum_weight_model_prediction_accuracy=_mean_or_zero(
            maximum_weight_model_accuracies
        ),
        macro_highest_confidence_model_prediction_accuracy=_mean_or_zero(
            highest_confidence_model_accuracies
        ),
        mean_any_model_or_combined_accuracy_gap=_mean_or_zero(accuracy_gaps),
        any_model_or_combined_accuracy_gap_standard_deviation=(
            float(std(accuracy_gaps)) if accuracy_gaps else 0.0
        ),
        mean_combined_to_any_model_or_combined_correct_ratio=_mean_or_zero(correct_ratios),
        minimum_combined_to_any_model_or_combined_correct_ratio=(
            min(correct_ratios) if correct_ratios else 0.0
        ),
    )
