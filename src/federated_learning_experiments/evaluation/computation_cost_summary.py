"""手法の比較に使う計算量のまとめ（積和演算の数の合計、内訳、標本あたり・保有モデル×標本あたり）。"""

from dataclasses import dataclass, fields

from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
)


def _validate_nonnegative_count(*, count: int, count_name: str) -> None:
    if type(count) is not int:
        raise TypeError(f"{count_name} must be builtin int")
    if count < 0:
        raise ValueError(f"{count_name} must be nonnegative")


@dataclass(frozen=True, kw_only=True)
class ServerComputationCounts:
    """サーバの、パラメータの積和演算の数（値1つの「重みを掛けて足す」「2乗して足す」を1と数える）。

    集約と統合は、手法の処理。パラメータ距離は、診断のための計算で、判定には使わない。
    """

    aggregation_parameter_multiply_accumulate_count: int
    consolidation_parameter_multiply_accumulate_count: int
    diagnostic_parameter_distance_multiply_accumulate_count: int

    def __post_init__(self) -> None:
        for count_field in fields(self):
            _validate_nonnegative_count(
                count=getattr(self, count_field.name), count_name=count_field.name
            )


@dataclass(frozen=True, kw_only=True)
class ComputationCostSummary:
    """run 1回の計算量のまとめ。単位は、積和演算の数。

    `processed_sample_count`は、全clientの、処理した標本数の合計。`held_model_sample_count`は、
    全client・全標本の、その時点の保有モデル数の合計（保有モデル×標本の延べ数）。
    clientの値は、モデルの計算（全結合層）。`forward`は順伝播（推論＋学習）の実測、
    `estimated_backward`は逆伝播の見積り。サーバの値は、集約と統合（診断のパラメータ距離は別）。
    比は、分母が0のとき、NaN。

    共有部の計算は、標本ごとに1回で、保有モデル数に依らない。概念固有部の計算は、保有モデルごとに
    行う。このため、「保有モデル×標本あたり」の値は、全体（共有部を含む）のものと、概念固有部だけの
    ものを、別に持つ。全体の値は、保有モデルが増えると、共有部の分が薄まって小さくなる。
    """

    processed_sample_count: int
    held_model_sample_count: int
    mean_held_model_count: float
    client_forward_multiply_accumulate_count: int
    client_estimated_backward_multiply_accumulate_count: int
    client_shared_part_forward_multiply_accumulate_count: int
    client_concept_specific_part_forward_multiply_accumulate_count: int
    server_multiply_accumulate_count: int
    server_diagnostic_multiply_accumulate_count: int
    client_forward_multiply_accumulate_count_per_processed_sample: float
    client_forward_multiply_accumulate_count_per_held_model_sample: float
    client_forward_and_backward_multiply_accumulate_count_per_processed_sample: float
    client_forward_and_backward_multiply_accumulate_count_per_held_model_sample: float
    # 共有部は標本あたり、概念固有部は保有モデル×標本あたり（順伝播）。
    client_shared_part_forward_multiply_accumulate_count_per_processed_sample: float
    client_concept_specific_part_forward_multiply_accumulate_count_per_held_model_sample: float


def _divide_or_nan(*, numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def summarize_computation_cost(
    *,
    model_computation_counts: ModelComputationCounts,
    server_computation_counts: ServerComputationCounts,
    processed_sample_count: int,
    held_model_sample_count: int,
) -> ComputationCostSummary:
    """計数から、計算量のまとめを作る。"""
    if type(model_computation_counts) is not ModelComputationCounts:
        raise TypeError("model_computation_counts must be exact ModelComputationCounts")
    if type(server_computation_counts) is not ServerComputationCounts:
        raise TypeError("server_computation_counts must be exact ServerComputationCounts")
    # 手で壊した計数も、値を使う前に検査し直す。
    server_computation_counts.__post_init__()
    for count_field in fields(ModelComputationCounts):
        _validate_nonnegative_count(
            count=getattr(model_computation_counts, count_field.name), count_name=count_field.name
        )
    _validate_nonnegative_count(count=processed_sample_count, count_name="processed_sample_count")
    _validate_nonnegative_count(count=held_model_sample_count, count_name="held_model_sample_count")
    shared_part_forward_count = (
        model_computation_counts.shared_part_training_forward_multiply_accumulate_count
        + model_computation_counts.shared_part_inference_forward_multiply_accumulate_count
    )
    concept_specific_part_forward_count = (
        model_computation_counts.concept_specific_part_training_forward_multiply_accumulate_count
        + model_computation_counts.concept_specific_part_inference_forward_multiply_accumulate_count
    )
    forward_count = shared_part_forward_count + concept_specific_part_forward_count
    estimated_backward_count = (
        model_computation_counts.shared_part_estimated_backward_multiply_accumulate_count
        + model_computation_counts.concept_specific_part_estimated_backward_multiply_accumulate_count
    )
    return ComputationCostSummary(
        processed_sample_count=processed_sample_count,
        held_model_sample_count=held_model_sample_count,
        mean_held_model_count=_divide_or_nan(
            numerator=held_model_sample_count, denominator=processed_sample_count
        ),
        client_forward_multiply_accumulate_count=forward_count,
        client_estimated_backward_multiply_accumulate_count=estimated_backward_count,
        client_shared_part_forward_multiply_accumulate_count=shared_part_forward_count,
        client_concept_specific_part_forward_multiply_accumulate_count=(
            concept_specific_part_forward_count
        ),
        server_multiply_accumulate_count=(
            server_computation_counts.aggregation_parameter_multiply_accumulate_count
            + server_computation_counts.consolidation_parameter_multiply_accumulate_count
        ),
        server_diagnostic_multiply_accumulate_count=(
            server_computation_counts.diagnostic_parameter_distance_multiply_accumulate_count
        ),
        client_forward_multiply_accumulate_count_per_processed_sample=_divide_or_nan(
            numerator=forward_count, denominator=processed_sample_count
        ),
        client_forward_multiply_accumulate_count_per_held_model_sample=_divide_or_nan(
            numerator=forward_count, denominator=held_model_sample_count
        ),
        client_forward_and_backward_multiply_accumulate_count_per_processed_sample=_divide_or_nan(
            numerator=forward_count + estimated_backward_count, denominator=processed_sample_count
        ),
        client_forward_and_backward_multiply_accumulate_count_per_held_model_sample=_divide_or_nan(
            numerator=forward_count + estimated_backward_count,
            denominator=held_model_sample_count,
        ),
        client_shared_part_forward_multiply_accumulate_count_per_processed_sample=_divide_or_nan(
            numerator=shared_part_forward_count, denominator=processed_sample_count
        ),
        client_concept_specific_part_forward_multiply_accumulate_count_per_held_model_sample=(
            _divide_or_nan(
                numerator=concept_specific_part_forward_count, denominator=held_model_sample_count
            )
        ),
    )
