"""計算量のまとめ: 手計算との照合、分母0のNaN、拒否。"""

from dataclasses import FrozenInstanceError, fields, replace
from math import isnan

import pytest

from federated_learning_experiments.evaluation.computation_cost_summary import (
    ComputationCostSummary,
    ServerComputationCounts,
    summarize_computation_cost,
)
from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
)

MODEL_COMPUTATION_COUNTS = ModelComputationCounts(
    shared_part_training_example_count=40,
    shared_part_inference_example_count=100,
    concept_specific_part_training_example_count=40,
    concept_specific_part_inference_example_count=250,
    shared_parameter_optimizer_step_count=4,
    concept_specific_parameter_optimizer_step_count=7,
    shared_part_training_forward_multiply_accumulate_count=4000,
    shared_part_inference_forward_multiply_accumulate_count=10000,
    concept_specific_part_training_forward_multiply_accumulate_count=800,
    concept_specific_part_inference_forward_multiply_accumulate_count=5000,
    shared_part_estimated_backward_multiply_accumulate_count=7000,
    concept_specific_part_estimated_backward_multiply_accumulate_count=1600,
)
SERVER_COMPUTATION_COUNTS = ServerComputationCounts(
    aggregation_parameter_multiply_accumulate_count=3000,
    consolidation_parameter_multiply_accumulate_count=600,
    diagnostic_parameter_distance_multiply_accumulate_count=90,
)
VALID_ARGUMENTS = dict(
    model_computation_counts=MODEL_COMPUTATION_COUNTS,
    server_computation_counts=SERVER_COMPUTATION_COUNTS,
    processed_sample_count=100,
    held_model_sample_count=250,
)


def test_summary_matches_hand_calculation():
    """合計、内訳、標本あたり・保有モデル×標本あたりの値が、手計算と一致する。"""
    summary = summarize_computation_cost(**VALID_ARGUMENTS)
    assert type(summary) is ComputationCostSummary
    # 順伝播: 共有部 4000+10000、概念固有部 800+5000。逆伝播の見積り: 7000+1600。
    assert summary == ComputationCostSummary(
        processed_sample_count=100,
        held_model_sample_count=250,
        mean_held_model_count=2.5,
        client_forward_multiply_accumulate_count=19800,
        client_estimated_backward_multiply_accumulate_count=8600,
        client_shared_part_forward_multiply_accumulate_count=14000,
        client_concept_specific_part_forward_multiply_accumulate_count=5800,
        # サーバ: 集約＋統合。診断のパラメータ距離は別。
        server_multiply_accumulate_count=3600,
        server_diagnostic_multiply_accumulate_count=90,
        client_forward_multiply_accumulate_count_per_processed_sample=198.0,
        client_forward_multiply_accumulate_count_per_held_model_sample=79.2,
        client_forward_and_backward_multiply_accumulate_count_per_processed_sample=284.0,
        client_forward_and_backward_multiply_accumulate_count_per_held_model_sample=113.6,
    )
    with pytest.raises(FrozenInstanceError):
        summary.processed_sample_count = 0
    # 標本数の計数と、optimizerの更新回数は、まとめの値を変えない。
    assert summary == summarize_computation_cost(
        **VALID_ARGUMENTS
        | dict(
            model_computation_counts=replace(
                MODEL_COMPUTATION_COUNTS,
                shared_part_training_example_count=0,
                concept_specific_part_inference_example_count=0,
                shared_parameter_optimizer_step_count=0,
            )
        )
    )


def test_ratios_are_nan_when_denominators_are_zero():
    """処理した標本がない・保有モデル数の合計が0のとき、その比は「なし」（NaN）。"""
    summary = summarize_computation_cost(
        **VALID_ARGUMENTS | dict(processed_sample_count=0, held_model_sample_count=0)
    )
    for ratio_name in (
        "mean_held_model_count",
        "client_forward_multiply_accumulate_count_per_processed_sample",
        "client_forward_multiply_accumulate_count_per_held_model_sample",
        "client_forward_and_backward_multiply_accumulate_count_per_processed_sample",
        "client_forward_and_backward_multiply_accumulate_count_per_held_model_sample",
    ):
        assert isnan(getattr(summary, ratio_name)), ratio_name
    assert summary.client_forward_multiply_accumulate_count == 19800
    # 片方だけ0。
    summary = summarize_computation_cost(**VALID_ARGUMENTS | dict(held_model_sample_count=0))
    assert summary.client_forward_multiply_accumulate_count_per_processed_sample == 198.0
    assert summary.mean_held_model_count == 0.0
    assert isnan(summary.client_forward_multiply_accumulate_count_per_held_model_sample)


@pytest.mark.parametrize(
    ("argument_name", "invalid_value", "expected_exception_type"),
    [
        ("model_computation_counts", None, TypeError),
        ("model_computation_counts", dict(), TypeError),
        (
            "model_computation_counts",
            replace(MODEL_COMPUTATION_COUNTS, shared_part_training_example_count=-1),
            ValueError,
        ),
        (
            "model_computation_counts",
            replace(
                MODEL_COMPUTATION_COUNTS,
                concept_specific_part_estimated_backward_multiply_accumulate_count=1.5,
            ),
            TypeError,
        ),
        ("server_computation_counts", None, TypeError),
        ("server_computation_counts", (3000, 600, 90), TypeError),
        ("processed_sample_count", True, TypeError),
        ("processed_sample_count", 100.0, TypeError),
        ("processed_sample_count", -1, ValueError),
        ("held_model_sample_count", None, TypeError),
        ("held_model_sample_count", -5, ValueError),
    ],
)
def test_summary_rejects_invalid_arguments(argument_name, invalid_value, expected_exception_type):
    with pytest.raises(expected_exception_type):
        summarize_computation_cost(**VALID_ARGUMENTS | {argument_name: invalid_value})


def test_summary_requires_keyword_arguments_and_server_counts_validate_fields():
    with pytest.raises(TypeError):
        summarize_computation_cost(MODEL_COMPUTATION_COUNTS, SERVER_COMPUTATION_COUNTS, 100, 250)
    for count_field in fields(ServerComputationCounts):
        for invalid_count, expected_exception_type in (
            (True, TypeError),
            (1.0, TypeError),
            (None, TypeError),
            (-1, ValueError),
        ):
            with pytest.raises(expected_exception_type):
                replace(SERVER_COMPUTATION_COUNTS, **{count_field.name: invalid_count})
    with pytest.raises(TypeError):
        ServerComputationCounts(3000, 600, 90)
    # 手で壊したサーバの計数も、拒否する。
    broken_server_counts = replace(SERVER_COMPUTATION_COUNTS)
    object.__setattr__(broken_server_counts, "aggregation_parameter_multiply_accumulate_count", -1)
    with pytest.raises(ValueError):
        summarize_computation_cost(
            **VALID_ARGUMENTS | dict(server_computation_counts=broken_server_counts)
        )
