"""統合で、同じ共有部を平均せずに保つ扱い（`consolidated_shared_feature_policy`）。

集約は全グローバルモデルへ同じ共有部を入れるので、統合のメンバーの共有部はビット単位で同じ。同じ値を件数で重み付けて
平均すると、数学的には同じ値だが、浮動小数点では変わりうる。新しい扱いは、そのときに代表の共有部をそのまま使う。
"""

from dataclasses import replace

import pytest
import torch
from behavior_golden_cases import build_case_settings

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.runtime.model_clustering_and_consolidation import (
    _compute_sample_weighted_mean_parameters,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    parameters_have_the_same_bits,
)

SHARED = "feature_extractor.hidden_layers.0.weight"
CONCEPT = "classification_layer.weight"


def parameters(shared_value, concept_value):
    return {
        SHARED: torch.full((2, 3), shared_value, dtype=torch.float32),
        CONCEPT: torch.full((1, 2), concept_value, dtype=torch.float32),
    }


def drifting_shared_value_and_weights():
    """同じ値の加重平均が、浮動小数点でビット単位に変わる値と件数（testが空振りでないように探す）。"""
    for shared_value in (0.1, 0.3, 0.7, 1.1, 0.123456):
        for weights in ((1, 2), (3, 4), (2, 5), (7, 3)):
            snapshots = {3: parameters(shared_value, 1.0), 5: parameters(shared_value, 2.0)}
            legacy, _ = _compute_sample_weighted_mean_parameters(
                model_cluster=(3, 5),
                parameter_snapshots_by_model_id=snapshots,
                aggregated_training_sample_counts_by_model_id=dict(zip((3, 5), weights)),
                keeps_identical_shared_features=False,
            )
            if not torch.equal(legacy[SHARED], snapshots[3][SHARED]):
                return shared_value, weights
    raise AssertionError("浮動小数点で変わる例が見つからない")


def test_identical_shared_parts_are_kept_and_concept_parts_are_averaged_as_before():
    shared_value, weights = drifting_shared_value_and_weights()
    snapshots = {3: parameters(shared_value, 1.0), 5: parameters(shared_value, 2.0)}
    counts = dict(zip((3, 5), weights))
    legacy, legacy_count = _compute_sample_weighted_mean_parameters(
        model_cluster=(3, 5),
        parameter_snapshots_by_model_id=snapshots,
        aggregated_training_sample_counts_by_model_id=counts,
        keeps_identical_shared_features=False,
    )
    kept, kept_count = _compute_sample_weighted_mean_parameters(
        model_cluster=(3, 5),
        parameter_snapshots_by_model_id=snapshots,
        aggregated_training_sample_counts_by_model_id=counts,
        keeps_identical_shared_features=True,
    )
    # 旧の扱いでは、共有部がビット単位で変わる（上で選んだ例）。新しい扱いでは、代表の値のまま。
    assert not torch.equal(legacy[SHARED], snapshots[3][SHARED])
    assert parameters_have_the_same_bits({SHARED: kept[SHARED]}, {SHARED: snapshots[3][SHARED]})
    assert kept[SHARED] is not snapshots[3][SHARED]
    # 概念固有部は、旧の扱いと同じ値。計算回数は、概念固有部だけ。
    assert parameters_have_the_same_bits({CONCEPT: kept[CONCEPT]}, {CONCEPT: legacy[CONCEPT]})
    assert list(kept) == list(legacy)
    assert legacy_count == 2 * (6 + 2)
    assert kept_count == 2 * 2


@pytest.mark.parametrize(
    "snapshots,counts,kept",
    [
        # 共有部が違う: 旧の扱いと同じ。
        ({3: parameters(1.0, 1.0), 5: parameters(1.5, 2.0)}, {3: 1, 5: 2}, False),
        # 代表の件数が0でも、正の件数のメンバーの共有部が代表と同じなら、保つ。
        (
            {3: parameters(0.3, 1.0), 5: parameters(0.3, 2.0), 7: parameters(0.3, 3.0)},
            {3: 0, 5: 3, 7: 4},
            True,
        ),
        # 代表の件数が0で、ほかのメンバーと違う共有部: 旧の扱いと同じ。
        (
            {3: parameters(9.0, 1.0), 5: parameters(0.3, 2.0), 7: parameters(0.3, 3.0)},
            {3: 0, 5: 3, 7: 4},
            False,
        ),
    ],
)
def test_shared_part_is_kept_only_when_contributing_members_match_the_representative(
    snapshots, counts, kept
):
    cluster = tuple(sorted(snapshots))
    legacy, legacy_count = _compute_sample_weighted_mean_parameters(
        model_cluster=cluster,
        parameter_snapshots_by_model_id=snapshots,
        aggregated_training_sample_counts_by_model_id=counts,
        keeps_identical_shared_features=False,
    )
    result, count = _compute_sample_weighted_mean_parameters(
        model_cluster=cluster,
        parameter_snapshots_by_model_id=snapshots,
        aggregated_training_sample_counts_by_model_id=counts,
        keeps_identical_shared_features=True,
    )
    assert parameters_have_the_same_bits({CONCEPT: result[CONCEPT]}, {CONCEPT: legacy[CONCEPT]})
    expected_shared = snapshots[cluster[0]][SHARED] if kept else legacy[SHARED]
    assert parameters_have_the_same_bits({SHARED: result[SHARED]}, {SHARED: expected_shared})
    contributing = sum(1 for model_id in cluster if counts[model_id] > 0)
    assert count == (contributing * 2 if kept else legacy_count)


def test_all_zero_weight_cluster_and_empty_shared_part_are_unchanged():
    zero = {3: parameters(1.0, 1.0), 5: parameters(2.0, 2.0)}
    independent = {
        3: {
            "model_owned_hidden_layers.0.weight": torch.full((2, 2), 0.3),
            CONCEPT: torch.full((1, 2), 1.0),
        },
        5: {
            "model_owned_hidden_layers.0.weight": torch.full((2, 2), 0.7),
            CONCEPT: torch.full((1, 2), 2.0),
        },
    }
    for snapshots, counts in ((zero, {3: 0, 5: 0}), (independent, {3: 1, 5: 2})):
        legacy = _compute_sample_weighted_mean_parameters(
            model_cluster=(3, 5),
            parameter_snapshots_by_model_id=snapshots,
            aggregated_training_sample_counts_by_model_id=counts,
            keeps_identical_shared_features=False,
        )
        kept = _compute_sample_weighted_mean_parameters(
            model_cluster=(3, 5),
            parameter_snapshots_by_model_id=snapshots,
            aggregated_training_sample_counts_by_model_id=counts,
            keeps_identical_shared_features=True,
        )
        assert parameters_have_the_same_bits(kept[0], legacy[0])
        assert kept[1] == legacy[1]


def test_parameters_have_the_same_bits_compares_names_dtypes_shapes_and_bytes():
    base = {SHARED: torch.tensor([0.0, float("nan")])}
    assert parameters_have_the_same_bits(base, {SHARED: torch.tensor([0.0, float("nan")])})
    assert not parameters_have_the_same_bits(base, {SHARED: torch.tensor([-0.0, float("nan")])})
    assert not parameters_have_the_same_bits(base, {"other": torch.tensor([0.0, float("nan")])})
    assert not parameters_have_the_same_bits(
        base, {SHARED: torch.tensor([0.0, float("nan")], dtype=torch.float64)}
    )
    assert not parameters_have_the_same_bits(base, {SHARED: torch.tensor([[0.0, float("nan")]])})
    assert parameters_have_the_same_bits({}, {})


@pytest.mark.parametrize("invalid_value", ["keep", None, True])
def test_invalid_consolidated_shared_feature_policy_is_rejected(invalid_value):
    run_settings = build_case_settings("fedsda", 8)
    with pytest.raises(RunSettingsValidationError) as error_information:
        replace(
            run_settings,
            run_participant_settings=replace(
                run_settings.run_participant_settings,
                consolidated_shared_feature_policy=invalid_value,
            ),
        )
    assert (
        error_information.value.configuration_parameter_name == "consolidated_shared_feature_policy"
    )
