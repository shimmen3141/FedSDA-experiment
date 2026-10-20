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


@pytest.mark.parametrize("keeps", [False, True])
def test_parameter_order_follows_the_first_contributing_member(keeps):
    """代表の件数が0で、メンバーの名前の順が違っても、結果の名前の順は、件数が正の最初のメンバーの順（旧の扱いと同じ）。"""
    representative = {CONCEPT: torch.full((1, 2), 1.0), SHARED: torch.full((2, 3), 0.3)}
    member = {SHARED: torch.full((2, 3), 0.3), CONCEPT: torch.full((1, 2), 2.0)}
    result, _ = _compute_sample_weighted_mean_parameters(
        model_cluster=(3, 5),
        parameter_snapshots_by_model_id={3: representative, 5: member},
        aggregated_training_sample_counts_by_model_id={3: 0, 5: 2},
        keeps_identical_shared_features=keeps,
    )
    assert list(result) == [SHARED, CONCEPT]


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


def count_consolidations_and_shared_changes(run_settings):
    """統合の後の送り直しの記録を包み、統合が起きた回数と、共有部が先に送った値とビット単位で違った回数を数える。"""
    from unittest.mock import patch

    import federated_learning_experiments.runtime.server_round_synchronization as round_module
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
        split_shared_and_concept_specific_parameters,
    )

    counts = dict(consolidations=0, shared_changes=0)
    original = round_module._record_consolidated_model_resend_volume

    def wrapped(**arguments):
        if arguments["model_id_mapping"]:
            counts["consolidations"] += 1
            repository = arguments["global_model_repository"]
            shared = split_shared_and_concept_specific_parameters(
                parameter_snapshot=repository.get_global_model_parameters(
                    model_id=repository.global_model_ids[0]
                )
            )[0]
            if not parameters_have_the_same_bits(
                shared, arguments["early_distributed_shared_parameters"]
            ):
                counts["shared_changes"] += 1
        return original(**arguments)

    with patch.object(round_module, "_record_consolidated_model_resend_volume", wrapped):
        execute_fedsda_experiment_run(run_settings=run_settings)
    return counts


@pytest.mark.slow
def test_keeping_identical_shared_features_removes_shared_resends():
    """統合が起きる全体runで、新しい扱いでは共有部の送り直しが0回、旧の扱いでは1回以上。"""
    # goldenのケースの条件で、seedだけを1にする（seed 0では、旧の扱いでも統合の後の共有部が変わらない。
    # 2026-10-20の下調べで、seed 1は統合2回とも、旧の扱いで共有部が変わった）。
    case_settings = build_case_settings("fedsda-sine2-adoption-without-validation", 8)
    run_settings = replace(
        case_settings,
        execution_settings=replace(
            case_settings.execution_settings,
            experiment_run_conditions=replace(
                case_settings.execution_settings.experiment_run_conditions, random_seed=1
            ),
        ),
    )
    results = {}
    for policy in (
        "average_shared_features_with_concept_specific_parts",
        "keep_identical_shared_features",
    ):
        results[policy] = count_consolidations_and_shared_changes(
            replace(
                run_settings,
                run_participant_settings=replace(
                    run_settings.run_participant_settings,
                    consolidated_shared_feature_policy=policy,
                ),
            )
        )
    legacy = results["average_shared_features_with_concept_specific_parts"]
    kept = results["keep_identical_shared_features"]
    assert legacy["consolidations"] > 0 and kept["consolidations"] > 0, results
    assert legacy["shared_changes"] > 0, results
    assert kept["shared_changes"] == 0, results


def test_default_and_all_fedsda_configurations_keep_identical_shared_features():
    """最終構成の既定と、FedSDAの全部の構成は、同じ共有部を保つ扱い。"""
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
        build_experiment_run_settings,
    )

    conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=2,
        per_client_sample_count=40,
        server_aggregation_interval_per_client_samples=10,
    )
    fedsda_count = 0
    for name in EXPERIMENT_CONFIGURATION_NAMES:
        settings = build_experiment_run_settings(
            configuration_name=name, experiment_run_conditions=conditions
        )
        if type(settings) is not FedsdaRunSettings:
            continue
        fedsda_count += 1
        assert (
            settings.run_participant_settings.consolidated_shared_feature_policy
            == "keep_identical_shared_features"
        ), name
    assert fedsda_count > 1
