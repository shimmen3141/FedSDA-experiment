"""構成`fedsda-with-candidate-validation`（2026-10-19までの既定）と、構成ごとの、候補の採否と共有部の値。

2026-10-20から、提案構成は、候補を検証せずに採用し、候補の学習では共有部を固定する。この構成は、
提案構成から、その2つの値だけを、これまでの既定（候補検証あり、共有部も更新する）へ戻す。
"""

import pytest
from test_fedsda_final_configuration_run_settings import collect_settings_types

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.runtime.fedavg_run_settings import FedavgRunSettings
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    CANDIDATE_VALIDATION_CONFIGURATION_NAME,
    EXPERIMENT_CONFIGURATION_NAMES,
    build_experiment_run_settings,
)

VALIDATION = "current_model_first_reuse_then_two_segment_candidate_validation"
WITHOUT_VALIDATION = "current_model_first_reuse_then_candidate_adoption_without_validation"
UPDATE = "update_shared_features_with_candidate"
FIXED = "keep_shared_features_fixed"
_CONDITIONS = ExperimentRunConditions(
    dataset_name="sine2",
    random_seed=0,
    client_count=2,
    per_client_sample_count=60,
    server_aggregation_interval_per_client_samples=10,
)
# 構成ごとの、候補の採否の方針と、候補の共有部の更新（ほかのFedSDAの構成は、提案構成の値を引き継ぐ）。
_EXPECTED_CANDIDATE_VALUES_BY_CONFIGURATION_NAME = {
    "fedsda-with-candidate-validation": (VALIDATION, UPDATE),
    "fedsda-without-candidate-validation": (WITHOUT_VALIDATION, UPDATE),
}


def fedsda_configuration_names():
    return [
        name
        for name in EXPERIMENT_CONFIGURATION_NAMES
        if type(
            build_experiment_run_settings(
                configuration_name=name, experiment_run_conditions=_CONDITIONS
            )
        )
        not in (FedavgRunSettings, FeddriftRunSettings)
    ]


def candidate_values(run_settings):
    client = run_settings.run_participant_settings.run_client_settings
    return (
        client.candidate_model_training_and_acceptance_settings.candidate_model_acceptance_policy,
        client.candidate_epoch_training_settings.candidate_shared_feature_update_policy,
    )


def test_configuration_name_is_registered_last():
    assert CANDIDATE_VALIDATION_CONFIGURATION_NAME == "fedsda-with-candidate-validation"
    assert EXPERIMENT_CONFIGURATION_NAMES[-1] == CANDIDATE_VALIDATION_CONFIGURATION_NAME


@pytest.mark.parametrize("configuration_name", fedsda_configuration_names())
def test_fedsda_configurations_have_the_expected_candidate_values_and_run(configuration_name):
    """FedSDAの全構成の、候補の採否の方針と共有部の値。小さい条件の実行が、例外を出さない。"""
    run_settings = build_experiment_run_settings(
        configuration_name=configuration_name, experiment_run_conditions=_CONDITIONS
    )
    assert candidate_values(run_settings) == _EXPECTED_CANDIDATE_VALUES_BY_CONFIGURATION_NAME.get(
        configuration_name, (WITHOUT_VALIDATION, FIXED)
    )
    execute_fedsda_experiment_run(run_settings=run_settings)


def test_with_candidate_validation_differs_from_the_default_only_in_the_five_values():
    """既定と比べて、候補の採否の方針と共有部と、警報の後の選び方と、「新しいモデル」の仮説の学習者の組と、
    初期モデルの損失の統計の推定の仕方と、評価のclientの選び方の6つだけが違う（保存の形で比べる）。"""
    default_mapping = convert_settings_to_plain_mapping(
        build_experiment_run_settings(
            configuration_name="fedsda", experiment_run_conditions=_CONDITIONS
        )
    )
    with_validation = build_experiment_run_settings(
        configuration_name=CANDIDATE_VALIDATION_CONFIGURATION_NAME,
        experiment_run_conditions=_CONDITIONS,
    )
    assert candidate_values(with_validation) == (VALIDATION, UPDATE)
    with_validation_mapping = convert_settings_to_plain_mapping(with_validation)
    client_path = ("run_participant_settings", "run_client_settings")
    for mapping in (default_mapping, with_validation_mapping):
        client = mapping
        for name in client_path:
            client = client[name]
        del client["candidate_model_training_and_acceptance_settings"][
            "candidate_model_acceptance_policy"
        ]
        del client["candidate_epoch_training_settings"]["candidate_shared_feature_update_policy"]
        del client["training_data_assignment_settings"]["alarm_interval_assignment_policy"]
        del client["training_data_assignment_settings"]["new_model_hypothesis_learners"]
        del mapping["run_participant_settings"]["initial_model_loss_statistics_estimation"]
        assert mapping["run_participant_settings"].pop(
            "cross_evaluation_client_selection_policy"
        ) == (
            "clients_holding_target_model"
            if mapping is with_validation_mapping
            else "clients_with_target_model_samples"
        )
    assert with_validation_mapping == default_mapping
    assert (
        with_validation.run_participant_settings.run_client_settings.training_data_assignment_settings.new_model_hypothesis_learners,
        with_validation.run_participant_settings.initial_model_loss_statistics_estimation,
    ) == ("reinitialized_learner", "resubstitution")
    assert (
        with_validation.run_participant_settings.run_client_settings.training_data_assignment_settings.alarm_interval_assignment_policy
        == "detector_change_span_and_mean_loss_increase_tolerance"
    )
    assert collect_settings_types(with_validation) == collect_settings_types(
        build_experiment_run_settings(
            configuration_name="fedsda", experiment_run_conditions=_CONDITIONS
        )
    )
