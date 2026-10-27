"""残る全構成の、候補の採否と共有部の値と、小さい条件の実行。"""

import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.runtime.fedavg_run_settings import FedavgRunSettings
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
    build_experiment_run_settings,
)

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


@pytest.mark.parametrize("configuration_name", fedsda_configuration_names())
def test_fedsda_configurations_have_the_expected_candidate_values_and_run(configuration_name):
    """FedSDAの全構成の、候補の採否の方針と共有部の値。小さい条件の実行が、例外を出さない。"""
    run_settings = build_experiment_run_settings(
        configuration_name=configuration_name, experiment_run_conditions=_CONDITIONS
    )
    assert isinstance(run_settings, FedsdaRunSettings)
    assert candidate_values(run_settings) == _EXPECTED_CANDIDATE_VALUES_BY_CONFIGURATION_NAME.get(
        configuration_name, (WITHOUT_VALIDATION, FIXED)
    )
    execute_fedsda_experiment_run(run_settings=run_settings)
