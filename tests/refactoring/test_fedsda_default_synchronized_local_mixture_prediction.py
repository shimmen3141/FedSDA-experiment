"""FedSDAの既定を、混ぜる予測・T120にすること（fedsda-default-synchronized-local-mixture-prediction）を確認する。"""

from dataclasses import replace

import pytest
from test_fedsda_run_client import (
    assert_run_client_uses_legacy_compatible_policies,
    use_legacy_compatible_policies,
)

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
    build_experiment_run_settings,
)

_CONDITIONS = ExperimentRunConditions(
    dataset_name="sine2",
    random_seed=0,
    client_count=2,
    per_client_sample_count=40,
    server_aggregation_interval_per_client_samples=10,
)


def replace_prediction(run_settings, **changes):
    participant = run_settings.run_participant_settings
    client = participant.run_client_settings
    return replace(
        run_settings,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                client,
                prediction_combination_settings=replace(
                    client.prediction_combination_settings, **changes
                ),
            ),
        ),
    )


def run_clients(run_settings):
    run = execute_fedsda_experiment_run(run_settings=run_settings)
    return run.measured_run.participants.client_operations


def test_legacy_compatible_entry_fixes_the_prediction_settings():
    legacy_compatible = use_legacy_compatible_policies(
        build_experiment_run_settings(
            configuration_name="fedsda", experiment_run_conditions=_CONDITIONS
        )
    )
    client = legacy_compatible.run_participant_settings.run_client_settings
    assert client.prediction_combination_settings.prediction_model_parameter_source == (
        "current_local_parameters"
    )
    assert (
        client.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        == (client.training_data_assignment_settings.pending_assignment_buffer_capacity_samples)
    )
    for run_client in run_clients(legacy_compatible):
        assert_run_client_uses_legacy_compatible_policies(run_client)


@pytest.mark.parametrize(
    "changes",
    [
        dict(prediction_model_parameter_source="mixture_of_synchronized_and_local_parameters"),
        dict(prediction_model_parameter_source="parameters_at_last_synchronization"),
        dict(fixed_share_weight_redistribution_time_scale_samples=120),
    ],
)
def test_legacy_compatible_check_rejects_other_prediction_settings(changes):
    legacy_compatible = use_legacy_compatible_policies(
        build_experiment_run_settings(
            configuration_name="fedsda", experiment_run_conditions=_CONDITIONS
        )
    )
    (run_client, *_) = run_clients(replace_prediction(legacy_compatible, **changes))
    with pytest.raises(AssertionError, match="実旧との照合には"):
        assert_run_client_uses_legacy_compatible_policies(run_client)


# 構成ごとの、予測に使うパラメータ（既定は混ぜる値。Tは全部の構成で120）。
_EXPECTED_PARAMETER_SOURCE_BY_CONFIGURATION_NAME = {
    "fedsda-single-model-prediction": "current_local_parameters",
    "fedsda-local-parameter-prediction": "current_local_parameters",
    "fedsda-synchronized-model-prediction": "parameters_at_last_synchronization",
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
        is FedsdaRunSettings
    ]


def test_every_fedsda_configuration_uses_the_new_prediction_defaults():
    names = fedsda_configuration_names()
    assert "fedsda-local-parameter-prediction" in names
    for name in names:
        client = build_experiment_run_settings(
            configuration_name=name, experiment_run_conditions=_CONDITIONS
        ).run_participant_settings.run_client_settings
        prediction = client.prediction_combination_settings
        assert prediction.prediction_model_parameter_source == (
            _EXPECTED_PARAMETER_SOURCE_BY_CONFIGURATION_NAME.get(
                name, "mixture_of_synchronized_and_local_parameters"
            )
        ), name
        assert prediction.fixed_share_weight_redistribution_time_scale_samples == 120, name
        assert (
            client.training_data_assignment_settings.pending_assignment_buffer_capacity_samples
            == 30
        )


@pytest.mark.parametrize("configuration_name", EXPERIMENT_CONFIGURATION_NAMES)
def test_every_configuration_builds_and_runs_with_the_new_defaults(configuration_name):
    run_settings = build_experiment_run_settings(
        configuration_name=configuration_name, experiment_run_conditions=_CONDITIONS
    )
    if type(run_settings) is FedsdaRunSettings:
        assert run_clients(run_settings)
