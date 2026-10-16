"""構成`fedsda-local-parameter-prediction`（既定から、予測に使うパラメータだけを局所にした構成）を確認する。"""

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME,
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
    build_experiment_run_settings,
)

_CONDITIONS = ExperimentRunConditions(
    dataset_name="sine2",
    random_seed=0,
    client_count=2,
    per_client_sample_count=40,
    server_aggregation_interval_per_client_samples=10,
)


def test_configuration_differs_from_proposed_only_in_prediction_model_parameter_source():
    assert LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME == "fedsda-local-parameter-prediction"
    proposed, local = (
        convert_settings_to_plain_mapping(
            build_experiment_run_settings(
                configuration_name=name, experiment_run_conditions=_CONDITIONS
            )
        )
        for name in (
            PROPOSED_FEDSDA_CONFIGURATION_NAME,
            LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME,
        )
    )
    holder = local["run_participant_settings"]["run_client_settings"][
        "prediction_combination_settings"
    ]
    assert holder["prediction_model_parameter_source"] == "current_local_parameters"
    holder["prediction_model_parameter_source"] = "mixture_of_synchronized_and_local_parameters"
    assert local == proposed
