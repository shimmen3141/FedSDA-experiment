"""最終構成のFedSDAの1 runに要る全部の設定（完全なrun設定）の検証。"""

from dataclasses import FrozenInstanceError, fields, replace

import pytest
from test_held_candidate_validation_progress import make_subclass_copy

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_consolidation_settings import (
    ModelConsolidationSettings,
)
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
    build_final_configuration_fedsda_run_settings,
)
from federated_learning_experiments.runtime.fedsda_run_metric_derivation import (
    derive_fedsda_run_metrics,
)
from federated_learning_experiments.runtime.fedsda_run_participant_factory import (
    FedsdaRunParticipantFactory,
    FedsdaRunParticipantSettings,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
from federated_learning_experiments.runtime.single_run_execution import (
    execute_stream_protocol_run,
)

FIELD_NAMES = (
    "execution_settings",
    "run_participant_settings",
    "model_consolidation_settings",
    "run_metric_settings",
)


def make_valid_run_settings(**conditions):
    return build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=ExperimentRunConditions(
            **dict(
                dataset_name="sine2",
                random_seed=0,
                client_count=2,
                per_client_sample_count=60,
                server_aggregation_interval_per_client_samples=10,
            )
            | conditions
        )
    )


def make_field_values(run_settings):
    return {field_name: getattr(run_settings, field_name) for field_name in FIELD_NAMES}


def test_run_settings_hold_the_four_parts_and_are_frozen_and_keyword_only():
    run_settings = make_valid_run_settings()
    assert tuple(settings_field.name for settings_field in fields(FedsdaRunSettings)) == FIELD_NAMES
    assert type(run_settings.execution_settings) is StreamProtocolExecutionSettings
    assert type(run_settings.run_participant_settings) is FedsdaRunParticipantSettings
    assert type(run_settings.model_consolidation_settings) is ModelConsolidationSettings
    assert type(run_settings.run_metric_settings) is RunMetricSettings
    with pytest.raises(FrozenInstanceError):
        run_settings.run_metric_settings = run_settings.run_metric_settings  # type: ignore[misc]
    with pytest.raises(TypeError):
        FedsdaRunSettings(*make_field_values(run_settings).values())  # type: ignore[misc]
    for missing_field_name in FIELD_NAMES:
        field_values = make_field_values(run_settings)
        del field_values[missing_field_name]
        with pytest.raises(TypeError):
            FedsdaRunSettings(**field_values)
    # 同じ部分からは、等しい値。
    assert FedsdaRunSettings(**make_field_values(run_settings)) == run_settings


@pytest.mark.parametrize("field_name", FIELD_NAMES)
def test_each_part_must_be_its_exact_settings_type(field_name):
    """各部分は、決まった設定型そのもの（None、ほかの設定型、派生型は、拒否する）。"""
    run_settings = make_valid_run_settings()
    other_field_name = FIELD_NAMES[(FIELD_NAMES.index(field_name) + 1) % len(FIELD_NAMES)]
    for invalid_value in (
        None,
        getattr(run_settings, other_field_name),
        make_subclass_copy(getattr(run_settings, field_name)),
        {},
    ):
        with pytest.raises(RunSettingsValidationError) as exception_info:
            FedsdaRunSettings(**make_field_values(run_settings) | {field_name: invalid_value})
        assert exception_info.value.configuration_parameter_name == field_name
        assert exception_info.value.specified_parameter_value is invalid_value


def mutate(settings, field_name, value):
    """frozenを回避して、検証を通らない値を入れた写しを作る。"""
    mutated_settings = replace(settings)
    object.__setattr__(mutated_settings, field_name, value)
    return mutated_settings


def test_parts_are_validated_again_when_run_settings_are_built():
    """frozenを回避して組み立てた、不正な部分を、生成時に拒否する（各部分の検証を、もう一度行う）。"""
    run_settings = make_valid_run_settings()
    execution_settings = run_settings.execution_settings
    run_participant_settings = run_settings.run_participant_settings
    invalid_parts = dict(
        execution_settings=[
            mutate(execution_settings, "execution_strategy", "unknown"),
            mutate(
                execution_settings,
                "experiment_run_conditions",
                mutate(execution_settings.experiment_run_conditions, "client_count", 0),
            ),
        ],
        run_participant_settings=[
            mutate(run_participant_settings, "hidden_layer_widths", ()),
            mutate(run_participant_settings, "maximum_evaluating_client_count_per_model", 0),
        ],
        model_consolidation_settings=[
            mutate(
                run_settings.model_consolidation_settings, "model_clustering_linkage", "unknown"
            ),
        ],
        run_metric_settings=[
            mutate(run_settings.run_metric_settings, "maximum_detection_delay_sample_count", -1),
        ],
    )
    assert set(invalid_parts) == set(FIELD_NAMES)
    for field_name, invalid_values in invalid_parts.items():
        for invalid_value in invalid_values:
            with pytest.raises((RunSettingsValidationError, TypeError, ValueError)):
                FedsdaRunSettings(**make_field_values(run_settings) | {field_name: invalid_value})


def test_replacing_a_part_builds_validated_run_settings():
    run_settings = make_valid_run_settings()
    replaced_run_settings = replace(
        run_settings,
        run_metric_settings=RunMetricSettings(
            maximum_detection_delay_sample_count=5, post_change_recovery_window_sample_count=3
        ),
    )
    assert replaced_run_settings != run_settings
    assert replaced_run_settings.execution_settings is run_settings.execution_settings
    with pytest.raises(RunSettingsValidationError):
        replace(run_settings, model_consolidation_settings=run_settings.run_metric_settings)


def test_run_settings_drive_a_whole_run_and_the_metric_derivation():
    """完全なrun設定の部分を、そのまま、全体runの実行と、指標の導出へ渡せる。"""
    run_settings = make_valid_run_settings(dataset_name="sea2")
    participant_factory = FedsdaRunParticipantFactory(
        run_participant_settings=run_settings.run_participant_settings
    )
    run_result = execute_stream_protocol_run(
        execution_settings=run_settings.execution_settings, participant_factory=participant_factory
    )
    participants = participant_factory.prepared_run_participants
    run_metrics = derive_fedsda_run_metrics(
        run_result=run_result,
        participants=participants,
        run_metric_settings=run_settings.run_metric_settings,
    )
    assert run_result.processed_sample_count_per_client == 60
    assert len(participants.client_operations) == 2
    assert run_metrics.processed_sample_count == 120
    assert 0.0 <= run_metrics.prediction_accuracy <= 1.0
    # 既定の隠れ層の幅（合成データは(32, 32)）で、初期モデルが作られている。
    global_model_repository = participants.server_operations.owners.global_model_repository
    parameter_snapshot = global_model_repository.get_global_model_parameters(
        model_id=global_model_repository.global_model_ids[0]
    )
    first_weight = next(iter(parameter_snapshot.values()))
    assert tuple(first_weight.shape) == (32, 3)
