"""警報の後の変化区間の長さと帰属先を、最小記述長の原理で選ぶ扱いの、設定・接続・記録を検証する。"""

from dataclasses import replace

import pytest

import federated_learning_experiments.runtime.observed_sample_processing as processing_module
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_field_validation import (
    RunSettingsValidationError,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME,
    MINIMUM_DESCRIPTION_LENGTH_POLICY_NAME,
    TrainingDataAssignmentSettings,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_record_tables import (
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)

LEGACY = DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME
MDL = MINIMUM_DESCRIPTION_LENGTH_POLICY_NAME
SELECTION_COLUMNS = (
    "client_id",
    "alarm_sample_index",
    "selected_hypothesis",
    "change_interval_sample_count",
    "destination_model_id",
    "description_length",
    "alarm_raised",
)


def build_small_settings(configuration_name="fedsda", *, alarm_interval_assignment_policy=None):
    settings = build_experiment_run_settings(
        configuration_name=configuration_name,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=0,
            client_count=2,
            per_client_sample_count=900,
            server_aggregation_interval_per_client_samples=50,
        ),
    )
    execution_settings = settings.execution_settings
    settings = replace(
        settings,
        execution_settings=replace(
            execution_settings,
            concept_schedule_settings=replace(
                execution_settings.concept_schedule_settings,
                minimum_sample_index_gap_before_change_trial=100,
                per_eligible_sample_concept_change_probability=0.015,
            ),
        ),
    )
    if alarm_interval_assignment_policy is None:
        return settings
    participant = settings.run_participant_settings
    client = participant.run_client_settings
    return replace(
        settings,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                client,
                training_data_assignment_settings=replace(
                    client.training_data_assignment_settings,
                    alarm_interval_assignment_policy=alarm_interval_assignment_policy,
                ),
            ),
        ),
    )


def test_policy_field_accepts_only_declared_values():
    settings = build_small_settings().run_participant_settings.run_client_settings.training_data_assignment_settings
    for policy in (LEGACY, MDL):
        assert (
            replace(
                settings, alarm_interval_assignment_policy=policy
            ).alarm_interval_assignment_policy
            == policy
        )
    with pytest.raises(RunSettingsValidationError):
        replace(settings, alarm_interval_assignment_policy="other")
    assert "alarm_interval_assignment_policy" in TrainingDataAssignmentSettings.__dataclass_fields__


def test_minimum_description_length_is_rejected_with_candidate_validation():
    with pytest.raises(RunSettingsValidationError) as error_information:
        build_small_settings(
            "fedsda-with-candidate-validation", alarm_interval_assignment_policy=MDL
        )
    assert (
        error_information.value.configuration_parameter_name == "alarm_interval_assignment_policy"
    )


def test_legacy_policy_never_calls_the_minimum_description_length_selection(monkeypatch):
    def fail(**arguments):
        raise AssertionError("the minimum description length selection must not be called")

    monkeypatch.setattr(processing_module, "select_alarm_interval_assignment", fail)
    run = execute_fedsda_experiment_run(
        run_settings=build_small_settings(alarm_interval_assignment_policy=LEGACY)
    )
    tables = build_fedsda_run_record_tables(experiment_run=run)
    assert all(
        tables[f"alarm_interval_assignment_selection.{column}"].size == 0
        for column in SELECTION_COLUMNS
    )
    assert tables["loss_change_alarm.alarm_sample_index"].size > 0


def test_minimum_description_length_selection_sets_the_change_interval_and_destination(monkeypatch):
    selections = []
    original = processing_module.select_alarm_interval_assignment

    def recording(**arguments):
        selection = original(**arguments)
        selections.append((arguments["client_id"], arguments["alarm_sample_index"], selection))
        return selection

    monkeypatch.setattr(processing_module, "select_alarm_interval_assignment", recording)
    run = execute_fedsda_experiment_run(
        run_settings=build_small_settings(alarm_interval_assignment_policy=MDL)
    )
    tables = build_fedsda_run_record_tables(experiment_run=run)
    assert selections
    # 記録の表は、選択の全部を持つ（clientと警報の標本の番号の組）。
    assert sorted(
        zip(
            tables["alarm_interval_assignment_selection.client_id"].tolist(),
            tables["alarm_interval_assignment_selection.alarm_sample_index"].tolist(),
            strict=True,
        )
    ) == sorted((client_id, alarm_sample_index) for client_id, alarm_sample_index, _ in selections)
    assert len(tables["alarm_interval_assignment_selection.client_id"]) == len(selections)
    # 警報にした選択では、警報の記録の変化点が、選んだ変化区間の先頭。
    alarm_change_points = {
        (int(client_id), int(alarm_index)): int(change_point)
        for client_id, alarm_index, change_point in zip(
            tables["loss_change_alarm.client_id"],
            tables["loss_change_alarm.alarm_sample_index"],
            tables["loss_change_alarm.estimated_change_point_sample_index"],
            strict=True,
        )
    }
    clients = run.measured_run.participants.client_operations
    outcomes = {}
    for client in clients:
        for record in client.owners.adaptation_record_store.get_state_snapshot().adaptation_records:
            outcomes[(client.client_id, record.adaptation_sample_index)] = record
    hypotheses = set()
    raised = 0
    for client_id, alarm_index, selection in selections:
        if (client_id, alarm_index) not in alarm_change_points:
            continue
        raised += 1
        hypotheses.add(selection.selected_hypothesis)
        assert (
            alarm_change_points[(client_id, alarm_index)]
            == alarm_index - selection.change_interval_sample_count + 1
        )
        record = outcomes[(client_id, alarm_index)]
        if selection.selected_hypothesis == "held_model":
            assert record.adaptation_outcome == "alarm_interval_held_model_reused"
            assert record.current_training_model_id == selection.destination_model_id
        elif selection.selected_hypothesis == "new_model":
            assert record.adaptation_outcome == "alarm_interval_candidate_adopted_as_new_model"
        else:
            assert record.adaptation_outcome in (
                "alarm_interval_current_model_maintained",
                "alarm_interval_candidate_adopted_as_new_model",
            )
    assert raised > 0
    assert hypotheses, hypotheses
