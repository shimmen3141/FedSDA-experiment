"""警報の後の変化区間の長さと帰属先を、最小記述長の原理で選ぶ扱いの、設定・接続・記録を検証する。"""

from dataclasses import replace

import pytest

import federated_learning_experiments.runtime.alarm_buffer_response as buffer_response_module
import federated_learning_experiments.runtime.alarm_change_interval_resolution as resolution_module
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
    # 基準版10の既定（「新しいモデル」の仮説は初期化し直した学習者だけ、初期モデルの統計は再代入の推定、
    # 監視の基準は警報まで保つ）を基点にする（このファイルのtestは、基準版10の最小記述長の選択を確かめる）。
    # FedSDA以外の構成は、そのまま。
    if not hasattr(settings.run_participant_settings, "initial_model_loss_statistics_estimation"):
        return settings
    participant = settings.run_participant_settings
    client = participant.run_client_settings
    assignment = client.training_data_assignment_settings
    return replace(
        settings,
        run_participant_settings=replace(
            participant,
            initial_model_loss_statistics_estimation="resubstitution",
            run_client_settings=replace(
                client,
                loss_change_detection_settings=replace(
                    client.loss_change_detection_settings,
                    loss_monitoring_baseline_update_policy="keep_baseline_until_alarm",
                ),
                training_data_assignment_settings=replace(
                    assignment,
                    alarm_interval_assignment_policy=assignment.alarm_interval_assignment_policy
                    if alarm_interval_assignment_policy is None
                    else alarm_interval_assignment_policy,
                    new_model_hypothesis_learners="reinitialized_learner",
                    change_interval_search="minimum_interval_multiples_with_forward_prequential_code",
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
    # 解決の処理ごとに、選択と、そのときの現在のモデルが履歴の基準を持つかを記録する。
    current_model_has_baseline_by_selection_id = {}
    resolving = []
    original_resolve = buffer_response_module.resolve_alarm_change_interval
    original_evaluate = resolution_module.evaluate_held_models_for_alarm_interval_reuse

    def recording_resolve(**arguments):
        resolving.append(arguments)
        try:
            return original_resolve(**arguments)
        finally:
            resolving.pop()

    def recording_evaluate(**arguments):
        assessment = original_evaluate(**arguments)
        context = resolving[-1]
        selection = context["alarm_interval_assignment_selection"]
        if selection is not None:
            current_model_id = context[
                "current_training_model_assignment"
            ].current_training_model_id
            current_model_has_baseline_by_selection_id[id(selection)] = any(
                model_id == current_model_id
                for model_id, _ in assessment.baseline_supported_interval_mean_losses_by_model_id
            )
        return assessment

    monkeypatch.setattr(buffer_response_module, "resolve_alarm_change_interval", recording_resolve)
    monkeypatch.setattr(
        resolution_module, "evaluate_held_models_for_alarm_interval_reuse", recording_evaluate
    )
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
        elif current_model_has_baseline_by_selection_id[id(selection)]:
            assert record.adaptation_outcome == "alarm_interval_current_model_maintained"
        else:
            # 現在のモデルが履歴の基準を持たなければ、新しいモデルになる（研究ツールと同じ。Requirement 1.3）。
            assert record.adaptation_outcome == "alarm_interval_candidate_adopted_as_new_model"
    assert raised > 0
    assert hypotheses, hypotheses


def test_current_model_selection_without_baseline_adopts_a_new_model(monkeypatch):
    """「現在のモデルのまま」を選んでも、現在のモデルが履歴の基準を持たなければ、新しいモデルになる。

    研究ツールと同じ扱い（Requirement 1.3）。小さい条件ではこの場面が起きないことがあるので、「現在のモデルの
    まま」を選んだ解決で、評価の結果の履歴の基準を持つモデルの列から現在のモデルを除いて、場面を作る。
    """
    from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
        AlarmIntervalModelReuseAssessment,
    )

    forced = []
    expected_adoption_count = 0
    resolving = []
    original_resolve = buffer_response_module.resolve_alarm_change_interval
    original_evaluate = resolution_module.evaluate_held_models_for_alarm_interval_reuse

    def recording_resolve(**arguments):
        resolving.append(arguments)
        try:
            return original_resolve(**arguments)
        finally:
            resolving.pop()

    def evaluate_without_current_model_baseline(**arguments):
        assessment = original_evaluate(**arguments)
        context = resolving[-1]
        selection = context["alarm_interval_assignment_selection"]
        nonlocal expected_adoption_count
        if selection is None:
            return assessment
        if selection.selected_hypothesis != "current_model":
            # 行き先が履歴の基準を持つモデルの列に無ければ（新しいモデルの選択を含む）、候補の採用になる。
            expected_adoption_count += all(
                model_id != selection.destination_model_id
                for model_id, _ in assessment.baseline_supported_interval_mean_losses_by_model_id
            )
            return assessment
        current_model_id = context["current_training_model_assignment"].current_training_model_id
        forced.append(current_model_id)
        expected_adoption_count += 1
        without_current_model = tuple(
            evaluated_model_loss
            for evaluated_model_loss in assessment.baseline_supported_interval_mean_losses_by_model_id
            if evaluated_model_loss[0] != current_model_id
        )
        return AlarmIntervalModelReuseAssessment(
            baseline_supported_interval_mean_losses_by_model_id=without_current_model,
            reusable_mean_losses_by_model_id=tuple(
                evaluated_model_loss
                for evaluated_model_loss in assessment.reusable_mean_losses_by_model_id
                if evaluated_model_loss[0] != current_model_id
            ),
        )

    monkeypatch.setattr(buffer_response_module, "resolve_alarm_change_interval", recording_resolve)
    monkeypatch.setattr(
        resolution_module,
        "evaluate_held_models_for_alarm_interval_reuse",
        evaluate_without_current_model_baseline,
    )
    run = execute_fedsda_experiment_run(
        run_settings=build_small_settings(alarm_interval_assignment_policy=MDL)
    )
    assert forced
    outcomes = [
        record.adaptation_outcome
        for client in run.measured_run.participants.client_operations
        for record in client.owners.adaptation_record_store.get_state_snapshot().adaptation_records
    ]
    # 現在のモデルの維持は1回も起きず、作った場面の数だけ、新しいモデルの選択などに加えて候補の採用が起きる。
    assert "alarm_interval_current_model_maintained" not in outcomes
    assert (
        outcomes.count("alarm_interval_candidate_adopted_as_new_model") == expected_adoption_count
    )


def test_default_and_configurations_use_minimum_description_length():
    """最終構成の既定は最小記述長の原理で選び、候補検証ありの構成だけが今の扱いを明示する。"""
    from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
    )

    checked = 0
    for configuration_name in EXPERIMENT_CONFIGURATION_NAMES:
        settings = build_small_settings(configuration_name)
        if type(settings) is not FedsdaRunSettings:
            continue
        checked += 1
        policy = settings.run_participant_settings.run_client_settings.training_data_assignment_settings.alarm_interval_assignment_policy
        expected = MDL
        assert policy == expected, configuration_name
    assert checked >= 10
