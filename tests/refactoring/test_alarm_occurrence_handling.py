"""警報1回ぶんの処理の接続を、実旧の警報解決（_resolve_drift）・session属性・帰属変更hookと照合する。"""

from dataclasses import FrozenInstanceError, fields
from inspect import Parameter, signature
from unittest.mock import Mock

import pytest
import torch
from test_adahedge_diagnostic_evidence import assert_adahedge_matches_legacy
from test_alarm_adaptation_recording import RECORDING_ORACLE_CASES
from test_alarm_buffer_response import assert_buffer_response_matches_legacy
from test_alarm_response_completion import (
    LEGACY_ACTION_BY_RESPONSE_OUTCOME,
    assert_response_completion_matches_legacy,
    build_response_completion_oracle,
    run_legacy_alarm_with_real_completion,
)
from test_held_adahedge_diagnostic_notification import get_diagnostic_collection_snapshot
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

import federated_learning_experiments.runtime.alarm_occurrence_handling as occurrence_module
from federated_drift_experiment.clients.fedsda import (
    RestartingSoftRoutingClassConditionalESRFedSDAClient,
)
from federated_drift_experiment.expert_routing import AdaHedgeRouter
from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import (
    AdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.evaluation.adaptation_record_store import (
    AdaptationRecord,
    AdaptationRecordStore,
)
from federated_learning_experiments.runtime.alarm_buffer_response import (
    respond_to_alarm_with_buffered_samples,
)
from federated_learning_experiments.runtime.alarm_occurrence_handling import (
    AlarmOccurrenceHandling,
    handle_alarm_occurrence,
)
from federated_learning_experiments.runtime.alarm_response_completion import (
    AlarmResponseCompletion,
)
from federated_learning_experiments.runtime.candidate_validation_session_holder import (
    CandidateValidationSessionHolder,
)

# 接続する既存の5段。新moduleが参照する名前で、この順に1回ずつ実行される。
ALARM_OCCURRENCE_STEP_NAMES = (
    "respond_to_alarm_with_buffered_samples",
    "complete_alarm_buffer_response",
    "record_completed_alarm_response",
    "apply_alarm_response_to_validation_session_holder",
    "notify_diagnostics_of_training_assignment_change",
)
ARGUMENT_NAMES_ADDED_TO_RESPONSE = (
    "validation_session_holder",
    "adaptation_record_store",
    "diagnostic_evidence_collection",
    "loss_change_monitor",
    "alarm_sample_index",
)
ARGUMENT_NAMES_DERIVED_FOR_RESPONSE = ("active_validation_session", "proposal_sample_index")
# 再始動が観測できるよう、警報の前に新旧の診断証拠へ与える損失。
DIAGNOSTIC_LOSSES_BEFORE_ALARM = {2: 0.1, -1: 0.8}
REUSED_HELD_MODEL_ORACLE_CASE = RECORDING_ORACLE_CASES[1]


def build_alarm_occurrence_oracle(
    *,
    monkeypatch,
    valid_run_settings_mapping,
    class_count=2,
    recording_oracle_case=REUSED_HELD_MODEL_ORACLE_CASE,
):
    """応答・完了のoracleへ、保持・記録・診断のownerと、実旧の再始動hookを加える。"""
    _, alarm_interval_resolution_case, estimated_change_span_sample_count = recording_oracle_case
    (
        response_arguments,
        completion_arguments,
        preparation_arguments,
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
    ) = build_response_completion_oracle(
        monkeypatch=monkeypatch,
        loss_change_detection_settings=valid_run_settings_mapping["loss_change_detection_settings"],
        class_count=class_count,
        alarm_interval_resolution_case=alarm_interval_resolution_case,
        earlier_sample_count=0,
        estimated_change_span_sample_count=estimated_change_span_sample_count,
    )
    # 上流oracleの記録用hookを残したまま、実旧の再始動（確定した切替ごとのAdaHedge再始動）を実行させる。
    legacy_client.expert_router = AdaHedgeRouter()
    legacy_client.context_expert_routers = {}
    legacy_client.shadow_meta_routers = {}
    legacy_client.routing_active_set = None
    record_local_model_change = legacy_client._on_local_model_change

    def record_change_then_restart_legacy_routers(previous_model_id, current_model_id):
        record_local_model_change(previous_model_id, current_model_id)
        RestartingSoftRoutingClassConditionalESRFedSDAClient._on_local_model_change(
            legacy_client, previous_model_id, current_model_id
        )

    legacy_client._on_local_model_change = record_change_then_restart_legacy_routers
    diagnostic_evidence_collection = AdaHedgeDiagnosticEvidenceCollection()
    global_diagnostic_evidence = diagnostic_evidence_collection.global_diagnostic_evidence
    diagnostic_weights = global_diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
        model_ids=tuple(DIAGNOSTIC_LOSSES_BEFORE_ALARM)
    )
    legacy_weights = legacy_client.expert_router.probabilities(DIAGNOSTIC_LOSSES_BEFORE_ALARM)
    global_diagnostic_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id=DIAGNOSTIC_LOSSES_BEFORE_ALARM,
        diagnostic_weights_by_model_id=diagnostic_weights,
    )
    legacy_client.expert_router.update(DIAGNOSTIC_LOSSES_BEFORE_ALARM, legacy_weights)
    handling_arguments = {
        argument_name: argument
        for argument_name, argument in response_arguments.items()
        if argument_name not in ARGUMENT_NAMES_DERIVED_FOR_RESPONSE
    }
    handling_arguments.update(
        validation_session_holder=CandidateValidationSessionHolder(),
        adaptation_record_store=AdaptationRecordStore(),
        diagnostic_evidence_collection=diagnostic_evidence_collection,
        loss_change_monitor=completion_arguments["loss_change_monitor"],
        alarm_sample_index=completion_arguments["alarm_sample_index"],
    )
    return (
        handling_arguments,
        response_arguments,
        completion_arguments,
        preparation_arguments,
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
    )


def snapshot_owners_updated_after_response(*, handling_arguments):
    """応答より後の段が更新するowner（監視、保留位置、記録、保持、診断）の状態。"""
    return (
        handling_arguments["loss_change_monitor"].get_state_snapshot(),
        handling_arguments["pending_training_assignment_buffer"].get_state_snapshot(),
        handling_arguments["adaptation_record_store"].get_state_snapshot(),
        handling_arguments["validation_session_holder"].held_validation_session,
        get_diagnostic_collection_snapshot(handling_arguments["diagnostic_evidence_collection"]),
    )


def test_alarm_occurrence_signature_extends_response_arguments():
    handling_parameters = signature(handle_alarm_occurrence).parameters
    response_parameters = signature(respond_to_alarm_with_buffered_samples).parameters
    assert set(handling_parameters) == (
        set(response_parameters) - set(ARGUMENT_NAMES_DERIVED_FOR_RESPONSE)
    ) | set(ARGUMENT_NAMES_ADDED_TO_RESPONSE)
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY and parameter.default is Parameter.empty
        for parameter in handling_parameters.values()
    )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("recording_oracle_case", RECORDING_ORACLE_CASES)
def test_alarm_occurrence_matches_real_legacy_alarm(
    class_count, recording_oracle_case, monkeypatch, valid_run_settings_mapping
):
    expected_response_outcome = recording_oracle_case[0]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        (
            handling_arguments,
            response_arguments,
            completion_arguments,
            preparation_arguments,
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
        ) = build_alarm_occurrence_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=class_count,
            recording_oracle_case=recording_oracle_case,
        )
        validation_session_holder = handling_arguments["validation_session_holder"]
        adaptation_record_store = handling_arguments["adaptation_record_store"]
        diagnostic_evidence_collection = handling_arguments["diagnostic_evidence_collection"]
        pending_training_assignment_buffer = handling_arguments[
            "pending_training_assignment_buffer"
        ]
        assert handling_arguments["detector_name"] == legacy_client._detector_label()
        initial_training_model_id = legacy_client.current_model_id
        initial_torch_random_state = torch.get_rng_state().clone()
        legacy_result = run_legacy_alarm_with_real_completion(
            response_arguments=response_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        expected_torch_random_state = torch.get_rng_state().clone()
        torch.set_rng_state(initial_torch_random_state)
        pending_sample_indices_before_alarm = (
            pending_training_assignment_buffer.get_state_snapshot().pending_sample_indices
        )
        alarm_occurrence_handling = handle_alarm_occurrence(**handling_arguments)
        # 応答より後の段（完了・記録・保持・通知）は乱数を消費しない。
        assert torch.equal(torch.get_rng_state(), expected_torch_random_state)
        expected_record_count = 1
        if expected_response_outcome == "alarm_during_candidate_validation":
            # 1回目の警報が開始して保持させたsessionを、2回目の警報が保持から読む。
            started_validation_session = validation_session_holder.held_validation_session
            assert started_validation_session is not None
            assert (
                alarm_occurrence_handling.alarm_response_completion.alarm_buffer_response.active_validation_session
                is started_validation_session
            )
            # 保留位置を消費した後の同じ位置で、保留標本なしの警報を受ける（上流の記録oracleと同じ形）。
            handling_arguments.update(
                pending_sample_observations=(),
                estimated_change_point_sample_index=handling_arguments["alarm_sample_index"],
            )
            response_arguments.update(
                active_validation_session=started_validation_session,
                pending_sample_observations=(),
                estimated_change_point_sample_index=handling_arguments["alarm_sample_index"],
            )
            legacy_result = run_legacy_alarm_with_real_completion(
                response_arguments=response_arguments,
                legacy_client=legacy_client,
                monkeypatch=monkeypatch,
            )
            pending_sample_indices_before_alarm = (
                pending_training_assignment_buffer.get_state_snapshot().pending_sample_indices
            )
            torch_random_state_before_second_alarm = torch.get_rng_state().clone()
            alarm_occurrence_handling = handle_alarm_occurrence(**handling_arguments)
            assert torch.equal(torch.get_rng_state(), torch_random_state_before_second_alarm)
            assert validation_session_holder.held_validation_session is started_validation_session
            expected_record_count = 2
    assert type(alarm_occurrence_handling) is AlarmOccurrenceHandling
    alarm_response_completion = alarm_occurrence_handling.alarm_response_completion
    adaptation_record = alarm_occurrence_handling.adaptation_record
    assert type(alarm_response_completion) is AlarmResponseCompletion
    assert type(adaptation_record) is AdaptationRecord
    alarm_buffer_response = alarm_response_completion.alarm_buffer_response
    assert alarm_buffer_response.response_outcome == expected_response_outcome
    # 応答と完了: 実旧のイベント・FIFO・検出器・学習状態と一致する。
    assert_response_completion_matches_legacy(
        response_completion=alarm_response_completion,
        completion_arguments=completion_arguments,
        legacy_client=legacy_client,
        legacy_result=legacy_result,
        pending_sample_indices_before_completion=pending_sample_indices_before_alarm,
    )
    if expected_response_outcome != "alarm_during_candidate_validation":
        assert_buffer_response_matches_legacy(
            response=alarm_buffer_response,
            response_arguments=response_arguments,
            preparation_arguments=preparation_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_result=legacy_result,
            initial_training_model_id=initial_training_model_id,
        )
    # 記録: 実旧の適応イベント・切替位置・再利用件数と一致する。
    assert legacy_result[1][0] == dict(
        position=adaptation_record.adaptation_sample_index,
        detector=adaptation_record.detector_name,
        action=LEGACY_ACTION_BY_RESPONSE_OUTCOME[adaptation_record.adaptation_outcome],
        old_model_id=adaptation_record.previous_training_model_id,
        new_model_id=adaptation_record.current_training_model_id,
        estimated_change_point=adaptation_record.estimated_change_point_sample_index,
        episode_id=adaptation_record.detection_episode_id,
    )
    adaptation_record_snapshot = adaptation_record_store.get_state_snapshot()
    assert len(adaptation_record_snapshot.adaptation_records) == expected_record_count
    assert adaptation_record_snapshot.adaptation_records[-1] == adaptation_record
    assert adaptation_record_snapshot.training_model_switch_sample_indices == tuple(
        legacy_client.local_switch_positions
    )
    assert (
        adaptation_record_snapshot.alternative_model_reuse_count
        == legacy_client.reuse_selection_counts["alternative_fit"]
    )
    assert (
        adaptation_record_snapshot.current_model_fit_count
        == legacy_client.reuse_selection_counts["current_fit"]
    )
    # 保持: 実旧のsession属性の有無と一致し、保持しているのは応答のsessionそのもの。
    assert (
        validation_session_holder.held_validation_session
        is alarm_buffer_response.active_validation_session
    )
    assert (validation_session_holder.held_validation_session is not None) == (
        legacy_client._forward_validation is not None
    )
    # 診断: 実旧の再始動hookを通した後のAdaHedge状態と一致し、再始動は他モデルの再利用のときだけ1回。
    global_diagnostic_evidence = diagnostic_evidence_collection.global_diagnostic_evidence
    assert_adahedge_matches_legacy(global_diagnostic_evidence, legacy_client.expert_router)
    assert global_diagnostic_evidence.concept_operation_restart_count == (
        1 if expected_response_outcome == "alarm_interval_held_model_reused" else 0
    )
    assert diagnostic_evidence_collection.created_true_concept_ids == ()


def test_alarm_occurrence_runs_each_step_once_in_order(monkeypatch, valid_run_settings_mapping):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        handling_arguments = build_alarm_occurrence_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )[0]
        step_calls = []
        real_steps = {
            step_name: getattr(occurrence_module, step_name)
            for step_name in ALARM_OCCURRENCE_STEP_NAMES
        }
        for step_name in ALARM_OCCURRENCE_STEP_NAMES:

            def record_step_call(*, step_name=step_name, **step_arguments):
                step_result = real_steps[step_name](**step_arguments)
                step_calls.append((step_name, step_arguments, step_result))
                return step_result

            monkeypatch.setattr(occurrence_module, step_name, record_step_call)
        alarm_occurrence_handling = handle_alarm_occurrence(**handling_arguments)
    assert tuple(step_name for step_name, _, _ in step_calls) == ALARM_OCCURRENCE_STEP_NAMES
    (
        (_, response_step_arguments, alarm_buffer_response),
        (_, completion_step_arguments, alarm_response_completion),
        (_, recording_step_arguments, adaptation_record),
        (_, holder_step_arguments, _),
        (_, notification_step_arguments, _),
    ) = step_calls
    # 応答: 進行中のsessionは保持から、提案位置は警報位置から与える。他は受け取った引数そのもの。
    assert response_step_arguments.pop("active_validation_session") is None
    assert (
        response_step_arguments.pop("proposal_sample_index")
        == (handling_arguments["alarm_sample_index"])
    )
    assert response_step_arguments.keys() == (
        handling_arguments.keys() - set(ARGUMENT_NAMES_ADDED_TO_RESPONSE)
    )
    # 完了・記録・保持・通知: 直前の段の結果と、受け取ったownerそのものを渡す。
    assert completion_step_arguments.pop("alarm_buffer_response") is alarm_buffer_response
    assert completion_step_arguments.keys() == {
        "alarm_sample_index",
        "estimated_change_point_sample_index",
        "detection_episode_id",
        "current_training_model_assignment",
        "loss_statistics_store",
        "loss_change_monitor",
        "pending_training_assignment_buffer",
    }
    assert recording_step_arguments.pop("alarm_response_completion") is alarm_response_completion
    assert recording_step_arguments.keys() == {"detector_name", "adaptation_record_store"}
    assert holder_step_arguments.pop("alarm_buffer_response") is alarm_buffer_response
    assert holder_step_arguments.keys() == {"validation_session_holder"}
    training_model_assignment_change = (
        alarm_buffer_response.change_interval_resolution.training_model_assignment_change
    )
    assert training_model_assignment_change is not None
    assert notification_step_arguments.pop("assignment_change") is training_model_assignment_change
    assert notification_step_arguments.keys() == {"diagnostic_evidence_collection"}
    for step_arguments in (
        response_step_arguments,
        completion_step_arguments,
        recording_step_arguments,
        holder_step_arguments,
        notification_step_arguments,
    ):
        for argument_name, argument in step_arguments.items():
            assert argument is handling_arguments[argument_name], argument_name
    assert alarm_occurrence_handling.alarm_response_completion is alarm_response_completion
    assert alarm_occurrence_handling.adaptation_record is adaptation_record


def make_uninitialized_subclass_instance(owner):
    """exact型の検査が拒否するべき、同じclassの派生型の値（初期化しない）。"""
    owner_subclass = type(f"{type(owner).__name__}Subclass", (type(owner),), {})
    return owner_subclass.__new__(owner_subclass)


def get_last_observed_sample_index(handling_arguments):
    return (
        handling_arguments["pending_training_assignment_buffer"]
        .get_state_snapshot()
        .last_observed_sample_index
    )


OWNER_ARGUMENT_NAMES_VALIDATED_BEFORE_RESPONSE = (
    "validation_session_holder",
    "adaptation_record_store",
    "diagnostic_evidence_collection",
    "loss_change_monitor",
    "pending_training_assignment_buffer",
)
# 条件名 -> (不正にする引数名, 正常な引数から不正な値を作る操作, 期待する例外)
INVALID_ALARM_OCCURRENCE_INPUT_CASES = {
    **{
        f"{owner_argument_name}_other_type": (
            owner_argument_name,
            lambda handling_arguments: object(),
            TypeError,
        )
        for owner_argument_name in OWNER_ARGUMENT_NAMES_VALIDATED_BEFORE_RESPONSE
    },
    **{
        f"{owner_argument_name}_subclass": (
            owner_argument_name,
            lambda handling_arguments, owner_argument_name=owner_argument_name: (
                make_uninitialized_subclass_instance(handling_arguments[owner_argument_name])
            ),
            TypeError,
        )
        for owner_argument_name in OWNER_ARGUMENT_NAMES_VALIDATED_BEFORE_RESPONSE
    },
    "alarm_sample_index_bool": ("alarm_sample_index", lambda handling_arguments: True, TypeError),
    "alarm_sample_index_float": (
        "alarm_sample_index",
        lambda handling_arguments: float(get_last_observed_sample_index(handling_arguments)),
        TypeError,
    ),
    "alarm_sample_index_negative": (
        "alarm_sample_index",
        lambda handling_arguments: -1,
        ValueError,
    ),
    "alarm_sample_index_before_last_observed": (
        "alarm_sample_index",
        lambda handling_arguments: get_last_observed_sample_index(handling_arguments) - 1,
        ValueError,
    ),
    "alarm_sample_index_after_last_observed": (
        "alarm_sample_index",
        lambda handling_arguments: get_last_observed_sample_index(handling_arguments) + 1,
        ValueError,
    ),
    "estimated_change_point_negative": (
        "estimated_change_point_sample_index",
        lambda handling_arguments: -1,
        ValueError,
    ),
    "estimated_change_point_bool": (
        "estimated_change_point_sample_index",
        lambda handling_arguments: False,
        TypeError,
    ),
    "detection_episode_id_negative": (
        "detection_episode_id",
        lambda handling_arguments: -1,
        ValueError,
    ),
    "detection_episode_id_float": (
        "detection_episode_id",
        lambda handling_arguments: 1.0,
        TypeError,
    ),
    "detector_name_none": ("detector_name", lambda handling_arguments: None, TypeError),
    "detector_name_str_subclass": (
        "detector_name",
        lambda handling_arguments: type("StrSubclass", (str,), {})("ClassESR"),
        TypeError,
    ),
    "detector_name_blank": ("detector_name", lambda handling_arguments: " \t\n", ValueError),
}


@pytest.mark.parametrize("invalid_case", INVALID_ALARM_OCCURRENCE_INPUT_CASES)
def test_alarm_occurrence_rejects_invalid_input_before_any_step(
    invalid_case, monkeypatch, valid_run_settings_mapping
):
    argument_name, make_invalid_argument, expected_exception = INVALID_ALARM_OCCURRENCE_INPUT_CASES[
        invalid_case
    ]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        handling_arguments = build_alarm_occurrence_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )[0]
    state_snapshot = snapshot_owners_updated_after_response(handling_arguments=handling_arguments)
    invalid_handling_arguments = handling_arguments | {
        argument_name: make_invalid_argument(handling_arguments)
    }
    step_calls = {
        step_name: Mock(side_effect=AssertionError(step_name))
        for step_name in ALARM_OCCURRENCE_STEP_NAMES
    }
    for step_name, step_call in step_calls.items():
        monkeypatch.setattr(occurrence_module, step_name, step_call)
    with pytest.raises(expected_exception):
        handle_alarm_occurrence(**invalid_handling_arguments)
    # 応答を含むどの段も呼ばれていない（検査がすべて最初の状態更新より前にある）。
    for step_call in step_calls.values():
        step_call.assert_not_called()
    assert (
        snapshot_owners_updated_after_response(handling_arguments=handling_arguments)
        == state_snapshot
    )


@pytest.mark.parametrize("failing_step_name", ALARM_OCCURRENCE_STEP_NAMES)
def test_failed_step_stops_alarm_occurrence_before_later_steps(
    failing_step_name, monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        handling_arguments = build_alarm_occurrence_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )[0]
        failing_step_position = ALARM_OCCURRENCE_STEP_NAMES.index(failing_step_name)
        later_step_calls = {
            step_name: Mock()
            for step_name in ALARM_OCCURRENCE_STEP_NAMES[failing_step_position + 1 :]
        }
        monkeypatch.setattr(
            occurrence_module, failing_step_name, Mock(side_effect=RuntimeError(failing_step_name))
        )
        for step_name, step_call in later_step_calls.items():
            monkeypatch.setattr(occurrence_module, step_name, step_call)
        with pytest.raises(RuntimeError, match=failing_step_name):
            handle_alarm_occurrence(**handling_arguments)
    # 失敗した段より後は実行しない。先行する段の更新は巻き戻さない（上流の既存契約）。
    for step_call in later_step_calls.values():
        step_call.assert_not_called()
    adaptation_record_count = len(
        handling_arguments["adaptation_record_store"].get_state_snapshot().adaptation_records
    )
    assert adaptation_record_count == (
        1
        if failing_step_position
        > ALARM_OCCURRENCE_STEP_NAMES.index("record_completed_alarm_response")
        else 0
    )


def test_alarm_occurrence_handling_record_is_frozen_and_keyword_only():
    assert tuple(record_field.name for record_field in fields(AlarmOccurrenceHandling)) == (
        "alarm_response_completion",
        "adaptation_record",
    )
    alarm_occurrence_handling = AlarmOccurrenceHandling(
        alarm_response_completion=None, adaptation_record=None
    )
    with pytest.raises(FrozenInstanceError):
        alarm_occurrence_handling.adaptation_record = None
    with pytest.raises(TypeError):
        AlarmOccurrenceHandling(None, None)
