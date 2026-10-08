"""命名承認用の下書き。完了した警報応答の記録だけを実旧イベントと照合する。"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from typing import get_args
from unittest.mock import patch

import pytest
import torch
from test_alarm_change_interval_resolution import (
    assert_alarm_change_interval_resolution_state_unchanged,
    snapshot_alarm_change_interval_resolution_state,
)
from test_alarm_response_completion import (
    LEGACY_ACTION_BY_RESPONSE_OUTCOME,
    build_response_completion_oracle,
    run_legacy_alarm_with_real_completion,
)
from test_joint_model_parameter_update import assert_nested_state_equal
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_learning_experiments.evaluation.adaptation_record_store import (
    AdaptationOutcome,
    AdaptationRecord,
    AdaptationRecordSnapshot,
    AdaptationRecordStore,
)
from federated_learning_experiments.runtime.alarm_adaptation_recording import (
    record_completed_alarm_response,
)
from federated_learning_experiments.runtime.alarm_buffer_response import (
    ALARM_BUFFER_RESPONSE_OUTCOMES,
    respond_to_alarm_with_buffered_samples,
)
from federated_learning_experiments.runtime.alarm_response_completion import (
    AlarmResponseCompletion,
    complete_alarm_buffer_response,
)

RECORDING_ORACLE_CASES = (
    ("alarm_change_interval_too_short", "current_model_maintained", 2),
    ("alarm_interval_held_model_reused", "other_model_reused", 3),
    ("alarm_interval_current_model_maintained", "current_model_maintained", 99),
    ("alarm_interval_candidate_validation_started", "no_model_fits", 3),
    ("alarm_during_candidate_validation", "no_model_fits", 3),
)


def build_completed_recording_oracle(
    *,
    monkeypatch,
    valid_run_settings_mapping,
    class_count=2,
    recording_oracle_case=RECORDING_ORACLE_CASES[1],
):
    """上流の応答・完了を呼出側で実行し、記録境界には完成した値だけを渡す。"""
    (
        expected_response_outcome,
        alarm_interval_resolution_case,
        estimated_change_span_sample_count,
    ) = recording_oracle_case
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
    initial_torch_random_state = torch.get_rng_state().clone()
    legacy_result = run_legacy_alarm_with_real_completion(
        response_arguments=response_arguments,
        legacy_client=legacy_client,
        monkeypatch=monkeypatch,
    )
    torch.set_rng_state(initial_torch_random_state)
    alarm_response_completion = complete_alarm_buffer_response(
        alarm_buffer_response=respond_to_alarm_with_buffered_samples(**response_arguments),
        **completion_arguments,
    )
    if expected_response_outcome == "alarm_during_candidate_validation":
        # 実際に開始したsessionを次の警報へ渡す。保留位置を消費後の同じ位置も許容される。
        response_arguments.update(
            active_validation_session=alarm_response_completion.alarm_buffer_response.active_validation_session,
            pending_sample_observations=(),
            estimated_change_point_sample_index=response_arguments["proposal_sample_index"],
        )
        completion_arguments["estimated_change_point_sample_index"] = response_arguments[
            "proposal_sample_index"
        ]
        legacy_result = run_legacy_alarm_with_real_completion(
            response_arguments=response_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        alarm_response_completion = complete_alarm_buffer_response(
            alarm_buffer_response=respond_to_alarm_with_buffered_samples(**response_arguments),
            **completion_arguments,
        )
    assert type(alarm_response_completion) is AlarmResponseCompletion
    assert (
        alarm_response_completion.alarm_buffer_response.response_outcome
        == expected_response_outcome
    )
    return (
        alarm_response_completion,
        response_arguments,
        completion_arguments,
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_result,
    )


def make_adaptation_record(**record_field_overrides):
    return AdaptationRecord(
        **(
            dict(
                adaptation_sample_index=9,
                detector_name=" ClassESR ",
                adaptation_outcome="alarm_interval_held_model_reused",
                previous_training_model_id=-1,
                current_training_model_id=3,
                estimated_change_point_sample_index=None,
                detection_episode_id=None,
            )
            | record_field_overrides
        )
    )


def assert_recording_preserves_upstream_state(
    *,
    alarm_response_completion,
    response_arguments,
    completion_arguments,
    resolution_arguments,
    shared_optimizer_owners,
    recording_operation,
):
    """監視/FIFO/帰属/標本/モデル/optimizer/乱数/sessionを記録前後で照合する。"""
    upstream_state_snapshot = snapshot_alarm_change_interval_resolution_state(
        resolution_arguments=resolution_arguments,
        shared_optimizer_owners=shared_optimizer_owners,
    )
    monitoring_state = completion_arguments["loss_change_monitor"].get_state_snapshot()
    pending_assignment_state = completion_arguments[
        "pending_training_assignment_buffer"
    ].get_state_snapshot()
    python_random_state = response_arguments["python_random_generator"].getstate()
    active_validation_session = (
        alarm_response_completion.alarm_buffer_response.active_validation_session
    )
    session_bindings = (
        None if active_validation_session is None else dict(vars(active_validation_session))
    )
    if active_validation_session is not None:
        candidate_training_state = active_validation_session.candidate_training_state
        candidate_parameter_snapshot = snapshot_parameter_values_and_gradients(
            tuple(candidate_training_state.candidate_classifier.parameters())
        )
        candidate_optimizer_snapshots = tuple(
            (
                optimizer_owner,
                optimizer_owner.parameter_optimizer,
                deepcopy(optimizer_owner.parameter_optimizer.state_dict()),
            )
            for optimizer_owner in (
                candidate_training_state.candidate_shared_parameter_optimizer_state,
                candidate_training_state.candidate_concept_specific_parameter_optimizer_state,
            )
        )
        candidate_loss_collection_state = (
            active_validation_session.post_alarm_candidate_loss_collection.get_state_snapshot()
        )
    with (
        patch.object(completion_arguments["loss_change_monitor"], "reset") as monitor_reset_calls,
        patch.object(
            completion_arguments["pending_training_assignment_buffer"],
            "drain_pending_sample_indices",
        ) as pending_index_drain_calls,
    ):
        recording_operation()
        monitor_reset_calls.assert_not_called()
        pending_index_drain_calls.assert_not_called()
    assert_alarm_change_interval_resolution_state_unchanged(upstream_state_snapshot)
    assert completion_arguments["loss_change_monitor"].get_state_snapshot() == monitoring_state
    assert (
        completion_arguments["pending_training_assignment_buffer"].get_state_snapshot()
        == pending_assignment_state
    )
    assert response_arguments["python_random_generator"].getstate() == python_random_state
    assert (
        alarm_response_completion.alarm_buffer_response.active_validation_session
        is active_validation_session
    )
    if active_validation_session is not None:
        assert all(
            vars(active_validation_session)[binding_name] is binding_value
            for binding_name, binding_value in session_bindings.items()
        )
        assert_parameter_values_and_gradients_unchanged(candidate_parameter_snapshot)
        for optimizer_owner, parameter_optimizer, optimizer_state in candidate_optimizer_snapshots:
            assert optimizer_owner.parameter_optimizer is parameter_optimizer
            assert_nested_state_equal(parameter_optimizer.state_dict(), optimizer_state)
        assert (
            active_validation_session.post_alarm_candidate_loss_collection.get_state_snapshot()
            == candidate_loss_collection_state
        )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("recording_oracle_case", RECORDING_ORACLE_CASES)
def test_completed_alarm_record_matches_all_real_legacy_event_fields(
    class_count,
    recording_oracle_case,
    monkeypatch,
    valid_run_settings_mapping,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        (
            alarm_response_completion,
            response_arguments,
            completion_arguments,
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
            legacy_result,
        ) = build_completed_recording_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=class_count,
            recording_oracle_case=recording_oracle_case,
        )
        adaptation_record_store = AdaptationRecordStore()

        def record_and_compare_legacy_event():
            adaptation_record = record_completed_alarm_response(
                alarm_response_completion=alarm_response_completion,
                detector_name=legacy_client._detector_label(),
                adaptation_record_store=adaptation_record_store,
            )
            assert type(adaptation_record) is AdaptationRecord
            # 全fieldの新旧対応をtest側だけに置き、新APIに旧actionのaliasを設けない。
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
            assert type(adaptation_record_snapshot) is AdaptationRecordSnapshot
            assert adaptation_record_snapshot.adaptation_records == (adaptation_record,)
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

        assert_recording_preserves_upstream_state(
            alarm_response_completion=alarm_response_completion,
            response_arguments=response_arguments,
            completion_arguments=completion_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            recording_operation=record_and_compare_legacy_event,
        )


def test_record_store_keeps_input_order_equal_positions_and_old_immutable_snapshot():
    # 警報応答の5結果は適応結果の先頭5値。候補検証の結果はその後に続く。
    assert get_args(AdaptationOutcome)[:5] == ALARM_BUFFER_RESPONSE_OUTCOMES
    adaptation_record_store = AdaptationRecordStore()
    empty_snapshot = adaptation_record_store.get_state_snapshot()
    first_adaptation_record = make_adaptation_record()
    adaptation_record_store.append_adaptation_record(adaptation_record=first_adaptation_record)
    first_snapshot = adaptation_record_store.get_state_snapshot()
    for adaptation_sample_index, adaptation_outcome in (
        (9, "alarm_interval_current_model_maintained"),
        (2, "alarm_during_candidate_validation"),
        (5, "alarm_change_interval_too_short"),
        (5, "alarm_interval_candidate_validation_started"),
        (9, "alarm_interval_held_model_reused"),
    ):
        adaptation_record_store.append_adaptation_record(
            adaptation_record=make_adaptation_record(
                adaptation_sample_index=adaptation_sample_index,
                adaptation_outcome=adaptation_outcome,
                previous_training_model_id=-1,
                current_training_model_id=3
                if adaptation_outcome == "alarm_interval_held_model_reused"
                else -1,
            )
        )
    final_snapshot = adaptation_record_store.get_state_snapshot()
    assert tuple(
        adaptation_record.adaptation_sample_index
        for adaptation_record in final_snapshot.adaptation_records
    ) == (9, 9, 2, 5, 5, 9)
    assert final_snapshot.training_model_switch_sample_indices == (9, 9)
    assert (
        final_snapshot.alternative_model_reuse_count,
        final_snapshot.current_model_fit_count,
    ) == (2, 1)
    assert empty_snapshot == AdaptationRecordSnapshot(
        adaptation_records=(),
        training_model_switch_sample_indices=(),
        alternative_model_reuse_count=0,
        current_model_fit_count=0,
    )
    assert first_snapshot.adaptation_records == (first_adaptation_record,)
    assert first_snapshot.training_model_switch_sample_indices == (9,)
    assert (
        first_snapshot.alternative_model_reuse_count,
        first_snapshot.current_model_fit_count,
    ) == (1, 0)
    assert first_adaptation_record.detector_name == " ClassESR "
    with pytest.raises(FrozenInstanceError):
        first_adaptation_record.detector_name = "changed"
    with pytest.raises(FrozenInstanceError):
        first_snapshot.alternative_model_reuse_count = 0
    with pytest.raises(TypeError):
        first_snapshot.adaptation_records[0] = first_adaptation_record
    with pytest.raises(TypeError):
        AdaptationRecord(*asdict(first_adaptation_record).values())
    # frozenを手動で破壊した借用入力が、owner内の保存済みcopyへ波及しない。
    object.__setattr__(first_adaptation_record, "adaptation_sample_index", 999)
    object.__setattr__(first_adaptation_record, "detector_name", "changed")
    assert first_snapshot.adaptation_records[0].adaptation_sample_index == 9
    assert first_snapshot.adaptation_records[0].detector_name == " ClassESR "
    assert final_snapshot.adaptation_records[0] == first_snapshot.adaptation_records[0]
    assert adaptation_record_store.get_state_snapshot() == final_snapshot


INVALID_ADAPTATION_RECORD_FIELDS = (
    ("adaptation_sample_index", True, TypeError),
    ("adaptation_sample_index", 1.0, TypeError),
    ("adaptation_sample_index", -1, ValueError),
    ("detector_name", None, TypeError),
    ("detector_name", "", ValueError),
    ("detector_name", " \t", ValueError),
    ("previous_training_model_id", True, TypeError),
    ("current_training_model_id", False, TypeError),
    ("previous_training_model_id", 1.0, TypeError),
    ("estimated_change_point_sample_index", False, TypeError),
    ("estimated_change_point_sample_index", -1, ValueError),
    ("estimated_change_point_sample_index", 1.0, TypeError),
    ("detection_episode_id", True, TypeError),
    ("detection_episode_id", -1, ValueError),
    ("detection_episode_id", 1.0, TypeError),
    ("adaptation_outcome", 1, TypeError),
    ("adaptation_outcome", "reuse", ValueError),
    ("adaptation_outcome", "alarm_interval_current_model_maintained", ValueError),
    # 負のIDは有効。同一IDの再利用結果という不整合を拒否する。
    ("current_training_model_id", -1, ValueError),
)


@pytest.mark.parametrize(
    "parameter_name,specified_value,expected_exception", INVALID_ADAPTATION_RECORD_FIELDS
)
def test_record_constructor_and_store_reject_invalid_fields_before_any_update(
    parameter_name,
    specified_value,
    expected_exception,
):
    with pytest.raises(ValueError):
        make_adaptation_record(adaptation_outcome="unknown", current_training_model_id=-1)
    valid_adaptation_record = make_adaptation_record()
    with pytest.raises(expected_exception):
        replace(valid_adaptation_record, **{parameter_name: specified_value})
    adaptation_record_store = AdaptationRecordStore()
    adaptation_record_store.append_adaptation_record(adaptation_record=valid_adaptation_record)
    previous_snapshot = adaptation_record_store.get_state_snapshot()
    invalid_adaptation_record = replace(valid_adaptation_record)
    # frozenを手動で破壊した入力も、append前にconstructor相当の全field検査を通す。
    object.__setattr__(invalid_adaptation_record, parameter_name, specified_value)
    with pytest.raises(expected_exception):
        adaptation_record_store.append_adaptation_record(
            adaptation_record=invalid_adaptation_record
        )
    assert adaptation_record_store.get_state_snapshot() == previous_snapshot
    with pytest.raises(TypeError):
        adaptation_record_store.append_adaptation_record(adaptation_record=object())
    assert adaptation_record_store.get_state_snapshot() == previous_snapshot


@pytest.mark.parametrize(
    "invalid_recording_case",
    [
        "completion_type",
        "store_type",
        "detector_type",
        "detector_empty",
        "detector_whitespace",
        "completion_bool_id",
        "completion_bool_baseline",
        "completion_inconsistent_ids",
        "completion_negative_optional_index",
        "completion_unknown_outcome",
    ],
)
def test_recording_rejects_invalid_input_before_updating_populated_store(
    invalid_recording_case,
    monkeypatch,
    valid_run_settings_mapping,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        (
            alarm_response_completion,
            response_arguments,
            completion_arguments,
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
            legacy_result,
        ) = build_completed_recording_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
        )
        adaptation_record_store = AdaptationRecordStore()
        adaptation_record_store.append_adaptation_record(adaptation_record=make_adaptation_record())
        previous_snapshot = adaptation_record_store.get_state_snapshot()
        recording_arguments = dict(
            alarm_response_completion=replace(alarm_response_completion),
            detector_name="ClassESR",
            adaptation_record_store=adaptation_record_store,
        )
        expected_exception = (
            TypeError
            if invalid_recording_case
            in (
                "completion_type",
                "store_type",
                "detector_type",
                "completion_bool_id",
                "completion_bool_baseline",
            )
            else ValueError
        )
        if invalid_recording_case in ("completion_type", "store_type"):
            recording_arguments[
                "alarm_response_completion"
                if invalid_recording_case == "completion_type"
                else "adaptation_record_store"
            ] = object()
        elif invalid_recording_case.startswith("detector_"):
            recording_arguments["detector_name"] = {
                "detector_type": 1,
                "detector_empty": "",
                "detector_whitespace": " \t",
            }[invalid_recording_case]
        elif invalid_recording_case == "completion_unknown_outcome":
            object.__setattr__(
                recording_arguments["alarm_response_completion"],
                "previous_training_model_id",
                alarm_response_completion.current_training_model_id,
            )
            object.__setattr__(
                recording_arguments["alarm_response_completion"],
                "alarm_buffer_response",
                replace(alarm_response_completion.alarm_buffer_response),
            )
            object.__setattr__(
                recording_arguments["alarm_response_completion"].alarm_buffer_response,
                "response_outcome",
                "unknown",
            )
        else:
            parameter_name, specified_value = {
                "completion_bool_id": ("previous_training_model_id", True),
                "completion_bool_baseline": ("loss_monitoring_baseline_mean_loss", True),
                "completion_inconsistent_ids": (
                    "previous_training_model_id",
                    alarm_response_completion.current_training_model_id,
                ),
                "completion_negative_optional_index": ("detection_episode_id", -1),
            }[invalid_recording_case]
            object.__setattr__(
                recording_arguments["alarm_response_completion"], parameter_name, specified_value
            )

        def reject_recording_before_append():
            with patch.object(
                adaptation_record_store, "append_adaptation_record"
            ) as append_record_calls:
                with pytest.raises(expected_exception):
                    record_completed_alarm_response(**recording_arguments)
                append_record_calls.assert_not_called()
            assert adaptation_record_store.get_state_snapshot() == previous_snapshot

        assert_recording_preserves_upstream_state(
            alarm_response_completion=alarm_response_completion,
            response_arguments=response_arguments,
            completion_arguments=completion_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            recording_operation=reject_recording_before_append,
        )
