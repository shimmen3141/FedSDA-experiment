"""候補検証sessionの保持と、警報応答の反映・標本ごとの進行・終端回収の接続を、実旧のsession保持と照合する。"""

from dataclasses import FrozenInstanceError, replace
from inspect import signature
from unittest.mock import Mock

import pytest
import torch
from test_adahedge_diagnostic_evidence import assert_adahedge_matches_legacy
from test_alarm_adaptation_recording import (
    RECORDING_ORACLE_CASES,
    build_completed_recording_oracle,
    make_adaptation_record,
)
from test_candidate_validation_adaptation_recording import (
    assert_random_states_unchanged,
    assert_recorded_event_matches_legacy,
    replace_frozen_fields,
    snapshot_random_states,
)
from test_held_adahedge_diagnostic_notification import get_diagnostic_collection_snapshot
from test_incomplete_post_alarm_candidate_validation_finalization import (
    assert_incomplete_validation_finalization_matches_legacy,
    build_incomplete_validation_finalization_oracle,
)
from test_post_alarm_candidate_validation_progress import (
    assert_validation_progress_matches_legacy,
    build_validation_progress_oracle,
    set_scripted_validation_losses,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

import federated_learning_experiments.runtime.held_candidate_validation_progress as held_progress_module
from federated_drift_experiment.clients.fedsda import (
    RestartingSoftRoutingClassConditionalESRFedSDAClient,
)
from federated_drift_experiment.expert_routing import AdaHedgeRouter
from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import (
    AdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.evaluation.adaptation_record_store import (
    AdaptationRecordStore,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)
from federated_learning_experiments.runtime.candidate_validation_session_holder import (
    CandidateValidationSessionHolder,
)
from federated_learning_experiments.runtime.held_candidate_validation_progress import (
    HeldCandidateValidationAdvance,
    HeldIncompleteCandidateValidationFinalization,
    advance_held_candidate_validation,
    apply_alarm_response_to_validation_session_holder,
    finalize_held_incomplete_candidate_validation,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
)


def make_started_validation_session(*, monkeypatch):
    """実際の候補・参照・損失収集を持つ、開始済みの候補検証sessionを1つ作る。"""
    return build_validation_progress_oracle(
        class_count=2, legacy_resolution_case="create", monkeypatch=monkeypatch
    )[0]["validation_session"]


def make_holder_holding(*, validation_session):
    validation_session_holder = CandidateValidationSessionHolder()
    if validation_session is not None:
        validation_session_holder.hold_validation_session(validation_session=validation_session)
    return validation_session_holder


# 再始動が観測できるよう、確定の前に新旧の診断証拠へ与える損失。
DIAGNOSTIC_LOSSES_BEFORE_VALIDATION = {4: 0.1, 9: 0.8}


def make_diagnostics_observing_losses(*, legacy_client=None):
    """損失を1回観測済みの診断証拠を作る。実旧clientを渡すと、実旧の再始動hookと同じ損失の実旧routerも置く。"""
    diagnostic_evidence_collection = AdaHedgeDiagnosticEvidenceCollection()
    global_diagnostic_evidence = diagnostic_evidence_collection.global_diagnostic_evidence
    global_diagnostic_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id=DIAGNOSTIC_LOSSES_BEFORE_VALIDATION,
        diagnostic_weights_by_model_id=global_diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
            model_ids=tuple(DIAGNOSTIC_LOSSES_BEFORE_VALIDATION)
        ),
    )
    if legacy_client is None:
        return diagnostic_evidence_collection
    # 上流oracleの記録用hookを残したまま、実旧の再始動（確定した切替ごとのAdaHedge再始動）を実行させる。
    legacy_client.expert_router = AdaHedgeRouter()
    legacy_client.context_expert_routers = {}
    legacy_client.shadow_meta_routers = {}
    legacy_client.routing_active_set = None
    legacy_client.expert_router.update(
        DIAGNOSTIC_LOSSES_BEFORE_VALIDATION,
        legacy_client.expert_router.probabilities(DIAGNOSTIC_LOSSES_BEFORE_VALIDATION),
    )
    record_local_model_change = legacy_client._on_local_model_change

    def record_change_then_restart_legacy_routers(previous_model_id, current_model_id):
        record_local_model_change(previous_model_id, current_model_id)
        RestartingSoftRoutingClassConditionalESRFedSDAClient._on_local_model_change(
            legacy_client, previous_model_id, current_model_id
        )

    legacy_client._on_local_model_change = record_change_then_restart_legacy_routers
    return diagnostic_evidence_collection


def make_record_store_with_alarm_record():
    adaptation_record_store = AdaptationRecordStore()
    adaptation_record_store.append_adaptation_record(adaptation_record=make_adaptation_record())
    return adaptation_record_store


def test_session_holder_holds_one_session_and_rejects_before_changing(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        validation_session = make_started_validation_session(monkeypatch=monkeypatch)
    validation_session_holder = CandidateValidationSessionHolder()
    assert validation_session_holder.held_validation_session is None
    with pytest.raises(LookupError):
        validation_session_holder.release_validation_session()
    for invalid_validation_session in (None, object(), "session"):
        with pytest.raises(TypeError):
            validation_session_holder.hold_validation_session(
                validation_session=invalid_validation_session
            )
        assert validation_session_holder.held_validation_session is None
    validation_session_holder.hold_validation_session(validation_session=validation_session)
    assert validation_session_holder.held_validation_session is validation_session
    # 進行中のsessionを黙って置き換えない。同じsessionの再保持も拒否する。
    for second_validation_session in (validation_session, replace(validation_session)):
        with pytest.raises(ValueError):
            validation_session_holder.hold_validation_session(
                validation_session=second_validation_session
            )
        assert validation_session_holder.held_validation_session is validation_session
    assert validation_session_holder.release_validation_session() is validation_session
    assert validation_session_holder.held_validation_session is None
    with pytest.raises(TypeError):
        validation_session_holder.hold_validation_session(validation_session)
    with pytest.raises(AttributeError):
        validation_session_holder.held_validation_session = validation_session


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("recording_oracle_case", RECORDING_ORACLE_CASES)
def test_alarm_response_updates_holder_like_legacy_session_attribute(
    class_count, recording_oracle_case, monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        alarm_response_completion, _, _, _, _, legacy_client, _ = build_completed_recording_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=class_count,
            recording_oracle_case=recording_oracle_case,
        )
    alarm_buffer_response = alarm_response_completion.alarm_buffer_response
    response_outcome = alarm_buffer_response.response_outcome
    # 候補検証中の警報は、開始済みのsessionを保持した状態で受ける。それ以外は空の保持で受ける。
    validation_session_holder = make_holder_holding(
        validation_session=alarm_buffer_response.active_validation_session
        if response_outcome == "alarm_during_candidate_validation"
        else None
    )
    apply_alarm_response_to_validation_session_holder(
        alarm_buffer_response=alarm_buffer_response,
        validation_session_holder=validation_session_holder,
    )
    assert (
        validation_session_holder.held_validation_session
        is alarm_buffer_response.active_validation_session
    )
    # 実旧は、候補検証を開始した警報と候補検証中の警報の後だけsessionを持つ。
    assert (validation_session_holder.held_validation_session is not None) == (
        legacy_client._forward_validation is not None
    )
    assert (validation_session_holder.held_validation_session is not None) == (
        response_outcome
        in ("alarm_interval_candidate_validation_started", "alarm_during_candidate_validation")
    )


# 手で足すsessionの代わりの値。応答と区間解決の同じ場所へ同一の値を入れるために使う。
INJECTED_SESSION_MARKER = object()

# 条件名 → (元にする警報応答の条件, 応答の前に保持させるsession, 応答を不正にする操作, 期待する例外)。
# 保持させるsession: "none"は空、"response"は応答のsession、"other"は別の開始済みsession。
INVALID_ALARM_RESPONSE_HOLDER_CASES = {
    "response_is_not_exact_record": (
        RECORDING_ORACLE_CASES[2],
        "none",
        lambda alarm_buffer_response: object(),
        TypeError,
    ),
    # 正式な5値でない結果種別。応答自身の検査の再実行だけが拒否する（通ると何もせず正常終了する）。
    "response_outcome_is_unknown": (
        RECORDING_ORACLE_CASES[2],
        "none",
        lambda alarm_buffer_response: replace_frozen_fields(
            alarm_buffer_response, response_outcome="maintain"
        ),
        ValueError,
    ),
    "validation_alarm_with_empty_holder": (
        RECORDING_ORACLE_CASES[4],
        "none",
        lambda alarm_buffer_response: alarm_buffer_response,
        ValueError,
    ),
    "validation_alarm_for_other_session": (
        RECORDING_ORACLE_CASES[4],
        "other",
        lambda alarm_buffer_response: alarm_buffer_response,
        ValueError,
    ),
    "started_validation_while_holding_other_session": (
        RECORDING_ORACLE_CASES[3],
        "other",
        lambda alarm_buffer_response: alarm_buffer_response,
        ValueError,
    ),
    "started_validation_while_holding_same_session": (
        RECORDING_ORACLE_CASES[3],
        "response",
        lambda alarm_buffer_response: alarm_buffer_response,
        ValueError,
    ),
    "maintained_model_while_holding_session": (
        RECORDING_ORACLE_CASES[2],
        "other",
        lambda alarm_buffer_response: alarm_buffer_response,
        ValueError,
    ),
    "too_short_interval_while_holding_session": (
        RECORDING_ORACLE_CASES[0],
        "other",
        lambda alarm_buffer_response: alarm_buffer_response,
        ValueError,
    ),
    # 開始の応答からsessionだけを外した、手で壊した応答。空の保持のまま通ると開始を見落とす。
    "started_response_with_removed_session": (
        RECORDING_ORACLE_CASES[3],
        "none",
        lambda alarm_buffer_response: replace_frozen_fields(
            alarm_buffer_response, active_validation_session=None
        ),
        ValueError,
    ),
    # 維持の応答へsessionを足した、手で壊した応答。通ると開始していないsessionを保持する。
    "maintained_response_with_added_session": (
        RECORDING_ORACLE_CASES[2],
        "none",
        lambda alarm_buffer_response: replace_frozen_fields(
            alarm_buffer_response, active_validation_session=object()
        ),
        ValueError,
    ),
    # 応答と区間解決の両方を手で差し替えた入力。応答自身の検査（両者のsessionが同一であること）は通る。
    # 再利用の応答へsessionを足す。通ると、開始していないsessionを保持する。
    "reused_response_and_resolution_with_added_session": (
        RECORDING_ORACLE_CASES[1],
        "none",
        lambda alarm_buffer_response: replace_frozen_fields(
            alarm_buffer_response,
            active_validation_session=INJECTED_SESSION_MARKER,
            change_interval_resolution=replace_frozen_fields(
                alarm_buffer_response.change_interval_resolution,
                started_validation_session=INJECTED_SESSION_MARKER,
            ),
        ),
        ValueError,
    ),
    # 開始の応答からsessionを外す。通ると、何も保持せずに開始を見落とす。
    "started_response_and_resolution_with_removed_session": (
        RECORDING_ORACLE_CASES[3],
        "none",
        lambda alarm_buffer_response: replace_frozen_fields(
            alarm_buffer_response,
            active_validation_session=None,
            change_interval_resolution=replace_frozen_fields(
                alarm_buffer_response.change_interval_resolution,
                started_validation_session=None,
            ),
        ),
        ValueError,
    ),
}


@pytest.mark.parametrize("invalid_case", INVALID_ALARM_RESPONSE_HOLDER_CASES)
def test_alarm_response_is_rejected_before_changing_holder(
    invalid_case, monkeypatch, valid_run_settings_mapping
):
    recording_oracle_case, held_session_source, make_invalid_response, expected_exception = (
        INVALID_ALARM_RESPONSE_HOLDER_CASES[invalid_case]
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        alarm_buffer_response = build_completed_recording_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            recording_oracle_case=recording_oracle_case,
        )[0].alarm_buffer_response
        held_validation_session = {
            "none": None,
            "response": alarm_buffer_response.active_validation_session,
            "other": make_started_validation_session(monkeypatch=monkeypatch),
        }[held_session_source]
    validation_session_holder = make_holder_holding(validation_session=held_validation_session)
    with pytest.raises(expected_exception):
        apply_alarm_response_to_validation_session_holder(
            alarm_buffer_response=make_invalid_response(alarm_buffer_response),
            validation_session_holder=validation_session_holder,
        )
    assert validation_session_holder.held_validation_session is held_validation_session
    with pytest.raises(TypeError):
        apply_alarm_response_to_validation_session_holder(
            alarm_buffer_response=alarm_buffer_response, validation_session_holder=object()
        )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize(
    "legacy_resolution_case", ("create", "reuse", "maintain", "create_rejected")
)
def test_completed_held_validation_is_recorded_then_released_like_legacy(
    class_count, legacy_resolution_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        progress_arguments, resolution_arguments, shared_optimizer_owners, legacy_client, _ = (
            build_validation_progress_oracle(
                class_count=class_count,
                legacy_resolution_case=legacy_resolution_case,
                monkeypatch=monkeypatch,
            )
        )
        set_scripted_validation_losses(
            progress_arguments=progress_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        validation_session = progress_arguments["validation_session"]
        validation_session_holder = make_holder_holding(validation_session=validation_session)
        adaptation_record_store = make_record_store_with_alarm_record()
        previous_state_snapshot = adaptation_record_store.get_state_snapshot()
        previous_model_id = legacy_client.current_model_id
        diagnostic_evidence_collection = make_diagnostics_observing_losses(
            legacy_client=legacy_client
        )
        # 通知の時点で、記録が追加済みで保持が空であること（記録→解除→通知の順）。
        notification_calls = []
        notify_diagnostics = held_progress_module.notify_diagnostics_of_training_assignment_change

        def record_notification_call(**notification_arguments):
            notification_calls.append(
                (
                    notification_arguments,
                    validation_session_holder.held_validation_session,
                    len(adaptation_record_store.get_state_snapshot().adaptation_records),
                )
            )
            return notify_diagnostics(**notification_arguments)

        monkeypatch.setattr(
            held_progress_module,
            "notify_diagnostics_of_training_assignment_change",
            record_notification_call,
        )
        # 解除の時点で適応記録が追加済みであること（旧のイベント記録→session解除の順）。
        record_counts_at_release = []
        release_validation_session = validation_session_holder.release_validation_session

        def record_release_of_validation_session():
            record_counts_at_release.append(
                len(adaptation_record_store.get_state_snapshot().adaptation_records)
            )
            return release_validation_session()

        monkeypatch.setattr(
            validation_session_holder,
            "release_validation_session",
            record_release_of_validation_session,
        )
        legacy_drift_type = legacy_client._observe_forward_validation(
            progress_arguments["input_features"], progress_arguments["observed_class_labels"], 57
        )
        held_validation_advance = advance_held_candidate_validation(
            validation_session_holder=validation_session_holder,
            adaptation_record_store=adaptation_record_store,
            diagnostic_evidence_collection=diagnostic_evidence_collection,
            **{
                argument_name: argument
                for argument_name, argument in progress_arguments.items()
                if argument_name != "validation_session"
            },
        )
        assert type(held_validation_advance) is HeldCandidateValidationAdvance
        # 診断: 通知は1回で、確定の結果の帰属変更そのものを渡す。実旧の再始動hookの後のAdaHedgeと一致し、
        # 再始動は学習帰属が変わる確定（候補の採用、他モデルの再利用）のときだけ1回。
        validation_resolution = (
            held_validation_advance.validation_progress.completed_validation.validation_resolution
        )
        assert len(notification_calls) == 1
        notification_arguments, held_session_at_notification, record_count_at_notification = (
            notification_calls[0]
        )
        assert notification_arguments.keys() == {
            "assignment_change",
            "diagnostic_evidence_collection",
        }
        assert (
            notification_arguments["assignment_change"]
            is validation_resolution.training_model_assignment_change
        )
        assert (
            notification_arguments["diagnostic_evidence_collection"]
            is diagnostic_evidence_collection
        )
        assert held_session_at_notification is None
        assert record_count_at_notification == len(previous_state_snapshot.adaptation_records) + 1
        global_diagnostic_evidence = diagnostic_evidence_collection.global_diagnostic_evidence
        assert_adahedge_matches_legacy(global_diagnostic_evidence, legacy_client.expert_router)
        assert global_diagnostic_evidence.concept_operation_restart_count == (
            1 if legacy_resolution_case in ("create", "reuse") else 0
        )
        assert (legacy_client.current_model_id != previous_model_id) == (
            legacy_resolution_case in ("create", "reuse")
        )
        assert validation_session_holder.held_validation_session is None
        assert legacy_client._forward_validation is None
        assert record_counts_at_release == [len(previous_state_snapshot.adaptation_records) + 1]
        assert_recorded_event_matches_legacy(
            adaptation_record=held_validation_advance.adaptation_record,
            adaptation_record_store=adaptation_record_store,
            previous_state_snapshot=previous_state_snapshot,
            legacy_client=legacy_client,
        )
        assert_validation_progress_matches_legacy(
            validation_progress=held_validation_advance.validation_progress,
            progress_arguments=progress_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            previous_model_id=previous_model_id,
            monkeypatch=monkeypatch,
        )
    with pytest.raises(FrozenInstanceError):
        held_validation_advance.adaptation_record = None
    with pytest.raises(TypeError):
        HeldCandidateValidationAdvance(held_validation_advance.validation_progress, None)


def test_unfinished_held_validation_keeps_session_and_adds_no_record(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        progress_arguments, _, _, legacy_client, _ = build_validation_progress_oracle(
            class_count=2, legacy_resolution_case="reuse", monkeypatch=monkeypatch
        )
        # 要求件数4のうち、まだ1件も観測していないsessionにする（新旧とも）。
        validation_session = replace(
            progress_arguments["validation_session"],
            post_alarm_candidate_loss_collection=PostAlarmCandidateLossCollection(
                candidate_model_training_and_acceptance_settings=progress_arguments[
                    "candidate_model_training_and_acceptance_settings"
                ],
                proposal_sample_index=53,
                reference_model_ids=(4, 9),
            ),
        )
        progress_arguments["validation_session"] = validation_session
        legacy_session = legacy_client._forward_validation
        # 損失の固定は実旧sessionの先頭の損失を読むので、実旧の観測済み損失を空にする前に行う。
        set_scripted_validation_losses(
            progress_arguments=progress_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        legacy_session.candidate_losses = []
        legacy_session.reference_losses = {
            model_id: [] for model_id in legacy_session.reference_losses
        }
        validation_session_holder = make_holder_holding(validation_session=validation_session)
        adaptation_record_store = make_record_store_with_alarm_record()
        previous_state_snapshot = adaptation_record_store.get_state_snapshot()
        diagnostic_evidence_collection = make_diagnostics_observing_losses()
        diagnostic_snapshot = get_diagnostic_collection_snapshot(diagnostic_evidence_collection)
        for sample_index in (54, 55, 56):
            assert (
                legacy_client._observe_forward_validation(
                    progress_arguments["input_features"],
                    progress_arguments["observed_class_labels"],
                    sample_index,
                )
                == 0
            )
            held_validation_advance = advance_held_candidate_validation(
                validation_session_holder=validation_session_holder,
                adaptation_record_store=adaptation_record_store,
                diagnostic_evidence_collection=diagnostic_evidence_collection,
                **{
                    argument_name: argument
                    for argument_name, argument in progress_arguments.items()
                    if argument_name != "validation_session"
                }
                | dict(sample_index=sample_index),
            )
            # 未到達の間は診断へ通知しない。
            assert (
                get_diagnostic_collection_snapshot(diagnostic_evidence_collection)
                == diagnostic_snapshot
            )
            assert held_validation_advance.adaptation_record is None
            assert held_validation_advance.validation_progress.completed_validation is None
            assert (
                held_validation_advance.validation_progress.session_to_continue
                is validation_session
            )
            assert validation_session_holder.held_validation_session is validation_session
            assert legacy_client._forward_validation is legacy_session
            assert adaptation_record_store.get_state_snapshot() == previous_state_snapshot
        assert (
            validation_session.post_alarm_candidate_loss_collection.validation_sample_count
            == len(legacy_session.candidate_losses)
            == 3
        )


def make_placeholder_arguments(*, operation, validation_session_holder, adaptation_record_store):
    """保持・記録・診断のowner以外をobject()にした引数。sessionがなければ他の引数は読まれない。"""
    placeholder_arguments = {
        argument_name: object() for argument_name in signature(operation).parameters
    } | dict(
        validation_session_holder=validation_session_holder,
        adaptation_record_store=adaptation_record_store,
    )
    # 診断のownerを受け取るのは進行だけ（終端回収は学習帰属を変えない）。
    if "diagnostic_evidence_collection" in placeholder_arguments:
        placeholder_arguments["diagnostic_evidence_collection"] = (
            make_diagnostics_observing_losses()
        )
    return placeholder_arguments


def test_operations_without_held_session_change_nothing():
    validation_session_holder = CandidateValidationSessionHolder()
    adaptation_record_store = make_record_store_with_alarm_record()
    previous_state_snapshot = adaptation_record_store.get_state_snapshot()
    random_states = snapshot_random_states()
    placeholder_arguments = make_placeholder_arguments(
        operation=advance_held_candidate_validation,
        validation_session_holder=validation_session_holder,
        adaptation_record_store=adaptation_record_store,
    )
    diagnostic_snapshot = get_diagnostic_collection_snapshot(
        placeholder_arguments["diagnostic_evidence_collection"]
    )
    held_validation_advance = advance_held_candidate_validation(**placeholder_arguments)
    assert (
        get_diagnostic_collection_snapshot(placeholder_arguments["diagnostic_evidence_collection"])
        == diagnostic_snapshot
    )
    assert held_validation_advance.adaptation_record is None
    assert (
        held_validation_advance.validation_progress.session_to_continue
        is held_validation_advance.validation_progress.completed_validation
        is None
    )
    assert (
        finalize_held_incomplete_candidate_validation(
            **make_placeholder_arguments(
                operation=finalize_held_incomplete_candidate_validation,
                validation_session_holder=validation_session_holder,
                adaptation_record_store=adaptation_record_store,
            )
        )
        is None
    )
    assert validation_session_holder.held_validation_session is None
    assert adaptation_record_store.get_state_snapshot() == previous_state_snapshot
    assert_random_states_unchanged(random_states=random_states)


@pytest.mark.parametrize(
    "operation,upstream_operation_name,invalid_owner_name",
    [
        (
            advance_held_candidate_validation,
            "advance_post_alarm_candidate_validation",
            "validation_session_holder",
        ),
        (
            advance_held_candidate_validation,
            "advance_post_alarm_candidate_validation",
            "adaptation_record_store",
        ),
        (
            advance_held_candidate_validation,
            "advance_post_alarm_candidate_validation",
            "diagnostic_evidence_collection",
        ),
        (
            finalize_held_incomplete_candidate_validation,
            "finalize_incomplete_post_alarm_candidate_validation",
            "validation_session_holder",
        ),
        (
            finalize_held_incomplete_candidate_validation,
            "finalize_incomplete_post_alarm_candidate_validation",
            "adaptation_record_store",
        ),
    ],
)
@pytest.mark.parametrize("invalid_owner_kind", ("other_type", "subclass"))
@pytest.mark.parametrize("session_is_held", (True, False))
def test_owner_types_are_rejected_before_upstream_updates(
    operation,
    upstream_operation_name,
    invalid_owner_name,
    invalid_owner_kind,
    session_is_held,
    monkeypatch,
):
    with torch.random.fork_rng(devices=[]):
        validation_session = make_started_validation_session(monkeypatch=monkeypatch)
    # 保持がないときも、ownerの型は同じく拒否する（保持がなければ上流は何もしないが、不正なownerを見逃さない）。
    held_validation_session = validation_session if session_is_held else None
    validation_session_holder = make_holder_holding(validation_session=held_validation_session)
    adaptation_record_store = make_record_store_with_alarm_record()
    previous_state_snapshot = adaptation_record_store.get_state_snapshot()
    # 上流の進行・終端回収は多くのownerを更新する。型の拒否はその呼出しより前でなければならない。
    monkeypatch.setattr(
        held_progress_module, upstream_operation_name, Mock(side_effect=AssertionError)
    )
    placeholder_arguments = make_placeholder_arguments(
        operation=operation,
        validation_session_holder=validation_session_holder,
        adaptation_record_store=adaptation_record_store,
    )
    # 不正にする引数をその操作が受け取ること（受け取らない引数のTypeErrorを拒否と取り違えない）。
    assert invalid_owner_name in placeholder_arguments
    owner_subclass = type("OwnerSubclass", (type(placeholder_arguments[invalid_owner_name]),), {})
    invalid_owner = (
        object() if invalid_owner_kind == "other_type" else owner_subclass.__new__(owner_subclass)
    )
    with pytest.raises(TypeError, match=f"{invalid_owner_name} must be exact"):
        operation(**placeholder_arguments | {invalid_owner_name: invalid_owner})
    assert validation_session_holder.held_validation_session is held_validation_session
    assert adaptation_record_store.get_state_snapshot() == previous_state_snapshot


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("observed_validation_sample_count", (0, 3))
def test_incomplete_held_validation_is_recorded_then_released_like_legacy(
    class_count, observed_validation_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        finalization_arguments, resolution_arguments, shared_optimizer_owners, legacy_client = (
            build_incomplete_validation_finalization_oracle(
                class_count=class_count,
                observed_validation_sample_count=observed_validation_sample_count,
                monkeypatch=monkeypatch,
            )
        )
        validation_session = finalization_arguments["validation_session"]
        assert type(validation_session) is PostAlarmCandidateValidationSession
        validation_session_holder = make_holder_holding(validation_session=validation_session)
        adaptation_record_store = make_record_store_with_alarm_record()
        previous_state_snapshot = adaptation_record_store.get_state_snapshot()
        legacy_client.finalize_incomplete_forward_validation()
        held_incomplete_finalization = finalize_held_incomplete_candidate_validation(
            validation_session_holder=validation_session_holder,
            adaptation_record_store=adaptation_record_store,
            **{
                argument_name: argument
                for argument_name, argument in finalization_arguments.items()
                if argument_name != "validation_session"
            },
        )
        assert type(held_incomplete_finalization) is HeldIncompleteCandidateValidationFinalization
        assert validation_session_holder.held_validation_session is None
        assert legacy_client._forward_validation is None
        assert_recorded_event_matches_legacy(
            adaptation_record=held_incomplete_finalization.adaptation_record,
            adaptation_record_store=adaptation_record_store,
            previous_state_snapshot=previous_state_snapshot,
            legacy_client=legacy_client,
        )
        assert_incomplete_validation_finalization_matches_legacy(
            incomplete_validation_finalization=held_incomplete_finalization.incomplete_validation_finalization,
            finalization_arguments=finalization_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
        )
        # 解除した後の再回収は何もしない（旧もsessionがなければ何もしない）。
        legacy_event_count = len(legacy_client.adaptation_events)
        legacy_client.finalize_incomplete_forward_validation()
        state_snapshot = adaptation_record_store.get_state_snapshot()
        assert (
            finalize_held_incomplete_candidate_validation(
                validation_session_holder=validation_session_holder,
                adaptation_record_store=adaptation_record_store,
                **{
                    argument_name: argument
                    for argument_name, argument in finalization_arguments.items()
                    if argument_name != "validation_session"
                },
            )
            is None
        )
        assert adaptation_record_store.get_state_snapshot() == state_snapshot
        assert len(legacy_client.adaptation_events) == legacy_event_count
    with pytest.raises(FrozenInstanceError):
        held_incomplete_finalization.adaptation_record = None
