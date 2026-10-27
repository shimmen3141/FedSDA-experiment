"""候補検証sessionの保持と、警報応答の反映・標本ごとの進行・終端回収の接続を、実旧のsession保持と照合する。"""

from dataclasses import replace
from inspect import signature
from unittest.mock import Mock

import pytest
import torch
from test_adahedge_weight_evidence import (
    get_adahedge_evidence_snapshot,
)
from test_alarm_adaptation_recording import (
    RECORDING_ORACLE_CASES,
    build_completed_recording_oracle,
    make_adaptation_record,
)
from test_candidate_validation_adaptation_recording import (
    assert_random_states_unchanged,
    replace_frozen_fields,
    snapshot_random_states,
)
from test_post_alarm_candidate_validation_progress import (
    build_validation_progress_oracle,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

import federated_learning_experiments.runtime.held_candidate_validation_progress as held_progress_module
from federated_drift_experiment.clients.fedsda import (
    RestartingSoftRoutingClassConditionalESRFedSDAClient,
)
from federated_drift_experiment.expert_routing import AdaHedgeRouter
from federated_learning_experiments.evaluation.adaptation_record_store import (
    AdaptationRecordStore,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.all_sample_adahedge_weight_evidence_holder import (
    AllSampleAdaHedgeWeightEvidenceHolder,
)
from federated_learning_experiments.runtime.candidate_validation_session_holder import (
    CandidateValidationSessionHolder,
)
from federated_learning_experiments.runtime.held_candidate_validation_progress import (
    advance_held_candidate_validation,
    apply_alarm_response_to_validation_session_holder,
    finalize_held_incomplete_candidate_validation,
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
    all_sample_adahedge_weight_evidence_holder = AllSampleAdaHedgeWeightEvidenceHolder()
    all_sample_weight_evidence = (
        all_sample_adahedge_weight_evidence_holder.all_sample_weight_evidence
    )
    all_sample_weight_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id=DIAGNOSTIC_LOSSES_BEFORE_VALIDATION,
        weights_by_model_id=all_sample_weight_evidence.get_weights_before_loss_observation(
            model_ids=tuple(DIAGNOSTIC_LOSSES_BEFORE_VALIDATION)
        ),
    )
    if legacy_client is None:
        return all_sample_adahedge_weight_evidence_holder
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
    return all_sample_adahedge_weight_evidence_holder


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


def make_subclass_copy(original):
    """同じ属性を持つ、派生型の値（exact型の検査が拒否するべき値）。"""
    subclass = type(type(original).__name__ + "Subclass", (type(original),), {})
    subclass_copy = object.__new__(subclass)
    for attribute_name, attribute in vars(original).items():
        object.__setattr__(subclass_copy, attribute_name, attribute)
    return subclass_copy


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
    # 応答の派生型（fieldは正しい）。exact型の検査だけが拒否する。
    "response_is_subclass_record": (
        RECORDING_ORACLE_CASES[2],
        "none",
        lambda alarm_buffer_response: make_subclass_copy(alarm_buffer_response),
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
    # 保持のownerは、別の型も派生型（同じsessionを保持している）も拒否する。
    for invalid_holder in (object(), make_subclass_copy(validation_session_holder)):
        with pytest.raises(TypeError, match="validation_session_holder must be exact"):
            apply_alarm_response_to_validation_session_holder(
                alarm_buffer_response=alarm_buffer_response,
                validation_session_holder=invalid_holder,
            )
    assert validation_session_holder.held_validation_session is held_validation_session


def make_placeholder_arguments(*, operation, validation_session_holder, adaptation_record_store):
    """保持・記録・診断のowner以外をobject()にした引数。sessionがなければ他の引数は読まれない。"""
    placeholder_arguments = {
        argument_name: object() for argument_name in signature(operation).parameters
    } | dict(
        validation_session_holder=validation_session_holder,
        adaptation_record_store=adaptation_record_store,
    )
    # 診断のownerを受け取るのは進行だけ（終端回収は学習帰属を変えない）。
    if "all_sample_adahedge_weight_evidence_holder" in placeholder_arguments:
        placeholder_arguments["all_sample_adahedge_weight_evidence_holder"] = (
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
    diagnostic_snapshot = get_adahedge_evidence_snapshot(
        placeholder_arguments[
            "all_sample_adahedge_weight_evidence_holder"
        ].all_sample_weight_evidence
    )
    held_validation_advance = advance_held_candidate_validation(**placeholder_arguments)
    assert (
        get_adahedge_evidence_snapshot(
            placeholder_arguments[
                "all_sample_adahedge_weight_evidence_holder"
            ].all_sample_weight_evidence
        )
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
            "all_sample_adahedge_weight_evidence_holder",
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
    # 保持がないときも、ownerの型は同じく拒否する（不正なownerを、候補検証が始まるまで見逃さない）。
    held_validation_session = validation_session if session_is_held else None
    validation_session_holder = make_holder_holding(validation_session=held_validation_session)
    adaptation_record_store = make_record_store_with_alarm_record()
    previous_state_snapshot = adaptation_record_store.get_state_snapshot()
    # 上流の進行・終端回収は多くのownerを更新する。型の拒否はその呼出しより前でなければならない。
    # 上流は「呼ばれたら失敗するmock」に差し替える（保持がなくても、型検査が抜ければ呼ばれて失敗する）。
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
