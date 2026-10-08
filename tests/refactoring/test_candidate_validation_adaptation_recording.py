"""候補検証の到達時の確定と未完了の終端回収の適応記録を、実旧のイベント・切替位置と照合する。"""

import random
from dataclasses import asdict, replace
from typing import get_args

import numpy as np
import pytest
import torch
from test_alarm_adaptation_recording import make_adaptation_record
from test_alarm_response_completion import LEGACY_ACTION_BY_RESPONSE_OUTCOME
from test_incomplete_post_alarm_candidate_validation_finalization import (
    assert_incomplete_validation_finalization_matches_legacy,
    build_incomplete_validation_finalization_oracle,
)
from test_post_alarm_candidate_validation_progress import (
    assert_validation_progress_matches_legacy,
    build_validation_progress_oracle,
    set_scripted_validation_losses,
)

from federated_learning_experiments.evaluation.adaptation_record_store import (
    TRAINING_MODEL_SWITCH_ADAPTATION_OUTCOMES,
    AdaptationOutcome,
    AdaptationRecord,
    AdaptationRecordStore,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.runtime.alarm_buffer_response import (
    ALARM_BUFFER_RESPONSE_OUTCOMES,
)
from federated_learning_experiments.runtime.candidate_validation_adaptation_recording import (
    ADAPTATION_OUTCOME_BY_VALIDATION_RESOLUTION_OUTCOME,
    record_completed_candidate_validation,
    record_incomplete_candidate_validation_finalization,
)
from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import (
    finalize_incomplete_post_alarm_candidate_validation,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import (
    advance_post_alarm_candidate_validation,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import (
    POST_ALARM_CANDIDATE_VALIDATION_RESOLUTION_OUTCOMES,
)

# 候補検証の適応結果から旧イベントのactionへの対応。未完了の棄却も旧はcreate_rejectedと記録する。
LEGACY_ACTION_BY_VALIDATION_ADAPTATION_OUTCOME = {
    "post_alarm_validation_candidate_adopted": "create",
    "post_alarm_validation_held_model_reused": "reuse",
    "post_alarm_validation_current_model_maintained": "maintain",
    "post_alarm_validation_candidate_rejected": "create_rejected",
    "post_alarm_validation_incomplete_candidate_rejected": "create_rejected",
}
# 上流oracleの旧確定条件名から、期待する適応結果への対応。
ADAPTATION_OUTCOME_BY_LEGACY_RESOLUTION_CASE = {
    "create": "post_alarm_validation_candidate_adopted",
    "reuse": "post_alarm_validation_held_model_reused",
    "maintain": "post_alarm_validation_current_model_maintained",
    "create_rejected": "post_alarm_validation_candidate_rejected",
}


def snapshot_random_states():
    return torch.get_rng_state().clone(), random.getstate(), np.random.get_state()


def assert_random_states_unchanged(*, random_states):
    torch_random_state, python_random_state, numpy_random_state = random_states
    assert torch.equal(torch.get_rng_state(), torch_random_state)
    assert random.getstate() == python_random_state
    assert np.random.get_state()[0] == numpy_random_state[0]
    assert np.array_equal(np.random.get_state()[1], numpy_random_state[1])
    assert np.random.get_state()[2:] == numpy_random_state[2:]


def make_populated_adaptation_record_store():
    """警報時の再利用1件を先に持つ記録owner。追加の順序と、拒否時の不変を確かめる。"""
    adaptation_record_store = AdaptationRecordStore()
    adaptation_record_store.append_adaptation_record(adaptation_record=make_adaptation_record())
    return adaptation_record_store


def assert_recorded_event_matches_legacy(
    *, adaptation_record, adaptation_record_store, previous_state_snapshot, legacy_client
):
    """返された記録と記録ownerの増分を、実旧の最後のイベントと切替位置に照合する。"""
    assert type(adaptation_record) is AdaptationRecord
    assert asdict(legacy_client.adaptation_events[-1]) == dict(
        position=adaptation_record.adaptation_sample_index,
        detector=adaptation_record.detector_name,
        action=LEGACY_ACTION_BY_VALIDATION_ADAPTATION_OUTCOME[adaptation_record.adaptation_outcome],
        old_model_id=adaptation_record.previous_training_model_id,
        new_model_id=adaptation_record.current_training_model_id,
        estimated_change_point=adaptation_record.estimated_change_point_sample_index,
        episode_id=adaptation_record.detection_episode_id,
    )
    assert adaptation_record.current_training_model_id == legacy_client.current_model_id
    state_snapshot = adaptation_record_store.get_state_snapshot()
    assert state_snapshot.adaptation_records == (
        *previous_state_snapshot.adaptation_records,
        adaptation_record,
    )
    assert state_snapshot.adaptation_records[-1] is not adaptation_record
    # 旧の切替位置は、候補採用と参照モデルの再利用のときだけ確定位置を1件追加する。
    assert state_snapshot.training_model_switch_sample_indices == (
        *previous_state_snapshot.training_model_switch_sample_indices,
        *legacy_client.local_switch_positions,
    )
    # 警報時の区間評価による2種類の件数は、候補検証の結果では増えない。
    assert (
        state_snapshot.alternative_model_reuse_count,
        state_snapshot.current_model_fit_count,
    ) == (
        previous_state_snapshot.alternative_model_reuse_count,
        previous_state_snapshot.current_model_fit_count,
    )


def complete_validation_in_both_implementations(
    *, class_count, legacy_resolution_case, pending_sample_count, monkeypatch
):
    """実旧の標本観測（到達時の確定を含む）と新の候補検証の進行を、同じ状態から実行する。"""
    progress_arguments, resolution_arguments, shared_optimizer_owners, legacy_client, _ = (
        build_validation_progress_oracle(
            class_count=class_count,
            legacy_resolution_case=legacy_resolution_case,
            pending_sample_count=pending_sample_count,
            monkeypatch=monkeypatch,
        )
    )
    set_scripted_validation_losses(
        progress_arguments=progress_arguments, legacy_client=legacy_client, monkeypatch=monkeypatch
    )
    previous_model_id = legacy_client.current_model_id
    legacy_drift_type = legacy_client._observe_forward_validation(
        progress_arguments["input_features"], progress_arguments["observed_class_labels"], 57
    )
    validation_progress = advance_post_alarm_candidate_validation(**progress_arguments)
    return (
        validation_progress,
        progress_arguments,
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_drift_type,
        previous_model_id,
    )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize(
    "legacy_resolution_case", ("create", "reuse", "maintain", "create_rejected")
)
@pytest.mark.parametrize("pending_sample_count", (0, 3))
def test_completed_validation_record_matches_real_legacy_event(
    class_count, legacy_resolution_case, pending_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        (
            validation_progress,
            progress_arguments,
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
            legacy_drift_type,
            previous_model_id,
        ) = complete_validation_in_both_implementations(
            class_count=class_count,
            legacy_resolution_case=legacy_resolution_case,
            pending_sample_count=pending_sample_count,
            monkeypatch=monkeypatch,
        )
        validation_completion = validation_progress.completed_validation
        adaptation_record_store = make_populated_adaptation_record_store()
        previous_state_snapshot = adaptation_record_store.get_state_snapshot()
        random_states = snapshot_random_states()
        adaptation_record = record_completed_candidate_validation(
            validation_completion=validation_completion,
            adaptation_record_store=adaptation_record_store,
        )
        assert_random_states_unchanged(random_states=random_states)
        assert (
            adaptation_record.adaptation_outcome
            == ADAPTATION_OUTCOME_BY_LEGACY_RESOLUTION_CASE[legacy_resolution_case]
        )
        assert_recorded_event_matches_legacy(
            adaptation_record=adaptation_record,
            adaptation_record_store=adaptation_record_store,
            previous_state_snapshot=previous_state_snapshot,
            legacy_client=legacy_client,
        )
        # 記録が読むのは完了情報の値と位置が同じであること。
        assert (
            adaptation_record.adaptation_sample_index
            in adaptation_record_store.get_state_snapshot().training_model_switch_sample_indices
        ) == (validation_completion.training_model_switch_sample_index is not None)
        # 記録の後も、完了情報と上流の全状態（保有モデル・統計・標本・計数・帰属）は実旧と一致したまま。
        assert validation_progress.completed_validation is validation_completion
        assert_validation_progress_matches_legacy(
            validation_progress=validation_progress,
            progress_arguments=progress_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            previous_model_id=previous_model_id,
            monkeypatch=monkeypatch,
        )


def finalize_incomplete_validation_in_both_implementations(
    *, class_count, observed_validation_sample_count, processed_sample_count, monkeypatch
):
    """実旧と新の未完了の終端回収を、同じ状態から実行する。"""
    finalization_arguments, resolution_arguments, shared_optimizer_owners, legacy_client = (
        build_incomplete_validation_finalization_oracle(
            class_count=class_count,
            observed_validation_sample_count=observed_validation_sample_count,
            processed_sample_count=processed_sample_count,
            monkeypatch=monkeypatch,
        )
    )
    legacy_client.finalize_incomplete_forward_validation()
    incomplete_validation_finalization = finalize_incomplete_post_alarm_candidate_validation(
        **finalization_arguments
    )
    return (
        incomplete_validation_finalization,
        finalization_arguments,
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
    )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("observed_validation_sample_count", (0, 3))
# 処理済み件数が提案位置以下のときは、旧の終端位置は提案位置になる。
@pytest.mark.parametrize("processed_sample_count", (50, 60))
def test_incomplete_validation_record_matches_real_legacy_event(
    class_count, observed_validation_sample_count, processed_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        (
            incomplete_validation_finalization,
            finalization_arguments,
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
        ) = finalize_incomplete_validation_in_both_implementations(
            class_count=class_count,
            observed_validation_sample_count=observed_validation_sample_count,
            processed_sample_count=processed_sample_count,
            monkeypatch=monkeypatch,
        )
        adaptation_record_store = make_populated_adaptation_record_store()
        previous_state_snapshot = adaptation_record_store.get_state_snapshot()
        random_states = snapshot_random_states()
        adaptation_record = record_incomplete_candidate_validation_finalization(
            incomplete_validation_finalization=incomplete_validation_finalization,
            adaptation_record_store=adaptation_record_store,
        )
        assert_random_states_unchanged(random_states=random_states)
        assert (
            adaptation_record.adaptation_outcome
            == "post_alarm_validation_incomplete_candidate_rejected"
        )
        assert adaptation_record.adaptation_sample_index == max(
            finalization_arguments["validation_session"].proposal_sample_index,
            processed_sample_count - 1,
        )
        assert legacy_client.local_switch_positions == []
        assert_recorded_event_matches_legacy(
            adaptation_record=adaptation_record,
            adaptation_record_store=adaptation_record_store,
            previous_state_snapshot=previous_state_snapshot,
            legacy_client=legacy_client,
        )
        assert_incomplete_validation_finalization_matches_legacy(
            incomplete_validation_finalization=incomplete_validation_finalization,
            finalization_arguments=finalization_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
        )


def test_adaptation_outcomes_cover_alarm_and_validation_results_exactly():
    validation_adaptation_outcomes = tuple(LEGACY_ACTION_BY_VALIDATION_ADAPTATION_OUTCOME)
    assert get_args(AdaptationOutcome) == (
        *ALARM_BUFFER_RESPONSE_OUTCOMES,
        *validation_adaptation_outcomes,
    )
    assert tuple(ADAPTATION_OUTCOME_BY_VALIDATION_RESOLUTION_OUTCOME) == (
        POST_ALARM_CANDIDATE_VALIDATION_RESOLUTION_OUTCOMES
    )
    assert (
        tuple(ADAPTATION_OUTCOME_BY_VALIDATION_RESOLUTION_OUTCOME.values())
        == validation_adaptation_outcomes[:4]
    )
    assert set(TRAINING_MODEL_SWITCH_ADAPTATION_OUTCOMES) < set(get_args(AdaptationOutcome))
    # 旧actionとの対応は、警報応答の5結果と合わせて全結果を覆う。
    assert set(LEGACY_ACTION_BY_RESPONSE_OUTCOME) | set(validation_adaptation_outcomes) == set(
        get_args(AdaptationOutcome)
    )


@pytest.mark.parametrize("adaptation_outcome", get_args(AdaptationOutcome))
def test_store_applies_switch_index_and_count_rules_for_every_outcome(adaptation_outcome):
    training_model_switches = adaptation_outcome in (
        "alarm_interval_held_model_reused",
        "post_alarm_validation_candidate_adopted",
        "post_alarm_validation_held_model_reused",
    )
    record_fields = dict(
        adaptation_sample_index=23,
        adaptation_outcome=adaptation_outcome,
        previous_training_model_id=4,
        current_training_model_id=-2 if training_model_switches else 4,
    )
    adaptation_record_store = AdaptationRecordStore()
    adaptation_record_store.append_adaptation_record(
        adaptation_record=make_adaptation_record(**record_fields)
    )
    state_snapshot = adaptation_record_store.get_state_snapshot()
    assert state_snapshot.training_model_switch_sample_indices == (
        (23,) if training_model_switches else ()
    )
    assert state_snapshot.alternative_model_reuse_count == (
        adaptation_outcome == "alarm_interval_held_model_reused"
    )
    assert state_snapshot.current_model_fit_count == (
        adaptation_outcome == "alarm_interval_current_model_maintained"
    )
    # 変更前後のIDが異なることと、帰属が変わる結果であることは必要十分。逆の組は作れない。
    with pytest.raises(ValueError):
        make_adaptation_record(
            **record_fields | dict(current_training_model_id=4 if training_model_switches else -2)
        )


def replace_frozen_fields(frozen_record, **field_values):
    """constructorの検査を通さずにfieldだけを差し替えたcopyを作る（手で壊した入力の再現）。"""
    replaced_record = replace(frozen_record)
    for field_name, field_value in field_values.items():
        object.__setattr__(replaced_record, field_name, field_value)
    return replaced_record


def replace_validation_resolution_fields(validation_completion, **field_values):
    return replace(
        validation_completion,
        validation_resolution=replace_frozen_fields(
            validation_completion.validation_resolution, **field_values
        ),
    )


def replace_decision_record_fields(completion_or_finalization, **field_values):
    return replace(
        completion_or_finalization,
        decision_record=replace_frozen_fields(
            completion_or_finalization.decision_record, **field_values
        ),
    )


# 条件名 → (元になる旧確定条件, 完了情報を不正にする操作, 期待する例外)。
INVALID_VALIDATION_COMPLETION_CASES = {
    "completion_is_not_exact_record": ("create", lambda completion: object(), TypeError),
    "decision_record_is_not_exact_record": (
        "create",
        lambda completion: replace(completion, decision_record=object()),
        TypeError,
    ),
    "resolution_is_not_exact_record": (
        "create",
        lambda completion: replace(completion, validation_resolution=object()),
        TypeError,
    ),
    # 結果種別の検査は、変更記録を持たない棄却を元にする。採用を元にすると、未知の値を別の結果として
    # 扱う誤りがあっても、変更記録との不対応で同じ例外になり区別できない。
    "resolution_outcome_is_not_str": (
        "create_rejected",
        lambda completion: replace_validation_resolution_fields(completion, resolution_outcome=1),
        TypeError,
    ),
    "resolution_outcome_is_unknown": (
        "create_rejected",
        lambda completion: replace_validation_resolution_fields(
            completion, resolution_outcome="create"
        ),
        ValueError,
    ),
    "previous_model_id_is_bool": (
        "maintain",
        lambda completion: replace(completion, previous_training_model_id=True),
        TypeError,
    ),
    "assigned_model_id_is_bool": (
        "maintain",
        lambda completion: replace_validation_resolution_fields(completion, assigned_model_id=True),
        TypeError,
    ),
    "assigned_model_is_not_resolved_training_model": (
        "create",
        lambda completion: replace_validation_resolution_fields(
            completion, assigned_model_id=completion.previous_training_model_id
        ),
        ValueError,
    ),
    # 変更記録がない維持・棄却で、保留標本の帰属先だけが学習帰属と違う。IDは変わらないので、
    # 記録のID整合の検査では拒否されない。
    "maintained_model_with_foreign_assigned_model": (
        "maintain",
        lambda completion: replace_validation_resolution_fields(
            completion, assigned_model_id=completion.previous_training_model_id + 100
        ),
        ValueError,
    ),
    "rejected_candidate_with_foreign_assigned_model": (
        "create_rejected",
        lambda completion: replace_validation_resolution_fields(
            completion, assigned_model_id=completion.previous_training_model_id + 100
        ),
        ValueError,
    ),
    "assignment_change_is_not_exact_record": (
        "create",
        lambda completion: replace_validation_resolution_fields(
            completion, training_model_assignment_change=object()
        ),
        TypeError,
    ),
    "assignment_change_has_bool_previous_model_id": (
        "create",
        lambda completion: replace_validation_resolution_fields(
            completion,
            training_model_assignment_change=TrainingModelAssignmentChange(
                previous_model_id=True,
                current_model_id=completion.validation_resolution.assigned_model_id,
            ),
        ),
        TypeError,
    ),
    "assignment_change_has_bool_current_model_id": (
        "create",
        lambda completion: replace_validation_resolution_fields(
            completion,
            training_model_assignment_change=TrainingModelAssignmentChange(
                previous_model_id=completion.previous_training_model_id, current_model_id=True
            ),
        ),
        TypeError,
    ),
    "assignment_change_starts_from_other_model": (
        "create",
        lambda completion: replace(
            completion, previous_training_model_id=completion.previous_training_model_id + 100
        ),
        ValueError,
    ),
    # 採用なのに変更記録がない。帰属先も変更前のモデルに合わせ、IDと結果の対応だけを破る。
    "adopted_candidate_without_assignment_change": (
        "create",
        lambda completion: replace_validation_resolution_fields(
            completion,
            training_model_assignment_change=None,
            assigned_model_id=completion.previous_training_model_id,
        ),
        ValueError,
    ),
    # 現行維持なのに変更記録がある。
    "maintained_model_with_assignment_change": (
        "maintain",
        lambda completion: replace_validation_resolution_fields(
            completion,
            training_model_assignment_change=TrainingModelAssignmentChange(
                previous_model_id=completion.previous_training_model_id,
                current_model_id=completion.previous_training_model_id + 100,
            ),
            assigned_model_id=completion.previous_training_model_id + 100,
        ),
        ValueError,
    ),
    # 棄却なのに、前後のIDが同じ変更記録がある。IDは変わらないので記録のID整合の検査では拒否されない。
    "rejected_candidate_with_unchanged_assignment_change": (
        "create_rejected",
        lambda completion: replace_validation_resolution_fields(
            completion,
            training_model_assignment_change=TrainingModelAssignmentChange(
                previous_model_id=completion.previous_training_model_id,
                current_model_id=completion.previous_training_model_id,
            ),
        ),
        ValueError,
    ),
    "resolution_sample_index_is_bool": (
        "create",
        lambda completion: replace_decision_record_fields(completion, resolution_sample_index=True),
        TypeError,
    ),
    "resolution_sample_index_is_negative": (
        "create",
        lambda completion: replace_decision_record_fields(completion, resolution_sample_index=-1),
        ValueError,
    ),
    # 確定位置は提案位置（警報位置）以降。判定記録の2つの位置の順序を破る。
    "resolution_sample_index_precedes_proposal": (
        "create",
        lambda completion: replace_decision_record_fields(
            completion,
            resolution_sample_index=completion.decision_record.proposal_sample_index - 1,
        ),
        ValueError,
    ),
    # 提案位置だけが負。確定位置は提案位置以降で非負なので、順序と記録の位置の検査では拒否されない。
    "proposal_sample_index_is_negative": (
        "create",
        lambda completion: replace_decision_record_fields(completion, proposal_sample_index=-1),
        ValueError,
    ),
    "proposal_sample_index_is_bool": (
        "create",
        lambda completion: replace_decision_record_fields(completion, proposal_sample_index=True),
        TypeError,
    ),
    "detector_name_is_blank": (
        "create",
        lambda completion: replace_decision_record_fields(completion, detector_name="  "),
        ValueError,
    ),
    "estimated_change_point_is_negative": (
        "create",
        lambda completion: replace(completion, estimated_change_point_sample_index=-1),
        ValueError,
    ),
    "detection_episode_id_is_bool": (
        "create",
        lambda completion: replace(completion, detection_episode_id=False),
        TypeError,
    ),
}


@pytest.mark.parametrize("invalid_case", INVALID_VALIDATION_COMPLETION_CASES)
def test_completed_validation_recording_rejects_invalid_input_before_updating_store(
    invalid_case, monkeypatch
):
    legacy_resolution_case, make_invalid_completion, expected_exception = (
        INVALID_VALIDATION_COMPLETION_CASES[invalid_case]
    )
    with torch.random.fork_rng(devices=[]):
        validation_progress = complete_validation_in_both_implementations(
            class_count=2,
            legacy_resolution_case=legacy_resolution_case,
            pending_sample_count=3,
            monkeypatch=monkeypatch,
        )[0]
    validation_completion = validation_progress.completed_validation
    adaptation_record_store = make_populated_adaptation_record_store()
    # 正常な完了情報は記録できる（不正化だけが拒否の原因であること）。
    record_completed_candidate_validation(
        validation_completion=validation_completion,
        adaptation_record_store=adaptation_record_store,
    )
    previous_state_snapshot = adaptation_record_store.get_state_snapshot()
    invalid_validation_completion = make_invalid_completion(validation_completion)
    with pytest.raises(expected_exception):
        record_completed_candidate_validation(
            validation_completion=invalid_validation_completion,
            adaptation_record_store=adaptation_record_store,
        )
    assert adaptation_record_store.get_state_snapshot() == previous_state_snapshot
    with pytest.raises(TypeError):
        record_completed_candidate_validation(
            validation_completion=validation_completion, adaptation_record_store=object()
        )


# 条件名 → (終端回収の結果を不正にする操作, 期待する例外)。
INVALID_INCOMPLETE_FINALIZATION_CASES = {
    "finalization_is_not_exact_record": (lambda finalization: object(), TypeError),
    "decision_record_is_not_exact_record": (
        lambda finalization: replace(finalization, decision_record=object()),
        TypeError,
    ),
    "current_model_id_is_bool": (
        lambda finalization: replace(finalization, current_training_model_id=True),
        TypeError,
    ),
    "finalization_sample_index_is_float": (
        lambda finalization: replace_decision_record_fields(
            finalization, finalization_sample_index=59.0
        ),
        TypeError,
    ),
    "finalization_sample_index_is_negative": (
        lambda finalization: replace_decision_record_fields(
            finalization, finalization_sample_index=-1
        ),
        ValueError,
    ),
    "finalization_sample_index_precedes_proposal": (
        lambda finalization: replace_decision_record_fields(
            finalization,
            finalization_sample_index=finalization.decision_record.proposal_sample_index - 1,
        ),
        ValueError,
    ),
    "proposal_sample_index_is_negative": (
        lambda finalization: replace_decision_record_fields(finalization, proposal_sample_index=-1),
        ValueError,
    ),
    "proposal_sample_index_is_none": (
        lambda finalization: replace_decision_record_fields(
            finalization, proposal_sample_index=None
        ),
        TypeError,
    ),
    "detector_name_is_not_str": (
        lambda finalization: replace_decision_record_fields(finalization, detector_name=None),
        TypeError,
    ),
    "estimated_change_point_is_bool": (
        lambda finalization: replace(finalization, estimated_change_point_sample_index=True),
        TypeError,
    ),
    "detection_episode_id_is_negative": (
        lambda finalization: replace(finalization, detection_episode_id=-1),
        ValueError,
    ),
}


@pytest.mark.parametrize("invalid_case", INVALID_INCOMPLETE_FINALIZATION_CASES)
def test_incomplete_validation_recording_rejects_invalid_input_before_updating_store(
    invalid_case, monkeypatch
):
    make_invalid_finalization, expected_exception = INVALID_INCOMPLETE_FINALIZATION_CASES[
        invalid_case
    ]
    with torch.random.fork_rng(devices=[]):
        incomplete_validation_finalization = finalize_incomplete_validation_in_both_implementations(
            class_count=2,
            observed_validation_sample_count=3,
            processed_sample_count=60,
            monkeypatch=monkeypatch,
        )[0]
    adaptation_record_store = make_populated_adaptation_record_store()
    record_incomplete_candidate_validation_finalization(
        incomplete_validation_finalization=incomplete_validation_finalization,
        adaptation_record_store=adaptation_record_store,
    )
    previous_state_snapshot = adaptation_record_store.get_state_snapshot()
    with pytest.raises(expected_exception):
        record_incomplete_candidate_validation_finalization(
            incomplete_validation_finalization=make_invalid_finalization(
                incomplete_validation_finalization
            ),
            adaptation_record_store=adaptation_record_store,
        )
    assert adaptation_record_store.get_state_snapshot() == previous_state_snapshot
    with pytest.raises(TypeError):
        record_incomplete_candidate_validation_finalization(
            incomplete_validation_finalization=incomplete_validation_finalization,
            adaptation_record_store=object(),
        )
