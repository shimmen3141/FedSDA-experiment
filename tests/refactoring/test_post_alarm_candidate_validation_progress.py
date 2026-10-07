"""候補検証の判定記録と通常進行を実旧の処理へ対照する。"""

import math
import random
from dataclasses import FrozenInstanceError, replace
from inspect import Parameter, signature
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_adopted_candidate_local_adoption import (
    assert_adoption_state_unchanged,
    snapshot_adoption_state,
)
from test_post_alarm_candidate_loss_evaluation import (
    assert_evaluation_matches_legacy_decision,
    capture_legacy_candidate_decision,
    make_acceptance_settings,
)
from test_post_alarm_candidate_validation_resolution import (
    assert_resolution_matches_legacy,
    build_resolution_oracle,
)

import federated_learning_experiments.runtime.post_alarm_candidate_validation_progress as progress_module
import federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation as observation_module
from federated_learning_experiments.learning.training.candidate_epoch_training import (
    CandidateEpochTrainingResult,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    evaluate_candidate_using_post_alarm_losses,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import (
    PostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.runtime.candidate_classifier_construction import (
    IndependentCandidateTrainingState,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import (
    PostAlarmCandidateValidationCompletion,
    PostAlarmCandidateValidationProgress,
    advance_post_alarm_candidate_validation,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
)
from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import (
    FixedPostAlarmReferenceModels,
)


def build_validation_progress_oracle(
    *,
    class_count,
    legacy_resolution_case,
    pending_sample_count=3,
    optimizer_variant="standard",
    monkeypatch,
):
    """実旧確定oracleに開始済みsessionを結合し、最後の標本位置を57へ揃える。"""
    resolution_arguments, shared_optimizer_owners, legacy_client, legacy_candidate_model = (
        build_resolution_oracle(
            class_count=class_count,
            legacy_resolution_case=legacy_resolution_case,
            pending_sample_count=pending_sample_count,
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
    )
    legacy_session = legacy_client._forward_validation
    legacy_session.proposal_position = 53
    # 旧observeはfloat32 tensorからitemを得る。制御lossも同じ実値へ揃える。
    legacy_session.candidate_losses = [
        float(torch.tensor(loss).item()) for loss in legacy_session.candidate_losses
    ]
    legacy_session.reference_losses = {
        model_id: [float(torch.tensor(loss).item()) for loss in losses]
        for model_id, losses in legacy_session.reference_losses.items()
    }
    settings = make_acceptance_settings(sample_count=4)
    collection = PostAlarmCandidateLossCollection(
        candidate_model_training_and_acceptance_settings=settings,
        proposal_sample_index=53,
        reference_model_ids=tuple(legacy_session.reference_models),
    )
    for sample_index in (54, 55, 56):
        collection.observe_losses_after_label_observation(
            sample_index=sample_index,
            candidate_loss=legacy_session.candidate_losses[0],
            reference_losses_by_model_id={
                model_id: losses[0] for model_id, losses in legacy_session.reference_losses.items()
            },
        )
    registry = resolution_arguments["held_model_training_state_registry"]
    validation_session = PostAlarmCandidateValidationSession(
        proposal_sample_index=53,
        estimated_change_point_sample_index=legacy_session.estimated_change_point,
        detection_episode_id=legacy_session.episode_id,
        detector_name=legacy_session.detector,
        initial_training_model_id=legacy_session.old_model_id,
        candidate_training_state=IndependentCandidateTrainingState(
            candidate_classifier=resolution_arguments["adopted_candidate_classifier"],
            candidate_shared_parameter_optimizer_state=shared_optimizer_owners[-1],
            candidate_concept_specific_parameter_optimizer_state=resolution_arguments[
                "candidate_concept_specific_parameter_optimizer_state"
            ],
        ),
        candidate_epoch_training_result=CandidateEpochTrainingResult(
            completed_epoch_count=1,
            candidate_trained_sample_count=resolution_arguments["candidate_trained_sample_count"],
            candidate_parameter_update_step_count=resolution_arguments[
                "candidate_parameter_update_step_count"
            ],
            validation_evaluated_sample_count=0,
        ),
        training_input_features=resolution_arguments["initial_statistics_input_features"],
        training_observed_class_labels=resolution_arguments[
            "initial_statistics_observed_class_labels"
        ],
        pending_assignment_training_samples=resolution_arguments[
            "pending_assignment_training_samples"
        ],
        fixed_reference_models=FixedPostAlarmReferenceModels(
            reference_classifiers_by_model_id={
                state.model_id: state.classifier
                for state in registry.snapshot_ordered_held_model_training_states()
            },
            reference_historical_mean_losses_by_model_id=dict(
                legacy_session.reference_historical_means
            ),
        ),
        post_alarm_candidate_loss_collection=collection,
    )
    progress_arguments = {
        name: resolution_arguments[name]
        for name in (
            "pending_assignment_sample_concept_ids",
            "temporary_model_id_allocator",
            "upload_delay_round_count",
            "held_model_training_state_registry",
            "loss_statistics_store",
            "training_sample_store",
            "model_training_and_assignment_counts_store",
            "current_training_model_assignment",
            "pending_model_upload_state",
        )
    }
    progress_arguments.update(
        validation_session=validation_session,
        sample_index=57,
        input_features=validation_session.training_input_features[:1],
        observed_class_labels=validation_session.training_observed_class_labels[:1],
        candidate_model_training_and_acceptance_settings=settings,
        maximum_reference_mean_loss_increase=legacy_client.distance_threshold,
        minimum_candidate_mean_loss_improvement=0.01,
    )
    legacy_session.candidate_losses.pop()
    for losses in legacy_session.reference_losses.values():
        losses.pop()
    return (
        progress_arguments,
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_candidate_model,
    )


def set_scripted_validation_losses(*, progress_arguments, legacy_client, monkeypatch):
    """経路検証専用のloss供給。Task3の実NN数値接続とは分ける。"""
    session = progress_arguments["validation_session"]
    legacy_session = legacy_client._forward_validation
    scripted_validation_losses_by_classifier_identity = {
        id(session.candidate_training_state.candidate_classifier): legacy_session.candidate_losses[
            0
        ]
    }
    for (
        model_id,
        classifier,
    ) in session.fixed_reference_models.reference_classifiers_by_model_id.items():
        scripted_validation_losses_by_classifier_identity[id(classifier)] = (
            legacy_session.reference_losses[model_id][0]
        )
    monkeypatch.setattr(
        observation_module,
        "_evaluate_single_sample_bounded_loss",
        lambda *, classifier, input_features, observed_class_labels: (
            scripted_validation_losses_by_classifier_identity[id(classifier)]
        ),
    )
    for model_module, loss in [(legacy_session.candidate, legacy_session.candidate_losses[0])] + [
        (model_module, legacy_session.reference_losses[model_id][0])
        for model_id, model_module in legacy_session.reference_models.items()
    ]:
        original_operation = model_module.per_sample_error
        monkeypatch.setattr(
            model_module,
            "per_sample_error",
            lambda x, y, loss=loss, original_operation=original_operation: (
                torch.tensor([loss])
                if x is progress_arguments["input_features"]
                else original_operation(x, y)
            ),
        )


def assert_validation_progress_matches_legacy(
    *,
    validation_progress,
    progress_arguments,
    resolution_arguments,
    shared_optimizer_owners,
    legacy_client,
    legacy_drift_type,
    previous_model_id,
    monkeypatch,
):
    completion = validation_progress.completed_validation
    assert validation_progress.session_to_continue is None
    assert type(completion) is PostAlarmCandidateValidationCompletion
    legacy_decision = legacy_client.provisional_model_decisions[-1]
    record = completion.decision_record
    evaluation = record.post_alarm_candidate_loss_evaluation
    assert (
        record.proposal_sample_index,
        record.resolution_sample_index,
        record.detector_name,
        record.candidate_training_interval_sample_count,
    ) == (
        legacy_decision.position,
        legacy_decision.resolution_position,
        legacy_decision.detector,
        legacy_decision.training_count,
    )
    assert record.candidate_training_interval_sample_count == legacy_decision.interval_count
    assert record.validation_completion_delay_sample_count == legacy_decision.resolution_delay
    assert record.full_validation_mean_loss_advantage == legacy_decision.full_margin
    assert record.second_segment_mean_loss_advantage == legacy_decision.recent_margin
    assert evaluation.candidate_accepted == legacy_decision.accepted
    assert evaluation.validation_sample_count == legacy_decision.validation_count
    assert evaluation.comparison_reference_model_id == legacy_decision.reference_model_id
    assert evaluation.candidate_full_interval_mean_loss == legacy_decision.candidate_mean_loss
    assert evaluation.reference_full_interval_mean_loss == legacy_decision.reference_mean_loss
    assert evaluation.candidate_second_segment_mean_loss == legacy_decision.candidate_recent_loss
    assert evaluation.reference_second_segment_mean_loss == legacy_decision.reference_recent_loss
    if math.isnan(legacy_decision.reference_excess):
        assert record.reference_mean_loss_difference_from_history is None
    else:
        assert (
            record.reference_mean_loss_difference_from_history == legacy_decision.reference_excess
        )
    # 既存helperで理由の正式名対応を含めて評価を対照する。
    snapshot = progress_arguments[
        "validation_session"
    ].post_alarm_candidate_loss_collection.get_state_snapshot()
    assert_evaluation_matches_legacy_decision(
        result=evaluation,
        evaluation_arguments=dict(
            candidate_model_training_and_acceptance_settings=progress_arguments[
                "candidate_model_training_and_acceptance_settings"
            ],
            candidate_losses=snapshot.candidate_losses,
            reference_losses_by_model_id=dict(snapshot.reference_losses_by_model_id),
            reference_historical_mean_losses_by_model_id=progress_arguments[
                "validation_session"
            ].fixed_reference_models.reference_historical_mean_losses_by_model_id,
            available_reference_model_ids=(4, 9),
            current_training_model_id=previous_model_id,
            maximum_reference_mean_loss_increase=progress_arguments[
                "maximum_reference_mean_loss_increase"
            ],
            minimum_candidate_mean_loss_improvement=progress_arguments[
                "minimum_candidate_mean_loss_improvement"
            ],
        ),
        monkeypatch=monkeypatch,
    )
    assert_resolution_matches_legacy(
        resolution=completion.validation_resolution,
        resolution_arguments=resolution_arguments,
        shared_optimizer_owners=shared_optimizer_owners,
        legacy_client=legacy_client,
        legacy_drift_type=legacy_drift_type,
        previous_model_id=previous_model_id,
    )
    legacy_adaptation_event = legacy_client.adaptation_events[-1]
    assert record.resolution_sample_index == legacy_adaptation_event.position
    assert record.detector_name == legacy_adaptation_event.detector
    assert (
        completion.validation_resolution.assigned_model_id == legacy_adaptation_event.new_model_id
    )
    assert completion.previous_training_model_id == legacy_adaptation_event.old_model_id
    assert (
        completion.estimated_change_point_sample_index
        == legacy_adaptation_event.estimated_change_point
    )
    assert completion.detection_episode_id == legacy_adaptation_event.episode_id
    assert completion.training_model_switch_sample_index == (
        57 if legacy_client.local_switch_positions else None
    )
    assert completion.detection_episode_operation_required == bool(
        legacy_client.local_switch_positions
    )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize(
    "legacy_resolution_case", ("create", "reuse", "maintain", "create_rejected")
)
@pytest.mark.parametrize("pending_sample_count", (0, 3))
def test_validation_progress_matches_legacy(
    class_count, legacy_resolution_case, pending_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        progress_arguments, resolution_arguments, shared_optimizer_owners, legacy_client, _ = (
            build_validation_progress_oracle(
                class_count=class_count,
                legacy_resolution_case=legacy_resolution_case,
                pending_sample_count=pending_sample_count,
                monkeypatch=monkeypatch,
            )
        )
        set_scripted_validation_losses(
            progress_arguments=progress_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        previous_model_id = legacy_client.current_model_id
        initial_rng_state = (
            torch.get_rng_state().clone(),
            random.getstate(),
            np.random.get_state(),
        )
        legacy_drift_type = legacy_client._observe_forward_validation(
            progress_arguments["input_features"], progress_arguments["observed_class_labels"], 57
        )
        validation_progress = advance_post_alarm_candidate_validation(**progress_arguments)
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
        assert torch.equal(torch.get_rng_state(), initial_rng_state[0])
        assert random.getstate() == initial_rng_state[1]
        expected_rng_state = np.random.get_state()
        assert expected_rng_state[0] == initial_rng_state[2][0]
        assert np.array_equal(expected_rng_state[1], initial_rng_state[2][1])
        assert expected_rng_state[2:] == initial_rng_state[2][2:]


def test_validation_progress_without_session_is_noop(monkeypatch):
    monkeypatch.setattr(
        progress_module,
        "observe_post_alarm_candidate_validation_sample",
        Mock(side_effect=AssertionError),
    )
    arguments = {
        name: object() for name in signature(advance_post_alarm_candidate_validation).parameters
    }
    arguments["validation_session"] = None
    initial_rng_state = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    result = advance_post_alarm_candidate_validation(**arguments)
    assert type(result) is PostAlarmCandidateValidationProgress
    assert result.session_to_continue is result.completed_validation is None
    assert torch.equal(torch.get_rng_state(), initial_rng_state[0])
    assert random.getstate() == initial_rng_state[1]
    expected_rng_state = np.random.get_state()
    assert expected_rng_state[0] == initial_rng_state[2][0]
    assert np.array_equal(expected_rng_state[1], initial_rng_state[2][1])
    assert expected_rng_state[2:] == initial_rng_state[2][2:]


def test_validation_progress_waits_without_resolution(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        progress_arguments, resolution_arguments, owners, legacy_client, _ = (
            build_validation_progress_oracle(
                class_count=2, legacy_resolution_case="reuse", monkeypatch=monkeypatch
            )
        )
        session = progress_arguments["validation_session"]
        collection = PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=progress_arguments[
                "candidate_model_training_and_acceptance_settings"
            ],
            proposal_sample_index=53,
            reference_model_ids=(4, 9),
        )
        progress_arguments["validation_session"] = replace(
            session, post_alarm_candidate_loss_collection=collection
        )
        set_scripted_validation_losses(
            progress_arguments=progress_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        monkeypatch.setattr(
            progress_module,
            "evaluate_candidate_using_post_alarm_losses",
            Mock(side_effect=AssertionError),
        )
        monkeypatch.setattr(
            progress_module,
            "apply_post_alarm_candidate_validation_resolution",
            Mock(side_effect=AssertionError),
        )
        previous_snapshot = snapshot_adoption_state(
            adoption_arguments=resolution_arguments, shared_optimizer_owners=owners
        )
        registry = progress_arguments["held_model_training_state_registry"]
        original_operation = registry.snapshot_ordered_held_model_training_states
        monkeypatch.setattr(
            registry, "snapshot_ordered_held_model_training_states", Mock(wraps=original_operation)
        )
        for sample_index in (54, 55, 56):
            registry.snapshot_ordered_held_model_training_states.reset_mock()
            result = advance_post_alarm_candidate_validation(
                **dict(progress_arguments, sample_index=sample_index)
            )
            assert result.session_to_continue is progress_arguments["validation_session"]
            assert result.completed_validation is None
            assert collection.validation_sample_count == sample_index - 53
            registry.snapshot_ordered_held_model_training_states.assert_not_called()
            assert_adoption_state_unchanged(
                previous_snapshot=previous_snapshot, adoption_arguments=resolution_arguments
            )


@pytest.mark.parametrize(
    "invalid_case",
    (
        "session",
        "settings",
        "forged_policy",
        "forged_count",
        "count_mismatch",
        "threshold_bool",
        "threshold_nan",
        "threshold_negative",
        "threshold_huge",
        "threshold_string",
        "maximum_threshold_inf",
        "registry",
        "assignment",
        "sample_index",
        "feature_shape",
        "feature_nan",
        "label_shape",
        "label_nan",
        "label_range",
        "already_ready",
    ),
)
def test_validation_progress_rejects_before_observation(invalid_case, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        progress_arguments, resolution_arguments, owners, _, _ = build_validation_progress_oracle(
            class_count=2, legacy_resolution_case="create", monkeypatch=monkeypatch
        )
        session = progress_arguments["validation_session"]
        if invalid_case == "session":
            progress_arguments["validation_session"] = object()
        elif invalid_case == "settings":
            progress_arguments["candidate_model_training_and_acceptance_settings"] = object()
        elif invalid_case.startswith("forged"):
            settings = make_acceptance_settings(sample_count=4)
            object.__setattr__(
                settings,
                "candidate_model_acceptance_policy"
                if invalid_case == "forged_policy"
                else "candidate_post_alarm_validation_sample_count",
                "bad" if invalid_case == "forged_policy" else True,
            )
            progress_arguments["candidate_model_training_and_acceptance_settings"] = settings
        elif invalid_case == "count_mismatch":
            progress_arguments["candidate_model_training_and_acceptance_settings"] = (
                make_acceptance_settings(sample_count=5)
            )
        elif invalid_case.startswith("threshold"):
            progress_arguments["minimum_candidate_mean_loss_improvement"] = {
                "threshold_bool": True,
                "threshold_nan": float("nan"),
                "threshold_negative": -1,
                "threshold_huge": 10**1000,
                "threshold_string": "0.01",
            }[invalid_case]
        elif invalid_case == "maximum_threshold_inf":
            progress_arguments["maximum_reference_mean_loss_increase"] = float("inf")
        elif invalid_case == "registry":
            progress_arguments["held_model_training_state_registry"] = object()
        elif invalid_case == "assignment":
            progress_arguments["current_training_model_assignment"] = object()
        elif invalid_case == "sample_index":
            progress_arguments["sample_index"] = 58
        elif invalid_case == "feature_shape":
            progress_arguments["input_features"] = torch.ones((2, 2))
        elif invalid_case == "feature_nan":
            progress_arguments["input_features"] = torch.tensor([[float("nan"), 0.0]])
        elif invalid_case == "label_shape":
            progress_arguments["observed_class_labels"] = torch.ones((1, 2))
        elif invalid_case == "label_nan":
            progress_arguments["observed_class_labels"] = torch.tensor([[float("nan")]])
        elif invalid_case == "label_range":
            progress_arguments["observed_class_labels"] = torch.tensor([[2.0]])
        elif invalid_case == "already_ready":
            session.post_alarm_candidate_loss_collection.observe_losses_after_label_observation(
                sample_index=57, candidate_loss=0.1, reference_losses_by_model_id={4: 0.2, 9: 0.3}
            )
        snapshot = session.post_alarm_candidate_loss_collection.get_state_snapshot()
        previous_snapshot = snapshot_adoption_state(
            adoption_arguments=resolution_arguments, shared_optimizer_owners=owners
        )
        initial_rng_state = (
            torch.get_rng_state().clone(),
            random.getstate(),
            np.random.get_state(),
        )
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            advance_post_alarm_candidate_validation(**progress_arguments)
        assert session.post_alarm_candidate_loss_collection.get_state_snapshot() == snapshot
        assert_adoption_state_unchanged(
            previous_snapshot=previous_snapshot, adoption_arguments=resolution_arguments
        )
        assert torch.equal(torch.get_rng_state(), initial_rng_state[0])
        assert random.getstate() == initial_rng_state[1]
        expected_rng_state = np.random.get_state()
        assert expected_rng_state[0] == initial_rng_state[2][0]
        assert np.array_equal(expected_rng_state[1], initial_rng_state[2][1])
        assert expected_rng_state[2:] == initial_rng_state[2][2:]


def test_validation_progress_records_are_immutable(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        arguments, _, _, legacy_client, _ = build_validation_progress_oracle(
            class_count=2, legacy_resolution_case="create", monkeypatch=monkeypatch
        )
        set_scripted_validation_losses(
            progress_arguments=arguments, legacy_client=legacy_client, monkeypatch=monkeypatch
        )
        result = advance_post_alarm_candidate_validation(**arguments)
        for record in (result, result.completed_validation):
            assert all(
                parameter.kind is Parameter.KEYWORD_ONLY and parameter.default is Parameter.empty
                for parameter in signature(type(record)).parameters.values()
            )
            for field_name in vars(record):
                with pytest.raises(FrozenInstanceError):
                    setattr(record, field_name, None)
            with pytest.raises(TypeError):
                type(record)(*vars(record).values())
        assert all(
            parameter.kind is Parameter.KEYWORD_ONLY and parameter.default is Parameter.empty
            for parameter in signature(advance_post_alarm_candidate_validation).parameters.values()
        )


@pytest.mark.parametrize("sample_count", (2, 4, 5))
@pytest.mark.parametrize(
    "legacy_resolution_case", ("create", "reuse", "maintain", "create_rejected")
)
def test_validation_decision_record_matches_legacy(
    sample_count, legacy_resolution_case, monkeypatch
):
    evaluation_arguments = dict(
        candidate_model_training_and_acceptance_settings=make_acceptance_settings(
            sample_count=sample_count
        ),
        candidate_losses=(0.01 if legacy_resolution_case == "create" else 0.9,) * sample_count,
        reference_losses_by_model_id={4: (0.2,) * sample_count, 9: (0.3,) * sample_count},
        reference_historical_mean_losses_by_model_id={
            "create": {},
            "reuse": {4: 0.2},
            "maintain": {9: 0.3},
            "create_rejected": {},
        }[legacy_resolution_case],
        available_reference_model_ids=(4, 9),
        current_training_model_id=9,
        maximum_reference_mean_loss_increase=0.0,
        minimum_candidate_mean_loss_improvement=0.01,
    )
    post_alarm_candidate_loss_evaluation = evaluate_candidate_using_post_alarm_losses(
        **evaluation_arguments
    )
    legacy_decision = capture_legacy_candidate_decision(
        evaluation_arguments=evaluation_arguments, monkeypatch=monkeypatch
    )
    decision_record = PostAlarmCandidateValidationDecisionRecord(
        proposal_sample_index=legacy_decision.position,
        resolution_sample_index=legacy_decision.resolution_position,
        detector_name=legacy_decision.detector,
        candidate_training_interval_sample_count=legacy_decision.training_count,
        post_alarm_candidate_loss_evaluation=post_alarm_candidate_loss_evaluation,
    )
    assert_evaluation_matches_legacy_decision(
        result=decision_record.post_alarm_candidate_loss_evaluation,
        evaluation_arguments=evaluation_arguments,
        monkeypatch=monkeypatch,
    )
    assert decision_record.proposal_sample_index == legacy_decision.position
    assert decision_record.resolution_sample_index == legacy_decision.resolution_position
    assert decision_record.detector_name == legacy_decision.detector
    assert (
        decision_record.candidate_training_interval_sample_count == legacy_decision.interval_count
    )
    assert (
        decision_record.candidate_training_interval_sample_count == legacy_decision.training_count
    )
    assert legacy_decision.validation_source == "forward"
    assert (
        decision_record.validation_completion_delay_sample_count == legacy_decision.resolution_delay
    )
    assert decision_record.full_validation_mean_loss_advantage == legacy_decision.full_margin
    assert decision_record.second_segment_mean_loss_advantage == legacy_decision.recent_margin
    if math.isnan(legacy_decision.reference_excess):
        assert decision_record.reference_mean_loss_difference_from_history is None
    else:
        assert (
            decision_record.reference_mean_loss_difference_from_history
            == legacy_decision.reference_excess
        )
    assert (
        decision_record.post_alarm_candidate_loss_evaluation is post_alarm_candidate_loss_evaluation
    )


def test_validation_decision_record_is_immutable():
    post_alarm_candidate_loss_evaluation = evaluate_candidate_using_post_alarm_losses(
        candidate_model_training_and_acceptance_settings=make_acceptance_settings(sample_count=2),
        candidate_losses=(0.1, 0.2),
        reference_losses_by_model_id={4: (0.3, 0.4)},
        reference_historical_mean_losses_by_model_id={4: 0.5},
        available_reference_model_ids=(4,),
        current_training_model_id=4,
        maximum_reference_mean_loss_increase=0.0,
        minimum_candidate_mean_loss_improvement=0.01,
    )
    decision_record = PostAlarmCandidateValidationDecisionRecord(
        proposal_sample_index=10,
        resolution_sample_index=12,
        detector_name="class_esr",
        candidate_training_interval_sample_count=11,
        post_alarm_candidate_loss_evaluation=post_alarm_candidate_loss_evaluation,
    )
    # 履歴との差は負にもなる。increaseという非負に見える名前にしない。
    assert decision_record.reference_mean_loss_difference_from_history < 0
    for field_name in vars(decision_record):
        with pytest.raises(FrozenInstanceError):
            setattr(decision_record, field_name, None)
    with pytest.raises(FrozenInstanceError):
        post_alarm_candidate_loss_evaluation.candidate_accepted = True
    with pytest.raises(TypeError):
        PostAlarmCandidateValidationDecisionRecord(
            10, 12, "class_esr", 11, post_alarm_candidate_loss_evaluation
        )
