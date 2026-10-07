"""未完了候補の不足判定記録を、実旧の終端確定記録へ対照する。"""

import math
import random
from dataclasses import FrozenInstanceError, fields, replace
from inspect import Parameter, signature
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import assert_held_model_states_match_legacy
from test_adopted_candidate_local_adoption import (
    assert_adoption_state_unchanged,
    assert_training_samples_match_legacy,
    snapshot_adoption_state,
)
from test_loss_statistics_model_id_reassignment import assert_store_statistics_match_legacy
from test_model_training_and_assignment_counts import assert_model_counts_match_legacy
from test_post_alarm_candidate_validation_progress import build_validation_progress_oracle
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

import federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization as finalization_module
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import (
    IncompletePostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)
from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import (
    IncompletePostAlarmCandidateValidationFinalization,
    finalize_incomplete_post_alarm_candidate_validation,
)


def build_incomplete_validation_finalization_oracle(
    *,
    class_count,
    observed_validation_sample_count=3,
    pending_sample_count=3,
    processed_sample_count=60,
    optimizer_variant="standard",
    monkeypatch,
):
    """既存の同じ候補・ownerへ、不足件数と終端引数を結合する。"""
    progress_arguments, resolution_arguments, shared_optimizer_owners, legacy_client, _ = (
        build_validation_progress_oracle(
            class_count=class_count,
            legacy_resolution_case="create",
            pending_sample_count=pending_sample_count,
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
    )
    validation_session = progress_arguments["validation_session"]
    legacy_session = legacy_client._forward_validation
    legacy_session.candidate_losses = legacy_session.candidate_losses[
        :observed_validation_sample_count
    ]
    legacy_session.reference_losses = {
        model_id: losses[:observed_validation_sample_count]
        for model_id, losses in legacy_session.reference_losses.items()
    }
    collection = PostAlarmCandidateLossCollection(
        candidate_model_training_and_acceptance_settings=progress_arguments[
            "candidate_model_training_and_acceptance_settings"
        ],
        proposal_sample_index=validation_session.proposal_sample_index,
        reference_model_ids=tuple(legacy_session.reference_models),
    )
    for sample_index in range(observed_validation_sample_count):
        collection.observe_losses_after_label_observation(
            sample_index=validation_session.proposal_sample_index + 1 + sample_index,
            candidate_loss=legacy_session.candidate_losses[sample_index],
            reference_losses_by_model_id={
                model_id: losses[sample_index]
                for model_id, losses in legacy_session.reference_losses.items()
            },
        )
    validation_session = replace(
        validation_session, post_alarm_candidate_loss_collection=collection
    )
    finalization_arguments = {
        name: progress_arguments[name]
        for name in (
            "pending_assignment_sample_concept_ids",
            "held_model_training_state_registry",
            "loss_statistics_store",
            "training_sample_store",
            "model_training_and_assignment_counts_store",
            "current_training_model_assignment",
        )
    }
    finalization_arguments.update(
        validation_session=validation_session, processed_sample_count=processed_sample_count
    )
    legacy_client.processed_samples = processed_sample_count
    return finalization_arguments, resolution_arguments, shared_optimizer_owners, legacy_client


def assert_incomplete_validation_finalization_matches_legacy(
    *,
    incomplete_validation_finalization,
    finalization_arguments,
    resolution_arguments,
    shared_optimizer_owners,
    legacy_client,
):
    decision_record = incomplete_validation_finalization.decision_record
    legacy_decision = legacy_client.provisional_model_decisions[-1]
    legacy_adaptation_event = legacy_client.adaptation_events[-1]
    assert (
        decision_record.proposal_sample_index,
        decision_record.finalization_sample_index,
        decision_record.detector_name,
        decision_record.candidate_training_interval_sample_count,
        decision_record.validation_sample_count,
    ) == (
        legacy_decision.position,
        legacy_decision.resolution_position,
        legacy_decision.detector,
        legacy_decision.interval_count,
        legacy_decision.validation_count,
    )
    assert (
        decision_record.candidate_training_interval_sample_count == legacy_decision.training_count
    )
    assert decision_record.finalization_delay_sample_count == legacy_decision.resolution_delay
    assert (
        legacy_decision.accepted is False and legacy_decision.reason == "insufficient_forward_data"
    )
    assert (
        legacy_decision.reference_model_id is None
        and legacy_decision.validation_source == "forward"
    )
    for field_name in (
        "candidate_mean_loss",
        "reference_mean_loss",
        "candidate_recent_loss",
        "reference_recent_loss",
        "full_margin",
        "recent_margin",
        "reference_excess",
        "reference_historical_mean",
    ):
        assert math.isnan(getattr(legacy_decision, field_name))
        assert not hasattr(decision_record, field_name)
    assert legacy_adaptation_event.action == "create_rejected"
    assert legacy_adaptation_event.position == decision_record.finalization_sample_index
    assert legacy_adaptation_event.detector == decision_record.detector_name
    assert (
        incomplete_validation_finalization.current_training_model_id
        == legacy_adaptation_event.old_model_id
        == legacy_adaptation_event.new_model_id
        == legacy_client.current_model_id
    )
    assert (
        incomplete_validation_finalization.estimated_change_point_sample_index
        == legacy_adaptation_event.estimated_change_point
    )
    assert (
        incomplete_validation_finalization.detection_episode_id
        == legacy_adaptation_event.episode_id
    )
    assert (
        finalization_arguments["current_training_model_assignment"].current_training_model_id
        == legacy_client.current_model_id
    )
    assert_held_model_states_match_legacy(
        registry=finalization_arguments["held_model_training_state_registry"],
        shared_optimizer_owners=shared_optimizer_owners[:-1],
        legacy_client=legacy_client,
        input_features=resolution_arguments["initial_statistics_input_features"],
    )
    assert_training_samples_match_legacy(
        training_sample_store=finalization_arguments["training_sample_store"],
        legacy_client=legacy_client,
    )
    assert_model_counts_match_legacy(
        counts_store=finalization_arguments["model_training_and_assignment_counts_store"],
        legacy_client=legacy_client,
    )
    assert_store_statistics_match_legacy(
        loss_statistics_store=finalization_arguments["loss_statistics_store"],
        legacy_client=legacy_client,
    )
    assert legacy_client.local_model_changes == [] and legacy_client.local_switch_positions == []


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize(
    "observed_validation_sample_count,processed_sample_count", ((0, 0), (1, 53), (3, 60))
)
@pytest.mark.parametrize("pending_sample_count", (0, 3))
def test_incomplete_validation_finalization_matches_legacy(
    class_count,
    observed_validation_sample_count,
    processed_sample_count,
    pending_sample_count,
    monkeypatch,
):
    with torch.random.fork_rng(devices=[]):
        finalization_arguments, resolution_arguments, shared_optimizer_owners, legacy_client = (
            build_incomplete_validation_finalization_oracle(
                class_count=class_count,
                observed_validation_sample_count=observed_validation_sample_count,
                pending_sample_count=pending_sample_count,
                processed_sample_count=processed_sample_count,
                monkeypatch=monkeypatch,
            )
        )
        validation_session = finalization_arguments["validation_session"]
        if observed_validation_sample_count == 1:
            validation_session = replace(
                validation_session,
                estimated_change_point_sample_index=None,
                detection_episode_id=None,
            )
            finalization_arguments["validation_session"] = validation_session
            legacy_client._forward_validation.estimated_change_point = None
            legacy_client._forward_validation.episode_id = None
        collection_state_snapshot = (
            validation_session.post_alarm_candidate_loss_collection.get_state_snapshot()
        )
        previous_snapshot = snapshot_adoption_state(
            adoption_arguments=resolution_arguments, shared_optimizer_owners=shared_optimizer_owners
        )
        reference_parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter
            for classifier in validation_session.fixed_reference_models.reference_classifiers_by_model_id.values()
            for parameter in classifier.parameters()
        )
        legacy_client.finalize_incomplete_forward_validation()
        incomplete_validation_finalization = finalize_incomplete_post_alarm_candidate_validation(
            **finalization_arguments
        )
        assert_incomplete_validation_finalization_matches_legacy(
            incomplete_validation_finalization=incomplete_validation_finalization,
            finalization_arguments=finalization_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
        )
        assert (
            validation_session.post_alarm_candidate_loss_collection.get_state_snapshot()
            == collection_state_snapshot
        )
        assert_parameter_values_and_gradients_unchanged(
            previous_snapshot["registration_state"]["parameter_snapshots"]
        )
        assert_parameter_values_and_gradients_unchanged(reference_parameter_snapshots)
        from test_joint_model_parameter_update import assert_nested_state_equal

        for owner, parameter_optimizer, optimizer_state in previous_snapshot["registration_state"][
            "optimizers"
        ]:
            assert owner.parameter_optimizer is parameter_optimizer
            assert_nested_state_equal(parameter_optimizer.state_dict(), optimizer_state)
        assert (
            resolution_arguments["temporary_model_id_allocator"].next_temporary_model_id
            == previous_snapshot["next_temporary_model_id"]
        )
        pending_upload_state = resolution_arguments["pending_model_upload_state"]
        assert (
            pending_upload_state.get_pending_model_upload(),
            pending_upload_state.remaining_upload_delay_round_count,
        ) == previous_snapshot["registration_state"]["pending_model_upload"]
        initial_rng_state = previous_snapshot["registration_state"]["random_states"]
        assert torch.equal(torch.get_rng_state(), initial_rng_state[0])
        assert random.getstate() == initial_rng_state[1]
        expected_rng_state = np.random.get_state()
        assert expected_rng_state[0] == initial_rng_state[2][0]
        assert np.array_equal(expected_rng_state[1], initial_rng_state[2][1])
        assert expected_rng_state[2:] == initial_rng_state[2][2:]


def test_incomplete_validation_finalization_without_session_is_noop(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        finalization_arguments, resolution_arguments, owners, _ = (
            build_incomplete_validation_finalization_oracle(class_count=2, monkeypatch=monkeypatch)
        )
        previous_snapshot = snapshot_adoption_state(
            adoption_arguments=resolution_arguments, shared_optimizer_owners=owners
        )
        monkeypatch.setattr(
            finalization_module,
            "absorb_assigned_training_samples_into_held_model",
            Mock(side_effect=AssertionError),
        )
        assert (
            finalize_incomplete_post_alarm_candidate_validation(
                **{
                    name: None if name == "validation_session" else object()
                    for name in signature(
                        finalize_incomplete_post_alarm_candidate_validation
                    ).parameters
                }
            )
            is None
        )
        assert_adoption_state_unchanged(
            previous_snapshot=previous_snapshot, adoption_arguments=resolution_arguments
        )


@pytest.mark.parametrize("class_count", (2, 4))
def test_incomplete_validation_finalization_uses_current_training_model(class_count, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        finalization_arguments, resolution_arguments, owners, legacy_client = (
            build_incomplete_validation_finalization_oracle(
                class_count=class_count, monkeypatch=monkeypatch
            )
        )
        finalization_arguments["current_training_model_assignment"].assign_model_for_training(
            model_id=4
        )
        legacy_client.current_model_id = 4
        assert finalization_arguments["validation_session"].initial_training_model_id == 9
        finalization_arguments["pending_assignment_sample_concept_ids"] = (None, 1, None)
        legacy_client._forward_validation.held_data = [
            (sample.input_features, sample.observed_class_labels, concept_id)
            for sample, concept_id in zip(
                finalization_arguments["validation_session"].pending_assignment_training_samples,
                (None, 1, None),
            )
        ]
        legacy_client.finalize_incomplete_forward_validation()
        result = finalize_incomplete_post_alarm_candidate_validation(**finalization_arguments)
        assert result.current_training_model_id == 4
        assert_incomplete_validation_finalization_matches_legacy(
            incomplete_validation_finalization=result,
            finalization_arguments=finalization_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=owners,
            legacy_client=legacy_client,
        )


@pytest.mark.parametrize(
    "invalid_case,expected_exception",
    [
        ("session", TypeError),
        ("assignment", TypeError),
        ("processed_bool", TypeError),
        ("processed_float", TypeError),
        ("processed_negative", ValueError),
        ("ready", ValueError),
        ("concept_list", TypeError),
        ("concept_short", ValueError),
        ("concept_bool", TypeError),
        ("concept_float", TypeError),
        ("samples_list", TypeError),
        ("sample_record", TypeError),
        ("features_type", TypeError),
        ("features_rows", ValueError),
        ("features_width", ValueError),
        ("features_empty", ValueError),
        ("features_rank", ValueError),
        ("features_nan", ValueError),
        ("features_dtype", ValueError),
        ("features_meta", ValueError),
        ("features_sparse", ValueError),
        ("labels_shape", ValueError),
        ("labels_fractional", ValueError),
        ("labels_out_of_range", ValueError),
        ("labels_nan", ValueError),
        ("missing_model", KeyError),
        ("missing_model_empty", KeyError),
    ]
    + [
        (owner_name, TypeError)
        for owner_name in (
            "held_model_training_state_registry",
            "loss_statistics_store",
            "training_sample_store",
            "model_training_and_assignment_counts_store",
        )
    ]
    + [
        (f"subclass:{owner_name}", TypeError)
        for owner_name in (
            "validation_session",
            "current_training_model_assignment",
            "held_model_training_state_registry",
            "loss_statistics_store",
            "training_sample_store",
            "model_training_and_assignment_counts_store",
        )
    ],
)
def test_incomplete_validation_finalization_rejects_before_absorption(
    invalid_case, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        finalization_arguments, resolution_arguments, owners, _ = (
            build_incomplete_validation_finalization_oracle(class_count=4, monkeypatch=monkeypatch)
        )
        validation_session = finalization_arguments["validation_session"]
        training_samples = validation_session.pending_assignment_training_samples
        if invalid_case.startswith("subclass:"):
            owner_name = invalid_case.removeprefix("subclass:")
            state_owner = finalization_arguments[owner_name]
            field_value = object.__new__(type("InputSubclass", (type(state_owner),), {}))
            field_value.__dict__.update(state_owner.__dict__)
            finalization_arguments[owner_name] = field_value
        elif invalid_case == "session":
            finalization_arguments["validation_session"] = object()
        elif invalid_case == "assignment":
            finalization_arguments["current_training_model_assignment"] = object()
        elif invalid_case in finalization_arguments:
            finalization_arguments[invalid_case] = object()
        elif invalid_case.startswith("processed"):
            finalization_arguments["processed_sample_count"] = {
                "processed_bool": True,
                "processed_float": 60.0,
                "processed_negative": -1,
            }[invalid_case]
        elif invalid_case == "ready":
            validation_session.post_alarm_candidate_loss_collection.observe_losses_after_label_observation(
                sample_index=57, candidate_loss=0.1, reference_losses_by_model_id={4: 0.2, 9: 0.3}
            )
        elif invalid_case.startswith("concept"):
            finalization_arguments["pending_assignment_sample_concept_ids"] = {
                "concept_list": [1, 1, 1],
                "concept_short": (1, 1),
                "concept_bool": (1, True, 1),
                "concept_float": (1, 1.0, 1),
            }[invalid_case]
        elif invalid_case.startswith("missing_model"):
            finalization_arguments["current_training_model_assignment"].assign_model_for_training(
                model_id=999
            )
            if invalid_case == "missing_model_empty":
                finalization_arguments["validation_session"] = replace(
                    validation_session, pending_assignment_training_samples=()
                )
                finalization_arguments["pending_assignment_sample_concept_ids"] = ()
        elif invalid_case == "samples_list":
            finalization_arguments["validation_session"] = replace(
                validation_session, pending_assignment_training_samples=list(training_samples)
            )
        elif invalid_case == "sample_record":
            finalization_arguments["validation_session"] = replace(
                validation_session,
                pending_assignment_training_samples=(
                    training_samples[0],
                    object(),
                    training_samples[2],
                ),
            )
        else:
            sample = training_samples[1]
            field_value = {
                "features_type": lambda: object(),
                "features_rows": lambda: sample.input_features.repeat(2, 1),
                "features_width": lambda: torch.ones((1, 3)),
                "features_empty": lambda: sample.input_features[:0],
                "features_rank": lambda: sample.input_features.flatten(),
                "features_nan": lambda: torch.full_like(sample.input_features, float("nan")),
                "features_dtype": lambda: sample.input_features.double(),
                "features_meta": lambda: sample.input_features.to("meta"),
                "features_sparse": lambda: sample.input_features.to_sparse(),
                "labels_shape": lambda: sample.observed_class_labels.repeat(1, 2),
                "labels_fractional": lambda: torch.tensor([[0.5]]),
                "labels_out_of_range": lambda: torch.tensor([[4.0]]),
                "labels_nan": lambda: torch.tensor([[float("nan")]]),
            }[invalid_case]()
            sample = ObservedTrainingSample(
                input_features=field_value
                if invalid_case.startswith("features")
                else sample.input_features,
                observed_class_labels=field_value
                if invalid_case.startswith("labels")
                else sample.observed_class_labels,
            )
            finalization_arguments["validation_session"] = replace(
                validation_session,
                pending_assignment_training_samples=(
                    training_samples[0],
                    sample,
                    training_samples[2],
                ),
            )
        previous_snapshot = snapshot_adoption_state(
            adoption_arguments=resolution_arguments, shared_optimizer_owners=owners
        )
        collection_state_snapshot = (
            validation_session.post_alarm_candidate_loss_collection.get_state_snapshot()
        )
        reference_parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter
            for classifier in validation_session.fixed_reference_models.reference_classifiers_by_model_id.values()
            for parameter in classifier.parameters()
        )
        with pytest.raises(expected_exception):
            finalize_incomplete_post_alarm_candidate_validation(**finalization_arguments)
        assert_adoption_state_unchanged(
            previous_snapshot=previous_snapshot, adoption_arguments=resolution_arguments
        )
        assert (
            validation_session.post_alarm_candidate_loss_collection.get_state_snapshot()
            == collection_state_snapshot
        )
        assert_parameter_values_and_gradients_unchanged(reference_parameter_snapshots)


def test_incomplete_validation_finalization_result_is_immutable(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        finalization_arguments, _, _, _ = build_incomplete_validation_finalization_oracle(
            class_count=2, monkeypatch=monkeypatch
        )
        result = finalize_incomplete_post_alarm_candidate_validation(**finalization_arguments)
        assert type(result) is IncompletePostAlarmCandidateValidationFinalization
        assert tuple(field.name for field in fields(result)) == (
            "decision_record",
            "current_training_model_id",
            "estimated_change_point_sample_index",
            "detection_episode_id",
        )
        for field_name in vars(result):
            with pytest.raises(FrozenInstanceError):
                setattr(result, field_name, None)
        for constructor in (
            IncompletePostAlarmCandidateValidationFinalization,
            finalize_incomplete_post_alarm_candidate_validation,
        ):
            assert all(
                parameter.kind is Parameter.KEYWORD_ONLY and parameter.default is Parameter.empty
                for parameter in signature(constructor).parameters.values()
            )
        with pytest.raises(TypeError):
            IncompletePostAlarmCandidateValidationFinalization(*vars(result).values())


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize(
    "observed_validation_sample_count,processed_sample_count", ((0, 0), (1, 53), (3, 60))
)
def test_incomplete_validation_decision_record_matches_legacy(
    class_count, observed_validation_sample_count, processed_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        progress_arguments, _, _, legacy_client, _ = build_validation_progress_oracle(
            class_count=class_count,
            legacy_resolution_case="create",
            pending_sample_count=0,
            monkeypatch=monkeypatch,
        )
        validation_session = progress_arguments["validation_session"]
        legacy_session = legacy_client._forward_validation
        legacy_session.candidate_losses = legacy_session.candidate_losses[
            :observed_validation_sample_count
        ]
        legacy_session.reference_losses = {
            model_id: losses[:observed_validation_sample_count]
            for model_id, losses in legacy_session.reference_losses.items()
        }
        collection = PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=progress_arguments[
                "candidate_model_training_and_acceptance_settings"
            ],
            proposal_sample_index=validation_session.proposal_sample_index,
            reference_model_ids=tuple(legacy_session.reference_models),
        )
        for sample_index in range(observed_validation_sample_count):
            collection.observe_losses_after_label_observation(
                sample_index=validation_session.proposal_sample_index + 1 + sample_index,
                candidate_loss=legacy_session.candidate_losses[sample_index],
                reference_losses_by_model_id={
                    model_id: losses[sample_index]
                    for model_id, losses in legacy_session.reference_losses.items()
                },
            )
        validation_session = replace(
            validation_session, post_alarm_candidate_loss_collection=collection
        )
        collection_state_snapshot = collection.get_state_snapshot()
        assert not collection_state_snapshot.ready_for_acceptance_evaluation
        assert collection_state_snapshot.validation_sample_count == observed_validation_sample_count
        legacy_client.processed_samples = processed_sample_count
        legacy_client.finalize_incomplete_forward_validation()
        legacy_decision = legacy_client.provisional_model_decisions[-1]
        decision_record = IncompletePostAlarmCandidateValidationDecisionRecord(
            proposal_sample_index=validation_session.proposal_sample_index,
            finalization_sample_index=max(
                validation_session.proposal_sample_index, processed_sample_count - 1
            ),
            detector_name=validation_session.detector_name,
            candidate_training_interval_sample_count=len(
                validation_session.training_input_features
            ),
            validation_sample_count=collection_state_snapshot.validation_sample_count,
        )
        assert decision_record.proposal_sample_index == legacy_decision.position
        assert decision_record.finalization_sample_index == legacy_decision.resolution_position
        assert decision_record.detector_name == legacy_decision.detector
        assert (
            decision_record.candidate_training_interval_sample_count
            == legacy_decision.interval_count
            == legacy_decision.training_count
        )
        assert decision_record.validation_sample_count == legacy_decision.validation_count
        assert (
            decision_record.validation_sample_count
            != decision_record.candidate_training_interval_sample_count
        )
        assert decision_record.finalization_delay_sample_count == legacy_decision.resolution_delay
        # 不足recordの型そのものが棄却・比較未成立を表す。旧理由やNaNを新fieldへ持ち込まない。
        assert type(decision_record) is IncompletePostAlarmCandidateValidationDecisionRecord
        assert legacy_decision.accepted is False
        assert legacy_decision.reason == "insufficient_forward_data"
        assert legacy_decision.reference_model_id is None
        assert legacy_decision.validation_source == "forward"
        for field_name in (
            "candidate_mean_loss",
            "reference_mean_loss",
            "candidate_recent_loss",
            "reference_recent_loss",
            "full_margin",
            "recent_margin",
            "reference_excess",
            "reference_historical_mean",
        ):
            assert math.isnan(getattr(legacy_decision, field_name))
            assert not hasattr(decision_record, field_name)
        assert not hasattr(decision_record, "reference_model_id")
        assert collection.get_state_snapshot() == collection_state_snapshot


def test_incomplete_validation_decision_record_is_immutable():
    decision_record = IncompletePostAlarmCandidateValidationDecisionRecord(
        proposal_sample_index=53,
        finalization_sample_index=59,
        detector_name="class_esr",
        candidate_training_interval_sample_count=11,
        validation_sample_count=3,
    )
    assert tuple(field.name for field in fields(decision_record)) == (
        "proposal_sample_index",
        "finalization_sample_index",
        "detector_name",
        "candidate_training_interval_sample_count",
        "validation_sample_count",
    )
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY and parameter.default is Parameter.empty
        for parameter in signature(
            IncompletePostAlarmCandidateValidationDecisionRecord
        ).parameters.values()
    )
    assert decision_record.finalization_delay_sample_count == 6
    for field_name in vars(decision_record):
        with pytest.raises(FrozenInstanceError):
            setattr(decision_record, field_name, None)
    with pytest.raises(FrozenInstanceError):
        decision_record.finalization_delay_sample_count = 0
    with pytest.raises(TypeError):
        IncompletePostAlarmCandidateValidationDecisionRecord(53, 59, "class_esr", 11, 3)
    for field_name in vars(decision_record):
        with pytest.raises(TypeError):
            IncompletePostAlarmCandidateValidationDecisionRecord(
                **{
                    name: value
                    for name, value in vars(decision_record).items()
                    if name != field_name
                }
            )
