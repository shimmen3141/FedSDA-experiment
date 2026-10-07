"""候補検証の開始を、候補学習が有効な実旧session開始へ対照する。"""

import random
from collections import defaultdict
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_candidate_epoch_training import assert_candidate_epoch_training_matches_legacy
from test_joint_model_parameter_update import assert_nested_state_equal
from test_post_alarm_reference_model_fixation import (
    STATISTICS_CASES,
    assert_fixed_references_match_legacy,
    build_fixation_oracle,
    set_overall_loss_statistics_in_both_implementations,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment import config
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
    start_post_alarm_candidate_validation_session,
)


def build_session_start_oracle(
    *,
    class_count,
    optimizer_variant,
    monkeypatch,
    candidate_training_strategy="validation_loss_early_stopping",
    maximum_epoch_count=3,
    maximum_batch_sample_count=3,
    consecutive_non_improving_epoch_limit=1,
    minimum_validation_loss_decrease=10,
    held_model_ids=(4, 9),
):
    fixation_arguments, adoption_arguments, shared_optimizer_owners, legacy_client = (
        build_fixation_oracle(
            class_count=class_count,
            optimizer_variant=optimizer_variant,
            held_model_ids=held_model_ids,
            monkeypatch=monkeypatch,
        )
    )
    monkeypatch.setattr(
        config,
        "NEW_MODEL_TRAINING",
        {
            "fixed_epoch_training": "fixed",
            "validation_loss_early_stopping": "early_stopping",
            "skip_training": "none",
        }[candidate_training_strategy],
    )
    monkeypatch.setattr(config, "NEW_MODEL_EPOCHS", maximum_epoch_count)
    monkeypatch.setattr(config, "CLIENT_BATCH_SIZE", maximum_batch_sample_count)
    monkeypatch.setattr(config, "NEW_MODEL_VALIDATION_FRACTION", 0.2)
    monkeypatch.setattr(
        config, "NEW_MODEL_EARLY_STOPPING_PATIENCE", consecutive_non_improving_epoch_limit
    )
    monkeypatch.setattr(
        config, "NEW_MODEL_EARLY_STOPPING_MIN_DELTA", minimum_validation_loss_decrease
    )
    legacy_client.phase_seconds = defaultdict(float)
    legacy_client.forward_validation_samples = 4
    session_start_arguments = dict(
        **fixation_arguments,
        architecture_reference_classifier=adoption_arguments["adopted_candidate_classifier"],
        # 構造参照と選択済みの値を別モデルにし、初期値選択の再実装を検出する。
        initial_candidate_parameter_snapshot=snapshot_classifier_parameters(
            classifier=fixation_arguments["held_model_training_state_registry"]
            .snapshot_ordered_held_model_training_states()[-1]
            .classifier
        ),
        parameter_optimizer_settings=(
            SgdParameterOptimizerSettings(learning_rate=0.01)
            if optimizer_variant == "sgd"
            else AdamParameterOptimizerSettings(
                learning_rate=0.01, weight_decay=0.001, adam_variant=optimizer_variant
            )
        ),
        candidate_epoch_training_settings=CandidateEpochTrainingSettings(
            candidate_training_strategy=candidate_training_strategy,
            maximum_epoch_count=maximum_epoch_count,
            maximum_batch_sample_count=maximum_batch_sample_count,
            validation_sample_fraction=0.2,
            consecutive_non_improving_epoch_limit=consecutive_non_improving_epoch_limit,
            minimum_validation_loss_decrease=minimum_validation_loss_decrease,
        ),
        candidate_model_training_and_acceptance_settings=CandidateModelTrainingAndAcceptanceSettings(
            candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
            candidate_post_alarm_validation_sample_count=4,
        ),
        current_training_model_assignment=adoption_arguments["current_training_model_assignment"],
        input_features=torch.arange(22, dtype=torch.float32).reshape(11, 2) / 7,
        observed_class_labels=(torch.arange(11) % class_count).float().reshape(11, 1),
        pending_assignment_training_samples=adoption_arguments[
            "pending_assignment_training_samples"
        ],
        proposal_sample_index=40,
        estimated_change_point_sample_index=35,
        detection_episode_id=7,
        detector_name=legacy_client._detector_label(),
    )
    return session_start_arguments, shared_optimizer_owners, legacy_client


def begin_candidate_validation_session_in_legacy_client(*, session_start_arguments, legacy_client):
    """学習も参照固定も実旧の開始関数で実行し、実epoch引数を記録する。"""
    with patch.object(
        legacy_client, "_update_new_model_epochs", wraps=legacy_client._update_new_model_epochs
    ) as legacy_epoch_training_calls:
        legacy_client._begin_forward_validation(
            session_start_arguments["input_features"],
            session_start_arguments["observed_class_labels"],
            [
                (training_sample.input_features, training_sample.observed_class_labels, 1)
                for training_sample in session_start_arguments[
                    "pending_assignment_training_samples"
                ]
            ],
            legacy_client.models[legacy_client.current_model_id].get_params(),
            session_start_arguments["proposal_sample_index"],
            session_start_arguments["estimated_change_point_sample_index"],
            session_start_arguments["detection_episode_id"],
        )
    return legacy_client._forward_validation, legacy_epoch_training_calls


def assert_started_session_matches_legacy(
    *,
    started_session,
    legacy_session,
    legacy_client,
    legacy_epoch_training_calls,
    session_start_arguments,
):
    assert type(started_session) is PostAlarmCandidateValidationSession
    assert_candidate_epoch_training_matches_legacy(
        candidate_training_state=started_session.candidate_training_state,
        legacy_candidate=legacy_session.candidate,
    )
    assert (
        started_session.candidate_training_state.candidate_classifier.training
        is legacy_session.candidate.training
    )
    for parameter_optimizer_state, legacy_parameter_optimizer in (
        (
            started_session.candidate_training_state.candidate_shared_parameter_optimizer_state,
            legacy_session.candidate.backbone.optimizer,
        ),
        (
            started_session.candidate_training_state.candidate_concept_specific_parameter_optimizer_state,
            legacy_session.candidate.head_optimizer,
        ),
    ):
        assert type(parameter_optimizer_state.parameter_optimizer) is type(
            legacy_parameter_optimizer
        )
        assert (
            parameter_optimizer_state.parameter_optimizer.defaults
            == legacy_parameter_optimizer.defaults
        )
    assert_fixed_references_match_legacy(
        fixed_reference_models=started_session.fixed_reference_models,
        legacy_reference_models=legacy_session.reference_models,
        registry=session_start_arguments["held_model_training_state_registry"],
        input_features=session_start_arguments["input_features"],
    )
    assert (
        started_session.fixed_reference_models.reference_historical_mean_losses_by_model_id
        == legacy_session.reference_historical_means
    )
    assert tuple(
        started_session.fixed_reference_models.reference_historical_mean_losses_by_model_id
    ) == tuple(legacy_session.reference_historical_means)
    assert started_session.proposal_sample_index == legacy_session.proposal_position
    assert (
        started_session.estimated_change_point_sample_index == legacy_session.estimated_change_point
    )
    assert started_session.detection_episode_id == legacy_session.episode_id
    assert started_session.detector_name == legacy_session.detector
    assert started_session.initial_training_model_id == legacy_session.old_model_id
    assert started_session.training_input_features is legacy_session.training_x
    assert started_session.training_observed_class_labels is legacy_session.training_y
    assert (
        started_session.pending_assignment_training_samples
        is session_start_arguments["pending_assignment_training_samples"]
    )
    assert type(started_session.pending_assignment_training_samples) is tuple
    assert len(started_session.pending_assignment_training_samples) == len(legacy_session.held_data)
    for training_sample, (input_features, observed_class_labels, _) in zip(
        started_session.pending_assignment_training_samples, legacy_session.held_data
    ):
        assert training_sample.input_features is input_features
        assert training_sample.observed_class_labels is observed_class_labels
    candidate_epoch_training_result = started_session.candidate_epoch_training_result
    assert candidate_epoch_training_result.completed_epoch_count == sum(
        legacy_epoch_training_calls.call_args_list[training_batch_index].args[2]
        for training_batch_index in range(len(legacy_epoch_training_calls.call_args_list))
    )
    assert (
        candidate_epoch_training_result.candidate_trained_sample_count
        == legacy_session.candidate_training_examples
        == legacy_client.compute_counters["training_examples"]
    )
    assert (
        candidate_epoch_training_result.candidate_parameter_update_step_count
        == legacy_session.candidate_optimizer_steps
        == legacy_client.compute_counters["optimizer_steps"]
    )
    assert (
        candidate_epoch_training_result.validation_evaluated_sample_count
        == legacy_client.compute_counters["initialization_examples"]
    )
    collection_state = started_session.post_alarm_candidate_loss_collection.get_state_snapshot()
    assert collection_state.proposal_sample_index == legacy_session.proposal_position
    assert collection_state.last_validation_sample_index is None
    assert collection_state.candidate_losses == tuple(legacy_session.candidate_losses) == ()
    assert collection_state.reference_losses_by_model_id == tuple(
        (model_id, tuple(losses)) for model_id, losses in legacy_session.reference_losses.items()
    )
    assert collection_state.required_validation_sample_count == legacy_session.target_count
    assert collection_state.validation_sample_count == 0
    assert not collection_state.ready_for_acceptance_evaluation
    held_parameter_storage_addresses = {
        parameter.data_ptr()
        for state in session_start_arguments[
            "held_model_training_state_registry"
        ].snapshot_ordered_held_model_training_states()
        for parameter in state.classifier.parameters()
    }
    reference_parameter_storage_addresses = {
        parameter.data_ptr()
        for classifier in started_session.fixed_reference_models.reference_classifiers_by_model_id.values()
        for parameter in classifier.parameters()
    }
    for parameter in started_session.candidate_training_state.candidate_classifier.parameters():
        assert parameter.data_ptr() not in held_parameter_storage_addresses
        assert parameter.data_ptr() not in reference_parameter_storage_addresses


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize(
    "candidate_training_strategy,maximum_epoch_count,maximum_batch_sample_count,"
    "consecutive_non_improving_epoch_limit,minimum_validation_loss_decrease",
    [
        (candidate_training_strategy, maximum_epoch_count, 3, 1, 10)
        for candidate_training_strategy in (
            "fixed_epoch_training",
            "validation_loss_early_stopping",
            "skip_training",
        )
        for maximum_epoch_count in (0, 3)
    ]
    + [("validation_loss_early_stopping", 30, 32, 3, 0.0001)],
)
def test_session_start_matches_legacy(
    class_count,
    optimizer_variant,
    candidate_training_strategy,
    maximum_epoch_count,
    maximum_batch_sample_count,
    consecutive_non_improving_epoch_limit,
    minimum_validation_loss_decrease,
    monkeypatch,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(821)
        session_start_arguments, shared_optimizer_owners, legacy_client = (
            build_session_start_oracle(
                class_count=class_count,
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
                candidate_training_strategy=candidate_training_strategy,
                maximum_epoch_count=maximum_epoch_count,
                maximum_batch_sample_count=maximum_batch_sample_count,
                consecutive_non_improving_epoch_limit=consecutive_non_improving_epoch_limit,
                minimum_validation_loss_decrease=minimum_validation_loss_decrease,
            )
        )
        registry = session_start_arguments["held_model_training_state_registry"]
        held_model_training_states = registry.snapshot_ordered_held_model_training_states()
        held_parameter_values_and_gradients = snapshot_parameter_values_and_gradients(
            parameter
            for state in held_model_training_states
            for parameter in state.classifier.parameters()
        )
        held_optimizer_state_snapshots = tuple(
            (owner, owner.parameter_optimizer, deepcopy(owner.parameter_optimizer.state_dict()))
            for owner in tuple(
                state.concept_specific_parameter_optimizer_state
                for state in held_model_training_states
            )
            + tuple(shared_optimizer_owners)
        )
        loss_statistics_snapshot = session_start_arguments[
            "loss_statistics_store"
        ].get_state_snapshot()
        initial_training_model_id = session_start_arguments[
            "current_training_model_assignment"
        ].current_training_model_id
        session_input_tensor_values = snapshot_parameter_values_and_gradients(
            (
                session_start_arguments["input_features"],
                session_start_arguments["observed_class_labels"],
            )
            + tuple(session_start_arguments["initial_candidate_parameter_snapshot"].values())
            + tuple(
                training_tensor
                for training_sample in session_start_arguments[
                    "pending_assignment_training_samples"
                ]
                for training_tensor in (
                    training_sample.input_features,
                    training_sample.observed_class_labels,
                )
            )
            + tuple(session_start_arguments["architecture_reference_classifier"].parameters())
        )
        initial_rng_state = torch.get_rng_state().clone()
        other_random_states = (random.getstate(), np.random.get_state())
        legacy_session, legacy_epoch_training_calls = (
            begin_candidate_validation_session_in_legacy_client(
                session_start_arguments=session_start_arguments, legacy_client=legacy_client
            )
        )
        expected_rng_state = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng_state)
        started_session = start_post_alarm_candidate_validation_session(**session_start_arguments)
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
        assert random.getstate() == other_random_states[0]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == other_random_states[1][0]
        assert np.array_equal(numpy_state[1], other_random_states[1][1])
        assert numpy_state[2:] == other_random_states[1][2:]
        assert_started_session_matches_legacy(
            started_session=started_session,
            legacy_session=legacy_session,
            legacy_client=legacy_client,
            legacy_epoch_training_calls=legacy_epoch_training_calls,
            session_start_arguments=session_start_arguments,
        )
        assert all(
            state is previous_state
            for state, previous_state in zip(
                registry.snapshot_ordered_held_model_training_states(), held_model_training_states
            )
        )
        assert_parameter_values_and_gradients_unchanged(held_parameter_values_and_gradients)
        assert_parameter_values_and_gradients_unchanged(session_input_tensor_values)
        for owner, previous_optimizer, previous_optimizer_state in held_optimizer_state_snapshots:
            assert owner.parameter_optimizer is previous_optimizer
            assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
        assert (
            session_start_arguments["loss_statistics_store"].get_state_snapshot()
            == loss_statistics_snapshot
        )
        assert (
            session_start_arguments["current_training_model_assignment"].current_training_model_id
            == initial_training_model_id
        )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("statistics_case", list(STATISTICS_CASES))
def test_session_start_retains_historical_mean_losses(class_count, statistics_case, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        session_start_arguments, _, legacy_client = build_session_start_oracle(
            class_count=class_count, optimizer_variant="standard", monkeypatch=monkeypatch
        )
        set_overall_loss_statistics_in_both_implementations(
            loss_statistics_store=session_start_arguments["loss_statistics_store"],
            legacy_client=legacy_client,
            statistics_by_model_id=STATISTICS_CASES[statistics_case],
        )
        initial_rng_state = torch.get_rng_state().clone()
        legacy_session, legacy_epoch_training_calls = (
            begin_candidate_validation_session_in_legacy_client(
                session_start_arguments=session_start_arguments, legacy_client=legacy_client
            )
        )
        expected_rng_state = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng_state)
        started_session = start_post_alarm_candidate_validation_session(**session_start_arguments)
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
        assert_started_session_matches_legacy(
            started_session=started_session,
            legacy_session=legacy_session,
            legacy_client=legacy_client,
            legacy_epoch_training_calls=legacy_epoch_training_calls,
            session_start_arguments=session_start_arguments,
        )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("held_model_ids", [(4,), (9, 4, -3)])
def test_session_start_preserves_optional_metadata_and_empty_pending_samples(
    class_count, held_model_ids, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        session_start_arguments, _, legacy_client = build_session_start_oracle(
            class_count=class_count,
            optimizer_variant="sgd",
            monkeypatch=monkeypatch,
            held_model_ids=held_model_ids,
        )
        session_start_arguments.update(
            estimated_change_point_sample_index=None,
            detection_episode_id=None,
            pending_assignment_training_samples=(),
        )
        initial_rng_state = torch.get_rng_state().clone()
        legacy_session, legacy_epoch_training_calls = (
            begin_candidate_validation_session_in_legacy_client(
                session_start_arguments=session_start_arguments, legacy_client=legacy_client
            )
        )
        expected_rng_state = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng_state)
        started_session = start_post_alarm_candidate_validation_session(**session_start_arguments)
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
        assert_started_session_matches_legacy(
            started_session=started_session,
            legacy_session=legacy_session,
            legacy_client=legacy_client,
            legacy_epoch_training_calls=legacy_epoch_training_calls,
            session_start_arguments=session_start_arguments,
        )
