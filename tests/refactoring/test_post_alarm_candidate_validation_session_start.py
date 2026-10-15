"""候補検証の開始を、候補学習が有効な実旧session開始へ対照する。"""

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_candidate_epoch_training import assert_candidate_epoch_training_matches_legacy
from test_joint_model_parameter_update import assert_nested_state_equal
from test_post_alarm_candidate_validation_sample_observation import (
    assert_collected_losses_match_legacy,
)
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
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.participating_model_training_batch import (
    ParticipatingModelTrainingBatch,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation import (
    observe_post_alarm_candidate_validation_sample,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
    start_post_alarm_candidate_validation_session,
)


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
def test_session_start_connects_to_validation_observation(
    class_count, optimizer_variant, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(827)
        session_start_arguments, shared_optimizer_owners, legacy_client = (
            build_session_start_oracle(
                class_count=class_count,
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
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
        candidate_training_state = started_session.candidate_training_state
        candidate_parameter_snapshot = snapshot_classifier_parameters(
            classifier=candidate_training_state.candidate_classifier
        )
        reference_parameter_snapshots_by_model_id = {
            model_id: snapshot_parameter_values_and_gradients(classifier.parameters())
            for model_id, classifier in started_session.fixed_reference_models.reference_classifiers_by_model_id.items()
        }
        collection = started_session.post_alarm_candidate_loss_collection
        validation_samples = tuple(
            (
                started_session.proposal_sample_index + 1 + sample_offset,
                started_session.training_input_features[sample_offset].reshape(1, -1) * 0.5
                + 0.1 * sample_offset,
                started_session.training_observed_class_labels[sample_offset].reshape(1, 1),
            )
            for sample_offset in range(legacy_session.target_count)
        )
        batch_input_features = torch.ones(4, 2) / 3
        batch_observed_class_labels = (torch.arange(4) % class_count).float().reshape(4, 1)
        session_input_tensor_values = snapshot_parameter_values_and_gradients(
            (
                started_session.training_input_features,
                started_session.training_observed_class_labels,
                batch_input_features,
                batch_observed_class_labels,
            )
            + tuple(session_start_arguments["initial_candidate_parameter_snapshot"].values())
            + tuple(session_start_arguments["architecture_reference_classifier"].parameters())
            + tuple(
                training_tensor
                for training_sample in started_session.pending_assignment_training_samples
                for training_tensor in (
                    training_sample.input_features,
                    training_sample.observed_class_labels,
                )
            )
            + tuple(
                training_tensor
                for _, input_features, observed_class_labels in validation_samples
                for training_tensor in (input_features, observed_class_labels)
            )
        )
        for sample_offset, (sample_index, input_features, observed_class_labels) in enumerate(
            validation_samples
        ):
            # 実旧の損失評価と収集だけを呼び、到達時の採否確定は起動しない。
            legacy_candidate_losses = legacy_session.candidate.per_sample_error(
                input_features, observed_class_labels
            )
            legacy_reference_losses_by_model_id = {
                model_id: reference_classifier.per_sample_error(
                    input_features, observed_class_labels
                )
                .reshape(-1)[0]
                .item()
                for model_id, reference_classifier in legacy_session.reference_models.items()
            }
            legacy_session.append_losses(
                legacy_candidate_losses.reshape(-1)[0].item(),
                legacy_reference_losses_by_model_id,
            )
            ready_for_acceptance_evaluation = observe_post_alarm_candidate_validation_sample(
                sample_index=sample_index,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
                candidate_classifier=candidate_training_state.candidate_classifier,
                reference_classifiers_by_model_id=started_session.fixed_reference_models.reference_classifiers_by_model_id,
                post_alarm_candidate_loss_collection=collection,
            )
            assert_collected_losses_match_legacy(
                post_alarm_candidate_loss_collection=collection, legacy_session=legacy_session
            )
            collection_state = collection.get_state_snapshot()
            assert collection_state.proposal_sample_index == started_session.proposal_sample_index
            assert collection_state.last_validation_sample_index == sample_index
            assert collection_state.validation_sample_count == sample_offset + 1
            assert collection_state.required_validation_sample_count == 4
            assert ready_for_acceptance_evaluation is (sample_offset == 3)
            if sample_offset == 0:
                # 最初の観測後に候補だけを継続学習し、以降の参照損失の固定を対照する。
                legacy_mean_training_loss = legacy_session.candidate.update(
                    batch_input_features, batch_observed_class_labels
                )
                candidate_mean_training_loss = perform_joint_model_parameter_update(
                    local_training_settings=LocalTrainingSettings(
                        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
                    ),
                    shared_feature_extractor=candidate_training_state.candidate_classifier.feature_extractor,
                    shared_parameter_optimizer=candidate_training_state.candidate_shared_parameter_optimizer_state.parameter_optimizer,
                    participating_training_batches=(
                        ParticipatingModelTrainingBatch(
                            classifier=candidate_training_state.candidate_classifier,
                            concept_specific_parameter_optimizer=candidate_training_state.candidate_concept_specific_parameter_optimizer_state.parameter_optimizer,
                            input_features=batch_input_features,
                            observed_class_labels=batch_observed_class_labels,
                        ),
                    ),
                    update_shared_features=True,
                )
                assert candidate_mean_training_loss == legacy_mean_training_loss
                assert any(
                    not torch.equal(parameter_value, candidate_parameter_snapshot[parameter_name])
                    for parameter_name, parameter_value in snapshot_classifier_parameters(
                        classifier=candidate_training_state.candidate_classifier
                    ).items()
                )
                assert_candidate_epoch_training_matches_legacy(
                    candidate_training_state=candidate_training_state,
                    legacy_candidate=legacy_session.candidate,
                )
            for reference_parameter_snapshots in reference_parameter_snapshots_by_model_id.values():
                assert_parameter_values_and_gradients_unchanged(reference_parameter_snapshots)
            assert_parameter_values_and_gradients_unchanged(held_parameter_values_and_gradients)
            assert_parameter_values_and_gradients_unchanged(session_input_tensor_values)
            assert all(
                state is previous_state
                for state, previous_state in zip(
                    registry.snapshot_ordered_held_model_training_states(),
                    held_model_training_states,
                )
            )
            for (
                owner,
                previous_optimizer,
                previous_optimizer_state,
            ) in held_optimizer_state_snapshots:
                assert owner.parameter_optimizer is previous_optimizer
                assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
            assert (
                session_start_arguments["loss_statistics_store"].get_state_snapshot()
                == loss_statistics_snapshot
            )
            assert (
                session_start_arguments[
                    "current_training_model_assignment"
                ].current_training_model_id
                == initial_training_model_id
            )
            assert legacy_client._forward_validation is legacy_session
            assert started_session.post_alarm_candidate_loss_collection is collection
            assert (
                started_session.training_input_features is session_start_arguments["input_features"]
            )
            assert (
                started_session.training_observed_class_labels
                is session_start_arguments["observed_class_labels"]
            )
            assert (
                started_session.pending_assignment_training_samples
                is session_start_arguments["pending_assignment_training_samples"]
            )
            assert torch.equal(torch.get_rng_state(), expected_rng_state)


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


@pytest.mark.parametrize(
    "invalid_case,field_value,expected_exception",
    [
        ("valid", None, None),
        ("skip_grad_disabled", None, None),
        ("epoch_zero_grad_disabled", None, None),
    ]
    + [
        (field_name, field_value, expected_exception)
        for field_name, field_value, expected_exception in (
            ("proposal_sample_index", True, TypeError),
            ("proposal_sample_index", np.int64(40), TypeError),
            ("proposal_sample_index", type("IntSubclass", (int,), {})(40), TypeError),
            ("proposal_sample_index", -1, ValueError),
            ("estimated_change_point_sample_index", False, TypeError),
            ("estimated_change_point_sample_index", 35.0, TypeError),
            ("estimated_change_point_sample_index", -1, ValueError),
            ("estimated_change_point_sample_index", 41, ValueError),
            ("detection_episode_id", True, TypeError),
            ("detection_episode_id", 7.0, TypeError),
            ("detection_episode_id", -1, ValueError),
            ("detector_name", None, TypeError),
            ("detector_name", type("StrSubclass", (str,), {})("detector"), TypeError),
            ("detector_name", " \t\n", ValueError),
            ("candidate_epoch_training_settings", object(), TypeError),
            ("candidate_model_training_and_acceptance_settings", object(), TypeError),
            ("held_model_training_state_registry", object(), TypeError),
            ("loss_statistics_store", object(), TypeError),
            ("current_training_model_assignment", object(), TypeError),
            ("parameter_optimizer_settings", object(), TypeError),
            ("initial_candidate_parameter_snapshot", [], TypeError),
            ("architecture_reference_classifier", object(), TypeError),
            ("epoch:candidate_training_strategy", "invalid", ValueError),
            ("epoch:maximum_epoch_count", -1, ValueError),
            ("epoch:maximum_batch_sample_count", 0, ValueError),
            ("epoch:validation_sample_fraction", 1.0, ValueError),
            ("epoch:consecutive_non_improving_epoch_limit", 0, ValueError),
            ("epoch:minimum_validation_loss_decrease", -1.0, ValueError),
            ("acceptance:candidate_model_acceptance_policy", "invalid", ValueError),
            ("acceptance:candidate_post_alarm_validation_sample_count", True, TypeError),
            ("acceptance:candidate_post_alarm_validation_sample_count", 1, ValueError),
            ("optimizer:learning_rate", float("nan"), ValueError),
            ("optimizer:weight_decay", -1.0, ValueError),
            ("optimizer:adam_variant", "invalid", ValueError),
            ("sgd_learning_rate", -1.0, ValueError),
            ("empty_registry", None, LookupError),
            ("missing_assignment", None, LookupError),
            ("held_dtype", None, ValueError),
            ("held_nonfinite", None, ValueError),
            ("held_input_width", None, ValueError),
            ("held_class_count", None, ValueError),
            ("reference_empty_features", None, ValueError),
            ("reference_dtype", None, ValueError),
            ("reference_nonfinite", None, ValueError),
            ("snapshot_missing_key", None, ValueError),
            ("snapshot_shape", None, ValueError),
            ("snapshot_dtype", None, ValueError),
            ("snapshot_nonfinite", None, ValueError),
            ("snapshot_tensor_type", None, TypeError),
            ("pending_container", None, TypeError),
            ("pending_record", None, TypeError),
            ("pending_tuple_subclass", None, TypeError),
            ("pending_record_subclass", None, TypeError),
            ("snapshot_dict_subclass", None, TypeError),
            ("snapshot_key_subclass", None, TypeError),
            ("pending_sample_count", None, ValueError),
            ("statistics:observed_loss_count", -1, ValueError),
            ("statistics:mean_loss", 2.0, ValueError),
            ("statistics:sum_squared_loss_deviations", -1.0, ValueError),
            ("grad_disabled", None, ValueError),
        )
    ]
    + [
        (f"subclass:{field_name}", None, TypeError)
        for field_name in (
            "candidate_epoch_training_settings",
            "candidate_model_training_and_acceptance_settings",
            "held_model_training_state_registry",
            "loss_statistics_store",
            "current_training_model_assignment",
            "parameter_optimizer_settings",
            "architecture_reference_classifier",
        )
    ]
    + [
        (f"{field_name}:{invalid_case}", None, expected_exception)
        for field_name in (
            "input_features",
            "observed_class_labels",
            "pending_features",
            "pending_labels",
        )
        for invalid_case, expected_exception in (
            ("type", TypeError),
            ("dtype", ValueError),
            ("device", ValueError),
            ("sparse", ValueError),
            ("nested", ValueError),
            ("empty", ValueError),
            ("ndim", ValueError),
            ("shape", ValueError),
            ("nonfinite", ValueError),
        )
    ]
    + [
        (f"{field_name}:{invalid_case}", None, ValueError)
        for field_name in ("observed_class_labels", "pending_labels")
        for invalid_case in ("fractional", "negative", "out_of_range")
    ],
)
def test_session_start_rejects_before_mutation(
    invalid_case, field_value, expected_exception, monkeypatch
):
    """各事前検査を通る前に、乱数消費と借用状態の変更を拒否する。"""
    with torch.random.fork_rng(devices=[]):
        session_start_arguments, shared_optimizer_owners, _ = build_session_start_oracle(
            class_count=4, optimizer_variant="standard", monkeypatch=monkeypatch
        )
        registry = session_start_arguments["held_model_training_state_registry"]
        loss_statistics_store = session_start_arguments["loss_statistics_store"]
        current_training_model_assignment = session_start_arguments[
            "current_training_model_assignment"
        ]
        architecture_reference_classifier = session_start_arguments[
            "architecture_reference_classifier"
        ]
        held_model_training_states = registry.snapshot_ordered_held_model_training_states()
        initial_candidate_parameter_snapshot = session_start_arguments[
            "initial_candidate_parameter_snapshot"
        ]
        parameter_name = next(iter(initial_candidate_parameter_snapshot))
        parameter_values = initial_candidate_parameter_snapshot[parameter_name]
        if invalid_case in session_start_arguments:
            session_start_arguments[invalid_case] = field_value
        elif invalid_case.startswith("subclass:"):
            field_name = invalid_case.removeprefix("subclass:")
            field_value = type("InputSubclass", (type(session_start_arguments[field_name]),), {})
            field_value = object.__new__(field_value)
            field_value.__dict__.update(session_start_arguments[field_name].__dict__)
            session_start_arguments[field_name] = field_value
        elif invalid_case.startswith(("epoch:", "acceptance:", "optimizer:")):
            object.__setattr__(
                session_start_arguments[
                    {
                        "epoch": "candidate_epoch_training_settings",
                        "acceptance": "candidate_model_training_and_acceptance_settings",
                        "optimizer": "parameter_optimizer_settings",
                    }[invalid_case.split(":")[0]]
                ],
                invalid_case.split(":")[1],
                field_value,
            )
        elif invalid_case == "sgd_learning_rate":
            session_start_arguments["parameter_optimizer_settings"] = SgdParameterOptimizerSettings(
                learning_rate=0.01
            )
            object.__setattr__(
                session_start_arguments["parameter_optimizer_settings"],
                "learning_rate",
                field_value,
            )
        elif invalid_case in ("skip_grad_disabled", "epoch_zero_grad_disabled"):
            object.__setattr__(
                session_start_arguments["candidate_epoch_training_settings"],
                "candidate_training_strategy"
                if invalid_case == "skip_grad_disabled"
                else "maximum_epoch_count",
                "skip_training" if invalid_case == "skip_grad_disabled" else 0,
            )
        elif invalid_case == "empty_registry":
            session_start_arguments["held_model_training_state_registry"] = type(registry)()
        elif invalid_case == "missing_assignment":
            current_training_model_assignment.assign_model_for_training(model_id=999)
        elif invalid_case in ("held_dtype", "reference_dtype"):
            (
                held_model_training_states[-1].classifier
                if invalid_case == "held_dtype"
                else architecture_reference_classifier
            ).double()
        elif invalid_case in ("held_nonfinite", "reference_nonfinite"):
            with torch.no_grad():
                next(
                    (
                        held_model_training_states[-1].classifier
                        if invalid_case == "held_nonfinite"
                        else architecture_reference_classifier
                    ).parameters()
                ).fill_(float("nan"))
        elif invalid_case == "held_input_width":
            held_model_training_states[-1].classifier.feature_extractor.input_feature_count = 3
        elif invalid_case == "held_class_count":
            held_model_training_states[-1].classifier.class_count = 2
        elif invalid_case == "reference_empty_features":
            architecture_reference_classifier = type(architecture_reference_classifier)(
                model_architecture_settings=architecture_reference_classifier.model_architecture_settings,
                input_feature_count=2,
                hidden_layer_widths=(),
                class_count=4,
            )
            session_start_arguments["architecture_reference_classifier"] = (
                architecture_reference_classifier
            )
            session_start_arguments["initial_candidate_parameter_snapshot"] = (
                snapshot_classifier_parameters(classifier=architecture_reference_classifier)
            )
        elif invalid_case.startswith("snapshot_"):
            if invalid_case == "snapshot_missing_key":
                del initial_candidate_parameter_snapshot[parameter_name]
            elif invalid_case == "snapshot_dict_subclass":
                session_start_arguments["initial_candidate_parameter_snapshot"] = type(
                    "DictSubclass", (dict,), {}
                )(initial_candidate_parameter_snapshot)
            elif invalid_case == "snapshot_key_subclass":
                initial_candidate_parameter_snapshot[
                    type("StrSubclass", (str,), {})(parameter_name)
                ] = initial_candidate_parameter_snapshot.pop(parameter_name)
            else:
                initial_candidate_parameter_snapshot[parameter_name] = {
                    "snapshot_shape": lambda: parameter_values.flatten()[:1],
                    "snapshot_dtype": lambda: parameter_values.double(),
                    "snapshot_nonfinite": lambda: torch.full_like(parameter_values, float("inf")),
                    "snapshot_tensor_type": lambda: torch.nn.Parameter(parameter_values.clone()),
                }[invalid_case]()
        elif invalid_case == "pending_container":
            session_start_arguments["pending_assignment_training_samples"] = list(
                session_start_arguments["pending_assignment_training_samples"]
            )
        elif invalid_case == "pending_record":
            session_start_arguments["pending_assignment_training_samples"] = (object(),)
        elif invalid_case == "pending_tuple_subclass":
            session_start_arguments["pending_assignment_training_samples"] = type(
                "TupleSubclass", (tuple,), {}
            )(session_start_arguments["pending_assignment_training_samples"])
        elif invalid_case == "pending_record_subclass":
            training_sample = session_start_arguments["pending_assignment_training_samples"][0]
            session_start_arguments["pending_assignment_training_samples"] = (
                type("SampleSubclass", (ObservedTrainingSample,), {})(
                    input_features=training_sample.input_features,
                    observed_class_labels=training_sample.observed_class_labels,
                ),
            )
        elif invalid_case == "pending_sample_count":
            session_start_arguments["pending_assignment_training_samples"] = (
                ObservedTrainingSample(
                    input_features=torch.zeros(2, 2), observed_class_labels=torch.zeros(2, 1)
                ),
            )
        elif invalid_case.startswith("statistics:"):
            # 非公開破損は保証対象外。公開getによる再検査だけを補助的に観測するfixture。
            object.__setattr__(
                loss_statistics_store._model_loss_statistics_by_model_id[
                    held_model_training_states[-1].model_id
                ].overall_loss_moments,
                invalid_case.split(":")[1],
                field_value,
            )
        elif ":" in invalid_case:
            field_name, invalid_case = invalid_case.split(":")
            if field_name.startswith("pending_"):
                training_sample = session_start_arguments["pending_assignment_training_samples"][0]
                training_tensor = (
                    training_sample.input_features
                    if field_name == "pending_features"
                    else training_sample.observed_class_labels
                )
            else:
                training_tensor = session_start_arguments[field_name]
            training_tensor = {
                "type": lambda: torch.nn.Parameter(training_tensor.clone()),
                "dtype": lambda: training_tensor.double(),
                "device": lambda: training_tensor.to("meta"),
                "sparse": lambda: training_tensor.to_sparse(),
                "nested": lambda: torch.nested.nested_tensor([training_tensor]),
                "empty": lambda: training_tensor[:0],
                "ndim": lambda: training_tensor.flatten(),
                "shape": lambda: torch.cat((training_tensor, training_tensor), dim=1),
                "nonfinite": lambda: torch.full_like(training_tensor, float("nan")),
                "fractional": lambda: torch.full_like(training_tensor, 0.5),
                "negative": lambda: torch.full_like(training_tensor, -1),
                "out_of_range": lambda: torch.full_like(training_tensor, 4),
            }[invalid_case]()
            if field_name.startswith("pending_"):
                session_start_arguments["pending_assignment_training_samples"] = (
                    ObservedTrainingSample(
                        input_features=training_tensor
                        if field_name == "pending_features"
                        else training_sample.input_features,
                        observed_class_labels=training_tensor
                        if field_name == "pending_labels"
                        else training_sample.observed_class_labels,
                    ),
                ) + session_start_arguments["pending_assignment_training_samples"][1:]
            else:
                session_start_arguments[field_name] = training_tensor
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
        loss_statistics_snapshot = deepcopy(loss_statistics_store.__dict__)
        initial_training_model_id = current_training_model_assignment.current_training_model_id
        session_input_tensor_values = snapshot_parameter_values_and_gradients(
            (
                session_start_arguments["input_features"],
                session_start_arguments["observed_class_labels"],
            )
            + tuple(
                parameter
                for parameter in initial_candidate_parameter_snapshot.values()
                if isinstance(parameter, torch.Tensor)
            )
            + tuple(
                training_tensor
                for training_sample in session_start_arguments[
                    "pending_assignment_training_samples"
                ]
                if type(training_sample) is ObservedTrainingSample
                for training_tensor in (
                    training_sample.input_features,
                    training_sample.observed_class_labels,
                )
            )
            + tuple(architecture_reference_classifier.parameters())
        )
        initial_rng_state = torch.get_rng_state().clone()
        other_random_states = (random.getstate(), np.random.get_state())
        with torch.set_grad_enabled(not invalid_case.endswith("grad_disabled")):
            if expected_exception is not None:
                with pytest.raises(expected_exception):
                    start_post_alarm_candidate_validation_session(**session_start_arguments)
                assert torch.equal(torch.get_rng_state(), initial_rng_state)
            else:
                started_session = start_post_alarm_candidate_validation_session(
                    **session_start_arguments
                )
                assert type(started_session) is PostAlarmCandidateValidationSession
                assert tuple(parameter_name.name for parameter_name in fields(started_session)) == (
                    "proposal_sample_index",
                    "estimated_change_point_sample_index",
                    "detection_episode_id",
                    "detector_name",
                    "initial_training_model_id",
                    "candidate_training_state",
                    "candidate_epoch_training_result",
                    "training_input_features",
                    "training_observed_class_labels",
                    "pending_assignment_training_samples",
                    "fixed_reference_models",
                    "post_alarm_candidate_loss_collection",
                )
                assert all(parameter_name.kw_only for parameter_name in fields(started_session))
                for parameter_name in fields(started_session):
                    with pytest.raises(FrozenInstanceError):
                        setattr(
                            started_session,
                            parameter_name.name,
                            getattr(started_session, parameter_name.name),
                        )
                with pytest.raises(TypeError):
                    PostAlarmCandidateValidationSession(
                        *(
                            getattr(started_session, parameter_name.name)
                            for parameter_name in fields(started_session)
                        )
                    )
                assert (
                    started_session.training_input_features
                    is session_start_arguments["input_features"]
                )
                assert (
                    started_session.training_observed_class_labels
                    is session_start_arguments["observed_class_labels"]
                )
                assert (
                    started_session.pending_assignment_training_samples
                    is session_start_arguments["pending_assignment_training_samples"]
                )
        assert random.getstate() == other_random_states[0]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == other_random_states[1][0]
        assert np.array_equal(numpy_state[1], other_random_states[1][1])
        assert numpy_state[2:] == other_random_states[1][2:]
        assert all(
            state is previous_state
            for state, previous_state in zip(
                registry.snapshot_ordered_held_model_training_states(), held_model_training_states
            )
        )
        for parameter, parameter_values, previous_gradients in (
            held_parameter_values_and_gradients + session_input_tensor_values
        ):
            if not parameter.is_nested:
                assert parameter.shape == parameter_values.shape
            assert parameter.dtype == parameter_values.dtype
            assert parameter.device == parameter_values.device
            assert parameter.layout == parameter_values.layout
            if parameter.device.type != "meta":
                if parameter.is_nested:
                    torch.testing.assert_close(
                        parameter.unbind(),
                        parameter_values.unbind(),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                else:
                    torch.testing.assert_close(
                        parameter.to_dense(),
                        parameter_values.to_dense(),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
            assert parameter.grad is previous_gradients[0]
            if previous_gradients[1] is not None:
                torch.testing.assert_close(
                    parameter.grad, previous_gradients[1], rtol=0, atol=0, equal_nan=True
                )
        for owner, previous_optimizer, previous_optimizer_state in held_optimizer_state_snapshots:
            assert owner.parameter_optimizer is previous_optimizer
            assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
        assert_nested_state_equal(loss_statistics_store.__dict__, loss_statistics_snapshot)
        assert (
            current_training_model_assignment.current_training_model_id == initial_training_model_id
        )
