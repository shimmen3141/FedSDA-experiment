"""警報後の損失収集を旧sessionと直接照合する。"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import inspect
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.fedsda import FedSDAClient
from federated_drift_experiment.provisional_model import (
    ForwardValidationSession, select_forward_fitting_reference,
    has_disjoint_validation_advantage,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import PostAlarmCandidateLossCollection


def make_loss_collection(target_count=3, proposal_sample_index=100, reference_model_ids=(9, -2, 3)):
    return PostAlarmCandidateLossCollection(
        candidate_model_training_and_acceptance_settings=CandidateModelTrainingAndAcceptanceSettings(
            candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
            candidate_post_alarm_validation_sample_count=target_count,
        ),
        proposal_sample_index=proposal_sample_index, reference_model_ids=reference_model_ids,
    )


def make_legacy_loss_session(target_count=3, proposal_sample_index=100, reference_model_ids=(9, -2, 3)):
    return ForwardValidationSession(
        proposal_position=proposal_sample_index, estimated_change_point=None,
        episode_id=None, old_model_id=9, detector="test", candidate=None,
        training_x=None, training_y=None, held_data=[],
        reference_models=dict.fromkeys(reference_model_ids), target_count=target_count,
    )


@pytest.mark.parametrize("target_count", [2, 3, 5])
@pytest.mark.parametrize("proposal_sample_index", [0, 100, 10**30])
def test_candidate_loss_collection_matches_legacy_session(target_count, proposal_sample_index):
    """固定参照順・同一系列長・float化・到達回を全観測で比較する。"""
    collection = make_loss_collection(target_count, proposal_sample_index)
    legacy_session = make_legacy_loss_session(target_count, proposal_sample_index)
    assert collection.validation_sample_count == 0
    assert not collection.ready_for_acceptance_evaluation
    assert collection.get_state_snapshot().last_validation_sample_index is None
    for observation_index in range(target_count):
        reference_losses_by_model_id = {3: .7, -2: .6, 9: .5}
        candidate_loss = 0 if observation_index % 2 else 1
        collection.observe_losses_after_label_observation(
            sample_index=proposal_sample_index + observation_index + 1,
            candidate_loss=candidate_loss,
            reference_losses_by_model_id=reference_losses_by_model_id,
        )
        legacy_session.append_losses(candidate_loss, reference_losses_by_model_id)
        state_snapshot = collection.get_state_snapshot()
        assert state_snapshot.candidate_losses == tuple(legacy_session.candidate_losses)
        assert all(type(candidate_loss) is float for candidate_loss in state_snapshot.candidate_losses)
        assert state_snapshot.reference_losses_by_model_id == tuple(
            (model_id, tuple(reference_losses)) for model_id, reference_losses in legacy_session.reference_losses.items()
        )
        assert state_snapshot.proposal_sample_index == proposal_sample_index
        assert state_snapshot.last_validation_sample_index == proposal_sample_index + observation_index + 1
        assert state_snapshot.validation_sample_count == collection.validation_sample_count == legacy_session.validation_count
        assert state_snapshot.required_validation_sample_count == target_count
        assert state_snapshot.ready_for_acceptance_evaluation == collection.ready_for_acceptance_evaluation == legacy_session.ready
    state_before_call = collection.get_state_snapshot()
    with pytest.raises(RuntimeError, match="ready"):
        collection.observe_losses_after_label_observation(
            sample_index=proposal_sample_index + target_count + 1,
            candidate_loss=.2, reference_losses_by_model_id={9:.5, -2:.6, 3:.7},
        )
    assert collection.get_state_snapshot() == state_before_call
    assert collection.get_state_snapshot() == state_before_call


@pytest.mark.parametrize("operation,invalid_value", [
    ("proposal_sample_index", True), ("proposal_sample_index", 1.0),
    ("proposal_sample_index", -1), ("proposal_sample_index", np.int64(1)),
    ("reference_model_ids", []), ("reference_model_ids", ()),
    ("reference_model_ids", (1, 1)), ("reference_model_ids", (True,)),
    ("reference_model_ids", (1.0,)), ("reference_model_ids", (np.int64(1),)),
])
def test_candidate_loss_collection_invalid_initial_conditions(operation, invalid_value):
    inputs_before_call = dict(proposal_sample_index=100, reference_model_ids=(9, -2, 3))
    inputs_before_call[operation] = invalid_value
    with pytest.raises((TypeError, ValueError), match=operation):
        make_loss_collection(**inputs_before_call)


@pytest.mark.parametrize("invalid_value", [None, {}, 2, SimpleNamespace(candidate_post_alarm_validation_sample_count=2)])
def test_candidate_loss_collection_invalid_initial_conditions_for_settings_type(invalid_value):
    with pytest.raises(TypeError, match="candidate_model_training_and_acceptance_settings"):
        PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=invalid_value,
            proposal_sample_index=0, reference_model_ids=(1,),
        )


@pytest.mark.parametrize("invalid_value", [True, False, 1, 0, 2.0, np.int64(2)])
def test_candidate_loss_collection_invalid_initial_conditions_for_modified_settings(invalid_value):
    candidate_model_training_and_acceptance_settings = CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
        candidate_post_alarm_validation_sample_count=2,
    )
    object.__setattr__(candidate_model_training_and_acceptance_settings, "candidate_post_alarm_validation_sample_count", invalid_value)
    with pytest.raises((TypeError, ValueError), match="candidate_post_alarm_validation_sample_count"):
        PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
            proposal_sample_index=0, reference_model_ids=(1,),
        )


@pytest.mark.parametrize("operation,invalid_value", [
    ("sample_index", True), ("sample_index", 101.0), ("sample_index", np.int64(101)),
    ("sample_index", -1), ("sample_index", 100), ("sample_index", 102),
    ("candidate_loss", True), ("candidate_loss", None), ("candidate_loss", ".2"),
    ("candidate_loss", np.float64(.2)), ("candidate_loss", float("nan")),
    ("candidate_loss", float("inf")), ("candidate_loss", -.1), ("candidate_loss", 1.1),
    ("candidate_loss", 10**300),
    ("reference_losses_by_model_id", None), ("reference_losses_by_model_id", []),
    ("reference_losses_by_model_id", {9:.5, -2:.6}),
    ("reference_losses_by_model_id", {9:.5, -2:.6, 3:.7, 4:.8}),
    ("reference_losses_by_model_id", {9:.5, -2:.6, 3:None}),
    ("reference_losses_by_model_id", {9:.5, -2:.6, 3:True}),
    ("reference_losses_by_model_id", {9:.5, -2:.6, 3:float("nan")}),
    ("reference_losses_by_model_id", {9:.5, -2:.6, 3:1.1}),
    ("reference_losses_by_model_id", {9:.5, -2:.6, 3.0:.7}),
])
def test_candidate_loss_collection_invalid_observation_is_atomic(operation, invalid_value):
    collection = make_loss_collection()
    state_before_call = collection.get_state_snapshot()
    inputs_before_call = dict(sample_index=101, candidate_loss=.2, reference_losses_by_model_id={9:.5, -2:.6, 3:.7})
    inputs_before_call[operation] = invalid_value
    with pytest.raises((TypeError, ValueError), match=operation):
        collection.observe_losses_after_label_observation(**inputs_before_call)
    assert collection.get_state_snapshot() == state_before_call
    collection.observe_losses_after_label_observation(
        sample_index=101, candidate_loss=.2, reference_losses_by_model_id={9:.5, -2:.6, 3:.7},
    )
    state_before_call = collection.get_state_snapshot()
    with pytest.raises(ValueError, match="sample_index"):
        collection.observe_losses_after_label_observation(
            sample_index=101, candidate_loss=.2, reference_losses_by_model_id={9:.5, -2:.6, 3:.7},
        )
    assert collection.get_state_snapshot() == state_before_call


def test_candidate_loss_collection_invalid_observation_is_atomic_for_legacy_partial_update():
    """LEGACY-004の不足参照入力を直接再現し、新境界では全系列を保持する。"""
    legacy_session = make_legacy_loss_session(reference_model_ids=(2, 1))
    with pytest.raises(KeyError):
        legacy_session.append_losses(.2, {2:.4})
    assert legacy_session.candidate_losses == [.2]
    assert legacy_session.reference_losses == {2:[.4], 1:[]}
    collection = make_loss_collection(reference_model_ids=(2, 1))
    state_before_call = collection.get_state_snapshot()
    with pytest.raises(ValueError, match="reference_losses_by_model_id"):
        collection.observe_losses_after_label_observation(
            sample_index=101, candidate_loss=.2, reference_losses_by_model_id={2:.4},
        )
    assert collection.get_state_snapshot() == state_before_call
