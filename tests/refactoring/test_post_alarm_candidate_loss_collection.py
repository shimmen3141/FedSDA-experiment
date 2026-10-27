"""警報後の損失収集を旧sessionと直接照合する。"""

import inspect
import random
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)


def make_loss_collection(target_count=3, proposal_sample_index=100, reference_model_ids=(9, -2, 3)):
    return PostAlarmCandidateLossCollection(
        candidate_model_training_and_acceptance_settings=CandidateModelTrainingAndAcceptanceSettings(
            candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
            candidate_post_alarm_validation_sample_count=target_count,
        ),
        proposal_sample_index=proposal_sample_index,
        reference_model_ids=reference_model_ids,
    )


@pytest.mark.parametrize(
    "operation,invalid_value",
    [
        ("proposal_sample_index", True),
        ("proposal_sample_index", 1.0),
        ("proposal_sample_index", -1),
        ("proposal_sample_index", np.int64(1)),
        ("reference_model_ids", []),
        ("reference_model_ids", ()),
        ("reference_model_ids", (1, 1)),
        ("reference_model_ids", (True,)),
        ("reference_model_ids", (1.0,)),
        ("reference_model_ids", (np.int64(1),)),
    ],
)
def test_candidate_loss_collection_invalid_initial_conditions(operation, invalid_value):
    inputs_before_call = dict(proposal_sample_index=100, reference_model_ids=(9, -2, 3))
    inputs_before_call[operation] = invalid_value
    with pytest.raises((TypeError, ValueError), match=operation):
        make_loss_collection(**inputs_before_call)


@pytest.mark.parametrize(
    "invalid_value", [None, {}, 2, SimpleNamespace(candidate_post_alarm_validation_sample_count=2)]
)
def test_candidate_loss_collection_invalid_initial_conditions_for_settings_type(invalid_value):
    with pytest.raises(TypeError, match="candidate_model_training_and_acceptance_settings"):
        PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=invalid_value,
            proposal_sample_index=0,
            reference_model_ids=(1,),
        )


@pytest.mark.parametrize("invalid_value", [True, False, 1, 0, 2.0, np.int64(2)])
def test_candidate_loss_collection_invalid_initial_conditions_for_modified_settings(invalid_value):
    candidate_model_training_and_acceptance_settings = CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
        candidate_post_alarm_validation_sample_count=2,
    )
    object.__setattr__(
        candidate_model_training_and_acceptance_settings,
        "candidate_post_alarm_validation_sample_count",
        invalid_value,
    )
    with pytest.raises(
        (TypeError, ValueError), match="candidate_post_alarm_validation_sample_count"
    ):
        PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
            proposal_sample_index=0,
            reference_model_ids=(1,),
        )


@pytest.mark.parametrize(
    "operation,invalid_value",
    [
        ("sample_index", True),
        ("sample_index", 101.0),
        ("sample_index", np.int64(101)),
        ("sample_index", -1),
        ("sample_index", 100),
        ("sample_index", 102),
        ("candidate_loss", True),
        ("candidate_loss", None),
        ("candidate_loss", ".2"),
        ("candidate_loss", np.float64(0.2)),
        ("candidate_loss", float("nan")),
        ("candidate_loss", float("inf")),
        ("candidate_loss", -0.1),
        ("candidate_loss", 1.1),
        ("candidate_loss", 10**300),
        ("reference_losses_by_model_id", None),
        ("reference_losses_by_model_id", []),
        ("reference_losses_by_model_id", {9: 0.5, -2: 0.6}),
        ("reference_losses_by_model_id", {9: 0.5, -2: 0.6, 3: 0.7, 4: 0.8}),
        ("reference_losses_by_model_id", {9: 0.5, -2: 0.6, 3: None}),
        ("reference_losses_by_model_id", {9: 0.5, -2: 0.6, 3: True}),
        ("reference_losses_by_model_id", {9: 0.5, -2: 0.6, 3: float("nan")}),
        ("reference_losses_by_model_id", {9: 0.5, -2: 0.6, 3: 1.1}),
        ("reference_losses_by_model_id", {9: 0.5, -2: 0.6, 3.0: 0.7}),
    ],
)
def test_candidate_loss_collection_invalid_observation_is_atomic(operation, invalid_value):
    collection = make_loss_collection()
    state_before_call = collection.get_state_snapshot()
    inputs_before_call = dict(
        sample_index=101, candidate_loss=0.2, reference_losses_by_model_id={9: 0.5, -2: 0.6, 3: 0.7}
    )
    inputs_before_call[operation] = invalid_value
    with pytest.raises((TypeError, ValueError), match=operation):
        collection.observe_losses_after_label_observation(**inputs_before_call)
    assert collection.get_state_snapshot() == state_before_call
    collection.observe_losses_after_label_observation(
        sample_index=101,
        candidate_loss=0.2,
        reference_losses_by_model_id={9: 0.5, -2: 0.6, 3: 0.7},
    )
    state_before_call = collection.get_state_snapshot()
    with pytest.raises(ValueError, match="sample_index"):
        collection.observe_losses_after_label_observation(
            sample_index=101,
            candidate_loss=0.2,
            reference_losses_by_model_id={9: 0.5, -2: 0.6, 3: 0.7},
        )
    assert collection.get_state_snapshot() == state_before_call


def test_candidate_loss_collection_snapshots_and_instances_are_independent():
    collection = make_loss_collection()
    reference_collection = make_loss_collection()
    reference_losses_by_model_id = {9: 0.5, -2: 0.6, 3: 0.7}
    collection.observe_losses_after_label_observation(
        sample_index=101,
        candidate_loss=0.2,
        reference_losses_by_model_id=reference_losses_by_model_id,
    )
    state_snapshot = collection.get_state_snapshot()
    reference_losses_by_model_id.clear()
    assert state_snapshot.reference_losses_by_model_id == ((9, (0.5,)), (-2, (0.6,)), (3, (0.7,)))
    for field_name in state_snapshot.__dataclass_fields__:
        with pytest.raises(FrozenInstanceError):
            setattr(state_snapshot, field_name, None)
    collection.observe_losses_after_label_observation(
        sample_index=102,
        candidate_loss=0.3,
        reference_losses_by_model_id={9: 0.4, -2: 0.5, 3: 0.6},
    )
    assert state_snapshot.validation_sample_count == 1
    assert state_snapshot.candidate_losses == (0.2,)
    assert reference_collection.validation_sample_count == 0
    assert reference_collection.get_state_snapshot().reference_losses_by_model_id == (
        (9, ()),
        (-2, ()),
        (3, ()),
    )
    assert collection.get_state_snapshot().validation_sample_count == 2


def test_candidate_loss_collection_preserves_global_numeric_state():
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    global_torch_random_state = torch.get_rng_state().clone()
    global_default_dtype = torch.get_default_dtype()
    global_default_device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            collection = make_loss_collection(2)
            collection.observe_losses_after_label_observation(
                sample_index=101,
                candidate_loss=0.123456789123,
                reference_losses_by_model_id={9: 0.5, -2: 0.6, 3: 0.7},
            )
            collection.observe_losses_after_label_observation(
                sample_index=102,
                candidate_loss=0.2,
                reference_losses_by_model_id={9: 0.5, -2: 0.6, 3: 0.7},
            )
            assert collection.get_state_snapshot().candidate_losses[0] == 0.123456789123
            assert torch.get_default_dtype() == torch.float64
            assert torch.empty(0).device.type == "meta"
    finally:
        torch.set_default_dtype(global_default_dtype)
    assert torch.get_default_dtype() == global_default_dtype
    assert torch.get_default_device() == global_default_device
    assert random.getstate() == global_python_random_state
    assert np.random.get_state()[0] == global_numpy_random_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], global_numpy_random_state[1])
    assert np.random.get_state()[2:] == global_numpy_random_state[2:]
    assert torch.equal(torch.get_rng_state(), global_torch_random_state)


def test_candidate_loss_collection_public_arguments_are_keyword_only():
    for operation, field_name in (
        (PostAlarmCandidateLossCollection, "candidate_model_training_and_acceptance_settings"),
        (PostAlarmCandidateLossCollection, "proposal_sample_index"),
        (PostAlarmCandidateLossCollection, "reference_model_ids"),
        (PostAlarmCandidateLossCollection.observe_losses_after_label_observation, "sample_index"),
        (PostAlarmCandidateLossCollection.observe_losses_after_label_observation, "candidate_loss"),
        (
            PostAlarmCandidateLossCollection.observe_losses_after_label_observation,
            "reference_losses_by_model_id",
        ),
    ):
        assert (
            inspect.signature(operation).parameters[field_name].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
    with pytest.raises(TypeError):
        make_loss_collection().observe_losses_after_label_observation(
            101, 0.2, {9: 0.5, -2: 0.6, 3: 0.7}
        )
