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
    ForwardValidationSession,
    select_forward_fitting_reference,
    has_disjoint_validation_advantage,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    evaluate_candidate_using_post_alarm_losses,
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


def make_legacy_loss_session(
    target_count=3, proposal_sample_index=100, reference_model_ids=(9, -2, 3)
):
    return ForwardValidationSession(
        proposal_position=proposal_sample_index,
        estimated_change_point=None,
        episode_id=None,
        old_model_id=9,
        detector="test",
        candidate=None,
        training_x=None,
        training_y=None,
        held_data=[],
        reference_models=dict.fromkeys(reference_model_ids),
        target_count=target_count,
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
        reference_losses_by_model_id = {3: 0.7, -2: 0.6, 9: 0.5}
        candidate_loss = 0 if observation_index % 2 else 1
        collection.observe_losses_after_label_observation(
            sample_index=proposal_sample_index + observation_index + 1,
            candidate_loss=candidate_loss,
            reference_losses_by_model_id=reference_losses_by_model_id,
        )
        legacy_session.append_losses(candidate_loss, reference_losses_by_model_id)
        state_snapshot = collection.get_state_snapshot()
        assert state_snapshot.candidate_losses == tuple(legacy_session.candidate_losses)
        assert all(
            type(candidate_loss) is float for candidate_loss in state_snapshot.candidate_losses
        )
        assert state_snapshot.reference_losses_by_model_id == tuple(
            (model_id, tuple(reference_losses))
            for model_id, reference_losses in legacy_session.reference_losses.items()
        )
        assert state_snapshot.proposal_sample_index == proposal_sample_index
        assert (
            state_snapshot.last_validation_sample_index
            == proposal_sample_index + observation_index + 1
        )
        assert (
            state_snapshot.validation_sample_count
            == collection.validation_sample_count
            == legacy_session.validation_count
        )
        assert state_snapshot.required_validation_sample_count == target_count
        assert (
            state_snapshot.ready_for_acceptance_evaluation
            == collection.ready_for_acceptance_evaluation
            == legacy_session.ready
        )
    state_before_call = collection.get_state_snapshot()
    with pytest.raises(RuntimeError, match="ready"):
        collection.observe_losses_after_label_observation(
            sample_index=proposal_sample_index + target_count + 1,
            candidate_loss=0.2,
            reference_losses_by_model_id={9: 0.5, -2: 0.6, 3: 0.7},
        )
    assert collection.get_state_snapshot() == state_before_call
    assert collection.get_state_snapshot() == state_before_call


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


def test_candidate_loss_collection_invalid_observation_is_atomic_for_legacy_partial_update():
    """LEGACY-004の不足参照入力を直接再現し、新境界では全系列を保持する。"""
    legacy_session = make_legacy_loss_session(reference_model_ids=(2, 1))
    with pytest.raises(KeyError):
        legacy_session.append_losses(0.2, {2: 0.4})
    assert legacy_session.candidate_losses == [0.2]
    assert legacy_session.reference_losses == {2: [0.4], 1: []}
    collection = make_loss_collection(reference_model_ids=(2, 1))
    state_before_call = collection.get_state_snapshot()
    with pytest.raises(ValueError, match="reference_losses_by_model_id"):
        collection.observe_losses_after_label_observation(
            sample_index=101,
            candidate_loss=0.2,
            reference_losses_by_model_id={2: 0.4},
        )
    assert collection.get_state_snapshot() == state_before_call


def make_legacy_observation_client(target_count=3):
    """旧clientはliveモデルでなく開始時snapshotを観測する。"""
    legacy_events = []
    legacy_session = make_legacy_loss_session(target_count)
    legacy_session.candidate = SimpleNamespace(
        per_sample_error=lambda x, y: (
            legacy_events.append(("candidate", int(x.item()))) or torch.tensor([0.2])
        ),
    )
    legacy_session.reference_models = {
        model_id: SimpleNamespace(
            per_sample_error=lambda x, y, model_id=model_id: (
                legacy_events.append((model_id, int(x.item())))
                or torch.tensor([{9: 0.5, -2: 0.6, 3: 0.7}[model_id]])
            )
        )
        for model_id in (9, -2, 3)
    }
    legacy_client = SimpleNamespace(
        _forward_validation=legacy_session,
        models={},
        _record_model_compute=lambda *x: None,
    )
    legacy_client._finalize_forward_validation = lambda sample_index: (
        legacy_events.append(("finalize", sample_index, legacy_session.validation_count))
        or setattr(legacy_client, "_forward_validation", None)
        or 2
    )
    return legacy_client, legacy_session, legacy_events


@pytest.mark.parametrize("target_count", [2, 3, 5])
def test_candidate_loss_collection_readiness_matches_legacy_client(monkeypatch, target_count):
    """同じ到達回にfinalizeし、消えたlive参照も固定順で観測される。"""
    monkeypatch.setattr(config, "NEW_MODEL_CREATION_POLICY", "forward_persistent")
    collection = make_loss_collection(target_count)
    legacy_client, legacy_session, legacy_events = make_legacy_observation_client(target_count)
    for observation_index in range(1, target_count + 1):
        sample_index = 100 + observation_index
        result = FedSDAClient._observe_forward_validation(
            legacy_client,
            torch.tensor([[float(sample_index)]]),
            torch.tensor([[0.0]]),
            sample_index,
        )
        collection.observe_losses_after_label_observation(
            sample_index=sample_index,
            candidate_loss=legacy_session.candidate_losses[-1],
            reference_losses_by_model_id={
                model_id: reference_losses[-1]
                for model_id, reference_losses in legacy_session.reference_losses.items()
            },
        )
        assert result == (2 if collection.ready_for_acceptance_evaluation else 0)
        state_snapshot = collection.get_state_snapshot()
        assert state_snapshot.candidate_losses == tuple(legacy_session.candidate_losses)
        assert state_snapshot.reference_losses_by_model_id == tuple(
            (model_id, tuple(reference_losses))
            for model_id, reference_losses in legacy_session.reference_losses.items()
        )
        assert legacy_events[(observation_index - 1) * 4 : observation_index * 4] == [
            ("candidate", sample_index),
            (9, sample_index),
            (-2, sample_index),
            (3, sample_index),
        ]
    assert legacy_events[-1] == ("finalize", 100 + target_count, target_count)
    state_before_call = deepcopy(legacy_events)
    assert (
        FedSDAClient._observe_forward_validation(
            legacy_client, torch.tensor([[999.0]]), torch.tensor([[0.0]]), 999
        )
        == 0
    )
    assert legacy_events == state_before_call


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


@pytest.mark.parametrize("candidate_loss", [0.1, 0.8])
@pytest.mark.parametrize("reference_historical_mean_losses_by_model_id", [{}, {9: 0.5}, {-2: 0.6}])
@pytest.mark.parametrize("available_reference_model_ids", [(9, -2, 3), ()])
def test_candidate_loss_collection_connects_to_acceptance_evaluation(
    candidate_loss, reference_historical_mean_losses_by_model_id, available_reference_model_ids
):
    """snapshot系列を明示入力し、旧適合選択と二分区間判定へ比較する。"""
    collection = make_loss_collection(3)
    legacy_session = make_legacy_loss_session(3)
    for sample_index in (101, 102, 103):
        reference_losses_by_model_id = {3: 0.7, -2: 0.6, 9: 0.5}
        collection.observe_losses_after_label_observation(
            sample_index=sample_index,
            candidate_loss=candidate_loss,
            reference_losses_by_model_id=reference_losses_by_model_id,
        )
        legacy_session.append_losses(candidate_loss, reference_losses_by_model_id)
    state_snapshot = collection.get_state_snapshot()
    result = evaluate_candidate_using_post_alarm_losses(
        candidate_model_training_and_acceptance_settings=CandidateModelTrainingAndAcceptanceSettings(
            candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
            candidate_post_alarm_validation_sample_count=3,
        ),
        candidate_losses=state_snapshot.candidate_losses,
        reference_losses_by_model_id=dict(state_snapshot.reference_losses_by_model_id),
        reference_historical_mean_losses_by_model_id=reference_historical_mean_losses_by_model_id,
        available_reference_model_ids=available_reference_model_ids,
        current_training_model_id=9,
        maximum_reference_mean_loss_increase=0.1,
        minimum_candidate_mean_loss_improvement=0.0001,
    )
    model_id = select_forward_fitting_reference(
        {
            model_id: reference_losses
            for model_id, reference_losses in legacy_session.reference_losses.items()
            if model_id in available_reference_model_ids
        },
        reference_historical_mean_losses_by_model_id,
        0.1,
        preferred_model_id=9,
    )
    assert result.reusable_reference_model_id == model_id
    if model_id is None:
        model_id = min(
            legacy_session.reference_losses,
            key=lambda model_id: sum(legacy_session.reference_losses[model_id]),
        )
        assert result.candidate_accepted == has_disjoint_validation_advantage(
            torch.tensor(legacy_session.candidate_losses, dtype=torch.float32),
            torch.tensor(legacy_session.reference_losses[model_id], dtype=torch.float32),
            0.0001,
        )
    else:
        assert not result.candidate_accepted
    assert result.comparison_reference_model_id == model_id
    assert result.validation_sample_count == 3
    assert collection.get_state_snapshot() == state_snapshot


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
