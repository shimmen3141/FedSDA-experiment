"""モデルを生成せず、三方式の初期値を旧helperと直接照合する。"""

import copy
from types import SimpleNamespace

import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.fedsda import FedSDAClient
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import (
    select_candidate_initial_parameter_snapshot,
)


def build_candidate_parameter_initialization_inputs():
    snapshots = {
        -7: {
            "backbone.float16": torch.tensor([1, 2], dtype=torch.float16, device="cpu"),
            "backbone.bfloat16": torch.tensor([1, 2], dtype=torch.bfloat16, device="cpu"),
            "backbone.float32": torch.tensor([1, 2], dtype=torch.float32, device="cpu"),
            "adapter.float64": torch.tensor(1, dtype=torch.float64, device="cpu"),
            "adapter.complex64": torch.tensor([1 + 2j], dtype=torch.complex64, device="cpu"),
            "adapter.complex128": torch.tensor([1 + 2j], dtype=torch.complex128, device="cpu"),
            "head.uint8": torch.tensor([1], dtype=torch.uint8, device="cpu"),
            "head.uint16": torch.tensor([1], dtype=torch.uint16, device="cpu"),
            "head.uint32": torch.tensor([1], dtype=torch.uint32, device="cpu"),
            "head.uint64": torch.tensor([1], dtype=torch.uint64, device="cpu"),
            "head.int8": torch.tensor([1], dtype=torch.int8, device="cpu"),
            "head.int16": torch.tensor([1], dtype=torch.int16, device="cpu"),
            "head.int32": torch.tensor([1], dtype=torch.int32, device="cpu"),
            "head.int64": torch.tensor([1], dtype=torch.int64, device="cpu"),
            "head.bool": torch.tensor([True], dtype=torch.bool, device="cpu"),
            "head.empty": torch.empty((0, 2), dtype=torch.float32, device="cpu"),
            "head.noncontiguous": torch.tensor([[1, 2], [3, 4]],
                dtype=torch.float32, device="cpu").t(),
        },
    }
    snapshots[4] = {
        parameter_name: torch.full_like(parameter_value, 0)
        for parameter_name, parameter_value in reversed(tuple(snapshots[-7].items()))
    }
    return snapshots


def run_legacy_candidate_parameter_initialization(
    *, source, snapshots, current_training_model_id, losses, monkeypatch,
):
    monkeypatch.setattr(config, "NEW_MODEL_INITIALIZATION", {
        "assigned_training_model": "current",
        "lowest_evaluated_mean_loss_model": "best_candidate",
        "equal_mean_of_available_models": "average",
    }[source])
    legacy_client = SimpleNamespace(
        models={model_id: SimpleNamespace(
            get_params=lambda legacy_parameters=parameter_snapshot: copy.deepcopy(legacy_parameters))
            for model_id, parameter_snapshot in snapshots.items()},
        current_model_id=current_training_model_id,
    )
    legacy_client._average_model_params = lambda: FedSDAClient._average_model_params(legacy_client)
    return FedSDAClient._select_initialization_params(legacy_client, losses)


def assert_candidate_initialization_parameters_match_legacy(*, actual, expected):
    if expected is None:
        assert actual is None
        return
    assert type(actual) is dict
    assert tuple(actual) == tuple(expected)
    for parameter_name in expected:
        assert type(actual[parameter_name]) is torch.Tensor
        assert actual[parameter_name].shape == expected[parameter_name].shape
        assert actual[parameter_name].dtype == expected[parameter_name].dtype
        assert actual[parameter_name].device == expected[parameter_name].device
        assert torch.equal(actual[parameter_name], expected[parameter_name])


@pytest.mark.parametrize("source", [
    "assigned_training_model", "lowest_evaluated_mean_loss_model", "equal_mean_of_available_models",
])
@pytest.mark.parametrize("losses", [(), ((-7, 0.8), (4, 0.1)), ((4, 0.2), (-7, 0.2))])
def test_candidate_parameter_initialization_matches_legacy(source, losses, monkeypatch):
    snapshots = build_candidate_parameter_initialization_inputs()
    initialization_settings = CandidateParameterInitializationSettings(
        candidate_parameter_initialization_source=source)
    expected = run_legacy_candidate_parameter_initialization(
        source=source, snapshots=snapshots, current_training_model_id=-7,
        losses=losses, monkeypatch=monkeypatch)
    actual = select_candidate_initial_parameter_snapshot(
        settings=initialization_settings, available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=-7, evaluated_mean_losses_by_model_id=losses)
    assert_candidate_initialization_parameters_match_legacy(actual=actual, expected=expected)


@pytest.mark.parametrize("losses", [((-7, 0.2), (4, 0.2)), ((4, 0.2), (-7, 0.2)), ((-7, 1),)])
def test_candidate_parameter_initialization_uses_evaluated_loss_order(losses, monkeypatch):
    snapshots = build_candidate_parameter_initialization_inputs()
    actual = select_candidate_initial_parameter_snapshot(
        settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="lowest_evaluated_mean_loss_model"),
        available_parameter_snapshots_by_model_id=snapshots, current_training_model_id=999,
        evaluated_mean_losses_by_model_id=losses)
    expected = run_legacy_candidate_parameter_initialization(
        source="lowest_evaluated_mean_loss_model", snapshots=snapshots,
        current_training_model_id=999, losses=losses, monkeypatch=monkeypatch)
    assert_candidate_initialization_parameters_match_legacy(actual=actual, expected=expected)
    assert_candidate_initialization_parameters_match_legacy(actual=actual, expected=snapshots[losses[0][0]])


@pytest.mark.parametrize("snapshots", [
    {},
    {4: {"backbone.weight": torch.tensor([1], dtype=torch.float32, device="cpu")}},
    {4: {"backbone.weight": torch.tensor([1], dtype=torch.float32, device="cpu")},
     -7: {"backbone.weight": torch.tensor([2**-24], dtype=torch.float32, device="cpu")},
     2: {"backbone.weight": torch.tensor([2**-24], dtype=torch.float32, device="cpu")}},
])
def test_candidate_parameter_initialization_matches_legacy_average_empty_single_and_rounding(snapshots, monkeypatch):
    actual = select_candidate_initial_parameter_snapshot(
        settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="equal_mean_of_available_models"),
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=999, evaluated_mean_losses_by_model_id=())
    expected = run_legacy_candidate_parameter_initialization(
        source="equal_mean_of_available_models", snapshots=snapshots,
        current_training_model_id=999, losses=(), monkeypatch=monkeypatch)
    assert_candidate_initialization_parameters_match_legacy(actual=actual, expected=expected)


@pytest.mark.parametrize("source", [
    "assigned_training_model", "lowest_evaluated_mean_loss_model", "equal_mean_of_available_models",
])
def test_candidate_parameter_initialization_preserves_independent_snapshots(source):
    snapshots = build_candidate_parameter_initialization_inputs()
    original = copy.deepcopy(snapshots)
    initialization_settings = CandidateParameterInitializationSettings(
        candidate_parameter_initialization_source=source)
    actual = select_candidate_initial_parameter_snapshot(
        settings=initialization_settings, available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=-7, evaluated_mean_losses_by_model_id=((-7, 0.1), (4, 0.9)))
    expected = select_candidate_initial_parameter_snapshot(
        settings=initialization_settings, available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=-7, evaluated_mean_losses_by_model_id=((-7, 0.1), (4, 0.9)))
    for parameter_name, parameter_value in actual.items():
        if parameter_value.numel():
            assert parameter_value.data_ptr() != expected[parameter_name].data_ptr()
            assert all(parameter_value.data_ptr() != parameter_snapshot[parameter_name].data_ptr()
                       for parameter_snapshot in snapshots.values())
    actual["backbone.float16"].fill_(9)
    assert not torch.equal(actual["backbone.float16"], expected["backbone.float16"])
    for model_id in snapshots:
        assert_candidate_initialization_parameters_match_legacy(
            actual=snapshots[model_id], expected=original[model_id])
