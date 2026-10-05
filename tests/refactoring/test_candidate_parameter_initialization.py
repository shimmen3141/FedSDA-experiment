"""モデルを生成せず、三方式の初期値を旧helperと直接照合する。"""

import copy
from dataclasses import FrozenInstanceError, MISSING, fields
import inspect
import random
from types import SimpleNamespace

import numpy as np
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
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    evaluate_candidate_using_post_alarm_losses,
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
            "head.noncontiguous": torch.tensor(
                [[1, 2], [3, 4]], dtype=torch.float32, device="cpu"
            ).t(),
        },
    }
    snapshots[4] = {
        parameter_name: torch.full_like(parameter_value, 0)
        for parameter_name, parameter_value in reversed(tuple(snapshots[-7].items()))
    }
    return snapshots


def run_legacy_candidate_parameter_initialization(
    *,
    source,
    snapshots,
    current_training_model_id,
    losses,
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "NEW_MODEL_INITIALIZATION",
        {
            "assigned_training_model": "current",
            "lowest_evaluated_mean_loss_model": "best_candidate",
            "equal_mean_of_available_models": "average",
        }[source],
    )
    legacy_client = SimpleNamespace(
        models={
            model_id: SimpleNamespace(
                get_params=lambda legacy_parameters=parameter_snapshot: copy.deepcopy(
                    legacy_parameters
                )
            )
            for model_id, parameter_snapshot in snapshots.items()
        },
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


@pytest.mark.parametrize(
    "source",
    [
        "assigned_training_model",
        "lowest_evaluated_mean_loss_model",
        "equal_mean_of_available_models",
    ],
)
@pytest.mark.parametrize("losses", [(), ((-7, 0.8), (4, 0.1)), ((4, 0.2), (-7, 0.2))])
def test_candidate_parameter_initialization_matches_legacy(source, losses, monkeypatch):
    snapshots = build_candidate_parameter_initialization_inputs()
    initialization_settings = CandidateParameterInitializationSettings(
        candidate_parameter_initialization_source=source
    )
    expected = run_legacy_candidate_parameter_initialization(
        source=source,
        snapshots=snapshots,
        current_training_model_id=-7,
        losses=losses,
        monkeypatch=monkeypatch,
    )
    actual = select_candidate_initial_parameter_snapshot(
        settings=initialization_settings,
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=-7,
        evaluated_mean_losses_by_model_id=losses,
    )
    assert_candidate_initialization_parameters_match_legacy(actual=actual, expected=expected)


@pytest.mark.parametrize("losses", [((-7, 0.2), (4, 0.2)), ((4, 0.2), (-7, 0.2)), ((-7, 1),)])
def test_candidate_parameter_initialization_uses_evaluated_loss_order(losses, monkeypatch):
    snapshots = build_candidate_parameter_initialization_inputs()
    actual = select_candidate_initial_parameter_snapshot(
        settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="lowest_evaluated_mean_loss_model"
        ),
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=999,
        evaluated_mean_losses_by_model_id=losses,
    )
    expected = run_legacy_candidate_parameter_initialization(
        source="lowest_evaluated_mean_loss_model",
        snapshots=snapshots,
        current_training_model_id=999,
        losses=losses,
        monkeypatch=monkeypatch,
    )
    assert_candidate_initialization_parameters_match_legacy(actual=actual, expected=expected)
    assert_candidate_initialization_parameters_match_legacy(
        actual=actual, expected=snapshots[losses[0][0]]
    )


@pytest.mark.parametrize(
    "snapshots",
    [
        {},
        {4: {"backbone.weight": torch.tensor([1], dtype=torch.float32, device="cpu")}},
        {
            4: {"backbone.weight": torch.tensor([1], dtype=torch.float32, device="cpu")},
            -7: {"backbone.weight": torch.tensor([2**-24], dtype=torch.float32, device="cpu")},
            2: {"backbone.weight": torch.tensor([2**-24], dtype=torch.float32, device="cpu")},
        },
    ],
)
def test_candidate_parameter_initialization_matches_legacy_average_empty_single_and_rounding(
    snapshots, monkeypatch
):
    actual = select_candidate_initial_parameter_snapshot(
        settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="equal_mean_of_available_models"
        ),
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=999,
        evaluated_mean_losses_by_model_id=(),
    )
    expected = run_legacy_candidate_parameter_initialization(
        source="equal_mean_of_available_models",
        snapshots=snapshots,
        current_training_model_id=999,
        losses=(),
        monkeypatch=monkeypatch,
    )
    assert_candidate_initialization_parameters_match_legacy(actual=actual, expected=expected)


@pytest.mark.parametrize(
    "source",
    [
        "assigned_training_model",
        "lowest_evaluated_mean_loss_model",
        "equal_mean_of_available_models",
    ],
)
def test_candidate_parameter_initialization_preserves_independent_snapshots(source):
    snapshots = build_candidate_parameter_initialization_inputs()
    original = copy.deepcopy(snapshots)
    initialization_settings = CandidateParameterInitializationSettings(
        candidate_parameter_initialization_source=source
    )
    actual = select_candidate_initial_parameter_snapshot(
        settings=initialization_settings,
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=-7,
        evaluated_mean_losses_by_model_id=((-7, 0.1), (4, 0.9)),
    )
    expected = select_candidate_initial_parameter_snapshot(
        settings=initialization_settings,
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=-7,
        evaluated_mean_losses_by_model_id=((-7, 0.1), (4, 0.9)),
    )
    for parameter_name, parameter_value in actual.items():
        if parameter_value.numel():
            assert parameter_value.data_ptr() != expected[parameter_name].data_ptr()
            assert all(
                parameter_value.data_ptr() != parameter_snapshot[parameter_name].data_ptr()
                for parameter_snapshot in snapshots.values()
            )
    actual["backbone.float16"].fill_(9)
    assert not torch.equal(actual["backbone.float16"], expected["backbone.float16"])
    for model_id in snapshots:
        assert_candidate_initialization_parameters_match_legacy(
            actual=snapshots[model_id], expected=original[model_id]
        )
    snapshots[-7]["backbone.float16"].fill_(7)
    assert_candidate_initialization_parameters_match_legacy(
        actual=expected,
        expected=select_candidate_initial_parameter_snapshot(
            settings=initialization_settings,
            available_parameter_snapshots_by_model_id=original,
            current_training_model_id=-7,
            evaluated_mean_losses_by_model_id=((-7, 0.1), (4, 0.9)),
        ),
    )


@pytest.mark.parametrize(
    "case,invalid_value",
    [
        ("settings", None),
        ("forged_settings", "current"),
        ("outer", []),
        ("model_id", True),
        ("snapshot", {}),
        ("snapshot", []),
        ("parameter_name", ""),
        ("parameter_name", 3),
        ("parameter_value", 3),
        ("parameter_value", torch.tensor([float("nan"), 0], device="cpu")),
        ("parameter_value", torch.tensor([float("inf"), 0], device="cpu")),
        ("parameter_value", torch.empty(2, device="meta")),
        ("parameter_value", torch.nn.Parameter(torch.zeros(2, device="cpu"))),
        (
            "parameter_value",
            torch.quantize_per_tensor(
                torch.zeros(2, device="cpu"), scale=0.1, zero_point=0, dtype=torch.qint8
            ),
        ),
        ("shape", torch.zeros(3, device="cpu")),
        ("dtype", torch.zeros(2, dtype=torch.float64, device="cpu")),
        ("missing_key", None),
        ("current_id", True),
        ("missing_current", 999),
        ("losses", []),
        ("losses", ((-7, 0.1), [4, 0.2])),
        ("losses", ((-7, 0.1), (4,))),
        ("losses", ((-7, 0.1), (True, 0.2))),
        ("losses", ((-7, 0.1), (-7, 0.2))),
        ("losses", ((-7, 0.1), (999, 0.2))),
        ("losses", ((-7, 0.1), (4, True))),
        ("losses", ((-7, 0.1), (4, float("nan")))),
        ("losses", ((-7, 0.1), (4, float("inf")))),
        ("losses", ((-7, 0.1), (4, 10**400))),
        ("losses", ((-7, 0.1), (4, -0.1))),
        ("losses", ((-7, 0.1), (4, 1.1))),
    ],
)
def test_candidate_parameter_initialization_rejects_invalid_inputs_without_mutation(
    case, invalid_value
):
    """選ばない末尾モデル・評価も検査し、拒否時に入力を変更しない。"""
    snapshots = build_candidate_parameter_initialization_inputs()
    initialization_settings = CandidateParameterInitializationSettings(
        candidate_parameter_initialization_source="assigned_training_model"
    )
    initialization_arguments = dict(
        settings=initialization_settings,
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=-7,
        evaluated_mean_losses_by_model_id=((-7, 0.1), (4, 0.2)),
    )
    if case == "settings":
        initialization_arguments["settings"] = invalid_value
    elif case == "forged_settings":
        object.__setattr__(
            initialization_settings, "candidate_parameter_initialization_source", invalid_value
        )
    elif case == "outer":
        initialization_arguments["available_parameter_snapshots_by_model_id"] = invalid_value
    elif case == "model_id":
        snapshots[invalid_value] = snapshots.pop(4)
    elif case == "snapshot":
        snapshots[4] = invalid_value
    elif case == "parameter_name":
        snapshots[4][invalid_value] = snapshots[4].pop("backbone.float32")
    elif case in ("parameter_value", "shape", "dtype"):
        snapshots[4]["backbone.float32"] = invalid_value
    elif case == "missing_key":
        snapshots[4].pop("backbone.float32")
    elif case in ("current_id", "missing_current"):
        initialization_arguments["current_training_model_id"] = invalid_value
    elif case == "losses":
        initialization_arguments["evaluated_mean_losses_by_model_id"] = invalid_value
    original = copy.deepcopy(snapshots)
    expected = copy.deepcopy(initialization_arguments)
    with pytest.raises((TypeError, ValueError)) as validation_error:
        select_candidate_initial_parameter_snapshot(**initialization_arguments)
    assert str(validation_error.value)
    assert initialization_arguments["settings"] == expected["settings"]
    assert (
        initialization_arguments["current_training_model_id"]
        == expected["current_training_model_id"]
    )
    assert (
        initialization_arguments["evaluated_mean_losses_by_model_id"]
        == expected["evaluated_mean_losses_by_model_id"]
    )
    if case == "outer":
        assert (
            initialization_arguments["available_parameter_snapshots_by_model_id"] == invalid_value
        )
    assert tuple(snapshots) == tuple(original)
    for model_id, parameter_snapshot in snapshots.items():
        if type(parameter_snapshot) is not dict:
            assert parameter_snapshot == original[model_id]
            continue
        assert tuple(parameter_snapshot) == tuple(original[model_id])
        for parameter_name, parameter_value in parameter_snapshot.items():
            expected = original[model_id][parameter_name]
            if isinstance(parameter_value, torch.Tensor):
                assert (
                    parameter_value.shape == expected.shape
                    and parameter_value.dtype == expected.dtype
                )
                assert parameter_value.device == expected.device
                if parameter_value.device.type != "meta":
                    if parameter_value.is_floating_point() or parameter_value.is_complex():
                        torch.testing.assert_close(
                            parameter_value, expected, rtol=0, atol=0, equal_nan=True
                        )
                    else:
                        assert torch.equal(parameter_value, expected)
            else:
                assert parameter_value == expected
    if case == "forged_settings":
        assert initialization_settings.candidate_parameter_initialization_source == invalid_value


def test_candidate_parameter_initialization_requires_explicit_keyword_settings():
    """設定に暗黙defaultを作らず、公開関数はkeyword指定だけを受ける。"""
    assert fields(CandidateParameterInitializationSettings)[0].default is MISSING
    assert fields(CandidateParameterInitializationSettings)[0].default_factory is MISSING
    assert str(inspect.signature(select_candidate_initial_parameter_snapshot)).startswith("(*,")
    with pytest.raises(TypeError):
        CandidateParameterInitializationSettings()
    with pytest.raises(TypeError):
        CandidateParameterInitializationSettings("assigned_training_model")
    initialization_settings = CandidateParameterInitializationSettings(
        candidate_parameter_initialization_source="assigned_training_model"
    )
    with pytest.raises(FrozenInstanceError):
        initialization_settings.candidate_parameter_initialization_source = (
            "equal_mean_of_available_models"
        )
    with pytest.raises(TypeError):
        select_candidate_initial_parameter_snapshot(initialization_settings, {}, -7, ())
    with pytest.raises(TypeError):
        select_candidate_initial_parameter_snapshot(settings=initialization_settings)


@pytest.mark.parametrize(
    "source",
    [
        "assigned_training_model",
        "lowest_evaluated_mean_loss_model",
        "equal_mean_of_available_models",
    ],
)
def test_candidate_parameter_initialization_preserves_independent_snapshots_with_aliased_keys(
    source,
):
    """入力の二つのkeyが同じstorageでも、返却はkeyごと・呼出ごとに独立する。"""
    parameter_value = torch.tensor([1.0, 2.0], device="cpu")
    snapshots = {-7: {"backbone.weight": parameter_value, "head.weight": parameter_value}}
    initialization_arguments = dict(
        settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source=source
        ),
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=-7,
        evaluated_mean_losses_by_model_id=((-7, 0.2),),
    )
    actual = select_candidate_initial_parameter_snapshot(**initialization_arguments)
    repeated_actual = select_candidate_initial_parameter_snapshot(**initialization_arguments)
    actual["head.weight"].fill_(9)
    assert torch.equal(actual["backbone.weight"], parameter_value)
    assert torch.equal(repeated_actual["head.weight"], parameter_value)
    parameter_value.fill_(3)
    assert actual["backbone.weight"].tolist() == [1.0, 2.0]
    assert repeated_actual["backbone.weight"].tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    "source",
    [
        "assigned_training_model",
        "lowest_evaluated_mean_loss_model",
        "equal_mean_of_available_models",
    ],
)
def test_candidate_parameter_initialization_preserves_gradients_and_shared_state(source):
    """CPU入力だけを使い、既定meta環境でもコピー・拒否・空結果が共有状態を保つ。"""
    snapshots = build_candidate_parameter_initialization_inputs()
    snapshots[-7]["backbone.float32"].requires_grad_(True)
    snapshots[-7]["adapter.complex64"].requires_grad_(True)
    snapshots[-7]["adapter.float64"] = snapshots[-7]["adapter.float64"].requires_grad_(True) * 2
    original = {
        parameter_name: parameter_value.detach().clone()
        for parameter_name, parameter_value in snapshots[-7].items()
    }
    initialization_arguments = dict(
        settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source=source
        ),
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=-7,
        evaluated_mean_losses_by_model_id=((-7, 0.1), (4, 0.2)),
    )
    initialization_settings = CandidateParameterInitializationSettings(
        candidate_parameter_initialization_source="equal_mean_of_available_models"
    )
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    global_torch_random_state = torch.random.get_rng_state().clone()
    global_default_dtype = torch.get_default_dtype()
    global_default_device = torch.get_default_device()
    global_grad_enabled = torch.is_grad_enabled()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"), torch.enable_grad():
            actual = select_candidate_initial_parameter_snapshot(**initialization_arguments)
            assert torch.is_grad_enabled()
            assert torch.get_default_device() == torch.device("meta")
            assert torch.get_default_dtype() == torch.float64
            assert all(
                parameter_value.device.type == "cpu"
                and not parameter_value.requires_grad
                and parameter_value.grad_fn is None
                for parameter_value in actual.values()
            )
            with pytest.raises(ValueError):
                select_candidate_initial_parameter_snapshot(
                    **{
                        **initialization_arguments,
                        "evaluated_mean_losses_by_model_id": ((999, 0.1),),
                    }
                )
            assert (
                select_candidate_initial_parameter_snapshot(
                    settings=initialization_settings,
                    available_parameter_snapshots_by_model_id={},
                    current_training_model_id=999,
                    evaluated_mean_losses_by_model_id=(),
                )
                is None
            )
            assert torch.is_grad_enabled() and torch.get_default_device() == torch.device("meta")
            with torch.no_grad():
                repeated_actual = select_candidate_initial_parameter_snapshot(
                    **initialization_arguments
                )
                assert not torch.is_grad_enabled()
                assert_candidate_initialization_parameters_match_legacy(
                    actual=repeated_actual, expected=actual
                )
    finally:
        torch.set_default_dtype(global_default_dtype)
    assert torch.get_default_dtype() == global_default_dtype
    assert torch.get_default_device() == global_default_device
    assert torch.is_grad_enabled() == global_grad_enabled
    assert random.getstate() == global_python_random_state
    assert np.random.get_state()[0] == global_numpy_random_state[0]
    assert np.array_equal(np.random.get_state()[1], global_numpy_random_state[1])
    assert np.random.get_state()[2:] == global_numpy_random_state[2:]
    assert torch.equal(torch.random.get_rng_state(), global_torch_random_state)
    assert_candidate_initialization_parameters_match_legacy(actual=snapshots[-7], expected=original)
    assert snapshots[-7]["backbone.float32"].requires_grad
    assert snapshots[-7]["backbone.float32"].grad is None
    assert snapshots[-7]["adapter.complex64"].requires_grad
    assert snapshots[-7]["adapter.complex64"].grad is None
    assert snapshots[-7]["adapter.float64"].requires_grad
    assert snapshots[-7]["adapter.float64"].grad_fn is not None


def test_candidate_parameter_initialization_connects_post_alarm_evaluated_losses(monkeypatch):
    """再利用も候補採用も失敗した参照を、公開評価の平均だけで初期値として選ぶ。"""
    snapshots = build_candidate_parameter_initialization_inputs()
    original = copy.deepcopy(snapshots)
    reference_losses_by_model_id = {-7: (0.6, 0.6), 4: (0.2, 0.2)}
    candidate_losses = (0.9, 0.9)
    candidate_model_training_and_acceptance_settings = CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
        candidate_post_alarm_validation_sample_count=2,
    )
    evaluation_results_by_model_id = {}
    for model_id, losses in reference_losses_by_model_id.items():
        result = evaluate_candidate_using_post_alarm_losses(
            candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
            candidate_losses=candidate_losses,
            reference_losses_by_model_id={model_id: losses},
            reference_historical_mean_losses_by_model_id={model_id: 0.1},
            available_reference_model_ids=(model_id,),
            current_training_model_id=-7,
            maximum_reference_mean_loss_increase=0.0,
            minimum_candidate_mean_loss_improvement=0.0,
        )
        assert result.reusable_reference_model_id is None and not result.candidate_accepted
        evaluation_results_by_model_id[model_id] = result
    losses = tuple(
        (model_id, result.reference_full_interval_mean_loss)
        for model_id, result in evaluation_results_by_model_id.items()
    )
    actual = select_candidate_initial_parameter_snapshot(
        settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="lowest_evaluated_mean_loss_model"
        ),
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=-7,
        evaluated_mean_losses_by_model_id=losses,
    )
    expected = run_legacy_candidate_parameter_initialization(
        source="lowest_evaluated_mean_loss_model",
        snapshots=snapshots,
        current_training_model_id=-7,
        losses=losses,
        monkeypatch=monkeypatch,
    )
    assert_candidate_initialization_parameters_match_legacy(actual=actual, expected=expected)
    assert_candidate_initialization_parameters_match_legacy(actual=actual, expected=snapshots[4])
    for model_id in snapshots:
        assert_candidate_initialization_parameters_match_legacy(
            actual=snapshots[model_id], expected=original[model_id]
        )
    assert reference_losses_by_model_id == {-7: (0.6, 0.6), 4: (0.2, 0.2)}
    assert candidate_losses == (0.9, 0.9)


def test_candidate_parameter_initialization_rejects_nonfinite_average_results(monkeypatch):
    """有限float32入力の旧平均overflowを実測し、新APIは補正せず拒否する。"""
    snapshots = {
        model_id: {
            "backbone.weight": torch.tensor(
                [torch.finfo(torch.float32).max], dtype=torch.float32, device="cpu"
            )
        }
        for model_id in (-7, 4)
    }
    original = copy.deepcopy(snapshots)
    expected = run_legacy_candidate_parameter_initialization(
        source="equal_mean_of_available_models",
        snapshots=snapshots,
        current_training_model_id=-7,
        losses=(),
        monkeypatch=monkeypatch,
    )
    assert torch.isinf(expected["backbone.weight"]).all().item()
    with pytest.raises(ValueError, match="平均.*非有限"):
        select_candidate_initial_parameter_snapshot(
            settings=CandidateParameterInitializationSettings(
                candidate_parameter_initialization_source="equal_mean_of_available_models"
            ),
            available_parameter_snapshots_by_model_id=snapshots,
            current_training_model_id=-7,
            evaluated_mean_losses_by_model_id=(),
        )
    for model_id in snapshots:
        assert torch.isfinite(snapshots[model_id]["backbone.weight"]).all().item()
        assert_candidate_initialization_parameters_match_legacy(
            actual=snapshots[model_id], expected=original[model_id]
        )
