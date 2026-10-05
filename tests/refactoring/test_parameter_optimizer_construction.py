"""旧optimizer builderと生成条件・外部同勾配更新を直接照合する。"""

import random
from dataclasses import MISSING, FrozenInstanceError, fields
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.models import SharedBackboneMLP
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training import parameter_optimizer_construction
from federated_learning_experiments.learning.training.parameter_optimizer_construction import (
    create_parameter_optimizer,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)


@pytest.mark.parametrize(
    "optimizer_settings",
    [
        AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0, adam_variant="standard"),
        AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.2, adam_variant="standard"
        ),
        AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0, adam_variant="amsgrad"),
        AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.2, adam_variant="amsgrad"
        ),
        AdamParameterOptimizerSettings(learning_rate=0, weight_decay=0.2, adam_variant="amsgrad"),
        SgdParameterOptimizerSettings(learning_rate=0.05),
        SgdParameterOptimizerSettings(learning_rate=0),
    ],
)
def test_parameter_optimizer_matches_legacy(optimizer_settings, monkeypatch):
    input_parameters = (
        torch.nn.Parameter(
            torch.tensor([[0.25, -0.75], [1.5, 0.0]], dtype=torch.float32, device="cpu"),
            requires_grad=False,
        ),
        torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device="cpu")),
    )
    input_parameters[0].grad = torch.full_like(input_parameters[0], 0.3)
    expected = tuple(
        torch.nn.Parameter(parameter.detach().clone(), requires_grad=parameter.requires_grad)
        for parameter in input_parameters
    )
    parameter_snapshot_before_call = tuple(
        parameter.detach().clone() for parameter in input_parameters
    )
    gradient_snapshot_before_call = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in input_parameters
    )
    original = tuple(parameter.grad for parameter in input_parameters)
    if type(optimizer_settings) is AdamParameterOptimizerSettings:
        monkeypatch.setattr(config, "OPTIMIZER", "adam")
        monkeypatch.setattr(config, "WEIGHT_DECAY", optimizer_settings.weight_decay)
        monkeypatch.setattr(config, "AMSGRAD", optimizer_settings.adam_variant == "amsgrad")
    else:
        monkeypatch.setattr(config, "OPTIMIZER", "sgd")
        # SGDは旧Adam専用条件が有効でもそれらを使わない。
        monkeypatch.setattr(config, "WEIGHT_DECAY", 0.9)
        monkeypatch.setattr(config, "AMSGRAD", True)
    legacy_optimizer = SharedBackboneMLP._build_component_optimizer(
        expected, optimizer_settings.learning_rate
    )
    optimizer = create_parameter_optimizer(
        parameters=input_parameters, optimizer_settings=optimizer_settings
    )
    assert type(optimizer) is type(legacy_optimizer)
    assert optimizer.state == legacy_optimizer.state == {}
    torch.testing.assert_close(optimizer.defaults, legacy_optimizer.defaults, rtol=0, atol=0)
    torch.testing.assert_close(
        optimizer.state_dict(), legacy_optimizer.state_dict(), rtol=0, atol=0
    )
    assert all(
        parameter is input_parameters[parameter_index]
        for parameter_index, parameter in enumerate(optimizer.param_groups[0]["params"])
    )
    assert all(
        parameter is expected[parameter_index]
        for parameter_index, parameter in enumerate(legacy_optimizer.param_groups[0]["params"])
    )
    for parameter_index, parameter in enumerate(input_parameters):
        torch.testing.assert_close(
            parameter, parameter_snapshot_before_call[parameter_index], rtol=0, atol=0
        )
        assert parameter.grad is original[parameter_index]
        torch.testing.assert_close(
            parameter.grad, gradient_snapshot_before_call[parameter_index], rtol=0, atol=0
        )
    for actual in (0.25, None, -0.5):
        for parameter_index, parameter in enumerate(input_parameters):
            parameter.grad = (
                None
                if parameter_index == 1 and actual is None
                else torch.full_like(parameter, 0.125 if actual is None else actual)
            )
        for parameter_index, parameter in enumerate(expected):
            parameter.grad = (
                None
                if input_parameters[parameter_index].grad is None
                else input_parameters[parameter_index].grad.detach().clone()
            )
        optimizer.step()
        legacy_optimizer.step()
        for parameter_index, parameter in enumerate(input_parameters):
            torch.testing.assert_close(parameter, expected[parameter_index], rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state_dict(), legacy_optimizer.state_dict(), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "configuration_parameter_name,specified_parameter_value",
    [
        ("learning_rate", True),
        ("learning_rate", np.float64(0.1)),
        ("learning_rate", -0.1),
        ("learning_rate", float("inf")),
        ("weight_decay", False),
        ("weight_decay", float("nan")),
        ("weight_decay", -1),
        ("adam_variant", "unknown"),
        ("adam_variant", True),
    ],
)
def test_parameter_optimizer_rejects_invalid_inputs_without_mutation_settings(
    configuration_parameter_name,
    specified_parameter_value,
    monkeypatch,
):
    input_parameters = (torch.nn.Parameter(torch.tensor([1.0, 2.0])),)
    input_parameters[0].grad = torch.tensor([0.3, -0.2])
    parameter_snapshot_before_call = input_parameters[0].detach().clone()
    gradient_snapshot_before_call = input_parameters[0].grad.clone()
    original = input_parameters[0].grad
    optimizer_settings = AdamParameterOptimizerSettings(
        learning_rate=0.01, weight_decay=0.1, adam_variant="amsgrad"
    )
    object.__setattr__(optimizer_settings, configuration_parameter_name, specified_parameter_value)
    expected = dict(vars(optimizer_settings))
    global_torch_random_state = torch.get_rng_state()
    monkeypatch.setattr(
        parameter_optimizer_construction,
        "Adam",
        Mock(side_effect=AssertionError("生成前検査が必要")),
    )
    with pytest.raises(ValueError, match=configuration_parameter_name):
        create_parameter_optimizer(
            parameters=input_parameters, optimizer_settings=optimizer_settings
        )
    parameter_optimizer_construction.Adam.assert_not_called()
    assert vars(optimizer_settings) == expected
    assert input_parameters[0].grad is original
    assert torch.equal(input_parameters[0], parameter_snapshot_before_call)
    assert torch.equal(input_parameters[0].grad, gradient_snapshot_before_call)
    assert torch.equal(torch.get_rng_state(), global_torch_random_state)
    # 同じ不正値が通常の設定構築でも拒否されることを確認する。
    with pytest.raises(ValueError, match=configuration_parameter_name):
        AdamParameterOptimizerSettings(**expected)


@pytest.mark.parametrize("specified_parameter_value", [True, -1, float("nan"), np.float32(0.1)])
def test_parameter_optimizer_rejects_invalid_inputs_without_mutation_sgd_settings(
    specified_parameter_value, monkeypatch
):
    input_parameters = (torch.nn.Parameter(torch.ones(1)),)
    optimizer_settings = SgdParameterOptimizerSettings(learning_rate=0.1)
    object.__setattr__(optimizer_settings, "learning_rate", specified_parameter_value)
    monkeypatch.setattr(
        parameter_optimizer_construction,
        "SGD",
        Mock(side_effect=AssertionError("生成前検査が必要")),
    )
    with pytest.raises(ValueError, match="learning_rate"):
        create_parameter_optimizer(
            parameters=input_parameters, optimizer_settings=optimizer_settings
        )
    parameter_optimizer_construction.SGD.assert_not_called()
    with pytest.raises(ValueError, match="learning_rate"):
        SgdParameterOptimizerSettings(learning_rate=specified_parameter_value)


@pytest.mark.parametrize("specified_parameter_value", [None, {}, "adam"])
def test_parameter_optimizer_rejects_invalid_inputs_without_mutation_settings_type(
    specified_parameter_value, monkeypatch
):
    input_parameters = (torch.nn.Parameter(torch.ones(1)),)
    monkeypatch.setattr(
        parameter_optimizer_construction,
        "Adam",
        Mock(side_effect=AssertionError("生成前検査が必要")),
    )
    monkeypatch.setattr(
        parameter_optimizer_construction,
        "SGD",
        Mock(side_effect=AssertionError("生成前検査が必要")),
    )
    with pytest.raises(ValueError, match="optimizer_settings"):
        create_parameter_optimizer(
            parameters=input_parameters, optimizer_settings=specified_parameter_value
        )
    parameter_optimizer_construction.Adam.assert_not_called()
    parameter_optimizer_construction.SGD.assert_not_called()


@pytest.mark.parametrize(
    "specified_parameter_value",
    ["list", "empty", "tensor", "dtype", "device", "layout", "duplicate"],
)
def test_parameter_optimizer_rejects_invalid_inputs_without_mutation_parameters(
    specified_parameter_value, monkeypatch
):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    parameter.grad = torch.tensor([0.25, -0.5])
    original = parameter.grad
    parameter_snapshot_before_call = parameter.detach().clone()
    gradient_snapshot_before_call = parameter.grad.clone()
    input_parameters = {
        "list": [parameter],
        "empty": (),
        "tensor": (parameter, torch.ones(1)),
        "dtype": (parameter, torch.nn.Parameter(torch.ones(1, dtype=torch.float64))),
        "device": (parameter, torch.nn.Parameter(torch.ones(1, device="meta"))),
        "layout": (parameter, torch.nn.Parameter(torch.ones(1, 2).to_sparse())),
        "duplicate": (parameter, parameter),
    }[specified_parameter_value]
    optimizer_settings = AdamParameterOptimizerSettings(
        learning_rate=0.01, weight_decay=0.1, adam_variant="standard"
    )
    expected = dict(vars(optimizer_settings))
    global_torch_random_state = torch.get_rng_state()
    monkeypatch.setattr(
        parameter_optimizer_construction,
        "Adam",
        Mock(side_effect=AssertionError("後段まで生成前検査が必要")),
    )
    with pytest.raises(
        ValueError,
        match="parameters"
        if specified_parameter_value in ("list", "empty")
        else r"parameters\[1\]",
    ):
        create_parameter_optimizer(
            parameters=input_parameters, optimizer_settings=optimizer_settings
        )
    parameter_optimizer_construction.Adam.assert_not_called()
    assert vars(optimizer_settings) == expected
    assert parameter.grad is original
    assert torch.equal(parameter, parameter_snapshot_before_call)
    assert torch.equal(parameter.grad, gradient_snapshot_before_call)
    assert torch.equal(torch.get_rng_state(), global_torch_random_state)


def test_parameter_optimizer_matches_legacy_accepts_special_parameter_shapes():
    input_parameters = (
        torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3).t()),
        torch.nn.Parameter(torch.tensor(1.0), requires_grad=False),
        torch.nn.Parameter(torch.empty(0, 2)),
    )
    assert not input_parameters[0].is_contiguous()
    for optimizer_settings in (
        SgdParameterOptimizerSettings(learning_rate=0.1),
        AdamParameterOptimizerSettings(learning_rate=0.1, weight_decay=0, adam_variant="standard"),
    ):
        optimizer = create_parameter_optimizer(
            parameters=input_parameters, optimizer_settings=optimizer_settings
        )
        assert optimizer.state == {}
        for parameter_index, parameter in enumerate(optimizer.param_groups[0]["params"]):
            assert parameter is input_parameters[parameter_index]
            assert parameter.grad is None
        assert not input_parameters[1].requires_grad


def test_parameter_optimizer_settings_are_immutable_and_explicit():
    for optimizer_settings in (
        SgdParameterOptimizerSettings(learning_rate=0),
        AdamParameterOptimizerSettings(learning_rate=0, weight_decay=0, adam_variant="standard"),
    ):
        for configuration_parameter_name in fields(optimizer_settings):
            assert configuration_parameter_name.kw_only
            assert configuration_parameter_name.default is MISSING
            assert configuration_parameter_name.default_factory is MISSING
        with pytest.raises(FrozenInstanceError):
            optimizer_settings.learning_rate = 0.5
        with pytest.raises(TypeError):
            type(optimizer_settings)()
        with pytest.raises(TypeError):
            type(optimizer_settings)(0.1)
    optimizer_settings = SgdParameterOptimizerSettings(learning_rate=0.1)
    assert tuple(vars(optimizer_settings)) == ("learning_rate",)
    for configuration_parameter_name in ("weight_decay", "adam_variant"):
        with pytest.raises(TypeError):
            SgdParameterOptimizerSettings(learning_rate=0.1, **{configuration_parameter_name: 0})
    with pytest.raises(TypeError):
        create_parameter_optimizer((torch.nn.Parameter(torch.ones(1)),), optimizer_settings)


def test_parameter_optimizer_state_is_independent():
    input_parameters = (torch.nn.Parameter(torch.ones(2)),)
    optimizer_settings = AdamParameterOptimizerSettings(
        learning_rate=0.1, weight_decay=0, adam_variant="amsgrad"
    )
    optimizer = create_parameter_optimizer(
        parameters=input_parameters, optimizer_settings=optimizer_settings
    )
    expected = create_parameter_optimizer(
        parameters=input_parameters, optimizer_settings=optimizer_settings
    )
    assert optimizer is not expected
    assert optimizer.state is not expected.state
    assert optimizer.param_groups[0] is not expected.param_groups[0]
    input_parameters[0].grad = torch.ones(2)
    optimizer.step()
    assert optimizer.state
    assert expected.state == {}
    expected.step()
    for configuration_parameter_name in ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        assert (
            optimizer.state[input_parameters[0]][configuration_parameter_name].data_ptr()
            != expected.state[input_parameters[0]][configuration_parameter_name].data_ptr()
        )


@pytest.mark.parametrize("specified_parameter_value", [True, False])
def test_parameter_optimizer_preserves_random_state_and_tensor_environment(
    specified_parameter_value,
):
    input_parameters = (torch.nn.Parameter(torch.ones(2, dtype=torch.float32, device="cpu")),)
    input_parameters[0].grad = torch.tensor([0.25, -0.5], dtype=torch.float32, device="cpu")
    parameter_snapshot_before_call = input_parameters[0].detach().clone()
    gradient_snapshot_before_call = input_parameters[0].grad.clone()
    original = (
        torch.get_default_dtype(),
        torch.get_default_device(),
        torch.is_grad_enabled(),
        input_parameters[0].grad,
    )
    global_torch_random_state = torch.get_rng_state()
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"), torch.set_grad_enabled(specified_parameter_value):
            for optimizer_settings in (
                SgdParameterOptimizerSettings(learning_rate=0.1),
                AdamParameterOptimizerSettings(
                    learning_rate=0.1, weight_decay=0.01, adam_variant="amsgrad"
                ),
            ):
                optimizer = create_parameter_optimizer(
                    parameters=input_parameters, optimizer_settings=optimizer_settings
                )
                assert optimizer.param_groups[0]["params"][0] is input_parameters[0]
                assert optimizer.state == {}
            # 拒否経路も同じ環境を保つ。
            with pytest.raises(ValueError, match="parameters"):
                create_parameter_optimizer(parameters=(), optimizer_settings=optimizer_settings)
            with pytest.raises(ValueError, match="optimizer_settings"):
                create_parameter_optimizer(parameters=input_parameters, optimizer_settings=None)
            assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_device() == torch.device("meta")
            assert torch.is_grad_enabled() is specified_parameter_value
            assert torch.equal(torch.get_rng_state(), global_torch_random_state)
            assert random.getstate() == global_python_random_state
            expected = np.random.get_state()
            assert expected[0] == global_numpy_random_state[0]
            assert np.array_equal(expected[1], global_numpy_random_state[1])
            assert expected[2:] == global_numpy_random_state[2:]
            assert input_parameters[0].grad is original[3]
            assert torch.equal(input_parameters[0], parameter_snapshot_before_call)
            assert torch.equal(input_parameters[0].grad, gradient_snapshot_before_call)
    finally:
        torch.set_default_dtype(original[0])
    assert torch.get_default_dtype() == original[0]
    assert torch.get_default_device() == original[1]
    assert torch.is_grad_enabled() == original[2]


def test_parameter_optimizer_uses_separate_model_parameter_groups():
    actual = ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=3,
    )
    parameter_snapshot_before_call = {
        parameter_name: parameter.detach().clone()
        for parameter_name, parameter in actual.state_dict().items()
    }
    original = tuple((parameter, parameter.grad) for parameter in actual.parameters())
    model_module_attributes_by_name = {
        model_module_name: tuple(vars(model_module))
        for model_module_name, model_module in actual.named_modules()
    }
    shared_feature_parameters = tuple(actual.feature_extractor.parameters())
    concept_specific_parameters = tuple(actual.residual_adapter.parameters()) + tuple(
        actual.classification_layer.parameters()
    )
    assert not {id(parameter) for parameter in shared_feature_parameters} & {
        id(parameter) for parameter in concept_specific_parameters
    }
    assert {
        id(parameter) for parameter in shared_feature_parameters + concept_specific_parameters
    } == {id(parameter) for parameter in actual.parameters()}
    optimizer_settings = AdamParameterOptimizerSettings(
        learning_rate=0.01, weight_decay=0.001, adam_variant="amsgrad"
    )
    for input_parameters in (shared_feature_parameters, concept_specific_parameters):
        optimizer = create_parameter_optimizer(
            parameters=input_parameters, optimizer_settings=optimizer_settings
        )
        for parameter_index, parameter in enumerate(optimizer.param_groups[0]["params"]):
            assert parameter is input_parameters[parameter_index]
        assert optimizer.state == {}
    for model_module_name, model_module in actual.named_modules():
        assert tuple(vars(model_module)) == model_module_attributes_by_name[model_module_name]
        assert not hasattr(model_module, "optimizer")
    for parameter_name, parameter in actual.state_dict().items():
        assert torch.equal(parameter, parameter_snapshot_before_call[parameter_name])
    for parameter, expected in original:
        assert parameter.grad is expected


def test_parameter_optimizer_matches_legacy_extreme_integer_constructor_only(monkeypatch):
    # builtin intは有限。float変換や上限を追加せず、生成契約だけを比較する。
    input_parameters = (torch.nn.Parameter(torch.tensor([1.0, -2.0])),)
    input_parameters[0].grad = torch.tensor([0.25, -0.5])
    parameter_snapshot_before_call = input_parameters[0].detach().clone()
    gradient_snapshot_before_call = input_parameters[0].grad.clone()
    original = input_parameters[0].grad
    optimizer_settings = AdamParameterOptimizerSettings(
        learning_rate=10**1000, weight_decay=0.001, adam_variant="amsgrad"
    )
    monkeypatch.setattr(config, "OPTIMIZER", "adam")
    monkeypatch.setattr(config, "WEIGHT_DECAY", 0.001)
    monkeypatch.setattr(config, "AMSGRAD", True)
    legacy_optimizer = SharedBackboneMLP._build_component_optimizer(
        input_parameters, optimizer_settings.learning_rate
    )
    optimizer = create_parameter_optimizer(
        parameters=input_parameters, optimizer_settings=optimizer_settings
    )
    assert type(optimizer.param_groups[0]["lr"]) is int
    assert optimizer.param_groups[0]["lr"] == legacy_optimizer.param_groups[0]["lr"] == 10**1000
    assert optimizer.state == legacy_optimizer.state == {}
    assert optimizer.param_groups[0]["params"][0] is input_parameters[0]
    assert input_parameters[0].grad is original
    assert torch.equal(input_parameters[0], parameter_snapshot_before_call)
    assert torch.equal(input_parameters[0].grad, gradient_snapshot_before_call)
