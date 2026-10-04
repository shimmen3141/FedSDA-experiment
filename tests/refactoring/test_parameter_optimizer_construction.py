"""旧optimizer builderと生成条件・外部同勾配更新を直接照合する。"""

import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.models import SharedBackboneMLP
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings, SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_construction import create_parameter_optimizer


@pytest.mark.parametrize("optimizer_settings", [
    AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0, adam_variant="standard"),
    AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0.2, adam_variant="standard"),
    AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0, adam_variant="amsgrad"),
    AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0.2, adam_variant="amsgrad"),
    AdamParameterOptimizerSettings(learning_rate=0, weight_decay=0.2, adam_variant="amsgrad"),
    SgdParameterOptimizerSettings(learning_rate=0.05),
    SgdParameterOptimizerSettings(learning_rate=0),
])
def test_parameter_optimizer_matches_legacy(optimizer_settings, monkeypatch):
    input_parameters = (
        torch.nn.Parameter(torch.tensor([[0.25, -0.75], [1.5, 0.0]],
            dtype=torch.float32, device="cpu"), requires_grad=False),
        torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device="cpu")),
    )
    input_parameters[0].grad = torch.full_like(input_parameters[0], 0.3)
    expected = tuple(torch.nn.Parameter(parameter.detach().clone(),
        requires_grad=parameter.requires_grad) for parameter in input_parameters)
    parameter_snapshot_before_call = tuple(parameter.detach().clone() for parameter in input_parameters)
    gradient_snapshot_before_call = tuple(None if parameter.grad is None else parameter.grad.detach().clone()
                                         for parameter in input_parameters)
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
    legacy_optimizer = SharedBackboneMLP._build_component_optimizer(expected, optimizer_settings.learning_rate)
    optimizer = create_parameter_optimizer(parameters=input_parameters, optimizer_settings=optimizer_settings)
    assert type(optimizer) is type(legacy_optimizer)
    assert optimizer.state == legacy_optimizer.state == {}
    torch.testing.assert_close(optimizer.defaults, legacy_optimizer.defaults, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), legacy_optimizer.state_dict(), rtol=0, atol=0)
    assert all(parameter is input_parameters[parameter_index]
               for parameter_index, parameter in enumerate(optimizer.param_groups[0]["params"]))
    assert all(parameter is expected[parameter_index]
               for parameter_index, parameter in enumerate(legacy_optimizer.param_groups[0]["params"]))
    for parameter_index, parameter in enumerate(input_parameters):
        torch.testing.assert_close(parameter, parameter_snapshot_before_call[parameter_index], rtol=0, atol=0)
        assert parameter.grad is original[parameter_index]
        torch.testing.assert_close(parameter.grad, gradient_snapshot_before_call[parameter_index], rtol=0, atol=0)
    for actual in (0.25, None, -0.5):
        for parameter_index, parameter in enumerate(input_parameters):
            parameter.grad = (None if parameter_index == 1 and actual is None else
                torch.full_like(parameter, 0.125 if actual is None else actual))
        for parameter_index, parameter in enumerate(expected):
            parameter.grad = (None if input_parameters[parameter_index].grad is None else
                input_parameters[parameter_index].grad.detach().clone())
        optimizer.step()
        legacy_optimizer.step()
        for parameter_index, parameter in enumerate(input_parameters):
            torch.testing.assert_close(parameter, expected[parameter_index], rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), legacy_optimizer.state_dict(), rtol=0, atol=0)
