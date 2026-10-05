"""実旧optimizerリセットへ参照・state・同勾配更新を照合する。"""

import random
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_joint_model_parameter_update import (
    assert_joint_update_states_equal,
    build_joint_update_oracle_pair,
    run_legacy_joint_update,
)

from federated_drift_experiment import config
from federated_drift_experiment.models import SharedBackboneMLP
from federated_learning_experiments.learning.training import parameter_optimizer_state
from federated_learning_experiments.learning.training.held_model_training_binding import (
    HeldModelTrainingBinding,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)

OPTIMIZER_SETTINGS_CASES = (
    AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0, adam_variant="standard"),
    AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0.2, adam_variant="standard"),
    AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0.2, adam_variant="amsgrad"),
    AdamParameterOptimizerSettings(learning_rate=0, weight_decay=0.2, adam_variant="amsgrad"),
    SgdParameterOptimizerSettings(learning_rate=0.05),
    SgdParameterOptimizerSettings(learning_rate=0),
)


def build_optimizer_state_oracle_pair(*, optimizer_settings, monkeypatch):
    monkeypatch.setattr(
        config,
        "OPTIMIZER",
        "sgd" if type(optimizer_settings) is SgdParameterOptimizerSettings else "adam",
    )
    if type(optimizer_settings) is AdamParameterOptimizerSettings:
        monkeypatch.setattr(config, "WEIGHT_DECAY", optimizer_settings.weight_decay)
        monkeypatch.setattr(config, "AMSGRAD", optimizer_settings.adam_variant == "amsgrad")
    input_parameters = tuple(
        torch.nn.Parameter(
            torch.tensor([float(parameter_index), -0.25], dtype=torch.float32, device="cpu")
        )
        for parameter_index in range(2)
    )
    legacy_parameters = tuple(
        torch.nn.Parameter(parameter.detach().clone()) for parameter in input_parameters
    )
    legacy_concept_parameters = (
        torch.nn.Parameter(torch.tensor([0.75], dtype=torch.float32, device="cpu")),
    )
    optimizer_state = ParameterOptimizerState(
        parameters=input_parameters, optimizer_settings=optimizer_settings
    )
    legacy_model = SimpleNamespace(
        backbone=SimpleNamespace(parameters=lambda: legacy_parameters),
        personalized_parameters=lambda: legacy_concept_parameters,
        _build_component_optimizer=SharedBackboneMLP._build_component_optimizer,
    )
    SharedBackboneMLP.reset_optimizer(legacy_model, lr=optimizer_settings.learning_rate)
    return optimizer_state, legacy_model


def assert_optimizer_state_matches_legacy(*, optimizer_state, legacy_model):
    legacy_optimizer = legacy_model.backbone.optimizer
    torch.testing.assert_close(
        optimizer_state.parameter_optimizer.state_dict(),
        legacy_optimizer.state_dict(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        optimizer_state.parameter_optimizer.defaults, legacy_optimizer.defaults, rtol=0, atol=0
    )
    for parameter, legacy_parameters in zip(
        optimizer_state.parameter_optimizer.param_groups[0]["params"],
        legacy_optimizer.param_groups[0]["params"],
    ):
        assert torch.equal(parameter, legacy_parameters)
        if parameter.grad is None:
            assert legacy_parameters.grad is None
        else:
            assert torch.equal(parameter.grad, legacy_parameters.grad)


@pytest.mark.parametrize("optimizer_settings", OPTIMIZER_SETTINGS_CASES)
@pytest.mark.parametrize("update_index", [0, 2])
def test_explicit_reset_matches_actual_legacy_and_preserves_values_and_gradients(
    optimizer_settings, update_index, monkeypatch
):
    optimizer_state, legacy_model = build_optimizer_state_oracle_pair(
        optimizer_settings=optimizer_settings, monkeypatch=monkeypatch
    )
    assert_optimizer_state_matches_legacy(
        optimizer_state=optimizer_state, legacy_model=legacy_model
    )
    for _ in range(update_index):
        for parameter, legacy_parameters in zip(
            optimizer_state.parameter_optimizer.param_groups[0]["params"],
            legacy_model.backbone.optimizer.param_groups[0]["params"],
        ):
            parameter.grad = torch.full_like(parameter, 0.125)
            legacy_parameters.grad = parameter.grad.clone()
        optimizer_state.parameter_optimizer.step()
        legacy_model.backbone.optimizer.step()
    input_parameters = tuple(optimizer_state.parameter_optimizer.param_groups[0]["params"])
    previous_parameter_optimizer = optimizer_state.parameter_optimizer
    previous_optimizer_state_dict = deepcopy(previous_parameter_optimizer.state_dict())
    previous_gradients = tuple(parameter.grad for parameter in input_parameters)
    parameter_values_before_reset = tuple(
        parameter.detach().clone() for parameter in input_parameters
    )
    python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state()
    optimizer_state.reset_parameter_optimizer()
    SharedBackboneMLP.reset_optimizer(legacy_model, lr=optimizer_settings.learning_rate)
    assert optimizer_state.parameter_optimizer is not previous_parameter_optimizer
    assert optimizer_state.parameter_optimizer.state == {}
    torch.testing.assert_close(
        previous_parameter_optimizer.state_dict(), previous_optimizer_state_dict, rtol=0, atol=0
    )
    assert random.getstate() == python_random_state
    assert torch.equal(torch.get_rng_state(), torch_random_state)
    for parameter_index, parameter in enumerate(input_parameters):
        assert (
            optimizer_state.parameter_optimizer.param_groups[0]["params"][parameter_index]
            is parameter
        )
        assert parameter.grad is previous_gradients[parameter_index]
        assert torch.equal(parameter, parameter_values_before_reset[parameter_index])
    assert_optimizer_state_matches_legacy(
        optimizer_state=optimizer_state, legacy_model=legacy_model
    )
    for gradient_value in (0.25, None, -0.5):
        for parameter_index, (parameter, legacy_parameters) in enumerate(
            zip(input_parameters, legacy_model.backbone.optimizer.param_groups[0]["params"])
        ):
            parameter.grad = (
                None
                if gradient_value is None and parameter_index == 1
                else torch.full_like(parameter, 0.1 if gradient_value is None else gradient_value)
            )
            legacy_parameters.grad = None if parameter.grad is None else parameter.grad.clone()
        optimizer_state.parameter_optimizer.step()
        legacy_model.backbone.optimizer.step()
        assert_optimizer_state_matches_legacy(
            optimizer_state=optimizer_state, legacy_model=legacy_model
        )


def test_current_optimizer_reference_is_readonly_and_learning_state_is_borrowed(monkeypatch):
    optimizer_state, legacy_model = build_optimizer_state_oracle_pair(
        optimizer_settings=OPTIMIZER_SETTINGS_CASES[0], monkeypatch=monkeypatch
    )
    previous_parameter_optimizer = optimizer_state.parameter_optimizer
    assert optimizer_state.parameter_optimizer is previous_parameter_optimizer
    for parameter in previous_parameter_optimizer.param_groups[0]["params"]:
        parameter.grad = torch.ones_like(parameter)
    previous_parameter_optimizer.step()
    assert len(optimizer_state.parameter_optimizer.state) == 2
    with pytest.raises(AttributeError):
        optimizer_state.parameter_optimizer = previous_parameter_optimizer
    previous_parameter_optimizer.param_groups[0]["lr"] = 0.99
    optimizer_state.reset_parameter_optimizer()
    assert optimizer_state.parameter_optimizer.param_groups[0]["lr"] == 0.01
    assert len(previous_parameter_optimizer.state) == 2
    previous_parameter_optimizer = optimizer_state.parameter_optimizer
    optimizer_state.reset_parameter_optimizer()
    assert optimizer_state.parameter_optimizer is not previous_parameter_optimizer
    assert optimizer_state.parameter_optimizer.state == {}


@pytest.mark.parametrize("invalid_value", [[], (), (object(),), None])
def test_invalid_parameters_are_rejected_by_existing_factory(invalid_value):
    with pytest.raises(ValueError):
        ParameterOptimizerState(
            parameters=invalid_value, optimizer_settings=OPTIMIZER_SETTINGS_CASES[0]
        )


@pytest.mark.parametrize("invalid_value", [None, {}, "adam", True])
def test_invalid_optimizer_settings_leave_input_parameters_unchanged(invalid_value):
    parameter = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float32, device="cpu"))
    parameter.grad = torch.tensor([0.5], dtype=torch.float32, device="cpu")
    previous_gradients = parameter.grad
    with pytest.raises(ValueError):
        ParameterOptimizerState(parameters=(parameter,), optimizer_settings=invalid_value)
    assert parameter.item() == 0.25
    assert parameter.grad is previous_gradients


def test_failed_reset_keeps_current_optimizer_and_state(monkeypatch):
    optimizer_state, legacy_model = build_optimizer_state_oracle_pair(
        optimizer_settings=OPTIMIZER_SETTINGS_CASES[0], monkeypatch=monkeypatch
    )
    previous_parameter_optimizer = optimizer_state.parameter_optimizer
    for parameter in previous_parameter_optimizer.param_groups[0]["params"]:
        parameter.grad = torch.ones_like(parameter)
    previous_parameter_optimizer.step()
    previous_optimizer_state_dict = deepcopy(previous_parameter_optimizer.state_dict())
    input_parameters = tuple(previous_parameter_optimizer.param_groups[0]["params"])
    previous_gradients = tuple(parameter.grad for parameter in input_parameters)
    parameter_values_before_reset = tuple(
        parameter.detach().clone() for parameter in input_parameters
    )
    monkeypatch.setattr(
        parameter_optimizer_state,
        "create_parameter_optimizer",
        Mock(side_effect=RuntimeError("generation failed")),
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        optimizer_state.reset_parameter_optimizer()
    assert optimizer_state.parameter_optimizer is previous_parameter_optimizer
    torch.testing.assert_close(
        previous_parameter_optimizer.state_dict(), previous_optimizer_state_dict, rtol=0, atol=0
    )
    for parameter_index, parameter in enumerate(input_parameters):
        assert torch.equal(parameter, parameter_values_before_reset[parameter_index])
        assert parameter.grad is previous_gradients[parameter_index]


def test_reset_revalidates_borrowed_parameter_contract_without_losing_optimizer(monkeypatch):
    optimizer_state, legacy_model = build_optimizer_state_oracle_pair(
        optimizer_settings=OPTIMIZER_SETTINGS_CASES[0], monkeypatch=monkeypatch
    )
    previous_parameter_optimizer = optimizer_state.parameter_optimizer
    parameter = previous_parameter_optimizer.param_groups[0]["params"][0]
    parameter.data = parameter.data.to(dtype=torch.float64)
    previous_optimizer_state_dict = deepcopy(previous_parameter_optimizer.state_dict())
    with pytest.raises(ValueError, match="CPU float32"):
        optimizer_state.reset_parameter_optimizer()
    assert optimizer_state.parameter_optimizer is previous_parameter_optimizer
    assert parameter.dtype == torch.float64
    torch.testing.assert_close(
        previous_parameter_optimizer.state_dict(), previous_optimizer_state_dict, rtol=0, atol=0
    )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("reset_case", ["none", "shared", "first_concept", "all"])
def test_optimizer_owners_reset_and_rebind_match_actual_joint_learning(
    class_count, optimizer_variant, reset_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(173)
        training_batches, shared_parameter_optimizer, legacy_client = (
            build_joint_update_oracle_pair(
                class_count=class_count,
                batch_sample_counts=(2, 5),
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        optimizer_settings = (
            SgdParameterOptimizerSettings(learning_rate=0.01)
            if optimizer_variant == "sgd"
            else AdamParameterOptimizerSettings(
                learning_rate=0.01, weight_decay=0.001, adam_variant=optimizer_variant
            )
        )
        shared_optimizer_state = ParameterOptimizerState(
            parameters=tuple(training_batches[0].classifier.feature_extractor.parameters()),
            optimizer_settings=optimizer_settings,
        )
        concept_specific_optimizer_states = tuple(
            ParameterOptimizerState(
                parameters=tuple(training_batch.classifier.residual_adapter.parameters())
                + tuple(training_batch.classifier.classification_layer.parameters()),
                optimizer_settings=optimizer_settings,
            )
            for training_batch in training_batches
        )
        local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        )
        for update_index, update_shared_features in enumerate((True, False, True)):
            if update_index == 1:
                previous_training_batches = tuple(
                    replace(
                        training_batch,
                        concept_specific_parameter_optimizer=optimizer_state.parameter_optimizer,
                    )
                    for training_batch, optimizer_state in zip(
                        training_batches, concept_specific_optimizer_states
                    )
                )
                previous_binding = HeldModelTrainingBinding(
                    model_id=4,
                    classifier=training_batches[0].classifier,
                    concept_specific_parameter_optimizer=concept_specific_optimizer_states[
                        0
                    ].parameter_optimizer,
                )
                previous_parameter_optimizer = shared_optimizer_state.parameter_optimizer
                previous_optimizer_state_dict = deepcopy(previous_parameter_optimizer.state_dict())
                previous_concept_optimizers = tuple(
                    optimizer_state.parameter_optimizer
                    for optimizer_state in concept_specific_optimizer_states
                )
                previous_concept_state_dicts = tuple(
                    deepcopy(optimizer_state.parameter_optimizer.state_dict())
                    for optimizer_state in concept_specific_optimizer_states
                )
                if reset_case in ("shared", "all"):
                    shared_optimizer_state.reset_parameter_optimizer()
                if reset_case in ("first_concept", "all"):
                    concept_specific_optimizer_states[0].reset_parameter_optimizer()
                if reset_case == "all":
                    concept_specific_optimizer_states[1].reset_parameter_optimizer()
                    for legacy_model in legacy_client.models.values():
                        legacy_model.reset_optimizer(lr=0.01)
                elif reset_case == "first_concept":
                    legacy_client.models[4].attach_backbone(legacy_client.models[4].backbone)
                elif reset_case == "shared":
                    legacy_client._shared_backbone().optimizer = legacy_client.models[
                        4
                    ]._build_component_optimizer(
                        legacy_client._shared_backbone().parameters(), 0.01
                    )
                assert (
                    previous_binding.concept_specific_parameter_optimizer
                    is previous_concept_optimizers[0]
                )
                assert (
                    previous_training_batches[0].concept_specific_parameter_optimizer
                    is previous_concept_optimizers[0]
                )
                torch.testing.assert_close(
                    previous_parameter_optimizer.state_dict(),
                    previous_optimizer_state_dict,
                    rtol=0,
                    atol=0,
                )
                for training_batch_index, optimizer_state in enumerate(
                    concept_specific_optimizer_states
                ):
                    torch.testing.assert_close(
                        previous_concept_optimizers[training_batch_index].state_dict(),
                        previous_concept_state_dicts[training_batch_index],
                        rtol=0,
                        atol=0,
                    )
                    if reset_case == "all" or (
                        reset_case == "first_concept" and training_batch_index == 0
                    ):
                        assert (
                            optimizer_state.parameter_optimizer
                            is not previous_concept_optimizers[training_batch_index]
                        )
                        assert optimizer_state.parameter_optimizer.state == {}
                    else:
                        assert (
                            optimizer_state.parameter_optimizer
                            is previous_concept_optimizers[training_batch_index]
                        )
                if reset_case not in ("shared", "all"):
                    assert (
                        shared_optimizer_state.parameter_optimizer is previous_parameter_optimizer
                    )
                else:
                    assert (
                        shared_optimizer_state.parameter_optimizer
                        is not previous_parameter_optimizer
                    )
                    assert shared_optimizer_state.parameter_optimizer.state == {}
            participating_training_batches = tuple(
                replace(
                    training_batch,
                    concept_specific_parameter_optimizer=optimizer_state.parameter_optimizer,
                )
                for training_batch, optimizer_state in zip(
                    training_batches, concept_specific_optimizer_states
                )
            )
            legacy_loss = run_legacy_joint_update(
                legacy_client=legacy_client, update_shared_features=update_shared_features
            )
            new_loss = perform_joint_model_parameter_update(
                local_training_settings=local_training_settings,
                shared_feature_extractor=participating_training_batches[
                    0
                ].classifier.feature_extractor,
                shared_parameter_optimizer=shared_optimizer_state.parameter_optimizer,
                participating_training_batches=participating_training_batches,
                update_shared_features=update_shared_features,
            )
            assert new_loss == legacy_loss
            assert_joint_update_states_equal(
                participating_training_batches=participating_training_batches,
                shared_parameter_optimizer=shared_optimizer_state.parameter_optimizer,
                legacy_client=legacy_client,
            )
