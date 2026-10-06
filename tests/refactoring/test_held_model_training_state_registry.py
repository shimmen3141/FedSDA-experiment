"""保有一覧の順序と、現在optimizerを取得する境界を検証する。"""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from test_joint_model_parameter_update import assert_nested_state_equal
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingState,
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)


class IntSubclass(int):
    """IDとして受理しない派生型。"""


def build_registry_classifier_and_owner():
    classifier = ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(5, 4),
        class_count=2,
    )
    optimizer_owner = ParameterOptimizerState(
        parameters=tuple(classifier.residual_adapter.parameters())
        + tuple(classifier.classification_layer.parameters()),
        optimizer_settings=AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.001, adam_variant="standard"
        ),
    )
    return classifier, optimizer_owner


def register_training_state(registry, model_id, classifier, optimizer_owner):
    return registry.register_held_model_training_state(
        model_id=model_id,
        classifier=classifier,
        concept_specific_parameter_optimizer_state=optimizer_owner,
    )


def test_registration_replacement_preserves_order_and_old_snapshots():
    with torch.random.fork_rng(devices=[]):
        registry = HeldModelTrainingStateRegistry()
        assert registry.snapshot_ordered_held_model_training_states() == ()
        assert registry.snapshot_ordered_held_model_training_bindings() == ()
        classifier, optimizer_owner = build_registry_classifier_and_owner()
        for model_id in (-7, 0, 5, 10**40):
            assert register_training_state(registry, model_id, classifier, optimizer_owner) is None
        snapshot = registry.snapshot_ordered_held_model_training_states()
        previous_binding = registry.snapshot_ordered_held_model_training_bindings()[0]
        previous_state = registry.get_held_model_training_state(model_id=-7)
        assert type(previous_state) is HeldModelTrainingState
        assert previous_state is snapshot[0]
        assert previous_state.classifier is classifier
        assert previous_state.concept_specific_parameter_optimizer_state is optimizer_owner
        with pytest.raises(FrozenInstanceError):
            previous_state.model_id = 8
        replacement_classifier, replacement_owner = build_registry_classifier_and_owner()
        register_training_state(registry, -7, replacement_classifier, replacement_owner)
        register_training_state(registry, -100, classifier, optimizer_owner)
        states = registry.snapshot_ordered_held_model_training_states()
        assert tuple(state.model_id for state in states) == (-7, 0, 5, 10**40, -100)
        assert tuple(state.model_id for state in snapshot) == (-7, 0, 5, 10**40)
        assert previous_state.classifier is classifier
        assert previous_binding.classifier is classifier
        assert states[0].classifier is replacement_classifier
        assert states[0].concept_specific_parameter_optimizer_state is replacement_owner
        bindings = registry.snapshot_ordered_held_model_training_bindings()
        assert tuple(binding.model_id for binding in bindings) == tuple(
            state.model_id for state in states
        )
        for state, binding in zip(states, bindings):
            assert binding.classifier is state.classifier
            assert binding.concept_specific_parameter_optimizer is (
                state.concept_specific_parameter_optimizer_state.parameter_optimizer
            )


@pytest.mark.parametrize(
    "invalid_model_id", [True, False, 1.0, "1", None, IntSubclass(1), np.int64(1)]
)
def test_invalid_id_never_replaces_or_creates_a_state(invalid_model_id):
    with torch.random.fork_rng(devices=[]):
        classifier, optimizer_owner = build_registry_classifier_and_owner()
        registry = HeldModelTrainingStateRegistry()
        register_training_state(registry, 1, classifier, optimizer_owner)
        snapshot = registry.snapshot_ordered_held_model_training_states()
        with pytest.raises(ValueError):
            register_training_state(registry, invalid_model_id, classifier, optimizer_owner)
        with pytest.raises(ValueError):
            registry.get_held_model_training_state(model_id=invalid_model_id)
        assert registry.snapshot_ordered_held_model_training_states() == snapshot


def test_missing_id_is_explicit_and_does_not_create_a_state():
    registry = HeldModelTrainingStateRegistry()
    with pytest.raises(KeyError) as invalid_case:
        registry.get_held_model_training_state(model_id=-7)
    assert invalid_case.value.args == (-7,)
    assert registry.snapshot_ordered_held_model_training_states() == ()


@pytest.mark.parametrize(
    "invalid_case",
    [
        "classifier",
        "owner",
        "unrelated_owner",
        "reverse",
        "missing",
        "duplicate",
        "dtype",
        "empty",
        "shared",
        "optimizer_type",
    ],
)
def test_invalid_registration_keeps_registry_and_learning_state(invalid_case):
    with torch.random.fork_rng(devices=[]):
        registry = HeldModelTrainingStateRegistry()
        classifier, optimizer_owner = build_registry_classifier_and_owner()
        register_training_state(registry, 4, classifier, optimizer_owner)
        replacement_classifier, replacement_owner = build_registry_classifier_and_owner()
        concept_parameters = tuple(replacement_classifier.residual_adapter.parameters()) + tuple(
            replacement_classifier.classification_layer.parameters()
        )
        if invalid_case == "classifier":
            replacement_classifier = object()
        elif invalid_case == "owner":
            replacement_owner = object()
        elif invalid_case == "unrelated_owner":
            replacement_owner = optimizer_owner
        elif invalid_case in ("reverse", "missing", "duplicate", "shared"):
            if invalid_case == "reverse":
                concept_parameters = tuple(reversed(concept_parameters))
            elif invalid_case == "missing":
                concept_parameters = concept_parameters[:-1]
            elif invalid_case == "duplicate":
                concept_parameters = (concept_parameters[0],) * len(concept_parameters)
            else:
                replacement_classifier.classification_layer.weight = next(
                    replacement_classifier.feature_extractor.parameters()
                )
                concept_parameters = tuple(
                    replacement_classifier.residual_adapter.parameters()
                ) + tuple(replacement_classifier.classification_layer.parameters())
            replacement_owner._parameter_optimizer.param_groups[0]["params"] = list(
                concept_parameters
            )
        elif invalid_case == "dtype":
            concept_parameters[0].data = concept_parameters[0].data.double()
        elif invalid_case == "empty":
            replacement_classifier.residual_adapter = torch.nn.Identity()
            replacement_classifier.classification_layer = torch.nn.Identity()
        elif invalid_case == "optimizer_type":
            replacement_owner._parameter_optimizer = torch.optim.AdamW(concept_parameters)
        snapshot = registry.snapshot_ordered_held_model_training_states()
        parameter_snapshots = snapshot_parameter_values_and_gradients(
            tuple(classifier.parameters())
            + (
                tuple(replacement_classifier.parameters())
                if type(replacement_classifier) is ResidualAdapterClassifier
                else ()
            )
        )
        optimizer_snapshot = deepcopy(optimizer_owner.parameter_optimizer.state_dict())
        rng_state = torch.get_rng_state().clone()
        with pytest.raises(ValueError):
            register_training_state(registry, 4, replacement_classifier, replacement_owner)
        assert registry.snapshot_ordered_held_model_training_states() == snapshot
        assert_parameter_values_and_gradients_unchanged(parameter_snapshots)
        assert_nested_state_equal(
            optimizer_owner.parameter_optimizer.state_dict(), optimizer_snapshot
        )
        assert torch.equal(torch.get_rng_state(), rng_state)


def test_reset_changes_new_bindings_and_keeps_prior_references_and_values():
    with torch.random.fork_rng(devices=[]):
        classifier, optimizer_owner = build_registry_classifier_and_owner()
        for parameter in classifier.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer_owner.parameter_optimizer.step()
        previous_optimizer = optimizer_owner.parameter_optimizer
        previous_optimizer_state = deepcopy(previous_optimizer.state_dict())
        parameter_snapshots = snapshot_parameter_values_and_gradients(classifier.parameters())
        rng_state = torch.get_rng_state().clone()
        registry = HeldModelTrainingStateRegistry()
        register_training_state(registry, -7, classifier, optimizer_owner)
        previous_state = registry.get_held_model_training_state(model_id=-7)
        previous_binding = registry.snapshot_ordered_held_model_training_bindings()[0]
        assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
        optimizer_owner.reset_parameter_optimizer()
        binding = registry.snapshot_ordered_held_model_training_bindings()[0]
        assert previous_state.concept_specific_parameter_optimizer_state is optimizer_owner
        assert binding.concept_specific_parameter_optimizer is optimizer_owner.parameter_optimizer
        assert binding.concept_specific_parameter_optimizer is not previous_optimizer
        assert previous_binding.concept_specific_parameter_optimizer is previous_optimizer
        assert binding.concept_specific_parameter_optimizer.state_dict()["state"] == {}
        assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
        assert_parameter_values_and_gradients_unchanged(parameter_snapshots)
        assert torch.equal(torch.get_rng_state(), rng_state)
