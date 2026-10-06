"""保有一覧の順序と、現在optimizerを取得する境界を検証する。"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from random import Random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_held_model_joint_training_iterations import (
    build_training_iteration_oracle_pair,
    run_legacy_training_iterations,
)
from test_joint_model_parameter_update import (
    assert_joint_update_states_equal,
    assert_nested_state_equal,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment import config
from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.held_model_joint_training_iterations import (
    perform_held_model_joint_training_iterations,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingState,
    HeldModelTrainingStateRegistry,
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


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_old_registration_order_and_training_with_current_optimizers_match(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        monkeypatch.setattr(config, "NEW_MODEL_LR", 0.01)
        (
            training_batches,
            shared_optimizer,
            legacy_client,
            ordered_training_samples,
            _,
        ) = build_training_iteration_oracle_pair(
            class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
        )
        optimizer_settings = (
            SgdParameterOptimizerSettings(learning_rate=0.01)
            if optimizer_variant == "sgd"
            else AdamParameterOptimizerSettings(
                learning_rate=0.01, weight_decay=0.001, adam_variant=optimizer_variant
            )
        )
        concept_owners = tuple(
            ParameterOptimizerState(
                parameters=tuple(training_batch.classifier.residual_adapter.parameters())
                + tuple(training_batch.classifier.classification_layer.parameters()),
                optimizer_settings=optimizer_settings,
            )
            for training_batch in training_batches
        )
        registry = HeldModelTrainingStateRegistry()
        legacy_models = tuple(legacy_client.models.values())
        legacy_registration_client = SimpleNamespace(
            models={},
            model_stats={},
            # 共有準備は別specで対照済み。ここでは準備済み実体を渡す。
            _prepare_model_for_registration=lambda classifier: classifier,
            _record_model_compute=lambda *args: None,
        )
        for model_id, sample_index in ((-7, 0), (4, 0), (-7, 1)):
            training_batch = training_batches[sample_index]
            BaseClient._register_trained_new_model(
                legacy_registration_client,
                model_id,
                legacy_models[sample_index],
                training_batch.input_features,
                training_batch.observed_class_labels,
                pending_ready=False,
            )
            register_training_state(
                registry, model_id, training_batch.classifier, concept_owners[sample_index]
            )
        legacy_client.models = legacy_registration_client.models
        assert tuple(legacy_client.models) == (-7, 4)
        assert tuple(
            state.model_id for state in registry.snapshot_ordered_held_model_training_states()
        ) == tuple(legacy_client.models)
        assert legacy_client.models[-7] is legacy_models[1]
        assert registry.get_held_model_training_state(model_id=-7).classifier is (
            training_batches[1].classifier
        )
        legacy_client.batch_size = 3
        python_random_generator = Random(731)
        previous_binding = None
        previous_optimizer_state = None
        for step_index in range(3):
            if step_index == 1:
                previous_binding = registry.snapshot_ordered_held_model_training_bindings()[0]
                previous_optimizer_state = deepcopy(
                    previous_binding.concept_specific_parameter_optimizer.state_dict()
                )
                # 実旧公開接続は共有optimizerを保持して個別optimizerをresetする。
                legacy_client.models[-7].attach_backbone(legacy_models[0].backbone)
                concept_owners[1].reset_parameter_optimizer()
            training_bindings = registry.snapshot_ordered_held_model_training_bindings()
            model_states_by_id = {binding.model_id: binding for binding in training_bindings}
            participating_training_batches = tuple(
                replace(
                    training_batch,
                    concept_specific_parameter_optimizer=model_states_by_id[
                        model_id
                    ].concept_specific_parameter_optimizer,
                )
                for model_id, training_batch in zip((4, -7), training_batches)
            )
            expected_losses, expected_random_state, sampled_batch_history = (
                run_legacy_training_iterations(
                    legacy_client=legacy_client,
                    iteration_count=1,
                    update_shared_features=update_shared_features,
                    initial_random_state=python_random_generator.getstate(),
                )
            )
            actual_losses = perform_held_model_joint_training_iterations(
                requested_joint_update_iteration_count=1,
                held_model_training_bindings=training_bindings,
                ordered_model_training_samples=ordered_training_samples,
                batch_sample_count=3,
                python_random_generator=python_random_generator,
                local_training_settings=LocalTrainingSettings(
                    local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                    shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
                ),
                shared_feature_extractor=training_batches[0].classifier.feature_extractor,
                shared_parameter_optimizer=shared_optimizer,
                update_shared_features=update_shared_features,
            )
            assert actual_losses == expected_losses
            assert python_random_generator.getstate() == expected_random_state
            assert len(sampled_batch_history) == 1
            assert_joint_update_states_equal(
                # 既存照合helperは旧dict順にzipする。学習順は標本store順のまま。
                participating_training_batches=tuple(reversed(participating_training_batches)),
                shared_parameter_optimizer=shared_optimizer,
                legacy_client=legacy_client,
            )
            if previous_binding is not None:
                assert previous_binding.concept_specific_parameter_optimizer is not (
                    concept_owners[1].parameter_optimizer
                )
                assert_nested_state_equal(
                    previous_binding.concept_specific_parameter_optimizer.state_dict(),
                    previous_optimizer_state,
                )


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
