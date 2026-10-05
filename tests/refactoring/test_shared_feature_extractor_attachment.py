"""共有特徴抽出部の再接続と概念固有部の保持を検証する。"""

import random
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_joint_model_parameter_update import (
    assert_joint_update_states_equal,
    build_joint_update_oracle_pair,
    run_legacy_joint_update,
)

from federated_drift_experiment.models import SharedFeatureBackbone
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
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


class SharedFeatureExtractorSubclass(SharedFeatureExtractor):
    pass


def build_attachment_classifier(*, class_count=2, hidden_layer_widths=(5, 4)):
    return ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=hidden_layer_widths,
        class_count=class_count,
    )


def snapshot_parameter_values_and_gradients(parameters):
    return tuple(
        (
            parameter,
            parameter.detach().clone(),
            (parameter.grad, None if parameter.grad is None else parameter.grad.detach().clone()),
        )
        for parameter in parameters
    )


def assert_parameter_values_and_gradients_unchanged(parameter_snapshot):
    for parameter, parameter_values, previous_gradients in parameter_snapshot:
        assert parameter.shape == parameter_values.shape
        assert parameter.device == parameter_values.device
        assert parameter.dtype == parameter_values.dtype
        if parameter.device.type != "meta":
            assert torch.equal(parameter, parameter_values)
        assert parameter.grad is previous_gradients[0]
        parameter_values = previous_gradients[1]
        if parameter_values is not None and parameter.device.type != "meta":
            assert torch.equal(parameter.grad, parameter_values)


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("hidden_layer_widths", [(), (3,), (5, 4)])
@pytest.mark.parametrize("same_reference", [True, False])
def test_attachment_preserves_concept_parameters_and_uses_target_forward(
    class_count, hidden_layer_widths, same_reference
):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier(
            class_count=class_count, hidden_layer_widths=hidden_layer_widths
        )
        previous_feature_extractor = classifier.feature_extractor
        shared_feature_extractor = (
            previous_feature_extractor
            if same_reference
            else SharedFeatureExtractor(
                input_feature_count=2, hidden_layer_widths=hidden_layer_widths
            )
        )
        previous_residual_adapter = classifier.residual_adapter
        previous_classification_layer = classifier.classification_layer
        previous_output_activation = classifier.output_activation
        for parameter in tuple(classifier.parameters()) + tuple(
            shared_feature_extractor.parameters()
        ):
            parameter.grad = torch.ones_like(parameter)
        previous_parameters = snapshot_parameter_values_and_gradients(classifier.parameters())
        shared_feature_extractor_snapshot = snapshot_parameter_values_and_gradients(
            shared_feature_extractor.parameters()
        )
        python_random_state = random.getstate()
        torch_random_state = torch.get_rng_state()
        assert (
            classifier.attach_shared_feature_extractor(
                shared_feature_extractor=shared_feature_extractor
            )
            is None
        )
        assert classifier.feature_extractor is shared_feature_extractor
        assert classifier.residual_adapter is previous_residual_adapter
        assert classifier.classification_layer is previous_classification_layer
        assert classifier.output_activation is previous_output_activation
        assert classifier.class_count == class_count
        assert_parameter_values_and_gradients_unchanged(previous_parameters)
        assert_parameter_values_and_gradients_unchanged(shared_feature_extractor_snapshot)
        assert random.getstate() == python_random_state
        assert torch.equal(torch.get_rng_state(), torch_random_state)
        input_features = torch.tensor([[0.5, -0.25]], dtype=torch.float32, device="cpu")
        expected_prediction = classifier.forward_from_shared_features(
            shared_feature_extractor(input_features)
        )
        assert torch.equal(classifier(input_features), expected_prediction)


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("existing_shared_optimizer", [True, False])
def test_attachment_and_explicit_optimizer_selection_match_legacy_joint_training(
    class_count, optimizer_variant, existing_shared_optimizer, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(719)
        training_batches, previous_parameter_optimizer, legacy_client = (
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
        local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        )
        concept_specific_optimizer_states = tuple(
            ParameterOptimizerState(
                parameters=tuple(training_batch.classifier.residual_adapter.parameters())
                + tuple(training_batch.classifier.classification_layer.parameters()),
                optimizer_settings=optimizer_settings,
            )
            for training_batch in training_batches
        )
        training_batches = tuple(
            replace(
                training_batch,
                concept_specific_parameter_optimizer=optimizer_state.parameter_optimizer,
            )
            for training_batch, optimizer_state in zip(
                training_batches, concept_specific_optimizer_states
            )
        )
        legacy_loss = run_legacy_joint_update(
            legacy_client=legacy_client, update_shared_features=True
        )
        new_loss = perform_joint_model_parameter_update(
            local_training_settings=local_training_settings,
            shared_feature_extractor=training_batches[0].classifier.feature_extractor,
            shared_parameter_optimizer=previous_parameter_optimizer,
            participating_training_batches=training_batches,
            update_shared_features=True,
        )
        assert new_loss == legacy_loss
        assert_joint_update_states_equal(
            participating_training_batches=training_batches,
            shared_parameter_optimizer=previous_parameter_optimizer,
            legacy_client=legacy_client,
        )
        previous_feature_extractor = training_batches[0].classifier.feature_extractor
        previous_parameters = snapshot_parameter_values_and_gradients(
            previous_feature_extractor.parameters()
        )
        parameter_snapshot = deepcopy(previous_parameter_optimizer.state_dict())
        legacy_feature_extractor = SharedFeatureBackbone(input_dim=2, hidden_dims=(5, 4))
        shared_feature_extractor = SharedFeatureExtractor(
            input_feature_count=2, hidden_layer_widths=(5, 4)
        )
        shared_feature_extractor.hidden_layers.load_state_dict(
            legacy_feature_extractor.net.state_dict()
        )
        shared_optimizer_state = ParameterOptimizerState(
            parameters=tuple(shared_feature_extractor.parameters()),
            optimizer_settings=optimizer_settings,
        )
        assert shared_optimizer_state.parameter_optimizer is not previous_parameter_optimizer
        assert all(
            parameter is parameters
            for parameter, parameters in zip(
                shared_optimizer_state.parameter_optimizer.param_groups[0]["params"],
                shared_feature_extractor.parameters(),
                strict=True,
            )
        )
        assert all(
            parameter is parameters
            for parameter, parameters in zip(
                previous_parameter_optimizer.param_groups[0]["params"],
                previous_feature_extractor.parameters(),
                strict=True,
            )
        )
        if existing_shared_optimizer:
            legacy_feature_extractor.optimizer = legacy_client.models[4]._build_component_optimizer(
                legacy_feature_extractor.parameters(), 0.01
            )
            for parameter in tuple(shared_feature_extractor.parameters()) + tuple(
                legacy_feature_extractor.parameters()
            ):
                parameter.grad = torch.ones_like(parameter)
            shared_optimizer_state.parameter_optimizer.step()
            legacy_feature_extractor.optimizer.step()
        shared_feature_extractor_snapshot = deepcopy(
            shared_optimizer_state.parameter_optimizer.state_dict()
        )
        previous_concept_optimizer_snapshots = tuple(
            (
                optimizer_state.parameter_optimizer,
                deepcopy(optimizer_state.parameter_optimizer.state_dict()),
            )
            for optimizer_state in concept_specific_optimizer_states
        )
        python_random_state = random.getstate()
        torch_random_state = torch.get_rng_state()
        for training_batch in training_batches:
            training_batch.classifier.attach_shared_feature_extractor(
                shared_feature_extractor=shared_feature_extractor
            )
        assert random.getstate() == python_random_state
        assert torch.equal(torch.get_rng_state(), torch_random_state)
        for optimizer_state, parameters in zip(
            concept_specific_optimizer_states, previous_concept_optimizer_snapshots
        ):
            assert optimizer_state.parameter_optimizer is parameters[0]
            torch.testing.assert_close(
                optimizer_state.parameter_optimizer.state_dict(), parameters[1], rtol=0, atol=0
            )
            optimizer_state.reset_parameter_optimizer()
            assert optimizer_state.parameter_optimizer is not parameters[0]
            assert optimizer_state.parameter_optimizer.state == {}
        for legacy_model in legacy_client.models.values():
            legacy_model.attach_backbone(legacy_feature_extractor)
        torch.testing.assert_close(
            shared_optimizer_state.parameter_optimizer.state_dict(),
            shared_feature_extractor_snapshot,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            legacy_feature_extractor.optimizer.state_dict(),
            shared_feature_extractor_snapshot,
            rtol=0,
            atol=0,
        )
        participating_training_batches = tuple(
            replace(
                training_batch,
                concept_specific_parameter_optimizer=optimizer_state.parameter_optimizer,
            )
            for training_batch, optimizer_state in zip(
                training_batches, concept_specific_optimizer_states
            )
        )
        for update_shared_features in (True, False, True):
            legacy_loss = run_legacy_joint_update(
                legacy_client=legacy_client, update_shared_features=update_shared_features
            )
            new_loss = perform_joint_model_parameter_update(
                local_training_settings=local_training_settings,
                shared_feature_extractor=shared_feature_extractor,
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
            assert_parameter_values_and_gradients_unchanged(previous_parameters)
            torch.testing.assert_close(
                previous_parameter_optimizer.state_dict(), parameter_snapshot, rtol=0, atol=0
            )
            for training_batch, parameters in zip(
                training_batches, previous_concept_optimizer_snapshots
            ):
                assert training_batch.concept_specific_parameter_optimizer is parameters[0]
                torch.testing.assert_close(
                    parameters[0].state_dict(), parameters[1], rtol=0, atol=0
                )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "none",
        "object",
        "subclass",
        "input_width",
        "hidden_width",
        "float64",
        "meta",
        "layers",
        "shape",
    ],
)
def test_invalid_attachment_preserves_connection_values_and_gradients(invalid_case):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier()
        previous_feature_extractor = classifier.feature_extractor
        invalid_value = SharedFeatureExtractor(input_feature_count=2, hidden_layer_widths=(5, 4))
        if invalid_case == "none":
            invalid_value = None
        elif invalid_case == "object":
            invalid_value = object()
        elif invalid_case == "subclass":
            invalid_value = SharedFeatureExtractorSubclass(
                input_feature_count=2, hidden_layer_widths=(5, 4)
            )
        elif invalid_case == "input_width":
            invalid_value = SharedFeatureExtractor(
                input_feature_count=3, hidden_layer_widths=(5, 4)
            )
        elif invalid_case == "hidden_width":
            invalid_value = SharedFeatureExtractor(
                input_feature_count=2, hidden_layer_widths=(5, 3)
            )
        elif invalid_case == "float64":
            invalid_value.to(dtype=torch.float64)
        elif invalid_case == "meta":
            invalid_value.to(device="meta")
        elif invalid_case == "layers":
            invalid_value.hidden_layers[1] = torch.nn.Identity()
        elif invalid_case == "shape":
            invalid_value.hidden_layers[0].weight = torch.nn.Parameter(
                torch.ones((1, 1), device="cpu", dtype=torch.float32)
            )
        for parameter in classifier.parameters():
            parameter.grad = torch.ones_like(parameter)
        previous_parameters = snapshot_parameter_values_and_gradients(classifier.parameters())
        shared_feature_extractor_snapshot = (
            snapshot_parameter_values_and_gradients(invalid_value.parameters())
            if isinstance(invalid_value, SharedFeatureExtractor)
            else ()
        )
        python_random_state = random.getstate()
        torch_random_state = torch.get_rng_state()
        with pytest.raises(ValueError):
            classifier.attach_shared_feature_extractor(shared_feature_extractor=invalid_value)
        assert classifier.feature_extractor is previous_feature_extractor
        assert_parameter_values_and_gradients_unchanged(previous_parameters)
        assert_parameter_values_and_gradients_unchanged(shared_feature_extractor_snapshot)
        assert random.getstate() == python_random_state
        assert torch.equal(torch.get_rng_state(), torch_random_state)
