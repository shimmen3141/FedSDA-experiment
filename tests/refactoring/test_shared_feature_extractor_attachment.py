"""共有特徴抽出部の再接続と概念固有部の保持を検証する。"""

import random

import pytest
import torch

from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
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
