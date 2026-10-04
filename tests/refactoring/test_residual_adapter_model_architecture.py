"""旧モデルの構築順・全state・forwardを学習なしで直接照合する。"""

import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.data.specs import DatasetSpec
from federated_drift_experiment.models import ResidualAdapterMLP, SharedFeatureBackbone
from federated_learning_experiments.learning.models.model_architecture_settings import ModelArchitectureSettings
from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor
from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier


def build_legacy_residual_adapter_classifier(
    *, input_feature_count, hidden_layer_widths, class_count, requested_rank,
    monkeypatch, shared_feature_extractor=None,
):
    monkeypatch.setattr(config, "dataset_spec", lambda dataset: DatasetSpec(
        input_dim=input_feature_count, num_concepts=2, num_classes=class_count,
        hidden_dims=hidden_layer_widths))
    monkeypatch.setattr(config, "SHARED_ADAPTER_RANK", requested_rank)
    monkeypatch.setattr(ResidualAdapterMLP, "_build_component_optimizer",
                        staticmethod(lambda parameters, lr: None))
    return ResidualAdapterMLP(
        input_dim=input_feature_count, dataset="sine2", backbone=shared_feature_extractor)


def map_residual_adapter_state_to_legacy_keys(*, state):
    return {name.replace("feature_extractor.hidden_layers.", "backbone.net.")
                .replace("residual_adapter.feature_compression.", "adapter.down.")
                .replace("residual_adapter.feature_expansion.", "adapter.up.")
                .replace("classification_layer.", "head."): value
            for name, value in state.items()}


def assert_residual_adapter_state_matches_legacy(*, actual, expected):
    actual = map_residual_adapter_state_to_legacy_keys(state=actual.state_dict())
    expected = expected.state_dict()
    assert tuple(actual) == tuple(expected)
    for name in expected:
        assert actual[name].shape == expected[name].shape
        assert actual[name].dtype == expected[name].dtype == torch.float32
        assert actual[name].device == expected[name].device == torch.device("cpu")
        assert torch.equal(actual[name], expected[name])


def capture_torch_random_state_after_model_construction(*, constructor, random_state):
    original = torch.get_rng_state()
    try:
        torch.set_rng_state(random_state)
        model = constructor()
        return model, torch.get_rng_state()
    finally:
        torch.set_rng_state(original)


@pytest.mark.parametrize("input_feature_count,hidden_layer_widths,class_count,requested_rank", [
    (2, (32, 32), 2, 8),
    (3, (5, 4), 4, 2),
    (2, (), 2, 9),
    (3, (), 3, 1),
    (1, (1,), 2, 1),
])
@pytest.mark.parametrize("shared", [False, True])
def test_residual_adapter_classifier_matches_legacy(
    input_feature_count, hidden_layer_widths, class_count, requested_rank, shared, monkeypatch,
):
    random_state = torch.Generator(device="cpu").manual_seed(41).get_state()
    shared_feature_extractor = None
    legacy_feature_extractor = None
    if shared:
        shared_feature_extractor, random_state = capture_torch_random_state_after_model_construction(
            constructor=lambda: SharedFeatureExtractor(input_feature_count=input_feature_count,
                hidden_layer_widths=hidden_layer_widths), random_state=random_state)
        legacy_feature_extractor, expected_random_state = capture_torch_random_state_after_model_construction(
            constructor=lambda: SharedFeatureBackbone(input_feature_count, hidden_layer_widths),
            random_state=torch.Generator(device="cpu").manual_seed(41).get_state())
        assert torch.equal(random_state, expected_random_state)
    actual, actual_random_state = capture_torch_random_state_after_model_construction(
        constructor=lambda: ResidualAdapterClassifier(
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=requested_rank),
            input_feature_count=input_feature_count, hidden_layer_widths=hidden_layer_widths,
            class_count=class_count, shared_feature_extractor=shared_feature_extractor),
        random_state=random_state)
    expected, expected_random_state = capture_torch_random_state_after_model_construction(
        constructor=lambda: build_legacy_residual_adapter_classifier(
            input_feature_count=input_feature_count, hidden_layer_widths=hidden_layer_widths,
            class_count=class_count, requested_rank=requested_rank, monkeypatch=monkeypatch,
            shared_feature_extractor=legacy_feature_extractor), random_state=random_state)
    assert torch.equal(actual_random_state, expected_random_state)
    assert_residual_adapter_state_matches_legacy(actual=actual, expected=expected)
    assert actual.residual_adapter.effective_rank == min(requested_rank,
        hidden_layer_widths[-1] if hidden_layer_widths else input_feature_count)
    for input_features in (
        torch.arange(input_feature_count, dtype=torch.float32, device="cpu"),
        torch.arange(3 * input_feature_count, dtype=torch.float32, device="cpu").reshape(3, -1),
    ):
        original = torch.get_rng_state()
        assert torch.equal(actual(input_features), expected(input_features))
        shared_features = actual.extract_shared_features(input_features)
        assert torch.equal(actual.forward_from_shared_features(shared_features), actual(input_features))
        assert torch.equal(actual.residual_adapter(shared_features), shared_features)
        assert torch.equal(torch.get_rng_state(), original)
    if shared:
        assert actual.feature_extractor is shared_feature_extractor


def test_residual_adapter_classifier_shares_only_feature_extractor():
    shared_feature_extractor = SharedFeatureExtractor(input_feature_count=2, hidden_layer_widths=(4,))
    settings = ModelArchitectureSettings(model_architecture_name="shared_backbone_residual_adapter",
        residual_adapter_requested_rank=2)
    first = ResidualAdapterClassifier(model_architecture_settings=settings, input_feature_count=2,
        hidden_layer_widths=(4,), class_count=2, shared_feature_extractor=shared_feature_extractor)
    second = ResidualAdapterClassifier(model_architecture_settings=settings, input_feature_count=2,
        hidden_layer_widths=(4,), class_count=2, shared_feature_extractor=shared_feature_extractor)
    assert first.feature_extractor is second.feature_extractor is shared_feature_extractor
    assert first.residual_adapter is not second.residual_adapter
    assert first.classification_layer is not second.classification_layer
    for parameter, expected in zip(first.residual_adapter.parameters(), second.residual_adapter.parameters()):
        assert parameter.data_ptr() != expected.data_ptr()
    assert first.classification_layer.weight.data_ptr() != second.classification_layer.weight.data_ptr()


@pytest.mark.parametrize("name", ["weight", "bias"])
def test_residual_adapter_model_rejects_invalid_inputs_without_consuming_randomness_unregistered_parameter(name):
    shared_feature_extractor = SharedFeatureExtractor(input_feature_count=2, hidden_layer_widths=(4,))
    linear_layer = shared_feature_extractor.hidden_layers[0]
    parameter = getattr(linear_layer, name).detach().clone()
    delattr(linear_layer, name)
    setattr(linear_layer, name, parameter)
    original = ({name: value.detach().clone() for name, value in shared_feature_extractor.state_dict().items()},
                parameter.detach().clone())
    random_state = torch.get_rng_state()
    try:
        with pytest.raises(ValueError, match="shared_feature_extractor"):
            ResidualAdapterClassifier(
                model_architecture_settings=ModelArchitectureSettings(
                    model_architecture_name="shared_backbone_residual_adapter",
                    residual_adapter_requested_rank=2),
                input_feature_count=2, hidden_layer_widths=(4,), class_count=2,
                shared_feature_extractor=shared_feature_extractor)
        assert torch.equal(torch.get_rng_state(), random_state)
        assert getattr(linear_layer, name) is parameter
        assert torch.equal(parameter, original[1])
        expected = shared_feature_extractor.state_dict()
        assert tuple(expected) == tuple(original[0])
        for name in original[0]:
            assert torch.equal(expected[name], original[0][name])
    finally:
        torch.set_rng_state(random_state)
