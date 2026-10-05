"""旧モデルの構築順・全state・forwardを学習なしで直接照合する。"""

import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.fedsda import _AdaHedgeRoutingFedSDAClientMixin
from federated_drift_experiment.data.specs import DatasetSpec
from federated_drift_experiment.models import ResidualAdapterMLP, SharedFeatureBackbone
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.nonlinear_residual_adapter import (
    NonlinearResidualAdapter,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    compute_model_mean_bounded_losses_after_label_observation,
    convert_model_outputs_to_prediction_probabilities,
    predict_class_labels_from_prediction_scores,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import (
    select_candidate_initial_parameter_snapshot,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)


def build_legacy_residual_adapter_classifier(
    *,
    input_feature_count,
    hidden_layer_widths,
    class_count,
    requested_rank,
    monkeypatch,
    shared_feature_extractor=None,
):
    monkeypatch.setattr(
        config,
        "dataset_spec",
        lambda dataset: DatasetSpec(
            input_dim=input_feature_count,
            num_concepts=2,
            num_classes=class_count,
            hidden_dims=hidden_layer_widths,
        ),
    )
    monkeypatch.setattr(config, "SHARED_ADAPTER_RANK", requested_rank)
    monkeypatch.setattr(
        ResidualAdapterMLP, "_build_component_optimizer", staticmethod(lambda parameters, lr: None)
    )
    return ResidualAdapterMLP(
        input_dim=input_feature_count, dataset="sine2", backbone=shared_feature_extractor
    )


def map_residual_adapter_state_to_legacy_keys(*, state):
    return {
        name.replace("feature_extractor.hidden_layers.", "backbone.net.")
        .replace("residual_adapter.feature_compression.", "adapter.down.")
        .replace("residual_adapter.feature_expansion.", "adapter.up.")
        .replace("classification_layer.", "head."): value
        for name, value in state.items()
    }


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


@pytest.mark.parametrize(
    "input_feature_count,hidden_layer_widths,class_count,requested_rank",
    [
        (2, (32, 32), 2, 8),
        (3, (5, 4), 4, 2),
        (2, (), 2, 9),
        (3, (), 3, 1),
        (1, (1,), 2, 1),
    ],
)
@pytest.mark.parametrize("shared", [False, True])
def test_residual_adapter_classifier_matches_legacy(
    input_feature_count,
    hidden_layer_widths,
    class_count,
    requested_rank,
    shared,
    monkeypatch,
):
    random_state = torch.Generator(device="cpu").manual_seed(41).get_state()
    shared_feature_extractor = None
    legacy_feature_extractor = None
    if shared:
        shared_feature_extractor, random_state = (
            capture_torch_random_state_after_model_construction(
                constructor=lambda: SharedFeatureExtractor(
                    input_feature_count=input_feature_count, hidden_layer_widths=hidden_layer_widths
                ),
                random_state=random_state,
            )
        )
        legacy_feature_extractor, expected_random_state = (
            capture_torch_random_state_after_model_construction(
                constructor=lambda: SharedFeatureBackbone(input_feature_count, hidden_layer_widths),
                random_state=torch.Generator(device="cpu").manual_seed(41).get_state(),
            )
        )
        assert torch.equal(random_state, expected_random_state)
    actual, actual_random_state = capture_torch_random_state_after_model_construction(
        constructor=lambda: ResidualAdapterClassifier(
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=requested_rank,
            ),
            input_feature_count=input_feature_count,
            hidden_layer_widths=hidden_layer_widths,
            class_count=class_count,
            shared_feature_extractor=shared_feature_extractor,
        ),
        random_state=random_state,
    )
    expected, expected_random_state = capture_torch_random_state_after_model_construction(
        constructor=lambda: build_legacy_residual_adapter_classifier(
            input_feature_count=input_feature_count,
            hidden_layer_widths=hidden_layer_widths,
            class_count=class_count,
            requested_rank=requested_rank,
            monkeypatch=monkeypatch,
            shared_feature_extractor=legacy_feature_extractor,
        ),
        random_state=random_state,
    )
    assert torch.equal(actual_random_state, expected_random_state)
    assert_residual_adapter_state_matches_legacy(actual=actual, expected=expected)
    assert actual.residual_adapter.effective_rank == min(
        requested_rank, hidden_layer_widths[-1] if hidden_layer_widths else input_feature_count
    )
    for input_features in (
        torch.arange(input_feature_count, dtype=torch.float32, device="cpu"),
        torch.arange(3 * input_feature_count, dtype=torch.float32, device="cpu").reshape(3, -1),
    ):
        original = torch.get_rng_state()
        assert torch.equal(actual(input_features), expected(input_features))
        shared_features = actual.extract_shared_features(input_features)
        assert torch.equal(
            actual.forward_from_shared_features(shared_features), actual(input_features)
        )
        assert torch.equal(actual.residual_adapter(shared_features), shared_features)
        assert torch.equal(torch.get_rng_state(), original)
    if shared:
        assert actual.feature_extractor is shared_feature_extractor


def test_residual_adapter_classifier_shares_only_feature_extractor():
    shared_feature_extractor = SharedFeatureExtractor(
        input_feature_count=2, hidden_layer_widths=(4,)
    )
    settings = ModelArchitectureSettings(
        model_architecture_name="shared_backbone_residual_adapter",
        residual_adapter_requested_rank=2,
    )
    first = ResidualAdapterClassifier(
        model_architecture_settings=settings,
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=2,
        shared_feature_extractor=shared_feature_extractor,
    )
    second = ResidualAdapterClassifier(
        model_architecture_settings=settings,
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=2,
        shared_feature_extractor=shared_feature_extractor,
    )
    assert first.feature_extractor is second.feature_extractor is shared_feature_extractor
    assert first.residual_adapter is not second.residual_adapter
    assert first.classification_layer is not second.classification_layer
    for parameter, expected in zip(
        first.residual_adapter.parameters(), second.residual_adapter.parameters()
    ):
        assert parameter.data_ptr() != expected.data_ptr()
    assert (
        first.classification_layer.weight.data_ptr()
        != second.classification_layer.weight.data_ptr()
    )
    assert first.classification_layer.bias.data_ptr() != second.classification_layer.bias.data_ptr()
    for parameter, expected in zip(
        first.feature_extractor.parameters(), second.feature_extractor.parameters()
    ):
        assert parameter is expected
        assert parameter.data_ptr() == expected.data_ptr()


@pytest.mark.parametrize("name", ["weight", "bias"])
def test_residual_adapter_model_rejects_invalid_inputs_without_consuming_randomness_unregistered_parameter(
    name,
):
    shared_feature_extractor = SharedFeatureExtractor(
        input_feature_count=2, hidden_layer_widths=(4,)
    )
    linear_layer = shared_feature_extractor.hidden_layers[0]
    parameter = getattr(linear_layer, name).detach().clone()
    delattr(linear_layer, name)
    setattr(linear_layer, name, parameter)
    original = (
        {
            name: value.detach().clone()
            for name, value in shared_feature_extractor.state_dict().items()
        },
        parameter.detach().clone(),
    )
    random_state = torch.get_rng_state()
    try:
        with pytest.raises(ValueError, match="shared_feature_extractor"):
            ResidualAdapterClassifier(
                model_architecture_settings=ModelArchitectureSettings(
                    model_architecture_name="shared_backbone_residual_adapter",
                    residual_adapter_requested_rank=2,
                ),
                input_feature_count=2,
                hidden_layer_widths=(4,),
                class_count=2,
                shared_feature_extractor=shared_feature_extractor,
            )
        assert torch.equal(torch.get_rng_state(), random_state)
        assert getattr(linear_layer, name) is parameter
        assert torch.equal(parameter, original[1])
        expected = shared_feature_extractor.state_dict()
        assert tuple(expected) == tuple(original[0])
        for name in original[0]:
            assert torch.equal(expected[name], original[0][name])
    finally:
        torch.set_rng_state(random_state)


@pytest.mark.parametrize(
    "name,invalid_value",
    [
        ("input_feature_count", True),
        ("input_feature_count", 0),
        ("hidden_layer_widths", [4]),
        ("hidden_layer_widths", (4, False)),
        ("class_count", True),
        ("class_count", 1),
        ("model_architecture_settings", object()),
        ("shared_feature_extractor", torch.nn.Identity()),
    ],
)
def test_residual_adapter_model_rejects_invalid_inputs_without_consuming_randomness_constructor(
    name, invalid_value
):
    constructor_arguments = dict(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=2,
    )
    constructor_arguments[name] = invalid_value
    random_state = torch.get_rng_state()
    with pytest.raises(ValueError, match=name):
        ResidualAdapterClassifier(**constructor_arguments)
    assert torch.equal(torch.get_rng_state(), random_state)


@pytest.mark.parametrize(
    "name,invalid_value",
    [
        ("model_architecture_name", "unknown"),
        ("residual_adapter_requested_rank", False),
        ("residual_adapter_requested_rank", 0),
    ],
)
def test_residual_adapter_model_rejects_invalid_inputs_without_consuming_randomness_forged_settings(
    name, invalid_value
):
    settings = ModelArchitectureSettings(
        model_architecture_name="shared_backbone_residual_adapter",
        residual_adapter_requested_rank=2,
    )
    object.__setattr__(settings, name, invalid_value)
    random_state = torch.get_rng_state()
    with pytest.raises(ValueError, match=name):
        ResidualAdapterClassifier(
            model_architecture_settings=settings,
            input_feature_count=2,
            hidden_layer_widths=(4,),
            class_count=2,
        )
    assert getattr(settings, name) == invalid_value
    assert torch.equal(torch.get_rng_state(), random_state)


@pytest.mark.parametrize(
    "case", ["declared_width", "layer_count", "activation", "weight_shape", "dtype", "bias"]
)
def test_residual_adapter_model_rejects_invalid_inputs_without_consuming_randomness_shared_structure(
    case,
):
    shared_feature_extractor = SharedFeatureExtractor(
        input_feature_count=2, hidden_layer_widths=(4,)
    )
    if case == "declared_width":
        shared_feature_extractor.output_feature_count = 3
    elif case == "layer_count":
        shared_feature_extractor.hidden_layers.append(torch.nn.Identity())
    elif case == "activation":
        shared_feature_extractor.hidden_layers[1] = torch.nn.Identity()
    elif case == "weight_shape":
        shared_feature_extractor.hidden_layers[0].weight = torch.nn.Parameter(torch.ones(3, 2))
    elif case == "dtype":
        shared_feature_extractor.double()
    else:
        shared_feature_extractor.hidden_layers[0].bias = None
    parameter_snapshot_before_call = {
        name: value.detach().clone()
        for name, value in shared_feature_extractor.state_dict().items()
    }
    original = tuple(shared_feature_extractor.modules())
    random_state = torch.get_rng_state()
    with pytest.raises(ValueError, match="shared_feature_extractor"):
        ResidualAdapterClassifier(
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=2,
            ),
            input_feature_count=2,
            hidden_layer_widths=(4,),
            class_count=2,
            shared_feature_extractor=shared_feature_extractor,
        )
    assert tuple(shared_feature_extractor.modules()) == original
    assert torch.equal(torch.get_rng_state(), random_state)
    for name, expected in parameter_snapshot_before_call.items():
        assert torch.equal(shared_feature_extractor.state_dict()[name], expected)


@pytest.mark.parametrize(
    "case", ["dtype", "width", "scalar", "rank_three", "device", "layout", "type"]
)
def test_residual_adapter_model_rejects_invalid_inputs_without_consuming_randomness_forward(case):
    classifier = ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=1,
        ),
        input_feature_count=2,
        hidden_layer_widths=(),
        class_count=2,
    )
    input_features = {
        "dtype": torch.ones(2, dtype=torch.float64),
        "width": torch.ones(3),
        "scalar": torch.tensor(1.0),
        "rank_three": torch.ones(1, 1, 2),
        "device": torch.ones(2, device="meta"),
        "layout": torch.ones(1, 2).to_sparse(),
        "type": [1.0, 2.0],
    }[case]
    parameter_snapshot_before_call = {
        name: value.detach().clone() for name, value in classifier.state_dict().items()
    }
    input_features_before_call = (
        input_features.clone() if isinstance(input_features, torch.Tensor) else list(input_features)
    )
    random_state = torch.get_rng_state()
    for constructor in (
        classifier,
        classifier.extract_shared_features,
        classifier.forward_from_shared_features,
        classifier.residual_adapter,
    ):
        with pytest.raises(ValueError):
            constructor(input_features)
    assert torch.equal(torch.get_rng_state(), random_state)
    for name, expected in parameter_snapshot_before_call.items():
        assert torch.equal(classifier.state_dict()[name], expected)
    if case == "device":
        assert input_features.device == input_features_before_call.device
        assert input_features.shape == input_features_before_call.shape
    elif case == "layout":
        assert torch.equal(input_features.to_dense(), input_features_before_call.to_dense())
    elif case == "type":
        assert input_features == input_features_before_call
    else:
        assert torch.equal(input_features, input_features_before_call)


@pytest.mark.parametrize("class_count", [2, 3])
@pytest.mark.parametrize("use_nonzero_expansion_weights", [False, True])
def test_residual_adapter_model_gradients_match_legacy(
    class_count, use_nonzero_expansion_weights, monkeypatch
):
    classifier = ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=class_count,
    )
    legacy_classifier = build_legacy_residual_adapter_classifier(
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=class_count,
        requested_rank=2,
        monkeypatch=monkeypatch,
    )
    # ReLUを有効にして、展開が非zeroなら圧縮層にも勾配が流れる標本を使う。
    with torch.no_grad():
        for parameter in classifier.parameters():
            parameter.fill_(0.1)
        if not use_nonzero_expansion_weights:
            classifier.residual_adapter.feature_expansion.weight.zero_()
            classifier.residual_adapter.feature_expansion.bias.zero_()
    legacy_classifier.load_state_dict(
        map_residual_adapter_state_to_legacy_keys(state=classifier.state_dict())
    )
    input_features = torch.tensor([[0.0, 0.0], [1.0, 2.0]], requires_grad=True)
    legacy_input_features = input_features.detach().clone().requires_grad_()
    parameter_snapshot_before_call = {
        name: value.detach().clone() for name, value in classifier.state_dict().items()
    }
    random_state = torch.get_rng_state()
    actual = classifier(input_features)
    expected = legacy_classifier(legacy_input_features)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    assert torch.equal(input_features.grad, legacy_input_features.grad)
    assert torch.count_nonzero(input_features.grad) > 0
    parameter_gradients_by_name = map_residual_adapter_state_to_legacy_keys(
        state={name: parameter.grad for name, parameter in classifier.named_parameters()}
    )
    legacy_parameter_gradients_by_name = {
        name: parameter.grad for name, parameter in legacy_classifier.named_parameters()
    }
    assert tuple(parameter_gradients_by_name) == tuple(legacy_parameter_gradients_by_name)
    for name, actual in parameter_gradients_by_name.items():
        assert actual is not None
        assert torch.equal(actual, legacy_parameter_gradients_by_name[name])
    assert (
        bool(torch.count_nonzero(classifier.residual_adapter.feature_compression.weight.grad))
        == use_nonzero_expansion_weights
    )
    for name, expected in parameter_snapshot_before_call.items():
        assert torch.equal(classifier.state_dict()[name], expected)
    assert torch.equal(torch.get_rng_state(), random_state)


@pytest.mark.parametrize("class_count", [2, 3])
def test_residual_adapter_model_accepts_empty_batches(class_count, monkeypatch):
    classifier = ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=class_count,
    )
    legacy_classifier = build_legacy_residual_adapter_classifier(
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=class_count,
        requested_rank=2,
        monkeypatch=monkeypatch,
    )
    legacy_classifier.load_state_dict(
        map_residual_adapter_state_to_legacy_keys(state=classifier.state_dict())
    )
    input_features = torch.empty(0, 2)
    shared_features = classifier.extract_shared_features(input_features)
    assert shared_features.shape == (0, 4)
    assert classifier.residual_adapter(shared_features).shape == (0, 4)
    actual = classifier(input_features)
    assert actual.shape == (0, 1 if class_count == 2 else class_count)
    assert torch.equal(actual, legacy_classifier(input_features))
    assert torch.equal(actual, classifier.forward_from_shared_features(shared_features))


def test_residual_adapter_classifier_reuses_extracted_shared_features(monkeypatch):
    classifier = ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=2,
    )
    monkeypatch.setattr(
        classifier.feature_extractor, "forward", Mock(wraps=classifier.feature_extractor.forward)
    )
    input_features = torch.ones(3, 2)
    expected = classifier(input_features)
    assert classifier.feature_extractor.forward.call_count == 1
    shared_features = classifier.extract_shared_features(input_features)
    shared_features_before_call = shared_features.detach().clone()
    actual = classifier.forward_from_shared_features(shared_features)
    assert classifier.feature_extractor.forward.call_count == 2
    assert torch.equal(actual, expected)
    assert torch.equal(shared_features, shared_features_before_call)


@pytest.mark.parametrize("name,invalid_value", [("feature_count", False), ("requested_rank", 0)])
def test_residual_adapter_model_rejects_invalid_inputs_without_consuming_randomness_adapter(
    name, invalid_value
):
    constructor_arguments = dict(feature_count=4, requested_rank=2)
    constructor_arguments[name] = invalid_value
    random_state = torch.get_rng_state()
    with pytest.raises(ValueError, match=name):
        NonlinearResidualAdapter(**constructor_arguments)
    assert torch.equal(torch.get_rng_state(), random_state)


@pytest.mark.parametrize("case", [True, False])
def test_residual_adapter_model_preserves_default_tensor_environment(case):
    input_features = torch.ones(2, 2, dtype=torch.float32, device="cpu")
    settings = ModelArchitectureSettings(
        model_architecture_name="shared_backbone_residual_adapter",
        residual_adapter_requested_rank=2,
    )
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    original = (torch.get_default_dtype(), torch.get_default_device(), torch.is_grad_enabled())
    random_state = torch.get_rng_state()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"), torch.set_grad_enabled(case):
            classifier, actual_random_state = capture_torch_random_state_after_model_construction(
                constructor=lambda: ResidualAdapterClassifier(
                    model_architecture_settings=settings,
                    input_feature_count=2,
                    hidden_layer_widths=(4,),
                    class_count=2,
                ),
                random_state=random_state,
            )
            assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_device() == torch.device("meta")
            assert torch.is_grad_enabled() is case
            for parameter in classifier.parameters():
                assert parameter.device == torch.device("cpu")
                assert parameter.dtype == torch.float32
            actual = classifier(input_features)
            assert actual.device == torch.device("cpu")
            assert actual.dtype == torch.float32
            assert actual.requires_grad is case
            assert torch.equal(torch.get_rng_state(), random_state)
            assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_device() == torch.device("meta")
            assert torch.is_grad_enabled() is case
        repeated_classifier, expected_random_state = (
            capture_torch_random_state_after_model_construction(
                constructor=lambda: ResidualAdapterClassifier(
                    model_architecture_settings=settings,
                    input_feature_count=2,
                    hidden_layer_widths=(4,),
                    class_count=2,
                ),
                random_state=random_state,
            )
        )
        assert torch.equal(actual_random_state, expected_random_state)
        for name, expected in repeated_classifier.state_dict().items():
            assert torch.equal(classifier.state_dict()[name], expected)
        assert torch.equal(actual, repeated_classifier(input_features))
        assert random.getstate() == global_python_random_state
        actual = np.random.get_state()
        assert actual[0] == global_numpy_random_state[0]
        assert np.array_equal(actual[1], global_numpy_random_state[1])
        assert actual[2:] == global_numpy_random_state[2:]
    finally:
        torch.set_default_dtype(original[0])
        torch.set_rng_state(random_state)
    assert torch.get_default_dtype() == original[0]
    assert torch.get_default_device() == original[1]
    assert torch.is_grad_enabled() == original[2]


@pytest.mark.parametrize("class_count", [2, 3])
def test_residual_adapter_classifier_connects_to_probability_calculations(class_count, monkeypatch):
    classifier = ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=class_count,
    )
    legacy_classifier = build_legacy_residual_adapter_classifier(
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=class_count,
        requested_rank=2,
        monkeypatch=monkeypatch,
    )
    legacy_classifier.load_state_dict(
        map_residual_adapter_state_to_legacy_keys(state=classifier.state_dict())
    )
    input_features = torch.tensor([[0.0, 1.0], [2.0, -1.0], [-2.0, 3.0]])
    model_outputs_by_model_id = {-7: classifier(input_features)}
    prediction_probabilities_by_model_id = convert_model_outputs_to_prediction_probabilities(
        model_outputs_by_model_id=model_outputs_by_model_id, class_count=class_count
    )
    expected = legacy_classifier(input_features)
    if class_count > 2:
        expected = torch.softmax(expected, dim=1)
    assert torch.equal(prediction_probabilities_by_model_id[-7], expected)
    assert not prediction_probabilities_by_model_id[-7].requires_grad
    assert torch.equal(
        predict_class_labels_from_prediction_scores(
            prediction_scores=prediction_probabilities_by_model_id[-7], class_count=class_count
        ),
        _AdaHedgeRoutingFedSDAClientMixin._routing_prediction(expected, class_count),
    )
    observed_class_labels = torch.tensor([[0.0], [1.0], [float(class_count - 1)]])
    actual = compute_model_mean_bounded_losses_after_label_observation(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        observed_class_labels=observed_class_labels,
        class_count=class_count,
    )
    assert actual == {
        -7: _AdaHedgeRoutingFedSDAClientMixin._routing_score_loss(
            expected, observed_class_labels, class_count
        )
    }


@pytest.mark.parametrize(
    "candidate_parameter_initialization_source",
    [
        "assigned_training_model",
        "lowest_evaluated_mean_loss_model",
        "equal_mean_of_available_models",
    ],
)
def test_residual_adapter_classifier_loads_selected_initial_parameters(
    candidate_parameter_initialization_source, monkeypatch
):
    settings = ModelArchitectureSettings(
        model_architecture_name="shared_backbone_residual_adapter",
        residual_adapter_requested_rank=2,
    )
    source_classifiers_by_model_id = {
        model_id: ResidualAdapterClassifier(
            model_architecture_settings=settings,
            input_feature_count=2,
            hidden_layer_widths=(4,),
            class_count=3,
        )
        for model_id in (-7, 4)
    }
    available_parameter_snapshots_by_model_id = {
        model_id: {name: value.detach().clone() for name, value in classifier.state_dict().items()}
        for model_id, classifier in source_classifiers_by_model_id.items()
    }
    actual = select_candidate_initial_parameter_snapshot(
        settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source=candidate_parameter_initialization_source
        ),
        available_parameter_snapshots_by_model_id=available_parameter_snapshots_by_model_id,
        current_training_model_id=-7,
        evaluated_mean_losses_by_model_id=((-7, 0.8), (4, 0.2)),
    )
    initialized_classifier = ResidualAdapterClassifier(
        model_architecture_settings=settings,
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=3,
    )
    initialized_classifier.load_state_dict(actual)
    legacy_classifier = build_legacy_residual_adapter_classifier(
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=3,
        requested_rank=2,
        monkeypatch=monkeypatch,
    )
    legacy_classifier.load_state_dict(map_residual_adapter_state_to_legacy_keys(state=actual))
    assert_residual_adapter_state_matches_legacy(
        actual=initialized_classifier, expected=legacy_classifier
    )
    input_features = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    assert torch.equal(initialized_classifier(input_features), legacy_classifier(input_features))
    for name, expected in actual.items():
        assert torch.equal(initialized_classifier.state_dict()[name], expected)
        assert initialized_classifier.state_dict()[name].data_ptr() != expected.data_ptr()
        assert not expected.requires_grad
        for model_id, classifier in source_classifiers_by_model_id.items():
            assert expected.data_ptr() != classifier.state_dict()[name].data_ptr()
            assert (
                expected.data_ptr()
                != available_parameter_snapshots_by_model_id[model_id][name].data_ptr()
            )
    # load先の更新は供給側モデルと選択snapshotへ逆流しない。
    with torch.no_grad():
        initialized_classifier.classification_layer.weight.add_(1.0)
    assert not torch.equal(
        initialized_classifier.classification_layer.weight, actual["classification_layer.weight"]
    )
    for model_id, classifier in source_classifiers_by_model_id.items():
        for name, expected in available_parameter_snapshots_by_model_id[model_id].items():
            assert torch.equal(classifier.state_dict()[name], expected)
