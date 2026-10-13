"""FedDriftの検出バッチの判定を、実旧のメソッドと照合する。"""

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.feddrift import FedDriftClient
from federated_drift_experiment.models import SimpleMLP
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)


def build_models(*, class_count, monkeypatch):
    """実旧と新の分類器を、同じパラメータから用意する。"""
    monkeypatch.setattr(
        config,
        "dataset_spec",
        lambda dataset: SimpleNamespace(
            input_dim=3, num_classes=class_count, hidden_dims=(6, 4), learning_rate=0.01
        ),
    )
    legacy_models = {}
    held_models = []
    with torch.random.fork_rng():
        for model_id, seed in ((9, 31), (-104, 73), (2, 97)):
            torch.manual_seed(seed)
            legacy = SimpleMLP(dataset="sine2")
            classifier = ModelArchitectureClassifier(
                model_architecture_settings=ModelArchitectureSettings(
                    model_architecture_name="independent_models_without_shared_backbone",
                    residual_adapter_requested_rank=8,
                ),
                input_feature_count=3,
                hidden_layer_widths=(6, 4),
                class_count=class_count,
            )
            for new_parameter, old_parameter in zip(
                classifier.parameters(), legacy.parameters(), strict=True
            ):
                new_parameter.data.copy_(old_parameter.data)
            legacy_models[model_id] = legacy
            held_models.append((model_id, classifier))
    return legacy_models, tuple(held_models)


def run_legacy_assessment(*, models, features, labels, previous_loss, threshold, monkeypatch):
    """旧の数値評価をそのまま通し、判定後の処理だけを代役にする。"""
    client = FedDriftClient.__new__(FedDriftClient)
    client.models = models
    client.current_model_id = next(iter(models))
    client.last_min_loss = previous_loss
    client.distance_threshold = threshold
    client.processed_samples = len(features) + 7
    client.verbose = False
    client.detected_event_positions = []
    client.local_switch_positions = []
    client.train_data_store = defaultdict(list)
    calls = []
    client._record_model_compute = lambda phase, count: calls.append((phase, count))
    client._spawn_new_model = lambda *args, **kwargs: (-100, 0.15)
    client._absorb_into_store = lambda *args: None
    client._store_evaluation_data = lambda *args: None
    observed_means = []
    original_mean = torch.mean

    def observe_mean(*args, **kwargs):
        mean = original_mean(*args, **kwargs)
        observed_means.append(float(mean.item()))
        return mean

    batch = [(x.unsqueeze(0), y.unsqueeze(0)) for x, y in zip(features, labels, strict=True)]
    with monkeypatch.context() as patch:
        patch.setattr(torch, "mean", observe_mean)
        drift_type = client._detect_and_assign(batch)
    return client, drift_type, tuple(zip(models, observed_means, strict=True)), calls


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("sample_count", (1, 3, 32))
@pytest.mark.parametrize("comparison", ("first", "below", "equal", "above"))
def test_assessment_matches_real_legacy(class_count, sample_count, comparison, monkeypatch):
    from federated_learning_experiments.methods.feddrift.detection_batch_assessment import (
        assess_feddrift_detection_batch,
    )

    legacy_models, held_models = build_models(class_count=class_count, monkeypatch=monkeypatch)
    features = torch.linspace(-1, 1, sample_count * 3).reshape(sample_count, 3)
    labels = (torch.arange(sample_count) % class_count).float().reshape(-1, 1)
    baseline, _, baseline_means, _ = run_legacy_assessment(
        models=legacy_models,
        features=features,
        labels=labels,
        previous_loss=None,
        threshold=0.0,
        monkeypatch=monkeypatch,
    )
    minimum = baseline.last_min_loss
    previous = {
        "first": None,
        "below": minimum,
        "equal": minimum,
        "above": 0.0,
    }[comparison]
    threshold = {"first": 0.1, "below": 0.1, "equal": 0.0, "above": 0.0}[comparison]
    _, drift_type, means, calls = run_legacy_assessment(
        models=legacy_models,
        features=features,
        labels=labels,
        previous_loss=previous,
        threshold=threshold,
        monkeypatch=monkeypatch,
    )
    result = assess_feddrift_detection_batch(
        held_models=held_models,
        input_features=features,
        observed_class_labels=labels,
        previous_minimum_mean_bounded_loss=previous,
        loss_increase_threshold=threshold,
    )
    assert result.model_mean_bounded_losses == means == baseline_means
    assert result.minimum_mean_bounded_loss == minimum
    assert result.minimum_loss_model_id == baseline.current_model_id
    assert result.drift_detected == (drift_type == 2)
    assert result.batch_sample_count == sample_count
    assert calls == [("detection", sample_count)] * len(held_models)


def test_equal_losses_keep_first_held_model_and_preserve_state(monkeypatch):
    from federated_learning_experiments.methods.feddrift.detection_batch_assessment import (
        assess_feddrift_detection_batch,
    )

    legacy_models, held_models = build_models(class_count=4, monkeypatch=monkeypatch)
    for legacy in legacy_models.values():
        legacy.load_state_dict(next(iter(legacy_models.values())).state_dict())
    first_state = held_models[0][1].state_dict()
    for _, classifier in held_models:
        classifier.load_state_dict(first_state)
    features = torch.ones(3, 3)
    labels = torch.zeros(3, 1)
    for i, (_, classifier) in enumerate(held_models):
        classifier.train(i % 2 == 0)
        for parameter in classifier.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
    before = tuple(
        (
            deepcopy(model.state_dict()),
            model.training,
            tuple(p.grad.clone() for p in model.parameters()),
        )
        for _, model in held_models
    )
    torch_state, python_state = torch.random.get_rng_state().clone(), random.getstate()
    result = assess_feddrift_detection_batch(
        held_models=held_models,
        input_features=features,
        observed_class_labels=labels,
        previous_minimum_mean_bounded_loss=None,
        loss_increase_threshold=0.0,
    )
    baseline, _, means, _ = run_legacy_assessment(
        models=legacy_models,
        features=features,
        labels=labels,
        previous_loss=None,
        threshold=0.0,
        monkeypatch=monkeypatch,
    )
    assert result.minimum_loss_model_id == baseline.current_model_id == 9
    assert result.model_mean_bounded_losses == means
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    for (_, model), (parameters, training, gradients) in zip(held_models, before, strict=True):
        assert model.training == training
        for name, value in model.state_dict().items():
            assert torch.equal(value, parameters[name])
        for parameter, gradient in zip(model.parameters(), gradients, strict=True):
            assert torch.equal(parameter.grad, gradient)


@pytest.mark.parametrize(
    "invalid_case",
    (
        "empty_models",
        "list_models",
        "duplicate_id",
        "bool_id",
        "wrong_classifier",
        "wrong_architecture",
        "different_classes",
        "different_width",
        "empty_batch",
        "wrong_features",
        "nan_features",
        "wrong_label_shape",
        "fractional_labels",
        "out_of_range_labels",
        "nan_previous",
        "negative_previous",
        "large_previous",
        "bool_previous",
        "int_previous",
        "negative_threshold",
        "nan_threshold",
        "inf_threshold",
        "bool_threshold",
        "int_threshold",
    ),
)
def test_invalid_inputs_are_rejected_before_any_forward(invalid_case, monkeypatch):
    from federated_learning_experiments.methods.feddrift.detection_batch_assessment import (
        assess_feddrift_detection_batch,
    )

    _, held_models = build_models(class_count=2, monkeypatch=monkeypatch)
    arguments = dict(
        held_models=held_models,
        input_features=torch.ones(3, 3),
        observed_class_labels=torch.zeros(3, 1),
        previous_minimum_mean_bounded_loss=0.2,
        loss_increase_threshold=0.1,
    )
    if invalid_case == "empty_models":
        arguments["held_models"] = ()
    elif invalid_case == "list_models":
        arguments["held_models"] = list(held_models)
    elif invalid_case in ("duplicate_id", "bool_id", "wrong_classifier"):
        arguments["held_models"] = held_models[:2] + (
            (
                held_models[0][0]
                if invalid_case == "duplicate_id"
                else True
                if invalid_case == "bool_id"
                else 22,
                object() if invalid_case == "wrong_classifier" else held_models[-1][1],
            ),
        )
    elif invalid_case == "wrong_architecture":
        held_models[-1][1].model_architecture_settings = ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=8,
        )
    elif invalid_case == "different_classes":
        held_models[-1][1].class_count = 4
    elif invalid_case == "different_width":
        held_models[-1][1].input_feature_count = 4
    elif invalid_case == "empty_batch":
        arguments["input_features"] = torch.zeros(0, 3)
        arguments["observed_class_labels"] = torch.zeros(0, 1)
    elif invalid_case == "wrong_features":
        arguments["input_features"] = torch.ones(3, 4)
    elif invalid_case == "nan_features":
        arguments["input_features"][0, 0] = float("nan")
    elif invalid_case == "wrong_label_shape":
        arguments["observed_class_labels"] = torch.zeros(3)
    elif invalid_case in ("fractional_labels", "out_of_range_labels"):
        arguments["observed_class_labels"][0, 0] = (
            0.5 if invalid_case == "fractional_labels" else 2.0
        )
    elif invalid_case.endswith("previous"):
        arguments["previous_minimum_mean_bounded_loss"] = {
            "nan_previous": float("nan"),
            "negative_previous": -0.1,
            "large_previous": 1.1,
            "bool_previous": True,
            "int_previous": 0,
        }[invalid_case]
    else:
        arguments["loss_increase_threshold"] = {
            "negative_threshold": -0.1,
            "nan_threshold": float("nan"),
            "inf_threshold": float("inf"),
            "bool_threshold": True,
            "int_threshold": 0,
        }[invalid_case]
    calls = []
    for _, model in held_models:
        monkeypatch.setattr(model, "forward", lambda *args: calls.append(True))
    with pytest.raises((TypeError, ValueError)):
        assess_feddrift_detection_batch(**arguments)
    assert calls == []


@pytest.mark.parametrize(
    "field,value",
    (
        ("minimum_loss_model_id", True),
        ("minimum_loss_model_id", 2),
        ("minimum_mean_bounded_loss", float("nan")),
        ("minimum_mean_bounded_loss", 0.4),
        ("drift_detected", 1),
        ("batch_sample_count", 0),
        ("batch_sample_count", True),
        ("model_mean_bounded_losses", ()),
        ("model_mean_bounded_losses", ((9, 0.2), (9, 0.3))),
        ("model_mean_bounded_losses", ((9, -0.1),)),
        ("model_mean_bounded_losses", ((9, 1.1),)),
        ("model_mean_bounded_losses", ((True, 0.2),)),
        ("model_mean_bounded_losses", [(9, 0.2)]),
    ),
)
def test_result_rejects_inconsistent_values(field, value):
    from federated_learning_experiments.methods.feddrift.detection_batch_assessment import (
        FeddriftDetectionBatchAssessment,
    )

    result = FeddriftDetectionBatchAssessment(
        model_mean_bounded_losses=((9, 0.2), (2, 0.2)),
        minimum_loss_model_id=9,
        minimum_mean_bounded_loss=0.2,
        drift_detected=False,
        batch_sample_count=3,
    )
    with pytest.raises((TypeError, ValueError)):
        replace(result, **{field: value})
