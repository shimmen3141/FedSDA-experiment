"""準備済みモデルの有界損失・副作用と旧実装の対応を検証する。"""

import random
import warnings
from copy import deepcopy

import numpy as np
import pytest
import torch
from test_joint_model_parameter_update import build_joint_update_oracle_pair
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    build_attachment_classifier,
    snapshot_parameter_values_and_gradients,
)

from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)


def build_bounded_loss_oracle_pair(*, class_count, sample_count, monkeypatch):
    training_batches, _, legacy_client = build_joint_update_oracle_pair(
        class_count=class_count,
        batch_sample_counts=(sample_count,),
        optimizer_variant="standard",
        monkeypatch=monkeypatch,
    )
    legacy_model = legacy_client.models[4]
    classifier = training_batches[0].classifier
    # ゼロ初期化adapterだけの対照にならないよう、実旧展開層から値を対応する。
    with torch.no_grad():
        legacy_model.adapter.up.weight.fill_(0.15)
        legacy_model.adapter.up.bias.fill_(0.05)
    classifier.residual_adapter.feature_expansion.load_state_dict(
        legacy_model.adapter.up.state_dict()
    )
    return classifier, legacy_model, training_batches[0]


@pytest.mark.parametrize("class_count", [2, 4, 10])
@pytest.mark.parametrize("sample_count", [1, 5])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_classifier_bounded_losses_match_legacy(
    class_count, sample_count, noncontiguous, training, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        classifier, legacy_model, training_batch = build_bounded_loss_oracle_pair(
            class_count=class_count, sample_count=sample_count, monkeypatch=monkeypatch
        )
        input_features = training_batch.input_features.flip(0)
        observed_class_labels = training_batch.observed_class_labels.flip(0)
        if noncontiguous:
            input_features = input_features.repeat_interleave(2, dim=1)[:, ::2]
            observed_class_labels = observed_class_labels.repeat_interleave(2, dim=1)[:, ::2]
            if sample_count > 1:
                assert not input_features.is_contiguous()
                assert not observed_class_labels.is_contiguous()
        classifier.train(training)
        legacy_model.train(training)
        # 親flagだけの検査にせず、子の異なるflagも保持する。
        classifier.residual_adapter.eval()
        for parameter in classifier.parameters():
            parameter.grad = torch.ones_like(parameter)
        state_before = snapshot_parameter_values_and_gradients(classifier.parameters())
        input_before = (input_features.clone(), observed_class_labels.clone())
        flags_before = tuple(module.training for module in classifier.modules())
        rng_before = torch.get_rng_state().clone()
        forward_calls = []
        hook = classifier.register_forward_hook(
            lambda module, args, output: forward_calls.append(torch.is_grad_enabled())
        )
        try:
            with torch.no_grad():
                expected = legacy_model.per_sample_error(input_features, observed_class_labels)
            result = evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            )
        finally:
            hook.remove()
        assert torch.equal(result, expected)
        assert result.shape == (sample_count,)
        assert result.dtype == torch.float32 and result.device.type == "cpu"
        assert not result.requires_grad and result.grad_fn is None
        assert forward_calls == [False]
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert flags_before == tuple(module.training for module in classifier.modules())
        assert_parameter_values_and_gradients_unchanged(state_before)
        result.fill_(0)
        assert torch.equal(input_features, input_before[0])
        assert torch.equal(observed_class_labels, input_before[1])
        assert_parameter_values_and_gradients_unchanged(state_before)


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("classifier", None),
        ("classifier", torch.nn.Linear(2, 1)),
        ("input_features", None),
        ("input_features", [[0.1, 0.2]]),
        ("input_features", torch.empty((2, 2), dtype=torch.float64)),
        ("input_features", torch.empty((2, 2), dtype=torch.int64)),
        ("input_features", torch.empty((2, 2), dtype=torch.bool)),
        ("input_features", torch.empty((2, 2), device="meta")),
        ("input_features", torch.ones((2, 2)).to_sparse()),
        ("input_features", torch.empty((0, 2))),
        ("input_features", torch.empty(2)),
        ("input_features", torch.empty((2, 2, 1))),
        ("input_features", torch.empty((2, 3))),
        ("input_features", torch.tensor([[float("nan"), 0.0], [0.0, 0.0]])),
        ("input_features", torch.tensor([[float("inf"), 0.0], [0.0, 0.0]])),
        ("observed_class_labels", None),
        ("observed_class_labels", [0, 1]),
        ("observed_class_labels", torch.empty((2, 1), dtype=torch.float64)),
        ("observed_class_labels", torch.zeros((2, 1), dtype=torch.int64)),
        ("observed_class_labels", torch.zeros((2, 1), dtype=torch.bool)),
        ("observed_class_labels", torch.empty((2, 1), device="meta")),
        ("observed_class_labels", torch.ones((2, 1)).to_sparse()),
        ("observed_class_labels", torch.zeros(2)),
        ("observed_class_labels", torch.zeros((2, 2))),
        ("observed_class_labels", torch.zeros((1, 1))),
        ("observed_class_labels", torch.empty((0, 1))),
        ("observed_class_labels", torch.tensor([[0.5], [1.0]])),
        ("observed_class_labels", torch.tensor([[-1.0], [1.0]])),
        ("observed_class_labels", torch.tensor([[2.0], [1.0]])),
        ("observed_class_labels", torch.tensor([[float("nan")], [1.0]])),
        ("observed_class_labels", torch.tensor([[float("inf")], [1.0]])),
    ],
)
def test_bounded_loss_evaluation_rejects_invalid_inputs_before_forward(field_name, invalid_value):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier()
        inputs = {
            "classifier": classifier,
            "input_features": torch.ones((2, 2)),
            "observed_class_labels": torch.tensor([[0.0], [1.0]]),
        }
        inputs[field_name] = invalid_value
        state_before = snapshot_parameter_values_and_gradients(classifier.parameters())
        rng_before = torch.get_rng_state().clone()
        forward_calls = []
        hook = classifier.register_forward_hook(
            lambda module, args, output: forward_calls.append(output)
        )
        try:
            with pytest.raises((TypeError, ValueError), match=field_name):
                evaluate_classifier_per_sample_bounded_losses(**inputs)
        finally:
            hook.remove()
        assert not forward_calls
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert_parameter_values_and_gradients_unchanged(state_before)


@pytest.mark.parametrize("operation", ["float64", "meta", "invalid_class_count", "nested"])
def test_bounded_loss_evaluation_rejects_invalid_classifier_parameters(operation):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier()
        input_features = torch.ones((2, 2))
        observed_class_labels = torch.zeros((2, 1))
        if operation == "invalid_class_count":
            classifier.class_count = True
        elif operation == "nested":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                input_features = torch.nested.nested_tensor([torch.ones(2), torch.ones(2)])
        else:
            classifier.to(dtype=torch.float64) if operation == "float64" else classifier.to("meta")
        forward_calls = []
        hook = classifier.register_forward_hook(
            lambda module, args, output: forward_calls.append(output)
        )
        try:
            with pytest.raises((TypeError, ValueError)):
                evaluate_classifier_per_sample_bounded_losses(
                    classifier=classifier,
                    input_features=input_features,
                    observed_class_labels=observed_class_labels,
                )
        finally:
            hook.remove()
        assert not forward_calls


@pytest.mark.parametrize(
    "invalid_value",
    [
        [[0.2], [0.3]],
        torch.zeros((2, 1), dtype=torch.float64),
        torch.zeros((2, 1), dtype=torch.int64),
        torch.empty((2, 1), device="meta"),
        torch.ones((2, 1)).to_sparse(),
        torch.zeros(2),
        torch.zeros((1, 1)),
        torch.zeros((2, 2)),
        torch.tensor([[float("nan")], [0.5]]),
        torch.tensor([[float("inf")], [0.5]]),
        torch.tensor([[-0.1], [0.5]]),
        torch.tensor([[1.1], [0.5]]),
    ],
)
def test_bounded_loss_evaluation_rejects_invalid_outputs(invalid_value):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier()
        state_before = snapshot_parameter_values_and_gradients(classifier.parameters())
        rng_before = torch.get_rng_state().clone()
        flags_before = tuple(module.training for module in classifier.modules())
        hook = classifier.register_forward_hook(lambda module, args, output: invalid_value)
        try:
            with torch.enable_grad():
                with pytest.raises((TypeError, ValueError), match="classifier_outputs"):
                    evaluate_classifier_per_sample_bounded_losses(
                        classifier=classifier,
                        input_features=torch.ones((2, 2)),
                        observed_class_labels=torch.zeros((2, 1)),
                    )
                assert torch.is_grad_enabled()
        finally:
            hook.remove()
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert tuple(module.training for module in classifier.modules()) == flags_before
        assert_parameter_values_and_gradients_unchanged(state_before)


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_bounded_loss_evaluation_preserves_environment_and_independent_results(
    class_count, grad_enabled
):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier(class_count=class_count)
        input_features = torch.ones((2, 2), requires_grad=True)
        observed_class_labels = torch.tensor([[0.0], [1.0]], requires_grad=True)
        input_features.grad = torch.ones_like(input_features)
        observed_class_labels.grad = torch.ones_like(observed_class_labels)
        for parameter in classifier.parameters():
            parameter.grad = torch.ones_like(parameter)
        state_before = snapshot_parameter_values_and_gradients(
            tuple(classifier.parameters()) + (input_features, observed_class_labels)
        )
        rng_before = (
            torch.get_rng_state().clone(),
            random.getstate(),
            deepcopy(np.random.get_state()),
        )
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with torch.device("meta"), torch.set_grad_enabled(grad_enabled):
                result = evaluate_classifier_per_sample_bounded_losses(
                    classifier=classifier,
                    input_features=input_features,
                    observed_class_labels=observed_class_labels,
                )
                assert torch.is_grad_enabled() == grad_enabled
                assert torch.empty(0).device.type == "meta"
            assert torch.get_default_dtype() == torch.float64
        finally:
            torch.set_default_dtype(default_dtype)
        assert not result.requires_grad and result.grad_fn is None
        assert torch.equal(torch.get_rng_state(), rng_before[0])
        assert random.getstate() == rng_before[1]
        numpy_state = np.random.get_state()
        assert np.array_equal(numpy_state[1], rng_before[2][1])
        assert (numpy_state[0], *numpy_state[2:]) == (rng_before[2][0], *rng_before[2][2:])
        assert_parameter_values_and_gradients_unchanged(state_before)
        result.fill_(1)
        assert_parameter_values_and_gradients_unchanged(state_before)


def test_classifier_bounded_losses_match_legacy_multiclass_logits():
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier(class_count=4)
        classifier_outputs = torch.tensor([[100.0, -100.0, 2.0, -2.0], [-3.0, 2.0, -1.0, 3.0]])
        hook = classifier.register_forward_hook(lambda module, args, output: classifier_outputs)
        try:
            with torch.inference_mode():
                result = evaluate_classifier_per_sample_bounded_losses(
                    classifier=classifier,
                    input_features=torch.ones((2, 2)),
                    observed_class_labels=torch.tensor([[0.0], [2.0]]),
                )
                assert torch.is_inference_mode_enabled()
        finally:
            hook.remove()
        expected = 1.0 - torch.softmax(classifier_outputs, dim=1)[torch.arange(2), [0, 2]]
        assert torch.equal(result, expected)


def test_bounded_loss_evaluation_rejects_invalid_outputs_from_finite_parameters():
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier(class_count=4)
        with torch.no_grad():
            for parameter in classifier.parameters():
                parameter.fill_(torch.finfo(torch.float32).max)
        state_before = snapshot_parameter_values_and_gradients(classifier.parameters())
        with pytest.raises(ValueError, match="classifier_outputs"):
            evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier,
                input_features=torch.ones((2, 2)),
                observed_class_labels=torch.zeros((2, 1)),
            )
        assert_parameter_values_and_gradients_unchanged(state_before)
