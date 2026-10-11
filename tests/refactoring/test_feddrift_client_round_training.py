"""独立モデルの更新とFedDriftのround学習を、実旧の処理へ照合する。"""

import random
from copy import deepcopy

import pytest
import torch

from federated_learning_experiments.learning.training.independent_model_parameter_update import (
    update_independent_model_parameters,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.runtime.feddrift_client_round_training import (
    train_feddrift_held_models_for_round,
)
from tests.refactoring.test_feddrift_detection_batch_processing import (
    build_processing_oracle,
    make_samples,
)


def assert_values_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_values_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_values_equal(a, b)
    else:
        assert actual == expected


def build_training_pair(monkeypatch, class_count, optimizer_name):
    old, _, owners = build_processing_oracle(
        class_count=class_count, previous_loss=None, threshold=1.0, monkeypatch=monkeypatch
    )
    registry = owners["held_model_training_state_registry"]
    if optimizer_name == "adam":
        settings = AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.0, adam_variant="standard"
        )
        for state in registry.snapshot_ordered_held_model_training_states():
            old.models[state.model_id].optimizer = torch.optim.Adam(
                old.models[state.model_id].parameters(), lr=0.01
            )
            registry.register_held_model_training_state(
                model_id=state.model_id,
                classifier=state.classifier,
                concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                    parameters=tuple(state.classifier.parameters()), optimizer_settings=settings
                ),
            )
    arguments = dict(
        held_model_training_state_registry=registry,
        training_sample_store=owners["training_sample_store"],
        model_training_and_assignment_counts_store=ModelTrainingAndAssignmentCountsStore(),
        python_random_generator=owners["python_random_generator"],
    )
    return old, arguments


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize("batch_size", (1, 2, 3))
def test_single_update_matches_real_legacy_loss_grad_parameters_and_optimizer(
    monkeypatch, class_count, optimizer_name, batch_size
):
    old, arguments = build_training_pair(monkeypatch, class_count, optimizer_name)
    state = arguments["held_model_training_state_registry"].get_held_model_training_state(
        model_id=9
    )
    classifier = state.classifier
    optimizer = state.concept_specific_parameter_optimizer_state.parameter_optimizer
    parameter_refs = tuple(classifier.parameters())
    classifier.eval()
    old.models[9].eval()
    samples = make_samples(sample_count=batch_size, class_count=class_count)
    x = torch.cat([s.input_features for s in samples])
    y = torch.cat([s.observed_class_labels for s in samples])
    rng_before = torch.get_rng_state()
    for _ in range(3):
        expected = old.models[9].update(x, y)
        actual = update_independent_model_parameters(
            classifier=classifier,
            parameter_optimizer=optimizer,
            input_features=x,
            observed_class_labels=y,
        )
        assert actual == expected
        assert all(p is q for p, q in zip(classifier.parameters(), parameter_refs, strict=True))
        for new, legacy in zip(classifier.parameters(), old.models[9].parameters(), strict=True):
            assert torch.equal(new, legacy)
            assert torch.equal(new.grad, legacy.grad)
        assert_values_equal(optimizer.state_dict(), old.models[9].optimizer.state_dict())
        assert not classifier.training
        assert state.concept_specific_parameter_optimizer_state.parameter_optimizer is optimizer
    assert torch.equal(torch.get_rng_state(), rng_before)
    if optimizer_name == "adam":
        assert optimizer.state


def prepare_round(monkeypatch, class_count, optimizer_name, batch_size):
    old, arguments = build_training_pair(monkeypatch, class_count, optimizer_name)
    old.batch_size = batch_size
    old.updates_per_sample = 2
    for mid, count in ((2, 5), (99, 5), (9, 7), (-104, 1)):
        samples = make_samples(sample_count=count, class_count=class_count)
        arguments["training_sample_store"].append_model_training_samples(
            model_id=mid, training_samples=samples
        )
        old.train_data_store[mid].extend(
            (s.input_features, s.observed_class_labels) for s in samples
        )
    old._test_updates = []
    for mid, model in old.models.items():
        original = model.update

        def observed_update(x, y, _mid=mid, _original=original):
            loss = _original(x, y)
            old._test_updates.append((_mid, x.clone(), y.clone(), loss))
            return loss

        monkeypatch.setattr(model, "update", observed_update)
    return old, arguments


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize("batch_size", (1, 3))
@pytest.mark.parametrize("k_steps", (0, 1, 3))
def test_round_matches_real_local_train_across_repeated_rounds(
    monkeypatch, class_count, optimizer_name, batch_size, k_steps
):
    old, arguments = prepare_round(monkeypatch, class_count, optimizer_name, batch_size)
    registry = arguments["held_model_training_state_registry"]
    states = registry.snapshot_ordered_held_model_training_states()
    population_before = arguments["training_sample_store"].snapshot_ordered_model_training_samples()
    refs = tuple(tuple(s.classifier.parameters()) for s in states)
    optimizers = tuple(
        s.concept_specific_parameter_optimizer_state.parameter_optimizer for s in states
    )
    sampled = []
    for state in states:
        state.classifier.register_forward_pre_hook(
            lambda _module, inputs, mid=state.model_id: sampled.append(
                (mid, inputs[0].detach().clone())
            )
        )
    counts = arguments["model_training_and_assignment_counts_store"]
    counts.record_assigned_sample_concept(model_id=9, observed_concept_id=-2)
    torch_before = torch.get_rng_state()
    for _ in range(2):
        before_update_count = len(old._test_updates)
        before_sampled_count = len(sampled)
        python_before = random.getstate()
        try:
            random.setstate(arguments["python_random_generator"].getstate())
            old.local_train(k_steps)
            expected_python_after = random.getstate()
        finally:
            random.setstate(python_before)
        actual_losses = train_feddrift_held_models_for_round(
            **arguments, batch_sample_count=batch_size, updates_per_sample=2, k_steps=k_steps
        )
        expected_updates = old._test_updates[before_update_count:]
        assert actual_losses == tuple(x[3] for x in expected_updates)
        assert tuple(x[0] for x in sampled[before_sampled_count:]) == tuple(
            x[0] for x in expected_updates
        )
        for actual, expected in zip(sampled[before_sampled_count:], expected_updates, strict=True):
            assert torch.equal(actual[1], expected[1])
        assert arguments["python_random_generator"].getstate() == expected_python_after
        for state in states:
            legacy_model = old.models[state.model_id]
            for new, legacy in zip(
                state.classifier.parameters(), legacy_model.parameters(), strict=True
            ):
                assert torch.equal(new, legacy)
                assert_values_equal(new.grad, legacy.grad)
            assert_values_equal(
                state.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
                legacy_model.optimizer.state_dict(),
            )
        snapshot = counts.snapshot_model_training_and_assignment_counts()
        assert snapshot.trained_sample_counts_by_model_id == dict(old.model_training_examples)
        assert snapshot.parameter_update_step_counts_by_model_id == dict(old.model_optimizer_steps)
        assert snapshot.assigned_sample_counts_by_model_and_concept_id == {9: {-2: 1}}
        assert (
            arguments["training_sample_store"].snapshot_ordered_model_training_samples()
            == population_before
        )
    assert torch.equal(torch.get_rng_state(), torch_before)
    for state, parameters, optimizer in zip(states, refs, optimizers, strict=True):
        assert registry.get_held_model_training_state(model_id=state.model_id) is state
        assert all(a is b for a, b in zip(state.classifier.parameters(), parameters, strict=True))
        assert state.concept_specific_parameter_optimizer_state.parameter_optimizer is optimizer


@pytest.mark.parametrize(
    "bad_kind",
    (
        "later_label",
        "later_nan",
        "later_shape",
        "later_frozen",
        "later_optimizer",
        "no_grad",
        "bool_steps",
        "negative_updates",
        "float_batch",
        "wrong_random",
        "wrong_counts",
    ),
)
def test_round_invalid_input_rejected_before_any_sampling_or_update(monkeypatch, bad_kind):
    _, arguments = prepare_round(monkeypatch, 2, "adam", 3)
    registry = arguments["held_model_training_state_registry"]
    later = registry.get_held_model_training_state(model_id=9)
    sample = (
        arguments["training_sample_store"]
        .snapshot_ordered_model_training_samples()[2]
        .training_samples[-1]
    )
    if bad_kind == "later_label":
        sample.observed_class_labels[0, 0] = 0.5
    elif bad_kind == "later_nan":
        sample.input_features[0, 0] = float("nan")
    elif bad_kind == "later_shape":
        sample.input_features.resize_(3)
    elif bad_kind == "later_frozen":
        next(later.classifier.parameters()).requires_grad_(False)
    elif bad_kind == "later_optimizer":
        later.concept_specific_parameter_optimizer_state.parameter_optimizer.param_groups[0][
            "params"
        ].reverse()
    inputs = dict(arguments, batch_sample_count=3, updates_per_sample=2, k_steps=1)
    if bad_kind == "bool_steps":
        inputs["k_steps"] = True
    elif bad_kind == "negative_updates":
        inputs["updates_per_sample"] = -1
    elif bad_kind == "float_batch":
        inputs["batch_sample_count"] = 3.0
    elif bad_kind == "wrong_random":
        inputs["python_random_generator"] = None
    elif bad_kind == "wrong_counts":
        inputs["model_training_and_assignment_counts_store"] = None
    states = registry.snapshot_ordered_held_model_training_states()
    values = tuple(tuple(p.detach().clone() for p in s.classifier.parameters()) for s in states)
    optimizer_states = tuple(
        deepcopy(s.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict())
        for s in states
    )
    python_before = arguments["python_random_generator"].getstate()
    torch_before = torch.get_rng_state()
    calls = []
    for s in states:
        s.classifier.register_forward_pre_hook(lambda *_: calls.append(True))
    with torch.set_grad_enabled(bad_kind != "no_grad"), pytest.raises((TypeError, ValueError)):
        train_feddrift_held_models_for_round(**inputs)
    assert not calls
    assert arguments["python_random_generator"].getstate() == python_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    for state, value, optimizer_state in zip(states, values, optimizer_states, strict=True):
        assert_values_equal(tuple(state.classifier.parameters()), value)
        assert_values_equal(
            state.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
            optimizer_state,
        )
    assert (
        arguments["model_training_and_assignment_counts_store"]
        .snapshot_model_training_and_assignment_counts()
        .parameter_update_step_counts_by_model_id
        == {}
    )


@pytest.mark.parametrize("no_op_kind", ("zero_updates", "no_models", "insufficient"))
def test_no_op_round_skips_payload_and_grad_validation(monkeypatch, no_op_kind):
    _, arguments = prepare_round(monkeypatch, 2, "adam", 3)
    if no_op_kind == "no_models":
        arguments[
            "held_model_training_state_registry"
        ]._held_model_training_states_by_model_id.clear()
    samples = arguments["training_sample_store"].snapshot_ordered_model_training_samples()
    for collection in samples:
        for sample in collection.training_samples:
            sample.input_features[0, 0] = float("nan")
    before = arguments["python_random_generator"].getstate()
    with torch.no_grad():
        result = train_feddrift_held_models_for_round(
            **arguments,
            batch_sample_count=100 if no_op_kind == "insufficient" else 3,
            updates_per_sample=0 if no_op_kind == "zero_updates" else 2,
            k_steps=1,
        )
    assert result == ()
    assert arguments["python_random_generator"].getstate() == before


@pytest.mark.parametrize(
    "bad_kind",
    (
        "nan",
        "label",
        "dtype",
        "shape",
        "frozen",
        "optimizer",
        "architecture",
        "no_grad",
        "class_count",
        "head_shape",
        "hidden_shape",
        "activation",
        "feature_extractor",
    ),
)
def test_single_update_rejects_before_forward_zero_grad_or_mutation(monkeypatch, bad_kind):
    _, arguments = build_training_pair(monkeypatch, 2, "adam")
    state = arguments["held_model_training_state_registry"].get_held_model_training_state(
        model_id=9
    )
    classifier = state.classifier
    optimizer = state.concept_specific_parameter_optimizer_state.parameter_optimizer
    x, y = torch.ones(2, 3), torch.zeros(2, 1)
    if bad_kind == "nan":
        x[0, 0] = float("nan")
    elif bad_kind == "label":
        y[0, 0] = 0.5
    elif bad_kind == "dtype":
        x = x.double()
    elif bad_kind == "shape":
        y = y.flatten()
    elif bad_kind == "frozen":
        next(classifier.parameters()).requires_grad_(False)
    elif bad_kind == "optimizer":
        optimizer = torch.optim.SGD((torch.nn.Parameter(torch.ones(1)),), lr=0.1)
    elif bad_kind == "architecture":
        classifier.model_architecture_settings = None
    elif bad_kind == "class_count":
        classifier.class_count = 4
    elif bad_kind == "head_shape":
        classifier.classification_layer.weight.data = torch.ones(1, 3)
    elif bad_kind == "hidden_shape":
        next(classifier.parameters()).data = torch.ones(6, 4)
    elif bad_kind == "activation":
        classifier.output_activation = torch.nn.Identity()
    elif bad_kind == "feature_extractor":
        classifier.feature_extractor.hidden_layer_widths = (3,)
    refs = tuple(classifier.parameters())
    values = tuple(p.detach().clone() for p in refs)
    for p in refs:
        p.grad = torch.ones_like(p)
    optimizer_before = deepcopy(optimizer.state_dict())
    rng_before = torch.get_rng_state()
    calls = []
    classifier.register_forward_pre_hook(lambda *_: calls.append("forward"))
    original_zero_grad = optimizer.zero_grad

    def zero_grad(*args, **kwargs):
        calls.append("zero_grad")
        return original_zero_grad(*args, **kwargs)

    monkeypatch.setattr(optimizer, "zero_grad", zero_grad)
    with torch.set_grad_enabled(bad_kind != "no_grad"), pytest.raises((TypeError, ValueError)):
        update_independent_model_parameters(
            classifier=classifier,
            parameter_optimizer=optimizer,
            input_features=x,
            observed_class_labels=y,
        )
    assert not calls
    assert_values_equal(optimizer.state_dict(), optimizer_before)
    assert all(
        torch.equal(p, v) and torch.equal(p.grad, torch.ones_like(p))
        for p, v in zip(refs, values, strict=True)
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
