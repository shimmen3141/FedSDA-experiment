"""組立済みFedDrift clientを実旧clientの生成・進行に照合する。"""

import random
from copy import deepcopy
from dataclasses import replace
from random import Random

import numpy as np
import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.feddrift import FedDriftClient
from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.runtime.feddrift_run_client import (
    assemble_feddrift_run_client,
)
from federated_learning_experiments.runtime.feddrift_run_client_settings import (
    FeddriftRunClientSettings,
)
from tests.refactoring.test_feddrift_client_round_training import assert_values_equal
from tests.refactoring.test_feddrift_detection_batch_assessment import build_models
from tests.refactoring.test_feddrift_detection_batch_processing import (
    assert_processing_matches_legacy,
)


def build_assembly_inputs(*, monkeypatch, class_count=2, optimizer_name="sgd", threshold=1.0):
    """実旧モデルと新初期モデルを同じ値から事前学習し、実旧__init__へ渡す。"""
    for name, value in (
        ("NEW_MODEL_TRAINING", "fixed"),
        ("OPTIMIZER", optimizer_name),
        ("CLIENT_BATCH_SIZE", 3),
        ("STORED_DATA_LIMIT", 5),
        ("EVAL_STORE_SAMPLE_SIZE", 2),
        ("FEDDRIFT_DETECTION_BATCH_SIZE", 3),
        ("UPDATES_PER_SAMPLE", 2),
        ("WEIGHT_DECAY", 0.0),
        ("AMSGRAD", False),
    ):
        monkeypatch.setattr(config, name, value)
    old_models, new_models = build_models(class_count=class_count, monkeypatch=monkeypatch)
    initial = new_models[0][1]
    optimizer_settings = (
        SgdParameterOptimizerSettings(learning_rate=0.01)
        if optimizer_name == "sgd"
        else AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.0, adam_variant="standard"
        )
    )
    optimizer_state = ParameterOptimizerState(
        parameters=tuple(initial.parameters()), optimizer_settings=optimizer_settings
    )
    old = old_models[9]
    old.optimizer = (
        torch.optim.SGD(old.parameters(), lr=0.01)
        if optimizer_name == "sgd"
        else torch.optim.Adam(old.parameters(), lr=0.01)
    )
    features = torch.linspace(-1, 1, 9).reshape(3, 3)
    labels = (torch.arange(3) % class_count).float().reshape(3, 1)
    old.update(features, labels)
    optimizer_state.parameter_optimizer.zero_grad()
    scores = initial(features)
    loss = (
        torch.nn.BCELoss()(scores, labels)
        if class_count == 2
        else torch.nn.CrossEntropyLoss()(scores, labels.view(-1).long())
    )
    loss.backward()
    optimizer_state.parameter_optimizer.step()
    initial.eval()
    old.eval()
    stats = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=3, mean_loss=0.25, sum_squared_loss_deviations=0.125
        ),
        class_loss_moments_by_class_id=(
            (
                0,
                BoundedLossMoments(
                    observed_loss_count=1, mean_loss=0.2, sum_squared_loss_deviations=0.0
                ),
            ),
        ),
    )
    legacy = FedDriftClient(
        6,
        {0: old},
        initial_stats={
            0: {
                "n": 3,
                "mean": 0.25,
                "M2": 0.125,
                "class_stats": {0: {"n": 1, "mean": 0.2, "M2": 0.0}},
            }
        },
        distance_threshold=threshold,
        verbose=False,
    )
    inputs = dict(
        client_id=6,
        initial_classifier=initial,
        initial_parameter_optimizer_state=optimizer_state,
        initial_loss_statistics=stats,
        run_client_settings=FeddriftRunClientSettings(
            model_comparison_criterion="bounded_loss",
            new_model_parameter_source="current_training_model",
            detection_reference_after_drift="new_model_batch_value",
            loss_increase_threshold=threshold,
            detection_batch_sample_count=3,
            local_training_batch_sample_count=3,
            updates_per_sample=2,
            maximum_stored_evaluation_sample_count_per_model=5,
            added_evaluation_batch_sample_count=2,
            new_model_optimizer_settings=optimizer_settings,
        ),
        python_random_generator=Random(87),
    )
    return legacy, inputs


def processing_owners(client):
    """上流の実旧対照helperへ同じowner名を対応付ける。"""
    owners = client.owners
    return dict(
        held_model_training_state_registry=owners.held_model_training_state_registry,
        loss_statistics_store=owners.loss_statistics_store,
        current_training_model_assignment=owners.current_training_model_assignment,
        training_sample_store=owners.training_sample_store,
        evaluation_sample_store=owners.evaluation_sample_store,
        temporary_model_id_allocator=owners.temporary_model_id_allocator,
        pending_model_upload_state=owners.pending_model_upload_state,
    )


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(before):
    assert random.getstate() == before[0]
    actual = np.random.get_state()
    assert actual[0] == before[1][0] and np.array_equal(actual[1], before[1][1])
    assert actual[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize("grad_enabled", (False, True))
def test_assembly_matches_real_init_and_isolates_state(
    monkeypatch, class_count, optimizer_name, grad_enabled
):
    legacy, inputs = build_assembly_inputs(
        monkeypatch=monkeypatch, class_count=class_count, optimizer_name=optimizer_name
    )
    before = rng_snapshot()
    random_before = inputs["python_random_generator"].getstate()
    with torch.set_grad_enabled(grad_enabled):
        client = assemble_feddrift_run_client(**inputs)
        other = assemble_feddrift_run_client(**inputs)
        assert torch.is_grad_enabled() == grad_enabled
    assert_rng_equal(before)
    assert inputs["python_random_generator"].getstate() == random_before
    assert client.client_id == legacy.client_id
    assert_processing_matches_legacy(
        legacy=legacy, processor=client.detection_batch_processor, owners=processing_owners(client)
    )
    assert client.stream_processor.processed_sample_count == 0
    assert client.stream_processor.sample_predictions == ()
    assert client.stream_processor.buffered_training_samples == ()
    assert client.stream_processor.completed_detection_batch_transitions == ()
    assert client.has_model_ready_for_server_registration() is False
    assert (
        client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts().trained_sample_counts_by_model_id
        == {}
    )
    state = client.owners.held_model_training_state_registry.get_held_model_training_state(
        model_id=0
    )
    other_state = other.owners.held_model_training_state_registry.get_held_model_training_state(
        model_id=0
    )
    assert state.classifier.training is False
    for actual, original, other_parameter, old_parameter in zip(
        state.classifier.parameters(),
        inputs["initial_classifier"].parameters(),
        other_state.classifier.parameters(),
        legacy.models[0].parameters(),
        strict=True,
    ):
        assert actual is not original and actual is not other_parameter
        assert (
            actual.data_ptr() != original.data_ptr()
            and actual.data_ptr() != other_parameter.data_ptr()
        )
        assert torch.equal(actual, old_parameter)
        assert actual.grad is None and old_parameter.grad is None
        assert original.grad is not None
    optimizer = state.concept_specific_parameter_optimizer_state.parameter_optimizer
    assert_values_equal(optimizer.state_dict(), legacy.models[0].optimizer.state_dict())
    if optimizer_name == "adam":
        assert optimizer.state
        assert_values_equal(
            optimizer.state_dict(),
            other_state.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
        )
    parameters = tuple(state.classifier.parameters())
    assert all(x is y for x, y in zip(optimizer.param_groups[0]["params"], parameters, strict=True))
    with torch.no_grad():
        parameters[0].add_(1)
    assert not torch.equal(parameters[0], next(inputs["initial_classifier"].parameters()))
    assert torch.equal(
        next(other_state.classifier.parameters()), next(inputs["initial_classifier"].parameters())
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("loss_increase_threshold", -0.1),
        ("loss_increase_threshold", float("nan")),
        ("loss_increase_threshold", True),
        ("loss_increase_threshold", 1),
        ("detection_batch_sample_count", 0),
        ("detection_batch_sample_count", True),
        ("local_training_batch_sample_count", 1.0),
        ("local_training_batch_sample_count", -1),
        ("updates_per_sample", -1),
        ("updates_per_sample", False),
        ("maximum_stored_evaluation_sample_count_per_model", 0),
        ("added_evaluation_batch_sample_count", -1),
        ("added_evaluation_batch_sample_count", True),
        ("new_model_optimizer_settings", None),
    ),
)
def test_settings_reject_invalid_values(monkeypatch, field, value):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        replace(inputs["run_client_settings"], **{field: value})


@pytest.mark.parametrize(
    "invalid",
    (
        "client_bool",
        "client_negative",
        "random",
        "settings",
        "tampered_settings",
        "classifier",
        "architecture",
        "shape",
        "frozen",
        "nan",
        "foreign_optimizer",
        "statistics",
        "statistics_class",
    ),
)
def test_assembly_rejects_before_clone_forward_or_rng(monkeypatch, invalid):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    classifier = inputs["initial_classifier"]
    if invalid == "client_bool":
        inputs["client_id"] = True
    elif invalid == "client_negative":
        inputs["client_id"] = -1
    elif invalid == "random":
        inputs["python_random_generator"] = object()
    elif invalid == "settings":
        inputs["run_client_settings"] = None
    elif invalid == "tampered_settings":
        object.__setattr__(inputs["run_client_settings"], "detection_batch_sample_count", False)
    elif invalid == "classifier":
        inputs["initial_classifier"] = None
    elif invalid == "architecture":
        classifier.model_architecture_settings = None
    elif invalid == "shape":
        classifier.classification_layer.weight = torch.nn.Parameter(torch.zeros(2, 4))
    elif invalid == "frozen":
        next(classifier.parameters()).requires_grad_(False)
    elif invalid == "nan":
        next(classifier.parameters()).data.fill_(float("nan"))
    elif invalid == "foreign_optimizer":
        inputs["initial_parameter_optimizer_state"] = ParameterOptimizerState(
            parameters=(torch.nn.Parameter(torch.ones(1)),),
            optimizer_settings=SgdParameterOptimizerSettings(learning_rate=0.01),
        )
    elif invalid == "statistics":
        inputs["initial_loss_statistics"] = None
    elif invalid == "statistics_class":
        inputs["initial_loss_statistics"] = replace(
            inputs["initial_loss_statistics"],
            class_loss_moments_by_class_id=(
                (2, inputs["initial_loss_statistics"].overall_loss_moments),
            ),
        )
    calls = []
    classifier.register_forward_pre_hook(lambda *args: calls.append("forward"))
    monkeypatch.setattr(
        "federated_learning_experiments.runtime.feddrift_run_client.deepcopy",
        lambda *args: calls.append("copy"),
    )
    before = rng_snapshot()
    params = tuple(p.detach().clone() for p in classifier.parameters())
    grads = tuple(None if p.grad is None else p.grad.clone() for p in classifier.parameters())
    with pytest.raises((TypeError, ValueError)):
        assemble_feddrift_run_client(**inputs)
    assert calls == []
    assert_rng_equal(before)
    for p, v, g in zip(classifier.parameters(), params, grads, strict=True):
        torch.testing.assert_close(p, v, equal_nan=True, rtol=0, atol=0)
        assert_values_equal(p.grad, g)


def run_pair(*, legacy, client, generator, old_operation, new_operation):
    """実旧のglobal Python RNGをrunのRandomへ対応付け、両経路を同じ3RNGから実行する。"""
    before = rng_snapshot()
    random.setstate(generator.getstate())
    try:
        old_result = old_operation()
        expected_python = random.getstate()
        expected_numpy = np.random.get_state()
        expected_torch = torch.get_rng_state().clone()
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])
    new_result = new_operation()
    assert generator.getstate() == expected_python
    assert torch.equal(torch.get_rng_state(), expected_torch)
    actual_numpy = np.random.get_state()
    assert (
        np.array_equal(actual_numpy[1], expected_numpy[1])
        and actual_numpy[2:] == expected_numpy[2:]
    )
    assert random.getstate() == before[0]
    assert_client_matches(legacy=legacy, client=client)
    return old_result, new_result


def assert_client_matches(*, legacy, client):
    assert_processing_matches_legacy(
        legacy=legacy, processor=client.detection_batch_processor, owners=processing_owners(client)
    )
    assert client.stream_processor.processed_sample_count == legacy.processed_samples
    predictions = client.stream_processor.sample_predictions
    assert tuple(p.prediction_model_id for p in predictions) == tuple(legacy.history_model_id)
    assert tuple(p.prediction_is_correct for p in predictions) == tuple(
        bool(v) for v in legacy.history_accuracy
    )
    assert tuple(p.observed_concept_id for p in predictions) == tuple(legacy.history_concept)
    kinds = {"current_model_retained": 0, "held_model_selected": 1, "new_model_created": 2}
    assert tuple(
        0
        if p.detection_batch_transition is None
        else kinds[p.detection_batch_transition.adaptation_kind]
        for p in predictions
    ) == tuple(legacy.history_drift_type)
    transitions = client.stream_processor.completed_detection_batch_transitions
    assert tuple(
        t.first_sample_index for t in transitions if t.adaptation_kind == "new_model_created"
    ) == tuple(legacy.detected_event_positions)
    assert tuple(
        t.last_sample_index for t in transitions if t.adaptation_kind != "current_model_retained"
    ) == tuple(legacy.local_switch_positions)
    assert len(client.stream_processor.buffered_training_samples) == len(legacy.detect_buffer)
    for sample, old in zip(
        client.stream_processor.buffered_training_samples, legacy.detect_buffer, strict=True
    ):
        assert_values_equal(sample.input_features, old[0])
        assert_values_equal(sample.observed_class_labels, old[1])
    for state in client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states():
        old = legacy.models[state.model_id]
        assert_values_equal(
            state.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
            old.optimizer.state_dict(),
        )
        for p, q in zip(state.classifier.parameters(), old.parameters(), strict=True):
            assert_values_equal(p.grad, q.grad)
    counts = client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
    # 旧は0 epochの新規生成でもゼロentryを作る。完了量ownerでは未学習IDの欠落を0へ対応付ける。
    for actual, expected in (
        (counts.trained_sample_counts_by_model_id, legacy.model_training_examples),
        (counts.parameter_update_step_counts_by_model_id, legacy.model_optimizer_steps),
    ):
        assert actual.keys() <= expected.keys()
        assert all(actual.get(mid, 0) == value for mid, value in expected.items())
    assert counts.assigned_sample_counts_by_model_and_concept_id == {}
    assert client.has_model_ready_for_server_registration() == (
        legacy.pending_model_params is not None and legacy.pending_model_ready
    )


def observation_block(*, start, count, class_count):
    # 低損失の先頭batchの後にラベルを反転し、新規モデル分岐も実経路で通す。
    return tuple(
        ObservedSample(
            feature_values=(float(i) / 10, -0.5, 0.25),
            class_label=(1 if class_count == 2 else 0) if i < 3 else (0 if class_count == 2 else 1),
        )
        for i in range(start, start + count)
    )


def snapshot_client(client, generator):
    owners = client.owners
    pending = owners.pending_model_upload_state.get_pending_model_upload()
    counts = owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
    return (
        owners.loss_statistics_store.get_state_snapshot(),
        owners.current_training_model_assignment.current_training_model_id,
        owners.temporary_model_id_allocator.next_temporary_model_id,
        generator.getstate(),
        client.detection_batch_processor.previous_minimum_mean_bounded_loss,
        tuple(
            (
                state.model_id,
                tuple(p.detach().clone() for p in state.classifier.parameters()),
                tuple(
                    None if p.grad is None else p.grad.clone()
                    for p in state.classifier.parameters()
                ),
                deepcopy(
                    state.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict()
                ),
            )
            for state in owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        ),
        tuple(
            (
                row.model_id,
                tuple(
                    (s.input_features.clone(), s.observed_class_labels.clone())
                    for s in row.training_samples
                ),
            )
            for row in owners.training_sample_store.snapshot_ordered_model_training_samples()
        ),
        tuple(
            (
                row.model_id,
                tuple(
                    (s.input_features.clone(), s.observed_class_labels.clone())
                    for s in row.evaluation_samples
                ),
            )
            for row in owners.evaluation_sample_store.snapshot_ordered_model_evaluation_samples()
        ),
        None if pending is None else (pending.model_id, deepcopy(pending.parameter_snapshot)),
        owners.pending_model_upload_state.has_ready_model_upload(),
        counts.trained_sample_counts_by_model_id,
        counts.parameter_update_step_counts_by_model_id,
        counts.assigned_sample_counts_by_model_and_concept_id,
    )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize("threshold", (0.0, 1.0))
def test_complete_client_flow_matches_real_legacy(
    monkeypatch, class_count, optimizer_name, threshold
):
    legacy, inputs = build_assembly_inputs(
        monkeypatch=monkeypatch,
        class_count=class_count,
        optimizer_name=optimizer_name,
        threshold=threshold,
    )
    with torch.no_grad():
        inputs["initial_classifier"].classification_layer.weight.zero_()
        inputs["initial_classifier"].classification_layer.bias.fill_(-8)
        inputs["initial_classifier"].classification_layer.bias[0] = 8
        old_head = [
            layer for layer in legacy.models[0].modules() if type(layer) is torch.nn.Linear
        ][-1]
        old_head.weight.zero_()
        old_head.bias.copy_(inputs["initial_classifier"].classification_layer.bias)
    client = assemble_feddrift_run_client(**inputs)
    generator = inputs["python_random_generator"]
    owners = client.owners
    losses = []
    wrapped_ids = set()
    for start, count in ((0, 2), (2, 4), (6, 1)):
        block = observation_block(start=start, count=count, class_count=class_count)
        concepts = tuple(i % 2 for i in range(start, start + count))
        data = [
            (
                torch.tensor((s.feature_values,), dtype=torch.float32),
                torch.tensor([[s.class_label]], dtype=torch.float32),
            )
            for s in block
        ]
        old_fired, new_fired = run_pair(
            legacy=legacy,
            client=client,
            generator=generator,
            old_operation=lambda data=data, concepts=concepts: legacy.process_batch(data, concepts),
            new_operation=lambda block=block, concepts=concepts: client.process_observed_block(
                observed_samples=block, evaluation_concept_ids=concepts
            ),
        )
        assert old_fired is new_fired
        for model_id, model in legacy.models.items():
            if model_id in wrapped_ids:
                continue
            real_update = model.update

            def record_update(x, y, real_update=real_update):
                result = real_update(x, y)
                losses.append(result)
                return result

            model.update = record_update
            wrapped_ids.add(model_id)
        for _ in range(2):
            losses.clear()
            _, actual_losses = run_pair(
                legacy=legacy,
                client=client,
                generator=generator,
                old_operation=lambda: legacy.local_train(k_steps=3),
                new_operation=client.train_local_round,
            )
            assert actual_losses == tuple(losses)
            assert client.owners is owners
    original_predictions = client.stream_processor.sample_predictions
    run_pair(
        legacy=legacy,
        client=client,
        generator=generator,
        old_operation=legacy.flush,
        new_operation=client.flush_detection_buffer,
    )
    assert client.stream_processor.sample_predictions == original_predictions
    assert client.flush_detection_buffer() is None
    if threshold == 0.0:
        assert any(
            t.adaptation_kind == "new_model_created"
            for t in client.stream_processor.completed_detection_batch_transitions
        )


@pytest.mark.parametrize(
    "invalid",
    (
        "list",
        "length",
        "concept_bool",
        "concept_type",
        "sample_type",
        "sample_tampered",
        "width",
        "nan",
        "label",
    ),
)
def test_invalid_later_block_value_is_rejected_before_any_update(monkeypatch, invalid):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    client = assemble_feddrift_run_client(**inputs)
    block = observation_block(start=0, count=3, class_count=2)
    concepts = (0, 1, 0)
    if invalid == "list":
        block = list(block)
    elif invalid == "length":
        concepts = (0, 1)
    elif invalid == "concept_bool":
        concepts = (0, 1, False)
    elif invalid == "concept_type":
        concepts = (0, 1, object())
    elif invalid == "sample_type":
        block = (*block[:2], None)
    elif invalid == "sample_tampered":
        object.__setattr__(block[-1], "class_label", True)
    elif invalid == "width":
        block = (*block[:2], replace(block[-1], feature_values=(1.0,)))
    elif invalid == "nan":
        block = (*block[:2], replace(block[-1], feature_values=(float("nan"), 0.0, 0.0)))
    elif invalid == "label":
        block = (*block[:2], replace(block[-1], class_label=2))
    calls = []
    state = client.owners.held_model_training_state_registry.get_held_model_training_state(
        model_id=0
    )
    state.classifier.register_forward_pre_hook(lambda *args: calls.append("forward"))
    before = snapshot_client(client, inputs["python_random_generator"])
    rng = rng_snapshot()
    generator_state = inputs["python_random_generator"].getstate()
    with pytest.raises((TypeError, ValueError)):
        client.process_observed_block(observed_samples=block, evaluation_concept_ids=concepts)
    assert calls == []
    assert_values_equal(snapshot_client(client, inputs["python_random_generator"]), before)
    assert_rng_equal(rng)
    assert inputs["python_random_generator"].getstate() == generator_state
    assert (
        client.stream_processor.processed_sample_count == 0
        and client.stream_processor.sample_predictions == ()
    )


def test_concepts_do_not_affect_processing_and_readiness_is_read_only(monkeypatch):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch, threshold=0.0)
    left = assemble_feddrift_run_client(**inputs)
    right_inputs = inputs | {"python_random_generator": Random(87)}
    right = assemble_feddrift_run_client(**right_inputs)
    torch_before = torch.get_rng_state().clone()
    block = observation_block(start=0, count=6, class_count=2)
    left.process_observed_block(observed_samples=block, evaluation_concept_ids=(0, 0, 0, 0, 0, 0))
    left.train_local_round()
    expected_torch = torch.get_rng_state().clone()
    torch.set_rng_state(torch_before)
    right.process_observed_block(
        observed_samples=block, evaluation_concept_ids=(None, -7, 4, 1, 2, 3)
    )
    right.train_local_round()
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert (
        inputs["python_random_generator"].getstate()
        == right_inputs["python_random_generator"].getstate()
    )
    assert_values_equal(
        snapshot_client(left, inputs["python_random_generator"]),
        snapshot_client(right, right_inputs["python_random_generator"]),
    )
    assert (
        left.detection_batch_processor.previous_minimum_mean_bounded_loss
        == right.detection_batch_processor.previous_minimum_mean_bounded_loss
    )
    for a, b in zip(
        left.stream_processor.sample_predictions,
        right.stream_processor.sample_predictions,
        strict=True,
    ):
        assert replace(a, observed_concept_id=None) == replace(b, observed_concept_id=None)
    assert left.process_observed_block(observed_samples=(), evaluation_concept_ids=()) is False
    before = snapshot_client(left, inputs["python_random_generator"])
    rng = rng_snapshot()
    for _ in range(3):
        assert type(left.has_model_ready_for_server_registration()) is bool
    assert_values_equal(snapshot_client(left, inputs["python_random_generator"]), before)
    assert_rng_equal(rng)


def test_zero_update_and_evaluation_addition_settings_are_accepted(monkeypatch):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    inputs["run_client_settings"] = replace(
        inputs["run_client_settings"], updates_per_sample=0, added_evaluation_batch_sample_count=0
    )
    client = assemble_feddrift_run_client(**inputs)
    client.process_observed_block(
        observed_samples=observation_block(start=0, count=3, class_count=2),
        evaluation_concept_ids=(None, None, None),
    )
    before = snapshot_client(client, inputs["python_random_generator"])
    assert client.train_local_round() == ()
    assert_values_equal(snapshot_client(client, inputs["python_random_generator"]), before)
    assert all(
        not row.evaluation_samples
        for row in client.owners.evaluation_sample_store.snapshot_ordered_model_evaluation_samples()
    )


def test_pending_readiness_and_mutable_owners_remain_independent(monkeypatch):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch, optimizer_name="adam")
    client = assemble_feddrift_run_client(**inputs)
    other = assemble_feddrift_run_client(**inputs)
    original = snapshot_client(other, inputs["python_random_generator"])
    state = client.owners.held_model_training_state_registry.get_held_model_training_state(
        model_id=0
    )
    snapshot = {
        name: value.detach().clone() for name, value in state.classifier.state_dict().items()
    }
    uploads = client.owners.pending_model_upload_state
    uploads.queue_model_upload(
        model_id=-106, parameter_snapshot=snapshot, upload_delay_round_count=1
    )
    assert client.has_model_ready_for_server_registration() is False
    uploads.advance_upload_readiness_at_round_boundary()
    ready_before = uploads.get_pending_model_upload()
    assert client.has_model_ready_for_server_registration() is True
    assert uploads.get_pending_model_upload() is ready_before
    next(iter(state.concept_specific_parameter_optimizer_state.parameter_optimizer.state.values()))[
        "exp_avg"
    ].add_(1)
    client.owners.model_training_and_assignment_counts_store.record_completed_model_training(
        model_id=0, trained_sample_count=3, parameter_update_step_count=1
    )
    assert_values_equal(snapshot_client(other, inputs["python_random_generator"]), original)
    assert_values_equal(
        inputs["initial_parameter_optimizer_state"].parameter_optimizer.state_dict(),
        other.owners.held_model_training_state_registry.get_held_model_training_state(
            model_id=0
        ).concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
    )
