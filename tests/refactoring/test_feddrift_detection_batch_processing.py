"""FedDriftの一バッチ処理を、実旧の状態更新と乱数に照合する。"""

import random
from copy import deepcopy
from dataclasses import replace
from random import Random

import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.feddrift import FedDriftClient
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (
    KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from tests.refactoring.test_feddrift_detection_batch_assessment import build_models


def build_processing_oracle(
    *, class_count, previous_loss, threshold, monkeypatch, current_model_id=9, added_count=3
):
    """実旧__init__と新ownerを同じモデル・固定条件から用意する。"""
    from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
        FeddriftDetectionBatchProcessor,
    )

    for name, value in (
        ("NEW_MODEL_TRAINING", "fixed"),
        ("OPTIMIZER", "sgd"),
        ("CLIENT_BATCH_SIZE", 2),
        ("STORED_DATA_LIMIT", 5),
        ("EVAL_STORE_SAMPLE_SIZE", added_count),
    ):
        monkeypatch.setattr(config, name, value)
    old_models, new_models = build_models(class_count=class_count, monkeypatch=monkeypatch)
    legacy = FedDriftClient(6, old_models, distance_threshold=threshold, verbose=False)
    legacy.current_model_id = current_model_id
    legacy.last_min_loss = previous_loss
    registry = HeldModelTrainingStateRegistry()
    optimizer_settings = SgdParameterOptimizerSettings(learning_rate=0.01)
    for model_id, classifier in new_models:
        registry.register_held_model_training_state(
            model_id=model_id,
            classifier=classifier,
            concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                parameters=tuple(classifier.parameters()), optimizer_settings=optimizer_settings
            ),
        )
    owners = dict(
        held_model_training_state_registry=registry,
        loss_statistics_store=ModelAndClassLossStatisticsStore(
            initial_loss_statistics_by_model_id={
                mid: ModelAndClassLossStatistics(
                    overall_loss_moments=BoundedLossMoments(
                        observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
                    )
                )
                for mid, _ in new_models
            }
        ),
        current_training_model_assignment=CurrentTrainingModelAssignment(
            initial_model_id=current_model_id
        ),
        training_sample_store=ModelTrainingSampleStore(),
        evaluation_sample_store=ModelEvaluationSampleStore(
            maximum_stored_sample_count_per_model=5, added_batch_sample_count=added_count
        ),
        temporary_model_id_allocator=TemporaryModelIdAllocator(client_id=6),
        pending_model_upload_state=PendingModelUploadState(
            pending_model_registration_policy=KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
        ),
        python_random_generator=Random(87),
        new_model_optimizer_settings=optimizer_settings,
        loss_increase_threshold=threshold,
        previous_minimum_mean_bounded_loss=previous_loss,
    )
    return legacy, FeddriftDetectionBatchProcessor(**owners), owners


def make_samples(*, sample_count, class_count):
    features = torch.linspace(-1, 1, sample_count * 3).reshape(sample_count, 3)
    labels = (torch.arange(sample_count) % class_count).float().reshape(-1, 1)
    return tuple(
        ObservedTrainingSample(input_features=x.unsqueeze(0), observed_class_labels=y.unsqueeze(0))
        for x, y in zip(features, labels, strict=True)
    )


def assert_processing_matches_legacy(*, legacy, processor, owners):
    """状態の対応付けをtest側で明示し、全部の値と順序を照合する。"""
    assert (
        owners["current_training_model_assignment"].current_training_model_id
        == legacy.current_model_id
    )
    assert processor.previous_minimum_mean_bounded_loss == legacy.last_min_loss
    assert owners["temporary_model_id_allocator"].next_temporary_model_id == legacy.next_temp_id
    states = owners[
        "held_model_training_state_registry"
    ].snapshot_ordered_held_model_training_states()
    assert tuple(x.model_id for x in states) == tuple(legacy.models)
    for state in states:
        for actual, expected in zip(
            state.classifier.parameters(), legacy.models[state.model_id].parameters(), strict=True
        ):
            assert torch.equal(actual, expected)
    for model_id, actual in owners["loss_statistics_store"].get_state_snapshot():
        expected = legacy.model_stats[model_id]
        overall = actual.overall_loss_moments
        assert (
            overall.observed_loss_count,
            overall.mean_loss,
            overall.sum_squared_loss_deviations,
        ) == (expected["n"], expected["mean"], expected["M2"])
        assert tuple(cid for cid, _ in actual.class_loss_moments_by_class_id) == tuple(
            expected.get("class_stats", {})
        )
        for cid, moments in actual.class_loss_moments_by_class_id:
            old = expected["class_stats"][cid]
            assert (
                moments.observed_loss_count,
                moments.mean_loss,
                moments.sum_squared_loss_deviations,
            ) == (old["n"], old["mean"], old["M2"])
    for store_name, snapshot_method, field, old_store in (
        (
            "training_sample_store",
            "snapshot_ordered_model_training_samples",
            "training_samples",
            legacy.train_data_store,
        ),
        (
            "evaluation_sample_store",
            "snapshot_ordered_model_evaluation_samples",
            "evaluation_samples",
            legacy.stored_data,
        ),
    ):
        collections = getattr(owners[store_name], snapshot_method)()
        assert tuple(x.model_id for x in collections) == tuple(old_store)
        for collection in collections:
            actual = getattr(collection, field)
            expected = old_store[collection.model_id]
            assert len(actual) == len(expected)
            for new, old in zip(actual, expected, strict=True):
                assert torch.equal(new.input_features, old[0])
                assert torch.equal(new.observed_class_labels, old[1])
    pending = owners["pending_model_upload_state"].get_pending_model_upload()
    assert (pending is None) == (legacy.pending_model_params is None)
    if pending is not None:
        assert pending.model_id in legacy.models
        assert (
            owners["pending_model_upload_state"].has_ready_model_upload()
            == legacy.pending_model_ready
        )
        for new, old in zip(
            pending.parameter_snapshot.values(), legacy.pending_model_params.values(), strict=True
        ):
            assert torch.equal(new, old)


def run_processing_pair(*, legacy, processor, owners, samples, processed_sample_count):
    legacy.processed_samples = processed_sample_count
    rng = torch.get_rng_state().clone()
    with torch.random.fork_rng():
        torch.set_rng_state(rng)
        python_state = random.getstate()
        random.setstate(owners["python_random_generator"].getstate())
        try:
            drift_type = legacy._detect_and_assign(
                [(s.input_features, s.observed_class_labels) for s in samples]
            )
            expected_python_state = random.getstate()
        finally:
            random.setstate(python_state)
        expected_torch_state = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        result = processor.process_detection_batch(
            training_samples=samples, processed_sample_count=processed_sample_count
        )
        assert torch.equal(torch.get_rng_state(), expected_torch_state)
        assert owners["python_random_generator"].getstate() == expected_python_state
    assert_processing_matches_legacy(legacy=legacy, processor=processor, owners=owners)
    assert (
        result.adaptation_kind
        == {0: "current_model_retained", 1: "held_model_selected", 2: "new_model_created"}[
            drift_type
        ]
    )
    assert result.assigned_model_id == legacy.current_model_id
    assert result.first_sample_index == processed_sample_count - len(samples)
    assert result.last_sample_index == processed_sample_count - 1
    assert result.next_previous_minimum_mean_bounded_loss == legacy.last_min_loss
    if drift_type == 2:
        assert legacy.detected_event_positions[-1] == result.first_sample_index
    if drift_type != 0:
        assert legacy.local_switch_positions[-1] == result.last_sample_index
    return result


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("sample_count", (1, 3, 8))
@pytest.mark.parametrize("comparison", ("first", "retained", "switched", "created"))
def test_processing_matches_real_legacy(class_count, sample_count, comparison, monkeypatch):
    legacy, processor, owners = build_processing_oracle(
        class_count=class_count,
        previous_loss=0.0 if comparison == "created" else None,
        threshold=0.0,
        monkeypatch=monkeypatch,
    )
    samples = make_samples(sample_count=sample_count, class_count=class_count)
    if comparison in ("retained", "switched"):
        losses = {
            mid: float(
                model.per_sample_error(
                    torch.cat([s.input_features for s in samples]),
                    torch.cat([s.observed_class_labels for s in samples]),
                )
                .mean()
                .item()
            )
            for mid, model in legacy.models.items()
        }
        best = min(losses, key=losses.get)
        current = best if comparison == "retained" else next(mid for mid in losses if mid != best)
        legacy.current_model_id = current
        owners["current_training_model_assignment"].assign_model_for_training(model_id=current)
    old_parameters = {
        id(p)
        for state in owners[
            "held_model_training_state_registry"
        ].snapshot_ordered_held_model_training_states()
        for p in state.classifier.parameters()
    }
    result = run_processing_pair(
        legacy=legacy,
        processor=processor,
        owners=owners,
        samples=samples,
        processed_sample_count=sample_count + 7,
    )
    if comparison == "created":
        created = owners["held_model_training_state_registry"].get_held_model_training_state(
            model_id=result.assigned_model_id
        )
        assert all(id(p) not in old_parameters for p in created.classifier.parameters())
        assert not created.concept_specific_parameter_optimizer_state.parameter_optimizer.state
        snapshot = (
            owners["pending_model_upload_state"].get_pending_model_upload().parameter_snapshot
        )
        assert all(
            value.data_ptr() != p.data_ptr()
            for value, p in zip(snapshot.values(), created.classifier.parameters(), strict=True)
        )


@pytest.mark.parametrize("added_count", (0, 3))
def test_multiple_batches_and_evaluation_capacity_match_legacy(added_count, monkeypatch):
    legacy, processor, owners = build_processing_oracle(
        class_count=4,
        previous_loss=None,
        threshold=1.0,
        monkeypatch=monkeypatch,
        current_model_id=-104,
        added_count=added_count,
    )
    for index in range(4):
        run_processing_pair(
            legacy=legacy,
            processor=processor,
            owners=owners,
            samples=make_samples(sample_count=4, class_count=4),
            processed_sample_count=(index + 1) * 4,
        )


def test_consecutive_creation_uses_next_id_and_replaces_ready_upload(monkeypatch):
    legacy, processor, owners = build_processing_oracle(
        class_count=2, previous_loss=0.0, threshold=0.0, monkeypatch=monkeypatch
    )
    with torch.no_grad():
        for model in legacy.models.values():
            for parameter in model.parameters():
                parameter.zero_()
            tuple(model.parameters())[-1].fill_(2.0)
        for state in owners[
            "held_model_training_state_registry"
        ].snapshot_ordered_held_model_training_states():
            for parameter in state.classifier.parameters():
                parameter.zero_()
            state.classifier.classification_layer.bias.fill_(2.0)
    for index, label in enumerate((1.0, 0.0)):
        samples = (
            ObservedTrainingSample(
                input_features=torch.zeros(1, 3), observed_class_labels=torch.tensor([[label]])
            ),
        )
        result = run_processing_pair(
            legacy=legacy,
            processor=processor,
            owners=owners,
            samples=samples,
            processed_sample_count=index + 1,
        )
        assert result.adaptation_kind == "new_model_created"
        assert result.assigned_model_id == -106 - index
    assert owners["evaluation_sample_store"].snapshot_ordered_model_evaluation_samples() == ()


def snapshot_processing_owners(processor, owners):
    """構造と値を文字列へ固定し、拒否前後の変化を検出する。"""
    return (
        repr(owners["loss_statistics_store"].get_state_snapshot()),
        repr(owners["training_sample_store"].snapshot_ordered_model_training_samples()),
        repr(owners["evaluation_sample_store"].snapshot_ordered_model_evaluation_samples()),
        owners["current_training_model_assignment"].current_training_model_id,
        owners["temporary_model_id_allocator"].next_temporary_model_id,
        owners["pending_model_upload_state"].get_pending_model_upload(),
        owners["python_random_generator"].getstate(),
        processor.previous_minimum_mean_bounded_loss,
        tuple(
            (state.model_id, repr(tuple(p.detach() for p in state.classifier.parameters())))
            for state in owners[
                "held_model_training_state_registry"
            ].snapshot_ordered_held_model_training_states()
        ),
    )


@pytest.mark.parametrize(
    "invalid_case",
    (
        "empty",
        "list",
        "record",
        "late_features",
        "late_labels",
        "multiple_rows",
        "negative_count",
        "bool_count",
        "small_count",
        "missing_current",
        "bool_current",
        "bad_optimizer",
        "bad_architecture",
        "bad_dimensions",
        "nan_parameter",
        "registry_collision",
        "statistics_collision",
        "training_collision",
        "pending_collision",
    ),
)
def test_invalid_batch_preserves_all_owners_and_rng(invalid_case, monkeypatch):
    _, processor, owners = build_processing_oracle(
        class_count=2, previous_loss=0.0, threshold=0.0, monkeypatch=monkeypatch
    )
    samples = make_samples(sample_count=3, class_count=2)
    count = 10
    states = owners[
        "held_model_training_state_registry"
    ].snapshot_ordered_held_model_training_states()
    if invalid_case == "empty":
        samples = ()
    elif invalid_case == "list":
        samples = list(samples)
    elif invalid_case == "record":
        samples = samples[:-1] + (object(),)
    elif invalid_case == "late_features":
        samples = samples[:-1] + (replace(samples[-1], input_features=torch.ones(1, 4)),)
    elif invalid_case == "late_labels":
        samples = samples[:-1] + (
            replace(samples[-1], observed_class_labels=torch.tensor([[0.5]])),
        )
    elif invalid_case == "multiple_rows":
        samples = (
            replace(
                samples[0], input_features=torch.ones(2, 3), observed_class_labels=torch.zeros(2, 1)
            ),
        )
    elif invalid_case.endswith("count"):
        count = {"negative_count": -1, "bool_count": True, "small_count": 2}[invalid_case]
    elif invalid_case == "missing_current":
        owners["current_training_model_assignment"].assign_model_for_training(model_id=999)
    elif invalid_case == "bool_current":
        owners["current_training_model_assignment"]._current_training_model_id = True
    elif invalid_case == "bad_optimizer":
        states[-1].concept_specific_parameter_optimizer_state.parameter_optimizer.param_groups[0][
            "params"
        ] = list(states[0].classifier.parameters())
    elif invalid_case == "bad_architecture":
        object.__setattr__(
            states[-1].classifier.model_architecture_settings, "model_architecture_name", "invalid"
        )
    elif invalid_case == "bad_dimensions":
        states[-1].classifier.hidden_layer_widths = (True,)
    elif invalid_case == "nan_parameter":
        with torch.no_grad():
            tuple(states[-1].classifier.parameters())[0].fill_(float("nan"))
    elif invalid_case == "registry_collision":
        owners["held_model_training_state_registry"].register_held_model_training_state(
            model_id=-106,
            classifier=states[0].classifier,
            concept_specific_parameter_optimizer_state=states[
                0
            ].concept_specific_parameter_optimizer_state,
        )
    elif invalid_case == "statistics_collision":
        owners["loss_statistics_store"].set_model_loss_statistics(
            model_id=-106,
            loss_statistics=owners["loss_statistics_store"].get_model_loss_statistics(model_id=9),
        )
    elif invalid_case == "training_collision":
        owners["training_sample_store"].append_model_training_samples(
            model_id=-106, training_samples=()
        )
    else:
        owners["pending_model_upload_state"].queue_ready_model_upload(
            model_id=-106, parameter_snapshot={"a": torch.ones(1)}
        )
    before = snapshot_processing_owners(processor, owners)
    torch_rng = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError)):
        processor.process_detection_batch(training_samples=samples, processed_sample_count=count)
    assert snapshot_processing_owners(processor, owners) == before
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize(
    "field,value",
    (
        ("loss_increase_threshold", True),
        ("loss_increase_threshold", -0.1),
        ("loss_increase_threshold", float("inf")),
        ("previous_minimum_mean_bounded_loss", True),
        ("previous_minimum_mean_bounded_loss", 1.1),
        ("new_model_optimizer_settings", object()),
        ("python_random_generator", object()),
        ("evaluation_sample_store", None),
    ),
)
def test_invalid_constructor_preserves_owners(field, value, monkeypatch):
    from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
        FeddriftDetectionBatchProcessor,
    )

    _, processor, owners = build_processing_oracle(
        class_count=2, previous_loss=0.0, threshold=0.0, monkeypatch=monkeypatch
    )
    before = snapshot_processing_owners(processor, owners)
    rng = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError)):
        FeddriftDetectionBatchProcessor(**(owners | {field: value}))
    assert snapshot_processing_owners(processor, owners) == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_creation_uses_current_even_when_another_model_has_lower_loss(monkeypatch):
    legacy, processor, owners = build_processing_oracle(
        class_count=4, previous_loss=0.0, threshold=0.0, monkeypatch=monkeypatch
    )
    samples = make_samples(sample_count=3, class_count=4)
    losses = {
        mid: float(
            model.per_sample_error(
                torch.cat([s.input_features for s in samples]),
                torch.cat([s.observed_class_labels for s in samples]),
            )
            .mean()
            .item()
        )
        for mid, model in legacy.models.items()
    }
    current_id = max(losses, key=losses.get)
    assert current_id != min(losses, key=losses.get)
    legacy.current_model_id = current_id
    owners["current_training_model_assignment"].assign_model_for_training(model_id=current_id)
    states = owners[
        "held_model_training_state_registry"
    ].snapshot_ordered_held_model_training_states()
    for state in states:
        state.classifier.train(state.model_id != current_id)
        for parameter in state.classifier.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
    original_modes = tuple(state.classifier.training for state in states)
    original_gradients = tuple(
        tuple(p.grad.clone() for p in state.classifier.parameters()) for state in states
    )
    result = run_processing_pair(
        legacy=legacy, processor=processor, owners=owners, samples=samples, processed_sample_count=3
    )
    assert result.adaptation_kind == "new_model_created"
    assert tuple(state.classifier.training for state in states) == original_modes
    for state, gradients in zip(states, original_gradients, strict=True):
        for parameter, gradient in zip(state.classifier.parameters(), gradients, strict=True):
            assert torch.equal(parameter.grad, gradient)


@pytest.mark.parametrize(
    "field,value",
    (
        ("adaptation_kind", "invalid"),
        ("assigned_model_id", True),
        ("assigned_model_id", 0),
        ("first_sample_index", -1),
        ("first_sample_index", True),
        ("last_sample_index", 1),
        ("next_previous_minimum_mean_bounded_loss", float("nan")),
    ),
)
def test_transition_rejects_invalid_values(field, value):
    from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
        FeddriftDetectionBatchTransition,
    )

    result = FeddriftDetectionBatchTransition(
        adaptation_kind="new_model_created",
        assigned_model_id=-100,
        first_sample_index=2,
        last_sample_index=3,
        next_previous_minimum_mean_bounded_loss=0.5,
    )
    with pytest.raises((TypeError, ValueError)):
        replace(result, **{field: value})


def test_non_drift_temporary_model_selection_skips_evaluation_sampling(monkeypatch):
    legacy, processor, owners = build_processing_oracle(
        class_count=2, previous_loss=None, threshold=0.0, monkeypatch=monkeypatch
    )
    with torch.no_grad():
        for model_id, model in legacy.models.items():
            for parameter in model.parameters():
                parameter.zero_()
            tuple(model.parameters())[-1].fill_(2.0 if model_id == -104 else -2.0)
        for state in owners[
            "held_model_training_state_registry"
        ].snapshot_ordered_held_model_training_states():
            for parameter in state.classifier.parameters():
                parameter.zero_()
            state.classifier.classification_layer.bias.fill_(
                2.0 if state.model_id == -104 else -2.0
            )
    random_state = owners["python_random_generator"].getstate()
    samples = (
        ObservedTrainingSample(
            input_features=torch.zeros(1, 3), observed_class_labels=torch.ones(1, 1)
        ),
    )
    result = run_processing_pair(
        legacy=legacy, processor=processor, owners=owners, samples=samples, processed_sample_count=1
    )
    assert result.adaptation_kind == "held_model_selected"
    assert result.assigned_model_id == -104
    assert owners["evaluation_sample_store"].snapshot_ordered_model_evaluation_samples() == ()
    assert owners["python_random_generator"].getstate() == random_state


def assert_optimizer_state_equal(actual, expected):
    """optimizerのTensor値・入れ子構造を省略や丸めなしで照合する。"""
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert tuple(actual) == tuple(expected)
        for key in actual:
            assert_optimizer_state_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for value, expected_value in zip(actual, expected, strict=True):
            assert_optimizer_state_equal(value, expected_value)
    else:
        assert actual == expected


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize("created", (False, True))
def test_existing_parameter_optimizer_identity_and_state_are_preserved(
    class_count, optimizer_name, created, monkeypatch
):
    legacy, processor, owners = build_processing_oracle(
        class_count=class_count,
        previous_loss=0.0 if created else None,
        threshold=0.0,
        monkeypatch=monkeypatch,
    )
    states = owners[
        "held_model_training_state_registry"
    ].snapshot_ordered_held_model_training_states()
    samples = make_samples(sample_count=3, class_count=class_count)
    features = torch.cat([sample.input_features for sample in samples])
    labels = torch.cat([sample.observed_class_labels for sample in samples])
    if optimizer_name == "adam":
        monkeypatch.setattr(config, "OPTIMIZER", "adam")
        monkeypatch.setattr(config, "WEIGHT_DECAY", 0.0)
        monkeypatch.setattr(config, "AMSGRAD", True)
        settings = AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.0, adam_variant="amsgrad"
        )
        for state in states:
            legacy.models[state.model_id].reset_optimizer()
            legacy.models[state.model_id].update(features, labels)
            optimizer_state = ParameterOptimizerState(
                parameters=tuple(state.classifier.parameters()), optimizer_settings=settings
            )
            optimizer = optimizer_state.parameter_optimizer
            optimizer.zero_grad()
            outputs = state.classifier(features)
            loss = (
                torch.nn.functional.binary_cross_entropy(outputs, labels)
                if class_count == 2
                else torch.nn.functional.cross_entropy(outputs, labels.reshape(-1).long())
            )
            loss.backward()
            optimizer.step()
            assert optimizer.state
            owners["held_model_training_state_registry"].register_held_model_training_state(
                model_id=state.model_id,
                classifier=state.classifier,
                concept_specific_parameter_optimizer_state=optimizer_state,
            )
        states = owners[
            "held_model_training_state_registry"
        ].snapshot_ordered_held_model_training_states()
    before = tuple(
        (
            state.classifier,
            tuple(state.classifier.parameters()),
            tuple(parameter.detach().clone() for parameter in state.classifier.parameters()),
            state.concept_specific_parameter_optimizer_state,
            state.concept_specific_parameter_optimizer_state.parameter_optimizer,
            deepcopy(
                state.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict()
            ),
        )
        for state in states
    )
    result = run_processing_pair(
        legacy=legacy, processor=processor, owners=owners, samples=samples, processed_sample_count=3
    )
    assert (result.adaptation_kind == "new_model_created") is created
    for state, (
        classifier,
        parameters,
        parameter_values,
        optimizer_state,
        optimizer,
        optimizer_values,
    ) in zip(states, before, strict=True):
        after = owners["held_model_training_state_registry"].get_held_model_training_state(
            model_id=state.model_id
        )
        assert after.classifier is classifier
        assert after.concept_specific_parameter_optimizer_state is optimizer_state
        assert after.concept_specific_parameter_optimizer_state.parameter_optimizer is optimizer
        assert all(
            actual is expected
            for actual, expected in zip(after.classifier.parameters(), parameters, strict=True)
        )
        for actual, expected in zip(after.classifier.parameters(), parameter_values, strict=True):
            assert torch.equal(actual, expected)
        assert_optimizer_state_equal(optimizer.state_dict(), optimizer_values)


def test_ready_upload_replaces_delayed_slot_without_waiting():
    state = PendingModelUploadState(
        pending_model_registration_policy=KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
    )
    state.queue_model_upload(
        model_id=-100, parameter_snapshot={"a": torch.ones(2)}, upload_delay_round_count=3
    )
    snapshot = {"a": torch.zeros(2)}
    state.queue_ready_model_upload(model_id=-101, parameter_snapshot=snapshot)
    assert state.has_ready_model_upload()
    assert state.remaining_upload_delay_round_count == 0
    assert state.get_pending_model_upload().model_id == -101
    assert state.get_pending_model_upload().parameter_snapshot is snapshot
    state.advance_upload_readiness_at_round_boundary()
    assert state.has_ready_model_upload()


@pytest.mark.parametrize(
    "model_id,snapshot",
    ((True, {"a": torch.ones(1)}), (-100, {}), (-100, {"a": torch.tensor([float("nan")])})),
)
def test_invalid_ready_upload_preserves_previous_slot(model_id, snapshot):
    state = PendingModelUploadState(
        pending_model_registration_policy=KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
    )
    state.queue_model_upload(
        model_id=-99, parameter_snapshot={"a": torch.ones(2)}, upload_delay_round_count=2
    )
    previous = state.get_pending_model_upload()
    with pytest.raises((TypeError, ValueError)):
        state.queue_ready_model_upload(model_id=model_id, parameter_snapshot=snapshot)
    assert state.get_pending_model_upload() is previous
    assert state.remaining_upload_delay_round_count == 2
