"""実旧の時刻ブロック処理・終端処理と新streamの値/状態を照合する。"""

import random
from dataclasses import replace

import pytest
import torch

from federated_drift_experiment import config
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.runtime.feddrift_client_stream_processing import (
    FeddriftClientStreamProcessor,
)
from tests.refactoring.test_feddrift_detection_batch_processing import (
    assert_processing_matches_legacy,
    build_processing_oracle,
    make_samples,
    snapshot_processing_owners,
)


def build_stream(*, monkeypatch, class_count=2, batch_size=3, previous_loss=None, threshold=1.0):
    monkeypatch.setattr(config, "FEDDRIFT_DETECTION_BATCH_SIZE", batch_size)
    legacy, processor, owners = build_processing_oracle(
        class_count=class_count,
        previous_loss=previous_loss,
        threshold=threshold,
        monkeypatch=monkeypatch,
    )
    stream = FeddriftClientStreamProcessor(
        detection_batch_processor=processor, detection_batch_sample_count=batch_size
    )
    legacy._test_predicted_classes = []
    original_record_prediction = legacy._record_prediction

    def record_prediction_with_class(x, y, concept_id):
        # 実旧の予測値を保存し、旧の本体も必ず呼ぶ。旧判断・割当を代役にしない。
        predicted = legacy.models[legacy.current_model_id].predict(x)
        legacy._test_predicted_classes.append(int(predicted.view(-1)[0].item()))
        original_record_prediction(x, y, concept_id)

    monkeypatch.setattr(legacy, "_record_prediction", record_prediction_with_class)
    return legacy, processor, owners, stream


def observations(*, start, count, class_count):
    return tuple(
        IndexedObservedTrainingSample(
            sample_index=start + i,
            training_sample=sample,
            observed_concept_id=(None, -2, 4)[i % 3],
        )
        for i, sample in enumerate(make_samples(sample_count=count, class_count=class_count))
    )


def run_pair(*, legacy, owners, stream, block=None):
    """実旧と新のTorch/Python RNGを同じ位置から進める（旧処理の代役なし）。"""
    python_before = random.getstate()
    try:
        random.setstate(owners["python_random_generator"].getstate())
        with torch.random.fork_rng():
            torch_before = torch.get_rng_state()
            if block is None:
                expected = legacy.flush()
            else:
                expected = legacy.process_batch(
                    [
                        (x.training_sample.input_features, x.training_sample.observed_class_labels)
                        for x in block
                    ],
                    [x.observed_concept_id for x in block],
                )
            torch_after = torch.get_rng_state()
            python_after = random.getstate()
            torch.set_rng_state(torch_before)
            if block is None:
                actual = stream.flush_detection_buffer()
            else:
                actual = stream.process_observations(indexed_observations=block)
                assert actual is expected
            assert torch.equal(torch.get_rng_state(), torch_after)
            assert owners["python_random_generator"].getstate() == python_after
    finally:
        random.setstate(python_before)
    return actual


def assert_stream_matches(*, legacy, processor, owners, stream):
    assert_processing_matches_legacy(legacy=legacy, processor=processor, owners=owners)
    assert stream.processed_sample_count == legacy.processed_samples
    assert len(stream.buffered_training_samples) == len(legacy.detect_buffer)
    for new, old in zip(stream.buffered_training_samples, legacy.detect_buffer, strict=True):
        assert torch.equal(new.input_features, old[0])
        assert torch.equal(new.observed_class_labels, old[1])
    records = stream.sample_predictions
    assert tuple(x.sample_index for x in records) == tuple(range(legacy.processed_samples))
    assert tuple(float(x.prediction_is_correct) for x in records) == tuple(legacy.history_accuracy)
    assert tuple(x.observed_concept_id for x in records) == tuple(legacy.history_concept)
    assert tuple(x.prediction_model_id for x in records) == tuple(legacy.history_model_id)
    assert tuple(x.predicted_class_id for x in records) == tuple(legacy._test_predicted_classes)
    drift_codes = {"current_model_retained": 0, "held_model_selected": 1, "new_model_created": 2}
    assert tuple(
        0
        if x.detection_batch_transition is None
        else drift_codes[x.detection_batch_transition.adaptation_kind]
        for x in records
    ) == tuple(legacy.history_drift_type)
    transitions = stream.completed_detection_batch_transitions
    assert tuple(
        x.first_sample_index for x in transitions if x.adaptation_kind == "new_model_created"
    ) == tuple(legacy.detected_event_positions)
    assert tuple(
        x.last_sample_index for x in transitions if x.adaptation_kind != "current_model_retained"
    ) == tuple(legacy.local_switch_positions)


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("batch_size", (1, 3, 5))
@pytest.mark.parametrize("previous_loss,threshold", ((None, 1.0), (0.0, 0.0)))
def test_real_legacy_stream_across_timesteps_and_flush(
    monkeypatch, class_count, batch_size, previous_loss, threshold
):
    legacy, processor, owners, stream = build_stream(
        monkeypatch=monkeypatch,
        class_count=class_count,
        batch_size=batch_size,
        previous_loss=previous_loss,
        threshold=threshold,
    )
    registry = owners["held_model_training_state_registry"]
    before = registry.snapshot_ordered_held_model_training_states()
    parameters = tuple(tuple(x.classifier.parameters()) for x in before)
    optimizer_states = tuple(
        x.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict()
        for x in before
    )
    for count in (2, 0, 7, 1, 2):
        block = observations(
            start=stream.processed_sample_count, count=count, class_count=class_count
        )
        run_pair(legacy=legacy, owners=owners, stream=stream, block=block)
        assert_stream_matches(legacy=legacy, processor=processor, owners=owners, stream=stream)
    before_flush_records = stream.sample_predictions
    run_pair(legacy=legacy, owners=owners, stream=stream)
    assert stream.sample_predictions == before_flush_records
    assert_stream_matches(legacy=legacy, processor=processor, owners=owners, stream=stream)
    assert run_pair(legacy=legacy, owners=owners, stream=stream) is None
    for state, refs, optimizer_state in zip(before, parameters, optimizer_states, strict=True):
        assert registry.get_held_model_training_state(model_id=state.model_id) is state
        assert all(new is old for new, old in zip(state.classifier.parameters(), refs, strict=True))
        assert (
            state.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict()
            == optimizer_state
        )


@pytest.mark.parametrize(
    "bad_kind",
    (
        "list",
        "none",
        "record",
        "bool_position",
        "gap",
        "repeat",
        "negative_position",
        "bool_concept",
        "float_concept",
        "sample",
        "features_shape",
        "two_samples",
        "features_nan",
        "features_float64",
        "labels_shape",
        "labels_nan",
        "labels_fraction",
        "labels_range",
        "labels_float64",
        "current_missing",
        "empty_registry",
        "parameter_nan",
    ),
)
def test_invalid_later_observation_rejected_before_forward_and_all_updates(monkeypatch, bad_kind):
    _, processor, owners, stream = build_stream(monkeypatch=monkeypatch, batch_size=2)
    stream.process_observations(indexed_observations=observations(start=0, count=1, class_count=2))
    block = observations(start=1, count=2, class_count=2)
    later = block[-1]
    if bad_kind in ("list", "none", "record"):
        block = {"list": list(block), "none": None, "record": (block[0], None)}[bad_kind]
    elif bad_kind in ("bool_position", "gap", "repeat", "negative_position"):
        later = replace(
            later,
            sample_index={"bool_position": True, "gap": 3, "repeat": 1, "negative_position": -1}[
                bad_kind
            ],
        )
    elif bad_kind in ("bool_concept", "float_concept"):
        later = replace(later, observed_concept_id=True if bad_kind == "bool_concept" else 2.0)
    elif bad_kind == "sample":
        later = replace(later, training_sample=None)
    elif bad_kind.startswith("features") or bad_kind in ("two_samples",):
        features = later.training_sample.input_features.clone()
        if bad_kind == "features_shape":
            features = features.view(-1)
        elif bad_kind == "two_samples":
            features = features.repeat(2, 1)
        elif bad_kind == "features_nan":
            features[0, 0] = float("nan")
        else:
            features = features.double()
        later = replace(
            later, training_sample=replace(later.training_sample, input_features=features)
        )
    elif bad_kind.startswith("labels"):
        labels = later.training_sample.observed_class_labels.clone()
        if bad_kind == "labels_shape":
            labels = labels.view(-1)
        elif bad_kind == "labels_float64":
            labels = labels.double()
        else:
            labels[0, 0] = {
                "labels_nan": float("nan"),
                "labels_fraction": 0.5,
                "labels_range": 2.0,
            }[bad_kind]
        later = replace(
            later, training_sample=replace(later.training_sample, observed_class_labels=labels)
        )
    elif bad_kind == "current_missing":
        owners["current_training_model_assignment"].assign_model_for_training(model_id=999)
    elif bad_kind == "empty_registry":
        owners["held_model_training_state_registry"]._held_model_training_states_by_model_id.clear()
    elif bad_kind == "parameter_nan":
        with torch.no_grad():
            next(processor.current_training_model_state.classifier.parameters()).view(-1)[0] = (
                float("nan")
            )
    if isinstance(block, tuple) and bad_kind != "record":
        block = (block[0], later)
    before = snapshot_processing_owners(processor, owners)
    stream_before = (
        stream.processed_sample_count,
        stream.buffered_training_samples,
        stream.sample_predictions,
        stream.completed_detection_batch_transitions,
    )
    torch_before = torch.get_rng_state()
    calls = []
    for state in owners[
        "held_model_training_state_registry"
    ].snapshot_ordered_held_model_training_states():
        state.classifier.register_forward_pre_hook(lambda *_: calls.append(True))
    with pytest.raises((TypeError, ValueError, LookupError)):
        stream.process_observations(indexed_observations=block)
    assert not calls
    assert snapshot_processing_owners(processor, owners) == before
    assert (
        stream.processed_sample_count,
        stream.buffered_training_samples,
        stream.sample_predictions,
        stream.completed_detection_batch_transitions,
    ) == stream_before
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize("bad_size", (True, 0, -1, 2.0, None))
def test_invalid_detection_size_does_not_mutate_owners(monkeypatch, bad_size):
    _, processor, owners, _ = build_stream(monkeypatch=monkeypatch)
    before = snapshot_processing_owners(processor, owners)
    with pytest.raises(ValueError):
        FeddriftClientStreamProcessor(
            detection_batch_processor=processor, detection_batch_sample_count=bad_size
        )
    assert snapshot_processing_owners(processor, owners) == before


def test_wrong_processor_rejected():
    with pytest.raises(TypeError):
        FeddriftClientStreamProcessor(
            detection_batch_processor=None, detection_batch_sample_count=3
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"sample_index": True},
        {"sample_index": -1},
        {"prediction_model_id": False},
        {"observed_concept_id": True},
        {"predicted_class_id": -1},
        {"observed_class_id": 0.0},
        {"prediction_is_correct": 1},
        {"prediction_is_correct": False},
        {"detection_batch_transition": object()},
    ),
)
def test_prediction_record_rejects_invalid_fields(changes):
    from federated_learning_experiments.runtime.feddrift_client_stream_processing import (
        FeddriftSamplePrediction,
    )

    values = dict(
        sample_index=0,
        observed_concept_id=None,
        prediction_model_id=-106,
        predicted_class_id=0,
        observed_class_id=0,
        prediction_is_correct=True,
    )
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        FeddriftSamplePrediction(**values)


def test_prediction_record_transition_position_and_snapshot_immutability(monkeypatch):
    from dataclasses import FrozenInstanceError

    from federated_learning_experiments.runtime.feddrift_client_stream_processing import (
        FeddriftSamplePrediction,
    )
    from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
        FeddriftDetectionBatchTransition,
    )

    transition = FeddriftDetectionBatchTransition(
        adaptation_kind="held_model_selected",
        assigned_model_id=2,
        first_sample_index=0,
        last_sample_index=1,
        next_previous_minimum_mean_bounded_loss=0.2,
    )
    with pytest.raises(ValueError):
        FeddriftSamplePrediction(
            sample_index=0,
            observed_concept_id=None,
            prediction_model_id=9,
            predicted_class_id=0,
            observed_class_id=0,
            prediction_is_correct=True,
            detection_batch_transition=transition,
        )
    _, _, _, stream = build_stream(monkeypatch=monkeypatch)
    block = observations(start=0, count=1, class_count=2)
    stream.process_observations(indexed_observations=block)
    snapshot = stream.sample_predictions
    buffer = stream.buffered_training_samples
    assert buffer[0] is block[0].training_sample
    with pytest.raises(FrozenInstanceError):
        snapshot[0].sample_index = 4
    stream.process_observations(indexed_observations=observations(start=1, count=3, class_count=2))
    assert len(snapshot) == len(buffer) == 1
    assert len(stream.sample_predictions) == 4


def test_flush_revalidates_borrowed_sample_before_updates(monkeypatch):
    _, processor, owners, stream = build_stream(monkeypatch=monkeypatch)
    stream.process_observations(indexed_observations=observations(start=0, count=1, class_count=2))
    stream.buffered_training_samples[0].input_features[0, 0] = float("nan")
    before = snapshot_processing_owners(processor, owners)
    records = stream.sample_predictions
    with pytest.raises(ValueError):
        stream.flush_detection_buffer()
    assert snapshot_processing_owners(processor, owners) == before
    assert stream.sample_predictions == records
    assert len(stream.buffered_training_samples) == 1


def test_incompatible_held_class_count_rejected_before_prediction(monkeypatch):
    from federated_learning_experiments.learning.training.parameter_optimizer_state import (
        ParameterOptimizerState,
    )

    _, processor, owners, stream = build_stream(monkeypatch=monkeypatch)
    current = processor.current_training_model_state.classifier
    other = type(current)(
        model_architecture_settings=current.model_architecture_settings,
        input_feature_count=current.input_feature_count,
        hidden_layer_widths=current.hidden_layer_widths,
        class_count=4,
    )
    owners["held_model_training_state_registry"].register_held_model_training_state(
        model_id=12,
        classifier=other,
        concept_specific_parameter_optimizer_state=ParameterOptimizerState(
            parameters=tuple(other.parameters()),
            optimizer_settings=owners["new_model_optimizer_settings"],
        ),
    )
    before = snapshot_processing_owners(processor, owners)
    calls = []
    current.register_forward_pre_hook(lambda *_: calls.append(True))
    with pytest.raises(ValueError):
        stream.process_observations(
            indexed_observations=observations(start=0, count=1, class_count=2)
        )
    assert not calls
    assert snapshot_processing_owners(processor, owners) == before
    assert stream.processed_sample_count == 0


@pytest.mark.parametrize("class_count", (2, 4))
def test_prediction_uses_pre_transition_current_model_and_flush_preserves_records(
    monkeypatch, class_count
):
    legacy, processor, owners, stream = build_stream(
        monkeypatch=monkeypatch,
        class_count=class_count,
        batch_size=3,
        previous_loss=0.0,
        threshold=0.0,
    )
    block = observations(start=0, count=2, class_count=class_count)
    run_pair(legacy=legacy, owners=owners, stream=stream, block=block)
    before = stream.sample_predictions
    assert tuple(x.prediction_model_id for x in before) == (9, 9)
    assert all(x.detection_batch_transition is None for x in before)
    result = run_pair(legacy=legacy, owners=owners, stream=stream)
    assert result.adaptation_kind == "new_model_created"
    assert (result.first_sample_index, result.last_sample_index) == (0, 1)
    assert stream.sample_predictions == before
    run_pair(
        legacy=legacy,
        owners=owners,
        stream=stream,
        block=observations(start=2, count=1, class_count=class_count),
    )
    assert stream.sample_predictions[-1].prediction_model_id == -106
    assert_stream_matches(legacy=legacy, processor=processor, owners=owners, stream=stream)


@pytest.mark.parametrize("class_count", (2, 4))
def test_external_current_assignment_is_read_before_next_prediction(monkeypatch, class_count):
    legacy, processor, owners, stream = build_stream(
        monkeypatch=monkeypatch, class_count=class_count
    )
    run_pair(
        legacy=legacy,
        owners=owners,
        stream=stream,
        block=observations(start=0, count=1, class_count=class_count),
    )
    legacy.current_model_id = -104
    owners["current_training_model_assignment"].assign_model_for_training(model_id=-104)
    run_pair(
        legacy=legacy,
        owners=owners,
        stream=stream,
        block=observations(start=1, count=1, class_count=class_count),
    )
    assert stream.sample_predictions[-1].prediction_model_id == -104
    assert_stream_matches(legacy=legacy, processor=processor, owners=owners, stream=stream)
