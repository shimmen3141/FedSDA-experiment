"""警報の5応答を実旧の処理と照合する。"""

import random
from collections import defaultdict, deque
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    assert_held_model_states_match_legacy,
)
from test_alarm_change_interval_resolution import (
    assert_alarm_change_interval_resolution_matches_legacy,
    assert_alarm_change_interval_resolution_state_unchanged,
    snapshot_alarm_change_interval_resolution_state,
    use_immediate_candidate_adoption,
)
from test_alarm_training_interval_preparation import (
    assert_evaluation_samples_match_legacy,
    build_alarm_preparation_oracle,
)
from test_assigned_training_sample_absorption import assert_absorption_matches_legacy
from test_joint_model_parameter_update import (
    run_legacy_joint_update,
)

import federated_learning_experiments.runtime.alarm_buffer_response as response_module
from federated_drift_experiment import config
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.participating_model_training_batch import (
    ParticipatingModelTrainingBatch,
)
from federated_learning_experiments.runtime.alarm_buffer_response import (
    AlarmBufferResponse,
)
from federated_learning_experiments.runtime.alarm_change_interval_resolution import (
    resolve_alarm_change_interval,
)
from federated_learning_experiments.runtime.alarm_training_interval_preparation import (
    prepare_alarm_training_intervals,
)


def build_buffer_response_oracle(
    *,
    monkeypatch,
    class_count=2,
    alarm_interval_resolution_case="current_model_maintained",
    earlier_sample_count=3,
    estimated_change_span_sample_count=3,
    minimum_change_interval_sample_count=3,
):
    monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
    (
        preparation_arguments,
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
    ) = build_alarm_preparation_oracle(
        monkeypatch=monkeypatch,
        class_count=class_count,
        alarm_interval_resolution_case=alarm_interval_resolution_case,
        earlier_sample_count=earlier_sample_count,
        estimated_change_span_sample_count=estimated_change_span_sample_count,
    )
    legacy_client._forward_validation = None
    legacy_client.verbose = False
    legacy_client.reuse_selection_counts = defaultdict(int)
    legacy_client.distance_threshold = resolution_arguments[
        "maximum_alarm_interval_mean_loss_increase"
    ]
    response_arguments = dict(resolution_arguments)
    del response_arguments["change_interval_training_samples"]
    del response_arguments["change_interval_sample_concept_ids"]
    response_arguments.update(preparation_arguments)
    response_arguments.update(
        active_validation_session=None,
        minimum_change_interval_sample_count=minimum_change_interval_sample_count,
    )
    return (
        response_arguments,
        preparation_arguments,
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
    )


def run_legacy_buffer_response(*, response_arguments, legacy_client, monkeypatch):
    global_python_random_state = random.getstate()
    legacy_adaptation_events = []
    detector_reset_calls = []
    span_calls = []

    def record_legacy_adaptation_event(**event_fields):
        legacy_adaptation_events.append(event_fields)

    def provide_estimated_change_span(sample_index):
        span_calls.append(sample_index)
        return response_arguments["estimated_change_span_sample_count"]

    legacy_client._estimated_new_concept_span = provide_estimated_change_span
    legacy_client._record_adaptation_event = record_legacy_adaptation_event
    legacy_client._reset_drift_detectors = lambda: detector_reset_calls.append(True)
    legacy_client.distance_threshold = response_arguments[
        "maximum_alarm_interval_mean_loss_increase"
    ]
    legacy_client.buffer = deque(
        (
            indexed_observation.training_sample.input_features,
            indexed_observation.training_sample.observed_class_labels,
            indexed_observation.observed_concept_id,
        )
        for indexed_observation in response_arguments["pending_sample_observations"]
    )
    monkeypatch.setattr(
        config,
        "MIN_DRIFT_DATA",
        response_arguments["minimum_change_interval_sample_count"],
    )
    try:
        random.setstate(response_arguments["python_random_generator"].getstate())
        with patch.object(
            legacy_client,
            "_update_new_model_epochs",
            wraps=legacy_client._update_new_model_epochs,
        ) as legacy_epoch_training_calls:
            legacy_drift_type = legacy_client._resolve_drift(
                response_arguments["proposal_sample_index"],
                response_arguments["estimated_change_point_sample_index"],
                response_arguments["detection_episode_id"],
            )
        legacy_python_random_state = random.getstate()
    finally:
        random.setstate(global_python_random_state)
    assert detector_reset_calls == [True]
    assert span_calls == (
        []
        if legacy_client._forward_validation is not None
        and legacy_adaptation_events[0]["action"] == "forward_validation_pending"
        else [response_arguments["proposal_sample_index"]]
    )
    return (
        legacy_drift_type,
        legacy_adaptation_events,
        legacy_epoch_training_calls,
        legacy_python_random_state,
    )


def assert_buffer_response_matches_legacy(
    *,
    response,
    response_arguments,
    preparation_arguments,
    resolution_arguments,
    shared_optimizer_owners,
    legacy_client,
    legacy_result,
    initial_training_model_id,
):
    (
        legacy_drift_type,
        legacy_adaptation_events,
        legacy_epoch_training_calls,
        legacy_python_random_state,
    ) = legacy_result
    assert type(response) is AlarmBufferResponse
    assert (
        response_arguments["current_training_model_assignment"].current_training_model_id
        == legacy_client.current_model_id
    )
    assert_absorption_matches_legacy(
        absorption_arguments=response_arguments, legacy_client=legacy_client
    )
    assert_evaluation_samples_match_legacy(
        preparation_arguments=preparation_arguments, legacy_client=legacy_client
    )
    assert_held_model_states_match_legacy(
        registry=response_arguments["held_model_training_state_registry"],
        shared_optimizer_owners=shared_optimizer_owners[:-1],
        legacy_client=legacy_client,
        input_features=response_arguments["pending_sample_observations"][
            0
        ].training_sample.input_features
        if response_arguments["pending_sample_observations"]
        else resolution_arguments["change_interval_training_samples"][0].input_features,
    )
    assert response_arguments["python_random_generator"].getstate() == legacy_python_random_state
    assert response.pending_assignment_buffer_should_be_cleared == (
        len(legacy_client.buffer) == 0
        and legacy_adaptation_events[0]["action"] != "insufficient_data"
    )
    if response.change_interval_resolution is not None:
        resolution_arguments["change_interval_training_samples"] = tuple(
            indexed_observation.training_sample
            for indexed_observation in response.prepared_alarm_training_intervals.change_interval_observations
        )
        resolution_arguments["change_interval_sample_concept_ids"] = tuple(
            indexed_observation.observed_concept_id
            for indexed_observation in response.prepared_alarm_training_intervals.change_interval_observations
        )
        if response.active_validation_session is not None:
            assert (
                response.active_validation_session.pending_assignment_training_samples
                == resolution_arguments["change_interval_training_samples"]
            )
            # 下位helperは呼出し元から借用したtuple自体の参照一致を検査する。
            resolution_arguments["change_interval_training_samples"] = (
                response.active_validation_session.pending_assignment_training_samples
            )
        assert_alarm_change_interval_resolution_matches_legacy(
            alarm_change_interval_resolution=response.change_interval_resolution,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            legacy_adaptation_events=legacy_adaptation_events,
            legacy_epoch_training_calls=legacy_epoch_training_calls,
            initial_training_model_id=initial_training_model_id,
        )
        assert (
            response.active_validation_session
            is response.change_interval_resolution.started_validation_session
        )
    else:
        assert legacy_drift_type == 0
        assert len(legacy_epoch_training_calls.call_args_list) == 0
        assert legacy_adaptation_events[0]["action"] == (
            "forward_validation_pending"
            if response.response_outcome == "alarm_during_candidate_validation"
            else "insufficient_data"
        )


def continue_joint_training_after_buffer_response(
    *, response_arguments, shared_optimizer_owners, legacy_client
):
    registry = response_arguments["held_model_training_state_registry"]
    training_sample_store = response_arguments["training_sample_store"]
    counts_store = response_arguments["model_training_and_assignment_counts_store"]
    local_training_settings = LocalTrainingSettings(
        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
    )
    legacy_training_batches = []
    legacy_client.updates_per_sample = 1
    legacy_client.backbone_gradient_diagnostics = defaultdict(float)
    legacy_client._sample_training_batches = lambda: legacy_training_batches
    update_shared_features = True
    change_interval_input_features = next(iter(legacy_client.train_data_store.values()))[0][0]
    # 両実装とも、各自の標本storeにある全標本をモデルごとの固定batchにする。
    training_bindings = registry.snapshot_ordered_held_model_training_bindings()
    model_training_sample_collections = (
        training_sample_store.snapshot_ordered_model_training_samples()
    )
    legacy_training_batches[:] = [
        (
            model_id,
            torch.cat(
                [legacy_training_sample[0] for legacy_training_sample in legacy_training_samples]
            ),
            torch.cat(
                [legacy_training_sample[1] for legacy_training_sample in legacy_training_samples]
            ),
        )
        for model_id, legacy_training_samples in legacy_client.train_data_store.items()
    ]
    assert tuple(training_binding.model_id for training_binding in training_bindings) == tuple(
        collection.model_id for collection in model_training_sample_collections
    )
    participating_training_batches = tuple(
        ParticipatingModelTrainingBatch(
            classifier=training_binding.classifier,
            concept_specific_parameter_optimizer=training_binding.concept_specific_parameter_optimizer,
            input_features=torch.cat(
                [training_sample.input_features for training_sample in collection.training_samples]
            ),
            observed_class_labels=torch.cat(
                [
                    training_sample.observed_class_labels
                    for training_sample in collection.training_samples
                ]
            ),
        )
        for training_binding, collection in zip(
            training_bindings, model_training_sample_collections
        )
    )
    expected_joint_loss = run_legacy_joint_update(
        legacy_client=legacy_client, update_shared_features=update_shared_features
    )
    actual_joint_loss = perform_joint_model_parameter_update(
        local_training_settings=local_training_settings,
        shared_feature_extractor=training_bindings[0].classifier.feature_extractor,
        shared_parameter_optimizer=shared_optimizer_owners[0].parameter_optimizer,
        participating_training_batches=participating_training_batches,
        update_shared_features=update_shared_features,
    )
    assert actual_joint_loss == expected_joint_loss
    for training_binding, training_batch in zip(training_bindings, participating_training_batches):
        counts_store.record_completed_model_training(
            model_id=training_binding.model_id,
            trained_sample_count=len(training_batch.input_features),
            parameter_update_step_count=1,
        )
    assert_held_model_states_match_legacy(
        registry=registry,
        shared_optimizer_owners=shared_optimizer_owners[:-1],
        legacy_client=legacy_client,
        input_features=change_interval_input_features,
    )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize(
    "alarm_interval_resolution_case",
    ["other_model_reused", "current_model_maintained", "no_model_fits"],
)
@pytest.mark.parametrize(
    "earlier_sample_count,estimated_change_span_sample_count,minimum_change_interval_sample_count",
    [(3, 3, 3), (3, 2, 3), (0, 99, 3), (3, 6, 6), (3, 6, 7), (3, 99, 3)],
)
def test_buffer_response_matches_real_legacy(
    class_count,
    alarm_interval_resolution_case,
    earlier_sample_count,
    estimated_change_span_sample_count,
    minimum_change_interval_sample_count,
    monkeypatch,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(811)
        (
            response_arguments,
            preparation_arguments,
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
        ) = build_buffer_response_oracle(
            monkeypatch=monkeypatch,
            class_count=class_count,
            alarm_interval_resolution_case=alarm_interval_resolution_case,
            earlier_sample_count=earlier_sample_count,
            estimated_change_span_sample_count=estimated_change_span_sample_count,
            minimum_change_interval_sample_count=minimum_change_interval_sample_count,
        )
        use_immediate_candidate_adoption(
            monkeypatch, response_arguments, resolution_arguments, legacy_client=legacy_client
        )
        initial_training_model_id = legacy_client.current_model_id
        initial_torch_random_state = torch.get_rng_state().clone()
        numpy_random_state = np.random.get_state()
        pending_assignment_snapshot = response_arguments[
            "pending_training_assignment_buffer"
        ].get_state_snapshot()
        legacy_result = run_legacy_buffer_response(
            response_arguments=response_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        expected_torch_random_state = torch.get_rng_state().clone()
        assert np.random.get_state()[0] == numpy_random_state[0]
        assert np.array_equal(np.random.get_state()[1], numpy_random_state[1])
        assert np.random.get_state()[2:] == numpy_random_state[2:]
        torch.set_rng_state(initial_torch_random_state)
        response = response_module.respond_to_alarm_with_buffered_samples(**response_arguments)
        assert torch.equal(torch.get_rng_state(), expected_torch_random_state)
        assert np.random.get_state()[0] == numpy_random_state[0]
        assert np.array_equal(np.random.get_state()[1], numpy_random_state[1])
        assert np.random.get_state()[2:] == numpy_random_state[2:]
        assert (
            response_arguments["pending_training_assignment_buffer"].get_state_snapshot()
            == pending_assignment_snapshot
        )
        assert_buffer_response_matches_legacy(
            response=response,
            response_arguments=response_arguments,
            preparation_arguments=preparation_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_result=legacy_result,
            initial_training_model_id=initial_training_model_id,
        )

        for _ in range(2):
            continue_joint_training_after_buffer_response(
                response_arguments=response_arguments,
                shared_optimizer_owners=shared_optimizer_owners,
                legacy_client=legacy_client,
            )


def test_empty_buffer_is_insufficient_without_candidate_work(monkeypatch):
    (
        response_arguments,
        _,
        _,
        shared_optimizer_owners,
        _,
    ) = build_buffer_response_oracle(monkeypatch=monkeypatch)
    response_arguments["pending_training_assignment_buffer"].drain_pending_sample_indices()
    response_arguments["pending_sample_observations"] = ()
    previous_snapshot = snapshot_alarm_change_interval_resolution_state(
        resolution_arguments=response_arguments,
        shared_optimizer_owners=shared_optimizer_owners,
    )
    with patch.object(
        response_module,
        "resolve_alarm_change_interval",
        side_effect=AssertionError("must not resolve"),
    ):
        response = response_module.respond_to_alarm_with_buffered_samples(**response_arguments)
    assert response.response_outcome == "alarm_change_interval_too_short"
    assert not response.pending_assignment_buffer_should_be_cleared
    assert response.prepared_alarm_training_intervals.change_interval_observations == ()
    assert_alarm_change_interval_resolution_state_unchanged(previous_snapshot)


@pytest.mark.parametrize(
    "invalid_case",
    [
        "session",
        "fifo",
        "observations",
        "record",
        "index_bool",
        "index_negative",
        "missing",
        "extra",
        "reverse",
        "minimum_bool",
        "minimum_float",
        "minimum_zero",
        "minimum_negative",
        "span_bool",
        "span_zero",
        "assignment",
    ],
)
def test_buffer_response_rejects_owned_inputs_before_updates(invalid_case, monkeypatch):
    (
        response_arguments,
        _,
        _,
        shared_optimizer_owners,
        _,
    ) = build_buffer_response_oracle(monkeypatch=monkeypatch)
    previous_snapshot = snapshot_alarm_change_interval_resolution_state(
        resolution_arguments=response_arguments,
        shared_optimizer_owners=shared_optimizer_owners,
    )
    evaluation_snapshot = response_arguments[
        "model_evaluation_sample_store"
    ].snapshot_ordered_model_evaluation_samples()
    python_random_state = response_arguments["python_random_generator"].getstate()
    pending_assignment_snapshot = response_arguments[
        "pending_training_assignment_buffer"
    ].get_state_snapshot()
    valid_response_arguments = dict(response_arguments)
    observations = response_arguments["pending_sample_observations"]
    if invalid_case == "session":
        response_arguments["active_validation_session"] = object()
    elif invalid_case == "fifo":
        response_arguments["pending_training_assignment_buffer"] = object()
    elif invalid_case == "observations":
        response_arguments["pending_sample_observations"] = list(observations)
    elif invalid_case == "record":
        response_arguments["pending_sample_observations"] = (object(),) + observations[1:]
    elif invalid_case == "index_bool":
        response_arguments["pending_sample_observations"] = (
            replace(observations[0], sample_index=True),
        ) + observations[1:]
    elif invalid_case == "index_negative":
        response_arguments["pending_sample_observations"] = (
            replace(observations[0], sample_index=-1),
        ) + observations[1:]
    elif invalid_case == "missing":
        response_arguments["pending_sample_observations"] = observations[:-1]
    elif invalid_case == "extra":
        response_arguments["pending_sample_observations"] = observations + (observations[-1],)
    elif invalid_case == "reverse":
        response_arguments["pending_sample_observations"] = observations[::-1]
    elif invalid_case.startswith("minimum_"):
        response_arguments["minimum_change_interval_sample_count"] = {
            "minimum_bool": True,
            "minimum_float": 2.0,
            "minimum_zero": 0,
            "minimum_negative": -1,
        }[invalid_case]
    elif invalid_case.startswith("span_"):
        response_arguments["estimated_change_span_sample_count"] = {
            "span_bool": True,
            "span_zero": 0,
        }[invalid_case]
    else:
        response_arguments["current_training_model_assignment"] = object()
    expected_exception = (
        ValueError
        if invalid_case
        in {
            "index_negative",
            "missing",
            "extra",
            "reverse",
            "minimum_zero",
            "minimum_negative",
            "span_zero",
        }
        else TypeError
    )
    with pytest.raises(expected_exception):
        response_module.respond_to_alarm_with_buffered_samples(**response_arguments)
    assert_alarm_change_interval_resolution_state_unchanged(
        previous_snapshot,
    )
    assert (
        valid_response_arguments[
            "model_evaluation_sample_store"
        ].snapshot_ordered_model_evaluation_samples()
        == evaluation_snapshot
    )
    assert valid_response_arguments["python_random_generator"].getstate() == python_random_state
    assert (
        valid_response_arguments["pending_training_assignment_buffer"].get_state_snapshot()
        == pending_assignment_snapshot
    )


def test_response_record_rejects_inconsistent_fields_and_is_frozen(monkeypatch):
    _, preparation_arguments, _, _, _ = build_buffer_response_oracle(
        monkeypatch=monkeypatch, minimum_change_interval_sample_count=999
    )
    prepared_intervals = prepare_alarm_training_intervals(**preparation_arguments)
    response = AlarmBufferResponse(
        response_outcome="alarm_change_interval_too_short",
        prepared_alarm_training_intervals=prepared_intervals,
        change_interval_resolution=None,
        active_validation_session=None,
    )
    with pytest.raises(FrozenInstanceError):
        response.response_outcome = "anything"
    with pytest.raises(TypeError):
        AlarmBufferResponse(
            response.response_outcome,
            response.prepared_alarm_training_intervals,
            None,
            None,
        )
    for response_outcome in [
        "forward_validation_pending",
        "insufficient_data",
        "reuse",
        "maintain",
        "create_pending",
        None,
    ]:
        with pytest.raises(ValueError):
            replace(response, response_outcome=response_outcome)
    with pytest.raises(ValueError):
        replace(response, response_outcome="alarm_during_candidate_validation")
    with pytest.raises(ValueError):
        replace(response, prepared_alarm_training_intervals=None)
    with pytest.raises(ValueError):
        replace(response, active_validation_session=object())
    with pytest.raises(ValueError):
        replace(response, response_outcome="alarm_interval_held_model_reused")

    # 本taskでは応答関数を使わず、既存の公開解決から全ての有効recordを構成する。
    for alarm_interval_resolution_case in (
        "other_model_reused",
        "current_model_maintained",
        "no_model_fits",
    ):
        _, preparation_arguments, resolution_arguments, _, _ = build_buffer_response_oracle(
            monkeypatch=monkeypatch,
            alarm_interval_resolution_case=alarm_interval_resolution_case,
        )
        prepared_intervals = prepare_alarm_training_intervals(**preparation_arguments)
        resolution_arguments["change_interval_training_samples"] = tuple(
            indexed_observation.training_sample
            for indexed_observation in prepared_intervals.change_interval_observations
        )
        resolution_arguments["change_interval_sample_concept_ids"] = tuple(
            indexed_observation.observed_concept_id
            for indexed_observation in prepared_intervals.change_interval_observations
        )
        change_interval_resolution = resolve_alarm_change_interval(**resolution_arguments)
        response = AlarmBufferResponse(
            response_outcome=change_interval_resolution.resolution_outcome,
            prepared_alarm_training_intervals=prepared_intervals,
            change_interval_resolution=change_interval_resolution,
            active_validation_session=change_interval_resolution.started_validation_session,
        )
        assert response.pending_assignment_buffer_should_be_cleared is True
        with pytest.raises(ValueError):
            replace(response, prepared_alarm_training_intervals=None)
        with pytest.raises(ValueError):
            replace(response, change_interval_resolution=None)
        with pytest.raises(ValueError):
            replace(response, active_validation_session=object())
        if response.active_validation_session is not None:
            session = response.active_validation_session
            response = AlarmBufferResponse(
                response_outcome="alarm_during_candidate_validation",
                prepared_alarm_training_intervals=None,
                change_interval_resolution=None,
                active_validation_session=session,
            )
            assert response.active_validation_session is session
            assert response.pending_assignment_buffer_should_be_cleared is True
            with pytest.raises(ValueError):
                replace(response, active_validation_session=None)
            with pytest.raises(ValueError):
                replace(response, prepared_alarm_training_intervals=prepared_intervals)


def test_prepare_precedes_resolution_and_preserves_original_metadata(monkeypatch):
    response_arguments, _, _, _, _ = build_buffer_response_oracle(
        monkeypatch=monkeypatch,
        alarm_interval_resolution_case="no_model_fits",
        earlier_sample_count=3,
    )
    operation_calls = []
    original_prepare = response_module.prepare_alarm_training_intervals
    original_resolve = response_module.resolve_alarm_change_interval

    def recorded_prepare(**arguments):
        prepared_intervals = original_prepare(**arguments)
        operation_calls.append(("prepare", prepared_intervals))
        return prepared_intervals

    def recorded_resolve(**arguments):
        operation_calls.append(("resolve", arguments))
        return original_resolve(**arguments)

    monkeypatch.setattr(response_module, "prepare_alarm_training_intervals", recorded_prepare)
    monkeypatch.setattr(response_module, "resolve_alarm_change_interval", recorded_resolve)
    response = response_module.respond_to_alarm_with_buffered_samples(**response_arguments)
    assert [operation_call[0] for operation_call in operation_calls] == [
        "prepare",
        "resolve",
    ]
    prepared_intervals = operation_calls[0][1]
    assert response.prepared_alarm_training_intervals is prepared_intervals
    forwarded_arguments = operation_calls[1][1]
    assert forwarded_arguments["change_interval_training_samples"] == tuple(
        indexed_observation.training_sample
        for indexed_observation in prepared_intervals.change_interval_observations
    )
    assert forwarded_arguments["change_interval_sample_concept_ids"] == tuple(
        indexed_observation.observed_concept_id
        for indexed_observation in prepared_intervals.change_interval_observations
    )
    for argument_name in (
        "proposal_sample_index",
        "estimated_change_point_sample_index",
        "detection_episode_id",
        "detector_name",
    ):
        assert forwarded_arguments[argument_name] == response_arguments[argument_name]


def test_failure_after_preparation_keeps_completed_earlier_updates(monkeypatch):
    response_arguments, _, _, _, _ = build_buffer_response_oracle(monkeypatch=monkeypatch)
    completed_preparations = []
    original_prepare = response_module.prepare_alarm_training_intervals

    def recorded_prepare(**arguments):
        prepared_intervals = original_prepare(**arguments)
        completed_preparations.append(prepared_intervals)
        return prepared_intervals

    monkeypatch.setattr(response_module, "prepare_alarm_training_intervals", recorded_prepare)
    response_arguments["maximum_alarm_interval_mean_loss_increase"] = -1.0
    pending_assignment_snapshot = response_arguments[
        "pending_training_assignment_buffer"
    ].get_state_snapshot()
    initial_loss_count = (
        response_arguments["loss_statistics_store"]
        .get_model_loss_statistics(
            model_id=response_arguments[
                "current_training_model_assignment"
            ].current_training_model_id
        )
        .overall_loss_moments.observed_loss_count
    )
    with pytest.raises(ValueError):
        response_module.respond_to_alarm_with_buffered_samples(**response_arguments)
    assert len(completed_preparations) == 1
    assert response_arguments["loss_statistics_store"].get_model_loss_statistics(
        model_id=response_arguments["current_training_model_assignment"].current_training_model_id
    ).overall_loss_moments.observed_loss_count == initial_loss_count + len(
        completed_preparations[0].earlier_observations
    )
    assert (
        response_arguments["pending_training_assignment_buffer"].get_state_snapshot()
        == pending_assignment_snapshot
    )
