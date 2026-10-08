"""警報区間の準備と実旧の前区間処理を照合する。"""

import random
from collections import Counter, defaultdict, deque
from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import assert_held_model_states_match_legacy
from test_adopted_candidate_local_adoption import assert_training_samples_match_legacy
from test_alarm_change_interval_resolution import (
    ALARM_INTERVAL_RESOLUTION_CASES,
    assert_alarm_change_interval_resolution_matches_legacy,
    assert_alarm_change_interval_resolution_state_unchanged,
    build_alarm_change_interval_resolution_oracle,
    snapshot_alarm_change_interval_resolution_state,
)
from test_assigned_training_sample_absorption import assert_absorption_matches_legacy
from test_joint_model_parameter_update import run_legacy_joint_update
from test_loss_statistics_model_id_reassignment import assert_store_statistics_match_legacy
from test_model_training_and_assignment_counts import assert_model_counts_match_legacy
from test_post_alarm_candidate_validation_sample_observation import (
    assert_collected_losses_match_legacy,
)

import federated_learning_experiments.runtime.alarm_training_interval_preparation as preparation_module
from federated_drift_experiment import config
from federated_learning_experiments.evaluation.model_evaluation_sample_records import (
    ObservedEvaluationSample,
)
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.participating_model_training_batch import (
    ParticipatingModelTrainingBatch,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import (
    PreparedAlarmTrainingIntervals,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    TrainingDataAssignmentSettings,
)
from federated_learning_experiments.runtime import (
    assigned_training_sample_absorption as absorption_module,
)
from federated_learning_experiments.runtime.alarm_change_interval_resolution import (
    resolve_alarm_change_interval,
)
from federated_learning_experiments.runtime.alarm_training_interval_preparation import (
    prepare_alarm_training_intervals,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation import (
    observe_post_alarm_candidate_validation_sample,
)


def test_alarm_preparation_records_are_immutable_and_borrow_payloads():
    training_sample = ObservedTrainingSample(
        input_features=torch.ones(1, 2), observed_class_labels=torch.zeros(1, 1)
    )
    indexed_observation = IndexedObservedTrainingSample(
        sample_index=7, training_sample=training_sample, observed_concept_id=None
    )
    earlier_observations = (indexed_observation,)
    prepared_intervals = PreparedAlarmTrainingIntervals(
        earlier_observations=earlier_observations,
        change_interval_observations=(),
        change_interval_start_sample_index=None,
        earlier_interval_absorbed_model_id=-3,
    )
    assert indexed_observation.training_sample is training_sample
    assert prepared_intervals.earlier_observations is earlier_observations
    assert type(prepared_intervals.earlier_observations) is tuple
    for record, expected_field_names in (
        (indexed_observation, ("sample_index", "training_sample", "observed_concept_id")),
        (
            prepared_intervals,
            (
                "earlier_observations",
                "change_interval_observations",
                "change_interval_start_sample_index",
                "earlier_interval_absorbed_model_id",
            ),
        ),
    ):
        assert tuple(field.name for field in fields(record)) == expected_field_names
        assert all(field.kw_only for field in fields(record))
        for field_name in expected_field_names:
            with pytest.raises(FrozenInstanceError):
                setattr(record, field_name, None)
    with pytest.raises(TypeError):
        IndexedObservedTrainingSample(7, training_sample, None)
    with pytest.raises(TypeError):
        PreparedAlarmTrainingIntervals(earlier_observations, (), None, -3)
    # payload自体は凍結しない。record constructorは境界検査を代行しない。
    training_sample.input_features.fill_(2)
    assert indexed_observation.training_sample.input_features[0, 0].item() == 2
    assert (
        IndexedObservedTrainingSample(
            sample_index=True, training_sample=None, observed_concept_id=False
        ).sample_index
        is True
    )


def build_alarm_preparation_oracle(
    *,
    monkeypatch,
    class_count=2,
    alarm_interval_resolution_case="current_model_maintained",
    earlier_sample_count=3,
    estimated_change_span_sample_count=None,
    added_batch_sample_count=2,
    maximum_stored_sample_count_per_model=3,
    pending_assignment_buffer_capacity_samples=4,
):
    resolution_arguments, shared_optimizer_owners, legacy_client, _ = (
        build_alarm_change_interval_resolution_oracle(
            class_count=class_count,
            alarm_interval_resolution_case=alarm_interval_resolution_case,
            monkeypatch=monkeypatch,
        )
    )
    change_interval_training_samples = resolution_arguments["change_interval_training_samples"]
    earlier_training_samples = tuple(
        ObservedTrainingSample(
            input_features=change_interval_training_samples[
                sample_offset % len(change_interval_training_samples)
            ].input_features
            + 0.1,
            observed_class_labels=change_interval_training_samples[
                sample_offset % len(change_interval_training_samples)
            ].observed_class_labels,
        )
        for sample_offset in range(earlier_sample_count)
    )
    pending_training_samples = earlier_training_samples + change_interval_training_samples
    pending_sample_concept_ids = (
        tuple(None if sample_offset % 2 else -7 for sample_offset in range(earlier_sample_count))
        + resolution_arguments["change_interval_sample_concept_ids"]
    )
    first_sample_index = (
        resolution_arguments["proposal_sample_index"] - len(pending_training_samples) + 1
    )
    pending_sample_observations = tuple(
        IndexedObservedTrainingSample(
            sample_index=first_sample_index + sample_offset,
            training_sample=training_sample,
            observed_concept_id=observed_concept_id,
        )
        for sample_offset, (training_sample, observed_concept_id) in enumerate(
            zip(pending_training_samples, pending_sample_concept_ids)
        )
    )
    pending_training_assignment_buffer = PendingTrainingAssignmentBuffer(
        training_data_assignment_settings=TrainingDataAssignmentSettings(
            pending_assignment_buffer_capacity_samples=pending_assignment_buffer_capacity_samples
        )
    )
    for indexed_observation in pending_sample_observations:
        pending_training_assignment_buffer.append_observed_sample_index(
            sample_index=indexed_observation.sample_index
        )
    model_evaluation_sample_store = ModelEvaluationSampleStore(
        maximum_stored_sample_count_per_model=maximum_stored_sample_count_per_model,
        added_batch_sample_count=added_batch_sample_count,
    )
    legacy_client.stored_data = defaultdict(list)
    legacy_client.stored_data_limit = maximum_stored_sample_count_per_model
    monkeypatch.setattr(config, "EVAL_STORE_SAMPLE_SIZE", added_batch_sample_count)
    preparation_arguments = dict(
        pending_training_assignment_buffer=pending_training_assignment_buffer,
        pending_sample_observations=pending_sample_observations,
        estimated_change_span_sample_count=len(change_interval_training_samples)
        if estimated_change_span_sample_count is None
        else estimated_change_span_sample_count,
        model_evaluation_sample_store=model_evaluation_sample_store,
        python_random_generator=random.Random(349),
        **{
            owner_name: resolution_arguments[owner_name]
            for owner_name in (
                "current_training_model_assignment",
                "held_model_training_state_registry",
                "training_sample_store",
                "model_training_and_assignment_counts_store",
                "loss_statistics_store",
            )
        },
    )
    return preparation_arguments, resolution_arguments, shared_optimizer_owners, legacy_client


def assert_evaluation_samples_match_legacy(*, preparation_arguments, legacy_client):
    collections = preparation_arguments[
        "model_evaluation_sample_store"
    ].snapshot_ordered_model_evaluation_samples()
    assert tuple(collection.model_id for collection in collections) == tuple(
        legacy_client.stored_data
    )
    for collection in collections:
        legacy_evaluation_samples = legacy_client.stored_data[collection.model_id]
        assert len(collection.evaluation_samples) == len(legacy_evaluation_samples)
        for evaluation_sample, legacy_evaluation_sample in zip(
            collection.evaluation_samples, legacy_evaluation_samples
        ):
            assert evaluation_sample.input_features is legacy_evaluation_sample[0]
            assert evaluation_sample.observed_class_labels is legacy_evaluation_sample[1]


def run_legacy_alarm_with_preparation_checkpoint(
    *,
    preparation_arguments,
    resolution_arguments,
    legacy_client,
    monkeypatch,
    minimum_change_sample_count,
):
    """実旧の保存と吸収を実行し、区間評価直前または不足終了直前に照合する。"""
    observations = preparation_arguments["pending_sample_observations"]
    estimated_span = preparation_arguments["estimated_change_span_sample_count"]
    change_sample_count = min(len(observations), estimated_span)
    legacy_adaptation_events = []
    preparation_checkpoint_calls = []
    detector_reset_calls = []
    global_python_random_state = random.getstate()

    def assert_preparation_checkpoint():
        assert_absorption_matches_legacy(
            absorption_arguments=preparation_arguments, legacy_client=legacy_client
        )
        assert_evaluation_samples_match_legacy(
            preparation_arguments=preparation_arguments, legacy_client=legacy_client
        )
        assert random.getstate() == preparation_arguments["python_random_generator"].getstate()
        preparation_checkpoint_calls.append(True)

    def record_legacy_adaptation_event(**event_fields):
        if event_fields["action"] == "insufficient_data":
            assert_preparation_checkpoint()
        legacy_adaptation_events.append(event_fields)

    first_model_id = next(iter(legacy_client.models))
    original_per_sample_error = legacy_client.models[first_model_id].per_sample_error

    def record_interval_evaluation(input_features, observed_class_labels):
        if len(input_features) == change_sample_count and not preparation_checkpoint_calls:
            assert_preparation_checkpoint()
        return original_per_sample_error(input_features, observed_class_labels)

    monkeypatch.setattr(config, "MIN_DRIFT_DATA", minimum_change_sample_count)
    legacy_client.verbose = False
    legacy_client._forward_validation = None
    legacy_client.reuse_selection_counts = Counter()
    legacy_client._estimated_new_concept_span = lambda sample_index: estimated_span
    legacy_client._reset_drift_detectors = lambda: detector_reset_calls.append(True)
    legacy_client._record_adaptation_event = record_legacy_adaptation_event
    legacy_client.distance_threshold = resolution_arguments[
        "maximum_alarm_interval_mean_loss_increase"
    ]
    legacy_client.buffer = deque(
        (
            observation.training_sample.input_features,
            observation.training_sample.observed_class_labels,
            observation.observed_concept_id,
        )
        for observation in observations
    )
    try:
        random.setstate(preparation_arguments["legacy_initial_python_random_state"])
        with (
            patch.object(
                legacy_client.models[first_model_id],
                "per_sample_error",
                side_effect=record_interval_evaluation,
            ),
            patch.object(
                legacy_client,
                "_update_new_model_epochs",
                wraps=legacy_client._update_new_model_epochs,
            ) as legacy_epoch_training_calls,
        ):
            legacy_drift_type = legacy_client._resolve_drift(
                resolution_arguments["proposal_sample_index"],
                resolution_arguments["estimated_change_point_sample_index"],
                resolution_arguments["detection_episode_id"],
            )
        assert preparation_checkpoint_calls == [True]
        assert detector_reset_calls == [True]
        # 最小件数未満で旧FIFOが残る挙動も境界対照で固定する。
        assert len(legacy_client.buffer) == (
            len(observations) if change_sample_count < minimum_change_sample_count else 0
        )
        return legacy_drift_type, legacy_adaptation_events, legacy_epoch_training_calls
    finally:
        random.setstate(global_python_random_state)


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize(
    "added_batch_sample_count,maximum_stored_sample_count_per_model", [(0, 3), (2, 3), (99, 2)]
)
@pytest.mark.parametrize("minimum_change_sample_count", [1, 999, "exact", "one_above"])
@pytest.mark.parametrize("negative_current_model", [False, True])
@pytest.mark.parametrize("existing_evaluation_samples", [False, True])
def test_alarm_preparation_matches_real_legacy_before_minimum_count_decision(
    class_count,
    added_batch_sample_count,
    maximum_stored_sample_count_per_model,
    minimum_change_sample_count,
    negative_current_model,
    existing_evaluation_samples,
    monkeypatch,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(873)
        preparation_arguments, resolution_arguments, shared_optimizer_owners, legacy_client = (
            build_alarm_preparation_oracle(
                monkeypatch=monkeypatch,
                class_count=class_count,
                added_batch_sample_count=added_batch_sample_count,
                maximum_stored_sample_count_per_model=maximum_stored_sample_count_per_model,
            )
        )
        if minimum_change_sample_count == "exact":
            minimum_change_sample_count = len(
                resolution_arguments["change_interval_training_samples"]
            )
        elif minimum_change_sample_count == "one_above":
            minimum_change_sample_count = (
                len(resolution_arguments["change_interval_training_samples"]) + 1
            )
        if negative_current_model:
            preparation_arguments, resolution_arguments, shared_optimizer_owners, legacy_client = (
                build_alarm_preparation_oracle(
                    monkeypatch=monkeypatch,
                    class_count=class_count,
                    alarm_interval_resolution_case="no_model_fits",
                    added_batch_sample_count=added_batch_sample_count,
                    maximum_stored_sample_count_per_model=maximum_stored_sample_count_per_model,
                )
            )
            preparation_arguments["current_training_model_assignment"].assign_model_for_training(
                model_id=-3
            )
            legacy_client.current_model_id = -3
        if existing_evaluation_samples:
            model_id = legacy_client.current_model_id
            training_sample = preparation_arguments["pending_sample_observations"][
                0
            ].training_sample
            evaluation_sample = ObservedEvaluationSample(
                input_features=training_sample.input_features,
                observed_class_labels=training_sample.observed_class_labels,
            )
            preparation_arguments[
                "model_evaluation_sample_store"
            ].sample_and_append_model_evaluation_samples(
                model_id=model_id,
                evaluation_samples=(evaluation_sample,),
                python_random_generator=preparation_arguments["python_random_generator"],
            )
            if model_id >= 0:
                legacy_client.stored_data[model_id] = (
                    [(training_sample.input_features, training_sample.observed_class_labels)]
                    if added_batch_sample_count
                    else []
                )
        initial_python_random_state = preparation_arguments["python_random_generator"].getstate()
        previous_snapshot = snapshot_alarm_change_interval_resolution_state(
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        pending_assignment_snapshot = preparation_arguments[
            "pending_training_assignment_buffer"
        ].get_state_snapshot()
        prepared_intervals = prepare_alarm_training_intervals(**preparation_arguments)
        assert (
            prepared_intervals.earlier_observations
            == preparation_arguments["pending_sample_observations"][:3]
        )
        assert (
            prepared_intervals.change_interval_observations
            == preparation_arguments["pending_sample_observations"][3:]
        )
        assert (
            prepared_intervals.earlier_interval_absorbed_model_id == legacy_client.current_model_id
        )
        assert (
            preparation_arguments["pending_training_assignment_buffer"].get_state_snapshot()
            == pending_assignment_snapshot
        )
        assert torch.equal(torch.get_rng_state(), previous_snapshot["random_states"][0])
        assert random.getstate() == previous_snapshot["random_states"][1]
        preparation_arguments["legacy_initial_python_random_state"] = initial_python_random_state
        run_legacy_alarm_with_preparation_checkpoint(
            preparation_arguments=preparation_arguments,
            resolution_arguments=resolution_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
            minimum_change_sample_count=minimum_change_sample_count,
        )


@pytest.mark.parametrize(
    "pending_sample_count,estimated_span", [(0, 1), (1, 1), (1, 2), (5, 1), (5, 5), (5, 6)]
)
@pytest.mark.parametrize("negative_current_model", [False, True])
def test_alarm_preparation_preserves_empty_and_capacity_plus_one_buffers(
    pending_sample_count, estimated_span, negative_current_model, monkeypatch
):
    preparation_arguments, resolution_arguments, shared_optimizer_owners, legacy_client = (
        build_alarm_preparation_oracle(monkeypatch=monkeypatch)
    )
    observations = preparation_arguments["pending_sample_observations"][:pending_sample_count]
    preparation_arguments["pending_sample_observations"] = observations
    pending_buffer = PendingTrainingAssignmentBuffer(
        training_data_assignment_settings=TrainingDataAssignmentSettings(
            pending_assignment_buffer_capacity_samples=4
        )
    )
    for observation in observations:
        pending_buffer.append_observed_sample_index(sample_index=observation.sample_index)
    preparation_arguments["pending_training_assignment_buffer"] = pending_buffer
    preparation_arguments["estimated_change_span_sample_count"] = estimated_span
    if negative_current_model:
        # 既存oracleに実在する負IDモデルを現行へ設定する。
        preparation_arguments, resolution_arguments, shared_optimizer_owners, legacy_client = (
            build_alarm_preparation_oracle(
                monkeypatch=monkeypatch, alarm_interval_resolution_case="no_model_fits"
            )
        )
        preparation_arguments["current_training_model_assignment"].assign_model_for_training(
            model_id=-3
        )
        preparation_arguments.update(
            pending_sample_observations=observations,
            pending_training_assignment_buffer=pending_buffer,
            estimated_change_span_sample_count=estimated_span,
        )
    pending_snapshot = pending_buffer.get_state_snapshot()
    evaluation_snapshot = preparation_arguments[
        "model_evaluation_sample_store"
    ].snapshot_ordered_model_evaluation_samples()
    python_random_state = preparation_arguments["python_random_generator"].getstate()
    prepared_intervals = prepare_alarm_training_intervals(**preparation_arguments)
    earlier_sample_count = max(0, pending_sample_count - estimated_span)
    assert prepared_intervals.earlier_observations == observations[:earlier_sample_count]
    assert prepared_intervals.change_interval_observations == observations[earlier_sample_count:]
    assert prepared_intervals.change_interval_start_sample_index == (
        observations[earlier_sample_count].sample_index if observations else None
    )
    assert pending_buffer.get_state_snapshot() == pending_snapshot
    if negative_current_model or not earlier_sample_count:
        assert (
            preparation_arguments[
                "model_evaluation_sample_store"
            ].snapshot_ordered_model_evaluation_samples()
            == evaluation_snapshot
        )
        assert preparation_arguments["python_random_generator"].getstate() == python_random_state
    if not earlier_sample_count:
        assert prepared_intervals.earlier_interval_absorbed_model_id is None


@pytest.mark.parametrize("invalid_sample_offset", [0, 2])
@pytest.mark.parametrize(
    "invalid_payload",
    ["shape", "dtype", "device", "nan", "label_range", "fractional_label", "feature_count"],
)
def test_alarm_preparation_rejects_every_earlier_sample_before_any_state_update(
    invalid_sample_offset, invalid_payload, monkeypatch
):
    preparation_arguments, resolution_arguments, shared_optimizer_owners, _ = (
        build_alarm_preparation_oracle(monkeypatch=monkeypatch)
    )
    observations = list(preparation_arguments["pending_sample_observations"])
    training_sample = observations[invalid_sample_offset].training_sample
    input_features = training_sample.input_features.clone()
    observed_class_labels = training_sample.observed_class_labels.clone()
    if invalid_payload == "shape":
        input_features = input_features.reshape(-1)
    elif invalid_payload == "dtype":
        input_features = input_features.double()
    elif invalid_payload == "device":
        input_features = input_features.to("meta")
    elif invalid_payload == "nan":
        input_features[0, 0] = float("nan")
    elif invalid_payload == "label_range":
        observed_class_labels.fill_(99)
    elif invalid_payload == "fractional_label":
        observed_class_labels.fill_(0.5)
    elif invalid_payload == "feature_count":
        input_features = torch.ones(1, 3)
    observations[invalid_sample_offset] = replace(
        observations[invalid_sample_offset],
        training_sample=ObservedTrainingSample(
            input_features=input_features, observed_class_labels=observed_class_labels
        ),
    )
    preparation_arguments["pending_sample_observations"] = tuple(observations)
    previous_snapshot = snapshot_alarm_change_interval_resolution_state(
        resolution_arguments=resolution_arguments, shared_optimizer_owners=shared_optimizer_owners
    )
    evaluation_snapshot = preparation_arguments[
        "model_evaluation_sample_store"
    ].snapshot_ordered_model_evaluation_samples()
    explicit_python_random_state = preparation_arguments["python_random_generator"].getstate()
    pending_snapshot = preparation_arguments[
        "pending_training_assignment_buffer"
    ].get_state_snapshot()
    with pytest.raises(ValueError):
        prepare_alarm_training_intervals(**preparation_arguments)
    assert_alarm_change_interval_resolution_state_unchanged(previous_snapshot)
    assert (
        preparation_arguments[
            "model_evaluation_sample_store"
        ].snapshot_ordered_model_evaluation_samples()
        == evaluation_snapshot
    )
    assert (
        preparation_arguments["python_random_generator"].getstate() == explicit_python_random_state
    )
    assert (
        preparation_arguments["pending_training_assignment_buffer"].get_state_snapshot()
        == pending_snapshot
    )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "owner",
        "owner_training_samples",
        "owner_assignment_counts",
        "owner_loss_statistics",
        "owner_current_assignment",
        "owner_registry",
        "owner_pending_buffer",
        "random",
        "list",
        "record",
        "bool_span",
        "zero_span",
        "negative_span",
        "bool_position",
        "negative_position",
        "bool_concept",
        "training_record",
        "missing",
        "extra",
        "duplicate",
        "reverse",
        "unheld",
    ],
)
def test_alarm_preparation_rejects_contract_errors_without_state_changes(invalid_case, monkeypatch):
    preparation_arguments, resolution_arguments, shared_optimizer_owners, _ = (
        build_alarm_preparation_oracle(monkeypatch=monkeypatch)
    )
    valid_preparation_arguments = dict(preparation_arguments)
    observations = preparation_arguments["pending_sample_observations"]
    if invalid_case == "owner":
        preparation_arguments["model_evaluation_sample_store"] = object()
    elif invalid_case.startswith("owner_"):
        preparation_arguments[
            {
                "owner_training_samples": "training_sample_store",
                "owner_assignment_counts": "model_training_and_assignment_counts_store",
                "owner_loss_statistics": "loss_statistics_store",
                "owner_current_assignment": "current_training_model_assignment",
                "owner_registry": "held_model_training_state_registry",
                "owner_pending_buffer": "pending_training_assignment_buffer",
            }[invalid_case]
        ] = object()
    elif invalid_case == "random":
        preparation_arguments["python_random_generator"] = random
    elif invalid_case == "list":
        preparation_arguments["pending_sample_observations"] = list(observations)
    elif invalid_case == "record":
        preparation_arguments["pending_sample_observations"] = (object(),) + observations[1:]
    elif invalid_case == "bool_span":
        preparation_arguments["estimated_change_span_sample_count"] = True
    elif invalid_case == "zero_span":
        preparation_arguments["estimated_change_span_sample_count"] = 0
    elif invalid_case == "negative_span":
        preparation_arguments["estimated_change_span_sample_count"] = -1
    elif invalid_case == "bool_position":
        preparation_arguments["pending_sample_observations"] = (
            replace(observations[0], sample_index=True),
        ) + observations[1:]
    elif invalid_case == "negative_position":
        preparation_arguments["pending_sample_observations"] = (
            replace(observations[0], sample_index=-1),
        ) + observations[1:]
    elif invalid_case == "bool_concept":
        preparation_arguments["pending_sample_observations"] = (
            replace(observations[0], observed_concept_id=True),
        ) + observations[1:]
    elif invalid_case == "training_record":
        preparation_arguments["pending_sample_observations"] = (
            replace(observations[0], training_sample=object()),
        ) + observations[1:]
    elif invalid_case == "missing":
        preparation_arguments["pending_sample_observations"] = observations[:-1]
    elif invalid_case == "extra":
        preparation_arguments["pending_sample_observations"] = observations + (observations[-1],)
    elif invalid_case == "duplicate":
        preparation_arguments["pending_sample_observations"] = (observations[0],) + observations[
            :-1
        ]
    elif invalid_case == "reverse":
        preparation_arguments["pending_sample_observations"] = observations[::-1]
    elif invalid_case == "unheld":
        preparation_arguments["current_training_model_assignment"].assign_model_for_training(
            model_id=999
        )
    previous_snapshot = snapshot_alarm_change_interval_resolution_state(
        resolution_arguments=resolution_arguments, shared_optimizer_owners=shared_optimizer_owners
    )
    evaluation_snapshot = valid_preparation_arguments[
        "model_evaluation_sample_store"
    ].snapshot_ordered_model_evaluation_samples()
    python_random_state = valid_preparation_arguments["python_random_generator"].getstate()
    pending_snapshot = valid_preparation_arguments[
        "pending_training_assignment_buffer"
    ].get_state_snapshot()
    with pytest.raises(
        {
            "owner": TypeError,
            "owner_training_samples": TypeError,
            "owner_assignment_counts": TypeError,
            "owner_loss_statistics": TypeError,
            "owner_current_assignment": TypeError,
            "owner_registry": TypeError,
            "owner_pending_buffer": TypeError,
            "random": TypeError,
            "list": TypeError,
            "record": TypeError,
            "bool_span": TypeError,
            "zero_span": ValueError,
            "negative_span": ValueError,
            "bool_position": TypeError,
            "negative_position": ValueError,
            "bool_concept": TypeError,
            "training_record": TypeError,
            "missing": ValueError,
            "extra": ValueError,
            "duplicate": ValueError,
            "reverse": ValueError,
            "unheld": LookupError,
        }[invalid_case]
    ):
        prepare_alarm_training_intervals(**preparation_arguments)
    assert_alarm_change_interval_resolution_state_unchanged(previous_snapshot)
    assert (
        valid_preparation_arguments[
            "model_evaluation_sample_store"
        ].snapshot_ordered_model_evaluation_samples()
        == evaluation_snapshot
    )
    assert valid_preparation_arguments["python_random_generator"].getstate() == python_random_state
    assert (
        valid_preparation_arguments["pending_training_assignment_buffer"].get_state_snapshot()
        == pending_snapshot
    )


def test_alarm_preparation_saves_evaluation_before_real_absorption(monkeypatch):
    preparation_arguments, _, _, _ = build_alarm_preparation_oracle(monkeypatch=monkeypatch)
    operation_calls = []
    original_save = ModelEvaluationSampleStore.sample_and_append_model_evaluation_samples
    original_absorption = preparation_module.absorb_assigned_training_samples_into_held_model
    original_bounded_loss_evaluation = (
        preparation_module.evaluate_classifier_per_sample_bounded_losses
    )
    classifier = (
        preparation_arguments["held_model_training_state_registry"]
        .get_held_model_training_state(
            model_id=preparation_arguments[
                "current_training_model_assignment"
            ].current_training_model_id
        )
        .classifier
    )

    def recorded_bounded_loss_evaluation(**operation_keyword_arguments):
        operation_calls.append(("loss", operation_keyword_arguments["classifier"]))
        return original_bounded_loss_evaluation(**operation_keyword_arguments)

    def recorded_save(self, **operation_keyword_arguments):
        assert operation_calls == [("loss", classifier)] * 3
        operation_calls.append("save")
        return original_save(self, **operation_keyword_arguments)

    def recorded_absorption(**operation_keyword_arguments):
        operation_calls.append("absorb")
        assert operation_calls == [("loss", classifier)] * 3 + ["save", "absorb"]
        return original_absorption(**operation_keyword_arguments)

    monkeypatch.setattr(
        ModelEvaluationSampleStore, "sample_and_append_model_evaluation_samples", recorded_save
    )
    monkeypatch.setattr(
        preparation_module,
        "evaluate_classifier_per_sample_bounded_losses",
        recorded_bounded_loss_evaluation,
    )
    monkeypatch.setattr(
        absorption_module,
        "evaluate_classifier_per_sample_bounded_losses",
        recorded_bounded_loss_evaluation,
    )
    monkeypatch.setattr(
        preparation_module, "absorb_assigned_training_samples_into_held_model", recorded_absorption
    )
    prepare_alarm_training_intervals(**preparation_arguments)
    assert (
        operation_calls
        == [("loss", classifier)] * 3 + ["save", "absorb"] + [("loss", classifier)] * 3
    )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize(
    "alarm_interval_resolution_case",
    ["other_model_reused", "current_model_maintained", "no_model_fits"],
)
def test_prepared_alarm_interval_continues_resolution_and_joint_training(
    class_count,
    alarm_interval_resolution_case,
    monkeypatch,
):
    update_shared_features = True
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(846)
        preparation_arguments, resolution_arguments, shared_optimizer_owners, legacy_client = (
            build_alarm_preparation_oracle(
                class_count=class_count,
                alarm_interval_resolution_case=alarm_interval_resolution_case,
                monkeypatch=monkeypatch,
            )
        )
        monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
        registry = resolution_arguments["held_model_training_state_registry"]
        training_sample_store = resolution_arguments["training_sample_store"]
        counts_store = resolution_arguments["model_training_and_assignment_counts_store"]
        local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        )
        legacy_training_batches = []
        legacy_client.updates_per_sample = 1
        legacy_client.backbone_gradient_diagnostics = defaultdict(float)
        legacy_client._sample_training_batches = lambda: legacy_training_batches
        change_interval_input_features = torch.cat(
            [
                training_sample.input_features
                for training_sample in resolution_arguments["change_interval_training_samples"]
            ]
        )

        def run_joint_update_in_both_implementations():
            # 両実装とも、各自の標本storeにある全標本をモデルごとの固定batchにする。
            training_bindings = registry.snapshot_ordered_held_model_training_bindings()
            model_training_sample_collections = (
                training_sample_store.snapshot_ordered_model_training_samples()
            )
            legacy_training_batches[:] = [
                (
                    model_id,
                    torch.cat(
                        [
                            legacy_training_sample[0]
                            for legacy_training_sample in legacy_training_samples
                        ]
                    ),
                    torch.cat(
                        [
                            legacy_training_sample[1]
                            for legacy_training_sample in legacy_training_samples
                        ]
                    ),
                )
                for model_id, legacy_training_samples in legacy_client.train_data_store.items()
            ]
            assert tuple(
                training_binding.model_id for training_binding in training_bindings
            ) == tuple(collection.model_id for collection in model_training_sample_collections)
            participating_training_batches = tuple(
                ParticipatingModelTrainingBatch(
                    classifier=training_binding.classifier,
                    concept_specific_parameter_optimizer=training_binding.concept_specific_parameter_optimizer,
                    input_features=torch.cat(
                        [
                            training_sample.input_features
                            for training_sample in collection.training_samples
                        ]
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
            for training_binding, training_batch in zip(
                training_bindings, participating_training_batches
            ):
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

        # 学習でparameterが変わった後のモデルで、区間を評価して解決する。
        run_joint_update_in_both_implementations()
        initial_training_model_id = legacy_client.current_model_id
        initial_python_random_state = preparation_arguments["python_random_generator"].getstate()
        prepared_intervals = prepare_alarm_training_intervals(**preparation_arguments)
        resolution_arguments["change_interval_training_samples"] = tuple(
            observation.training_sample
            for observation in prepared_intervals.change_interval_observations
        )
        resolution_arguments["change_interval_sample_concept_ids"] = tuple(
            observation.observed_concept_id
            for observation in prepared_intervals.change_interval_observations
        )
        preparation_arguments["legacy_initial_python_random_state"] = initial_python_random_state
        initial_torch_random_state = torch.get_rng_state().clone()
        (
            legacy_drift_type,
            legacy_adaptation_events,
            legacy_epoch_training_calls,
        ) = run_legacy_alarm_with_preparation_checkpoint(
            preparation_arguments=preparation_arguments,
            legacy_client=legacy_client,
            resolution_arguments=resolution_arguments,
            monkeypatch=monkeypatch,
            minimum_change_sample_count=1,
        )
        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        torch.set_rng_state(initial_torch_random_state)
        alarm_change_interval_resolution = resolve_alarm_change_interval(**resolution_arguments)
        assert (
            alarm_change_interval_resolution.resolution_outcome
            == ALARM_INTERVAL_RESOLUTION_CASES[alarm_interval_resolution_case][
                "expected_resolution_outcome"
            ]
        )
        assert_alarm_change_interval_resolution_matches_legacy(
            alarm_change_interval_resolution=alarm_change_interval_resolution,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            legacy_adaptation_events=legacy_adaptation_events,
            legacy_epoch_training_calls=legacy_epoch_training_calls,
            initial_training_model_id=initial_training_model_id,
        )
        # 吸収した区間の標本を含むbatchで学習を継続する。
        for _ in range(2):
            run_joint_update_in_both_implementations()
        assert_training_samples_match_legacy(
            training_sample_store=training_sample_store, legacy_client=legacy_client
        )
        assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)
        assert_store_statistics_match_legacy(
            loss_statistics_store=resolution_arguments["loss_statistics_store"],
            legacy_client=legacy_client,
        )
        assert_evaluation_samples_match_legacy(
            preparation_arguments=preparation_arguments, legacy_client=legacy_client
        )
        if alarm_change_interval_resolution.started_validation_session is not None:
            # 準備→候補開始→共同更新の後、実旧と新の候補へ同じ1標本を観測させる。
            training_sample = prepared_intervals.change_interval_observations[-1].training_sample
            assert (
                legacy_client._observe_forward_validation(
                    training_sample.input_features,
                    training_sample.observed_class_labels,
                    resolution_arguments["proposal_sample_index"] + 1,
                )
                == 0
            )
            assert (
                observe_post_alarm_candidate_validation_sample(
                    sample_index=resolution_arguments["proposal_sample_index"] + 1,
                    input_features=training_sample.input_features,
                    observed_class_labels=training_sample.observed_class_labels,
                    candidate_classifier=alarm_change_interval_resolution.started_validation_session.candidate_training_state.candidate_classifier,
                    reference_classifiers_by_model_id=alarm_change_interval_resolution.started_validation_session.fixed_reference_models.reference_classifiers_by_model_id,
                    post_alarm_candidate_loss_collection=alarm_change_interval_resolution.started_validation_session.post_alarm_candidate_loss_collection,
                )
                is False
            )
            assert_collected_losses_match_legacy(
                post_alarm_candidate_loss_collection=alarm_change_interval_resolution.started_validation_session.post_alarm_candidate_loss_collection,
                legacy_session=legacy_client._forward_validation,
            )
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]
