"""旧clientの学習要求列と独立したcounter/予算を照合する。"""

import random
from dataclasses import MISSING, FrozenInstanceError, fields
from types import MethodType, SimpleNamespace

import pytest
import torch
from test_held_model_joint_training_iterations import (
    build_training_iteration_oracle_pair,
    run_legacy_training_iterations,
)
from test_joint_model_parameter_update import assert_joint_update_states_equal

from federated_drift_experiment import config
from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.training.held_model_joint_training_iterations import (
    perform_held_model_joint_training_iterations,
)
from federated_learning_experiments.learning.training.local_training_request_schedule import (
    LocalTrainingRequestSchedule,
)
from federated_learning_experiments.learning.training.local_training_schedule_settings import (
    LocalTrainingScheduleSettings,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)


def build_training_schedule_oracle_pair(*, interval_count, iteration_budget, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_UPDATE_INTERVAL", interval_count)
    schedule_settings = LocalTrainingScheduleSettings(
        training_requests_per_update_interval=interval_count,
        joint_update_iterations_per_training_request=iteration_budget,
    )
    schedule = LocalTrainingRequestSchedule(local_training_schedule_settings=schedule_settings)
    legacy_client = SimpleNamespace(_pending_updates=0, updates_per_sample=iteration_budget)
    legacy_client.train_step = MethodType(BaseClient.train_step, legacy_client)
    legacy_client.flush_pending_updates = MethodType(
        BaseClient.flush_pending_updates, legacy_client
    )
    legacy_training_calls = []
    legacy_client.train_all_held_models = lambda *, count_multiplier: legacy_training_calls.append(
        (count_multiplier, legacy_client.updates_per_sample * count_multiplier)
    )
    return schedule_settings, schedule, legacy_client, legacy_training_calls


def execute_scheduled_training_request(
    *, schedule, schedule_settings, training_event, original_training_function
):
    if training_event == "request":
        requested_iteration_count = schedule.record_training_request()
        if (
            schedule.pending_training_request_count
            < schedule_settings.training_requests_per_update_interval
        ):
            return None
    else:
        if schedule.pending_training_request_count == 0:
            return None
        requested_iteration_count = schedule.calculate_pending_joint_update_iteration_count()
    pending_request_count = schedule.pending_training_request_count
    actual_losses = original_training_function(requested_iteration_count)
    schedule.acknowledge_completed_training_requests(
        completed_training_request_count=pending_request_count
    )
    return actual_losses


@pytest.mark.parametrize("interval_count", [1, 2, 5])
@pytest.mark.parametrize("iteration_budget", [0, 1, 3])
def test_training_request_schedule_matches_legacy(interval_count, iteration_budget, monkeypatch):
    schedule_settings, schedule, legacy_client, legacy_training_calls = (
        build_training_schedule_oracle_pair(
            interval_count=interval_count,
            iteration_budget=iteration_budget,
            monkeypatch=monkeypatch,
        )
    )
    new_training_calls = []
    training_events = (
        "flush",
        "request",
        "request",
        "flush",
        "flush",
        *("request",) * 7,
        "flush",
        "flush",
    )
    for training_event in training_events:
        if training_event == "request":
            legacy_client.train_step()
        else:
            legacy_client.flush_pending_updates()
        execute_scheduled_training_request(
            schedule=schedule,
            schedule_settings=schedule_settings,
            training_event=training_event,
            original_training_function=lambda requested_iteration_count: new_training_calls.append(
                (schedule.pending_training_request_count, requested_iteration_count)
            ),
        )
        assert schedule.pending_training_request_count == legacy_client._pending_updates
        assert new_training_calls == legacy_training_calls
        pending_count_before = schedule.pending_training_request_count
        assert (
            schedule.calculate_pending_joint_update_iteration_count()
            == pending_count_before * iteration_budget
        )
        assert schedule.pending_training_request_count == pending_count_before


@pytest.mark.parametrize("interval_count", [1, 3])
@pytest.mark.parametrize("iteration_budget", [0, 2])
def test_training_request_schedule_reports_whether_update_interval_is_reached(
    interval_count, iteration_budget
):
    """間隔への到達の読取りは状態を変えず、一要求あたりの回数が0でも件数だけで決まる。"""
    schedule = LocalTrainingRequestSchedule(
        local_training_schedule_settings=LocalTrainingScheduleSettings(
            training_requests_per_update_interval=interval_count,
            joint_update_iterations_per_training_request=iteration_budget,
        )
    )
    assert schedule.has_pending_requests_reaching_update_interval() is False
    for recorded_request_count in range(1, interval_count + 3):
        schedule.record_training_request()
        assert schedule.has_pending_requests_reaching_update_interval() is (
            recorded_request_count >= interval_count
        )
        assert schedule.pending_training_request_count == recorded_request_count
    schedule.acknowledge_completed_training_requests(
        completed_training_request_count=interval_count + 2
    )
    assert schedule.has_pending_requests_reaching_update_interval() is False


@pytest.mark.parametrize(
    "invalid_field",
    ["training_requests_per_update_interval", "joint_update_iterations_per_training_request"],
)
@pytest.mark.parametrize(
    "invalid_value", [True, False, -1, 1.5, "2", None, type("DerivedInteger", (int,), {})(1)]
)
def test_training_request_schedule_rejects_invalid_settings(invalid_field, invalid_value):
    keyword_arguments = dict(
        training_requests_per_update_interval=2, joint_update_iterations_per_training_request=1
    )
    keyword_arguments[invalid_field] = invalid_value
    with pytest.raises(ValueError, match=invalid_field):
        LocalTrainingScheduleSettings(**keyword_arguments)
    schedule_settings = LocalTrainingScheduleSettings(
        training_requests_per_update_interval=2, joint_update_iterations_per_training_request=1
    )
    object.__setattr__(schedule_settings, invalid_field, invalid_value)
    with pytest.raises(ValueError, match=invalid_field):
        LocalTrainingRequestSchedule(local_training_schedule_settings=schedule_settings)


def test_training_request_schedule_rejects_zero_interval_and_wrong_settings_type():
    with pytest.raises(ValueError, match="training_requests_per_update_interval"):
        LocalTrainingScheduleSettings(
            training_requests_per_update_interval=0, joint_update_iterations_per_training_request=1
        )
    with pytest.raises(ValueError, match="local_training_schedule_settings"):
        LocalTrainingRequestSchedule(local_training_schedule_settings=None)


@pytest.mark.parametrize(
    "invalid_ack_count",
    [True, False, -1, 0, 1, 3, 2.0, None, type("DerivedInteger", (int,), {})(2)],
)
def test_training_request_schedule_rejects_invalid_acknowledgement(invalid_ack_count):
    schedule = LocalTrainingRequestSchedule(
        local_training_schedule_settings=LocalTrainingScheduleSettings(
            training_requests_per_update_interval=3, joint_update_iterations_per_training_request=1
        )
    )
    schedule.record_training_request()
    schedule.record_training_request()
    with pytest.raises(ValueError, match="completed_training_request_count"):
        schedule.acknowledge_completed_training_requests(
            completed_training_request_count=invalid_ack_count
        )
    assert schedule.pending_training_request_count == 2
    schedule.acknowledge_completed_training_requests(completed_training_request_count=2)
    with pytest.raises(ValueError, match="completed_training_request_count"):
        schedule.acknowledge_completed_training_requests(completed_training_request_count=2)
    assert schedule.pending_training_request_count == 0


@pytest.mark.parametrize("iteration_budget", [0, 2])
def test_training_request_schedule_retains_pending_requests_on_failure(
    iteration_budget, monkeypatch
):
    schedule_settings, schedule, legacy_client, legacy_training_calls = (
        build_training_schedule_oracle_pair(
            interval_count=2, iteration_budget=iteration_budget, monkeypatch=monkeypatch
        )
    )
    failure_armed = True
    original_training_function = legacy_client.train_all_held_models

    def fail_training(*, count_multiplier):
        original_training_function(count_multiplier=count_multiplier)
        if failure_armed:
            raise RuntimeError("学習失敗")

    legacy_client.train_all_held_models = fail_training
    legacy_client.train_step()
    schedule.record_training_request()
    with pytest.raises(RuntimeError, match="学習失敗"):
        legacy_client.train_step()
    assert schedule.record_training_request() == 2 * iteration_budget
    # 外側の実行が例外になれば成功確認へ進まずcounterを残す。
    with pytest.raises(RuntimeError, match="学習失敗"):
        execute_scheduled_training_request(
            schedule=schedule,
            schedule_settings=schedule_settings,
            training_event="flush",
            original_training_function=lambda _: (_ for _ in ()).throw(RuntimeError("学習失敗")),
        )
    assert schedule.pending_training_request_count == legacy_client._pending_updates == 2
    failure_armed = False
    legacy_client.flush_pending_updates()
    new_training_calls = []
    execute_scheduled_training_request(
        schedule=schedule,
        schedule_settings=schedule_settings,
        training_event="flush",
        original_training_function=lambda requested_iteration_count: new_training_calls.append(
            (schedule.pending_training_request_count, requested_iteration_count)
        ),
    )
    assert schedule.pending_training_request_count == legacy_client._pending_updates == 0
    assert new_training_calls == legacy_training_calls[-1:]


def test_training_schedule_settings_are_frozen_and_explicit():
    schedule_settings = LocalTrainingScheduleSettings(
        training_requests_per_update_interval=2, joint_update_iterations_per_training_request=0
    )
    for record_field in fields(LocalTrainingScheduleSettings):
        assert record_field.default is MISSING
        assert record_field.default_factory is MISSING
        assert record_field.kw_only
        assert "parameter_unit" in record_field.metadata
    with pytest.raises(FrozenInstanceError):
        schedule_settings.training_requests_per_update_interval = 10
    schedule = LocalTrainingRequestSchedule(local_training_schedule_settings=schedule_settings)
    assert schedule.pending_training_request_count == 0
    with pytest.raises(AttributeError):
        schedule.pending_training_request_count = 10


def test_training_request_schedule_supports_large_counts():
    iteration_budget = 10**1000
    schedule = LocalTrainingRequestSchedule(
        local_training_schedule_settings=LocalTrainingScheduleSettings(
            training_requests_per_update_interval=2,
            joint_update_iterations_per_training_request=iteration_budget,
        )
    )
    assert schedule.record_training_request() == 0
    assert schedule.calculate_pending_joint_update_iteration_count() == iteration_budget
    assert schedule.record_training_request() == 2 * iteration_budget
    schedule.acknowledge_completed_training_requests(completed_training_request_count=2)
    assert schedule.calculate_pending_joint_update_iteration_count() == 0


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "sgd"])
@pytest.mark.parametrize("interval_count", [1, 3])
@pytest.mark.parametrize("iteration_budget", [0, 2])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_training_request_schedule_integrates_with_joint_training(
    class_count,
    optimizer_variant,
    interval_count,
    iteration_budget,
    update_shared_features,
    monkeypatch,
):
    global_python_random_state, global_torch_random_state = random.getstate(), torch.get_rng_state()
    try:
        (
            participating_training_batches,
            shared_parameter_optimizer,
            legacy_client,
            ordered_model_training_samples,
            held_model_training_bindings,
        ) = build_training_iteration_oracle_pair(
            class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
        )
        monkeypatch.setattr(config, "LOCAL_UPDATE_INTERVAL", interval_count)
        legacy_client.updates_per_sample = iteration_budget
        legacy_client._pending_updates = 0
        legacy_client.batch_size = 3
        legacy_client.train_step = MethodType(BaseClient.train_step, legacy_client)
        legacy_client.flush_pending_updates = MethodType(
            BaseClient.flush_pending_updates, legacy_client
        )
        schedule_settings = LocalTrainingScheduleSettings(
            training_requests_per_update_interval=interval_count,
            joint_update_iterations_per_training_request=iteration_budget,
        )
        schedule = LocalTrainingRequestSchedule(local_training_schedule_settings=schedule_settings)
        python_random_generator = random.Random(731)
        expected_random_state = python_random_generator.getstate()
        legacy_training_calls, new_training_calls, expected_losses = [], [], []

        def original_training_function(*, count_multiplier):
            nonlocal expected_random_state
            legacy_training_calls.append((count_multiplier, iteration_budget * count_multiplier))
            actual_losses, expected_random_state, _ = run_legacy_training_iterations(
                legacy_client=legacy_client,
                iteration_count=count_multiplier,
                update_shared_features=update_shared_features,
                initial_random_state=python_random_generator.getstate(),
            )
            expected_losses.append(actual_losses)

        legacy_client.train_all_held_models = original_training_function
        keyword_arguments = dict(
            held_model_training_bindings=held_model_training_bindings,
            ordered_model_training_samples=ordered_model_training_samples,
            batch_sample_count=3,
            python_random_generator=python_random_generator,
            local_training_settings=LocalTrainingSettings(
                local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
            ),
            shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
            shared_parameter_optimizer=shared_parameter_optimizer,
            update_shared_features=update_shared_features,
        )
        training_events = ("flush", "request", "flush", "flush", *("request",) * 5, "flush")
        for training_event in training_events:
            if training_event == "request":
                legacy_client.train_step()
            else:
                legacy_client.flush_pending_updates()
            actual_losses = execute_scheduled_training_request(
                schedule=schedule,
                schedule_settings=schedule_settings,
                training_event=training_event,
                original_training_function=lambda requested_iteration_count: (
                    new_training_calls.append(
                        (schedule.pending_training_request_count, requested_iteration_count)
                    )
                    or perform_held_model_joint_training_iterations(
                        requested_joint_update_iteration_count=requested_iteration_count,
                        **keyword_arguments,
                    )
                ),
            )
            if actual_losses is not None:
                assert actual_losses == expected_losses[-1]
            assert schedule.pending_training_request_count == legacy_client._pending_updates
            assert new_training_calls == legacy_training_calls
            assert python_random_generator.getstate() == expected_random_state
            assert_joint_update_states_equal(
                participating_training_batches=participating_training_batches,
                shared_parameter_optimizer=shared_parameter_optimizer,
                legacy_client=legacy_client,
            )
            assert random.getstate() == global_python_random_state
    finally:
        random.setstate(global_python_random_state)
        torch.set_rng_state(global_torch_random_state)


def test_training_request_schedule_consumes_ineligible_attempts(monkeypatch):
    schedule_settings, schedule, legacy_client, _ = build_training_schedule_oracle_pair(
        interval_count=3, iteration_budget=2, monkeypatch=monkeypatch
    )
    python_random_generator = random.Random(731)
    initial_random_state = python_random_generator.getstate()
    # 通常の正当な抽出入力に参加者なし。使わないNN/設定は読まれない。
    keyword_arguments = dict(
        held_model_training_bindings=(),
        ordered_model_training_samples=(),
        batch_sample_count=3,
        python_random_generator=python_random_generator,
        local_training_settings=None,
        shared_feature_extractor=None,
        shared_parameter_optimizer=None,
        update_shared_features=True,
    )
    for _ in range(3):
        legacy_client.train_step()
        actual_losses = execute_scheduled_training_request(
            schedule=schedule,
            schedule_settings=schedule_settings,
            training_event="request",
            original_training_function=lambda requested_iteration_count: (
                perform_held_model_joint_training_iterations(
                    requested_joint_update_iteration_count=requested_iteration_count,
                    **keyword_arguments,
                )
            ),
        )
    assert actual_losses == ()
    assert schedule.pending_training_request_count == legacy_client._pending_updates == 0
    assert python_random_generator.getstate() == initial_random_state
