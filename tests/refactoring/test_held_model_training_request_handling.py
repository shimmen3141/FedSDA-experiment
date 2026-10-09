"""学習要求の記録・共同学習・学習量の計数の接続を、実旧clientの学習stepと照合する。"""

import random
from collections import defaultdict

import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    assert_held_model_states_match_legacy,
)
from test_assigned_training_sample_absorption import (
    assert_absorption_matches_legacy,
    assert_absorption_state_unchanged,
    build_absorption_oracle,
    snapshot_absorption_state,
)
from test_held_candidate_validation_progress import make_subclass_copy

import federated_learning_experiments.learning.training.held_model_joint_training_iterations as joint_training_iteration_module
import federated_learning_experiments.runtime.held_model_training_request_handling as training_request_module
from federated_drift_experiment import config
from federated_learning_experiments.learning.training.local_training_request_schedule import (
    LocalTrainingRequestSchedule,
)
from federated_learning_experiments.learning.training.local_training_schedule_settings import (
    LocalTrainingScheduleSettings,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.runtime.assigned_training_sample_absorption import (
    absorb_assigned_training_samples_into_held_model,
)
from federated_learning_experiments.runtime.held_model_training_request_handling import (
    record_training_request_and_train_held_models_when_due,
    train_held_models_for_pending_training_requests,
)

# 学習要求（旧train_step）と明示の消化（旧flush_pending_updates）の列。空の消化、端数の消化、連続した要求を含む。
TRAINING_EVENTS = (
    "flush",
    "request",
    "request",
    "flush",
    "flush",
    *("request",) * 5,
    "flush",
)
TRAINING_FUNCTIONS = {
    "request": record_training_request_and_train_held_models_when_due,
    "flush": train_held_models_for_pending_training_requests,
}
UNHELD_MODEL_ID = 55
OWNER_ARGUMENT_NAMES = (
    "local_training_request_schedule",
    "held_model_training_state_registry",
    "training_sample_store",
    "model_training_and_assignment_counts_store",
)


def build_training_request_oracle(
    *,
    class_count,
    monkeypatch,
    update_interval,
    iterations_per_request,
    batch_sample_count,
    optimizer_variant="standard",
):
    """吸収のoracleの新ownerと実旧clientへ、学習要求の管理と学習の設定を加える。

    モデル9へ2標本を吸収して、標本数をモデル4が3件、モデル9が5件にする（batchの件数で参加モデルが変わる）。
    保有していないモデル55にも6標本を持たせる（どのbatchの件数でも標本は足りるが、学習に参加しない）。
    """
    (
        absorption_arguments,
        adoption_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_training_samples,
    ) = build_absorption_oracle(
        class_count=class_count,
        monkeypatch=monkeypatch,
        absorbed_sample_count=2,
        optimizer_variant=optimizer_variant,
    )
    legacy_client._absorb_into_store(absorption_arguments["model_id"], legacy_training_samples)
    absorb_assigned_training_samples_into_held_model(**absorption_arguments)
    absorption_arguments["training_sample_store"].append_model_training_samples(
        model_id=UNHELD_MODEL_ID,
        training_samples=absorption_arguments["assigned_training_samples"] * 3,
    )
    legacy_client.train_data_store[UNHELD_MODEL_ID].extend(legacy_training_samples * 3)
    monkeypatch.setattr(config, "LOCAL_UPDATE_INTERVAL", update_interval)
    monkeypatch.setattr(config, "SHARED_BACKBONE_TRAINING", "joint")
    monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
    # 実旧の学習stepが読む属性。計算量と時間の記録は対象外なので、入れ物だけ与える。
    legacy_client._pending_updates = 0
    legacy_client.updates_per_sample = iterations_per_request
    legacy_client.batch_size = batch_sample_count
    legacy_client.backbone_gradient_diagnostics = defaultdict(float)
    legacy_client.phase_seconds = defaultdict(float)
    registry = absorption_arguments["held_model_training_state_registry"]
    training_arguments = dict(
        local_training_request_schedule=LocalTrainingRequestSchedule(
            local_training_schedule_settings=LocalTrainingScheduleSettings(
                training_requests_per_update_interval=update_interval,
                joint_update_iterations_per_training_request=iterations_per_request,
            )
        ),
        held_model_training_state_registry=registry,
        training_sample_store=absorption_arguments["training_sample_store"],
        model_training_and_assignment_counts_store=absorption_arguments[
            "model_training_and_assignment_counts_store"
        ],
        batch_sample_count=batch_sample_count,
        python_random_generator=random.Random(977),
        local_training_settings=LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        ),
        shared_feature_extractor=registry.get_held_model_training_state(
            model_id=absorption_arguments["model_id"]
        ).classifier.feature_extractor,
        shared_parameter_optimizer=shared_optimizer_owners[0].parameter_optimizer,
    )
    return (
        training_arguments,
        absorption_arguments,
        adoption_arguments,
        shared_optimizer_owners,
        legacy_client,
    )


def run_legacy_training_event(*, legacy_client, training_event, legacy_random_state):
    """実旧の学習stepまたは明示の消化を、指定の乱数状態から実行し、実行後の乱数状態を返す。"""
    global_python_random_state = random.getstate()
    try:
        random.setstate(legacy_random_state)
        if training_event == "request":
            legacy_client.train_step()
        else:
            legacy_client.flush_pending_updates()
        return random.getstate()
    finally:
        random.setstate(global_python_random_state)


def snapshot_training_request_state(
    *, training_arguments, absorption_arguments, shared_optimizer_owners
):
    return (
        snapshot_absorption_state(
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        ),
        training_arguments["python_random_generator"].getstate(),
    )


def assert_training_request_state_unchanged(
    *, state_snapshot, training_arguments, absorption_arguments
):
    absorption_snapshot, python_random_generator_state = state_snapshot
    assert_absorption_state_unchanged(
        previous_snapshot=absorption_snapshot, valid_absorption_arguments=absorption_arguments
    )
    assert training_arguments["python_random_generator"].getstate() == python_random_generator_state


def spy_on_joint_training_iterations(monkeypatch, *, observed_state_reader=lambda: None):
    """共同学習の反復の呼出しを、引数と呼出し時点の状態つきで記録する。実物をそのまま実行する。"""
    perform_joint_training_iterations = (
        training_request_module.perform_held_model_joint_training_iterations
    )
    joint_training_calls = []

    def record_joint_training_call(**joint_training_arguments):
        observed_state = observed_state_reader()
        completed_joint_update_losses = perform_joint_training_iterations(
            **joint_training_arguments
        )
        joint_training_calls.append(
            (joint_training_arguments, observed_state, completed_joint_update_losses)
        )
        return completed_joint_update_losses

    monkeypatch.setattr(
        training_request_module,
        "perform_held_model_joint_training_iterations",
        record_joint_training_call,
    )
    return joint_training_calls


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("update_interval", (1, 2, 3))
@pytest.mark.parametrize("iterations_per_request", (0, 1, 2))
@pytest.mark.parametrize("batch_sample_count", (1, 3, 4, 6))
def test_training_request_handling_matches_real_legacy_training_steps(
    class_count, update_interval, iterations_per_request, batch_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(541)
        (
            training_arguments,
            absorption_arguments,
            adoption_arguments,
            shared_optimizer_owners,
            legacy_client,
        ) = build_training_request_oracle(
            class_count=class_count,
            monkeypatch=monkeypatch,
            update_interval=update_interval,
            iterations_per_request=iterations_per_request,
            batch_sample_count=batch_sample_count,
        )
        joint_training_calls = spy_on_joint_training_iterations(monkeypatch)
        schedule = training_arguments["local_training_request_schedule"]
        python_random_generator = training_arguments["python_random_generator"]
        legacy_random_state = python_random_generator.getstate()
        global_python_random_state = random.getstate()
        # batchの件数が1・3なら両モデル、4ならモデル9だけが参加し、6ならどのモデルも参加しない。
        participating_model_count = {1: 2, 3: 2, 4: 1, 6: 0}[batch_sample_count]
        for training_event in TRAINING_EVENTS:
            legacy_optimizer_step_count = legacy_client.compute_counters["optimizer_steps"]
            legacy_random_state = run_legacy_training_event(
                legacy_client=legacy_client,
                training_event=training_event,
                legacy_random_state=legacy_random_state,
            )
            joint_training_call_count = len(joint_training_calls)
            completed_joint_update_losses = TRAINING_FUNCTIONS[training_event](**training_arguments)
            # 保留中の要求の件数、乱数、学習量の計数、標本、モデルとoptimizerが実旧と一致する。
            assert schedule.pending_training_request_count == legacy_client._pending_updates
            assert python_random_generator.getstate() == legacy_random_state
            assert_absorption_matches_legacy(
                absorption_arguments=absorption_arguments, legacy_client=legacy_client
            )
            assert_held_model_states_match_legacy(
                registry=training_arguments["held_model_training_state_registry"],
                shared_optimizer_owners=shared_optimizer_owners[:-1],
                legacy_client=legacy_client,
                input_features=adoption_arguments["initial_statistics_input_features"],
            )
            # 戻り値は、共同学習の反復が返した損失を呼出しの順に並べたもの。件数は実旧の更新回数と対応する。
            assert type(completed_joint_update_losses) is tuple
            assert completed_joint_update_losses == tuple(
                joint_update_loss
                for _, _, joint_update_losses in joint_training_calls[joint_training_call_count:]
                for joint_update_loss in joint_update_losses
            )
            assert len(completed_joint_update_losses) * participating_model_count == (
                legacy_client.compute_counters["optimizer_steps"] - legacy_optimizer_step_count
            )
        assert random.getstate() == global_python_random_state
        # 列の最後は消化なので、保留は残らない。
        assert schedule.pending_training_request_count == 0


@pytest.mark.parametrize("optimizer_variant", ("amsgrad", "sgd"))
def test_training_request_handling_matches_real_legacy_for_other_optimizers(
    optimizer_variant, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(541)
        (
            training_arguments,
            absorption_arguments,
            adoption_arguments,
            shared_optimizer_owners,
            legacy_client,
        ) = build_training_request_oracle(
            class_count=4,
            monkeypatch=monkeypatch,
            update_interval=2,
            iterations_per_request=2,
            batch_sample_count=3,
            optimizer_variant=optimizer_variant,
        )
        legacy_random_state = training_arguments["python_random_generator"].getstate()
        for training_event in ("request", "request", "request", "flush"):
            legacy_random_state = run_legacy_training_event(
                legacy_client=legacy_client,
                training_event=training_event,
                legacy_random_state=legacy_random_state,
            )
            TRAINING_FUNCTIONS[training_event](**training_arguments)
            assert training_arguments["python_random_generator"].getstate() == legacy_random_state
            assert_absorption_matches_legacy(
                absorption_arguments=absorption_arguments, legacy_client=legacy_client
            )
            assert_held_model_states_match_legacy(
                registry=training_arguments["held_model_training_state_registry"],
                shared_optimizer_owners=shared_optimizer_owners[:-1],
                legacy_client=legacy_client,
                input_features=adoption_arguments["initial_statistics_input_features"],
            )


@pytest.mark.parametrize("training_event", ("request", "flush"))
def test_training_runs_before_counts_are_recorded_and_requests_are_acknowledged(
    training_event, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(541)
        training_arguments, _, _, _, _ = build_training_request_oracle(
            class_count=2,
            monkeypatch=monkeypatch,
            update_interval=3,
            iterations_per_request=2,
            batch_sample_count=4,
        )
        schedule = training_arguments["local_training_request_schedule"]
        registry = training_arguments["held_model_training_state_registry"]
        training_sample_store = training_arguments["training_sample_store"]
        counts_store = training_arguments["model_training_and_assignment_counts_store"]
        # 先に2件を保留し、対象の呼出しで3件目の要求（または2件の消化）にする。
        for _ in range(2):
            assert (
                record_training_request_and_train_held_models_when_due(**training_arguments) == ()
            )
        counts_before = counts_store.snapshot_model_training_and_assignment_counts()
        joint_training_calls = spy_on_joint_training_iterations(
            monkeypatch,
            observed_state_reader=lambda: (
                schedule.pending_training_request_count,
                counts_store.snapshot_model_training_and_assignment_counts(),
            ),
        )
        # 計数への反映の時点で、保留はまだ消化されていない。
        record_completed_model_training = counts_store.record_completed_model_training
        pending_request_counts_at_recording = []
        monkeypatch.setattr(
            counts_store,
            "record_completed_model_training",
            lambda **recording_arguments: (
                pending_request_counts_at_recording.append(schedule.pending_training_request_count),
                record_completed_model_training(**recording_arguments),
            )[1],
        )
        completed_joint_update_losses = TRAINING_FUNCTIONS[training_event](**training_arguments)
    pending_request_count = 3 if training_event == "request" else 2
    completed_joint_update_count = pending_request_count * 2
    # 共同学習の反復は、共同更新1回ぶんずつ、保留中の要求のぶんの回数だけ呼ばれる。
    assert len(joint_training_calls) == completed_joint_update_count
    assert completed_joint_update_losses == tuple(
        joint_update_loss
        for _, _, joint_update_losses in joint_training_calls
        for joint_update_loss in joint_update_losses
    )
    assert len(completed_joint_update_losses) == completed_joint_update_count
    registered_bindings = tuple(
        (binding.model_id, binding.classifier, binding.concept_specific_parameter_optimizer)
        for binding in registry.snapshot_ordered_held_model_training_bindings()
    )
    for completed_call_count, (joint_training_arguments, observed_state, _) in enumerate(
        joint_training_calls
    ):
        # 各回の時点で、保留は未消化。計数は、それまでに完了した回のぶんだけ増えている
        # （batchが4件なので参加はモデル9だけ）。
        observed_pending_request_count, observed_counts = observed_state
        assert observed_pending_request_count == pending_request_count
        assert observed_counts.trained_sample_counts_by_model_id == {
            4: counts_before.trained_sample_counts_by_model_id[4],
            9: counts_before.trained_sample_counts_by_model_id[9] + 4 * completed_call_count,
        }
        assert observed_counts.parameter_update_step_counts_by_model_id == {
            4: counts_before.parameter_update_step_counts_by_model_id[4],
            9: counts_before.parameter_update_step_counts_by_model_id[9] + completed_call_count,
        }
        assert joint_training_arguments["requested_joint_update_iteration_count"] == 1
        assert joint_training_arguments["update_shared_features"] is True
        assert (
            tuple(
                (binding.model_id, binding.classifier, binding.concept_specific_parameter_optimizer)
                for binding in joint_training_arguments["held_model_training_bindings"]
            )
            == registered_bindings
        )
        assert (
            joint_training_arguments["ordered_model_training_samples"]
            == training_sample_store.snapshot_ordered_model_training_samples()
        )
        for argument_name in (
            "batch_sample_count",
            "python_random_generator",
            "local_training_settings",
            "shared_feature_extractor",
            "shared_parameter_optimizer",
        ):
            assert joint_training_arguments[argument_name] is training_arguments[argument_name]
        assert set(joint_training_arguments) == {
            "requested_joint_update_iteration_count",
            "held_model_training_bindings",
            "ordered_model_training_samples",
            "update_shared_features",
            "batch_sample_count",
            "python_random_generator",
            "local_training_settings",
            "shared_feature_extractor",
            "shared_parameter_optimizer",
        }
    # 計数への反映は共同更新1回につき1回で、どの時点でも保留は未消化。
    assert pending_request_counts_at_recording == (
        [pending_request_count] * completed_joint_update_count
    )
    # 完了後: 完了した更新の回数ぶんだけモデル9の計数が増え、割当概念の計数は変わらず、保留は消化される。
    counts_after = counts_store.snapshot_model_training_and_assignment_counts()
    assert counts_after.trained_sample_counts_by_model_id == {
        4: counts_before.trained_sample_counts_by_model_id[4],
        9: counts_before.trained_sample_counts_by_model_id[9] + 4 * completed_joint_update_count,
    }
    assert counts_after.parameter_update_step_counts_by_model_id == {
        4: counts_before.parameter_update_step_counts_by_model_id[4],
        9: counts_before.parameter_update_step_counts_by_model_id[9] + completed_joint_update_count,
    }
    assert (
        counts_after.assigned_sample_counts_by_model_and_concept_id
        == counts_before.assigned_sample_counts_by_model_and_concept_id
    )
    assert schedule.pending_training_request_count == 0


@pytest.mark.parametrize("training_event", ("request", "flush"))
@pytest.mark.parametrize("owner_argument_name", OWNER_ARGUMENT_NAMES)
@pytest.mark.parametrize("invalid_owner_kind", ("other_type", "subclass"))
def test_training_request_handling_rejects_invalid_owner_before_any_update(
    training_event, owner_argument_name, invalid_owner_kind, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(541)
        training_arguments, absorption_arguments, _, shared_optimizer_owners, _ = (
            build_training_request_oracle(
                class_count=2,
                monkeypatch=monkeypatch,
                update_interval=1,
                iterations_per_request=1,
                batch_sample_count=3,
            )
        )
        schedule = training_arguments["local_training_request_schedule"]
        # 消化でも学習へ進む状態にするため、要求を1件保留しておく（間隔1なので、通常の経路では残らない）。
        schedule.record_training_request()
        joint_training_calls = spy_on_joint_training_iterations(monkeypatch)
        state_snapshot = snapshot_training_request_state(
            training_arguments=training_arguments,
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        invalid_owner = (
            object()
            if invalid_owner_kind == "other_type"
            else make_subclass_copy(training_arguments[owner_argument_name])
        )
        with pytest.raises(TypeError, match=owner_argument_name):
            TRAINING_FUNCTIONS[training_event](
                **training_arguments | {owner_argument_name: invalid_owner}
            )
        assert joint_training_calls == []
        assert schedule.pending_training_request_count == 1
        assert_training_request_state_unchanged(
            state_snapshot=state_snapshot,
            training_arguments=training_arguments,
            absorption_arguments=absorption_arguments,
        )


@pytest.mark.parametrize("training_event", ("request", "flush"))
def test_failed_training_keeps_pending_requests_and_counts(training_event, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(541)
        training_arguments, absorption_arguments, _, shared_optimizer_owners, legacy_client = (
            build_training_request_oracle(
                class_count=2,
                monkeypatch=monkeypatch,
                update_interval=2,
                iterations_per_request=1,
                batch_sample_count=3,
            )
        )
        schedule = training_arguments["local_training_request_schedule"]
        assert record_training_request_and_train_held_models_when_due(**training_arguments) == ()
        legacy_client.train_step()
        state_snapshot = snapshot_training_request_state(
            training_arguments=training_arguments,
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        # batchの件数が不正なら、共同学習の反復が抽出より前に拒否する。実旧も学習の中で失敗する。
        with pytest.raises(ValueError, match="batch_sample_count"):
            TRAINING_FUNCTIONS[training_event](**training_arguments | dict(batch_sample_count=0))
        legacy_client.batch_size = 0
        with pytest.raises((RuntimeError, ValueError)):
            run_legacy_training_event(
                legacy_client=legacy_client,
                training_event=training_event,
                legacy_random_state=random.getstate(),
            )
        # 要求は記録されたまま残り（実旧と同じ件数）、計数・モデル・乱数は変わらない。
        assert schedule.pending_training_request_count == legacy_client._pending_updates
        assert schedule.pending_training_request_count == (2 if training_event == "request" else 1)
        assert_training_request_state_unchanged(
            state_snapshot=state_snapshot,
            training_arguments=training_arguments,
            absorption_arguments=absorption_arguments,
        )


@pytest.mark.parametrize("training_event", ("request", "flush"))
def test_training_inputs_are_not_read_when_no_training_is_due(training_event, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(541)
        training_arguments, absorption_arguments, _, shared_optimizer_owners, _ = (
            build_training_request_oracle(
                class_count=2,
                monkeypatch=monkeypatch,
                update_interval=3,
                iterations_per_request=1,
                batch_sample_count=3,
            )
        )
        schedule = training_arguments["local_training_request_schedule"]
        joint_training_calls = spy_on_joint_training_iterations(monkeypatch)
        state_snapshot = snapshot_training_request_state(
            training_arguments=training_arguments,
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        # 学習へ進まないときは、学習へ渡すだけの値を読まない（不正でも成功する）。
        unread_arguments = training_arguments | dict(
            batch_sample_count=0,
            python_random_generator=object(),
            local_training_settings=object(),
            shared_feature_extractor=object(),
            shared_parameter_optimizer=object(),
        )
        assert TRAINING_FUNCTIONS[training_event](**unread_arguments) == ()
        assert joint_training_calls == []
        assert schedule.pending_training_request_count == (1 if training_event == "request" else 0)
        assert_training_request_state_unchanged(
            state_snapshot=state_snapshot,
            training_arguments=training_arguments,
            absorption_arguments=absorption_arguments,
        )


@pytest.mark.parametrize("training_event", ("request", "flush"))
def test_training_failure_after_completed_updates_matches_real_legacy(training_event, monkeypatch):
    """2回目の共同更新を、抽出の後・更新の前に失敗させる。完了した1回ぶんの学習と計数は残り、保留は消化されない。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(541)
        (
            training_arguments,
            absorption_arguments,
            adoption_arguments,
            shared_optimizer_owners,
            legacy_client,
        ) = build_training_request_oracle(
            class_count=4,
            monkeypatch=monkeypatch,
            update_interval=2,
            iterations_per_request=2,
            batch_sample_count=3,
        )
        schedule = training_arguments["local_training_request_schedule"]
        python_random_generator = training_arguments["python_random_generator"]
        legacy_random_state = run_legacy_training_event(
            legacy_client=legacy_client,
            training_event="request",
            legacy_random_state=python_random_generator.getstate(),
        )
        assert record_training_request_and_train_held_models_when_due(**training_arguments) == ()
        counts_before = training_arguments[
            "model_training_and_assignment_counts_store"
        ].snapshot_model_training_and_assignment_counts()
        # 実旧: 2回目の抽出を実行した直後に失敗させる（更新は1回だけ完了する）。
        sample_legacy_training_batches = legacy_client._sample_training_batches
        legacy_sampling_calls = []

        def fail_after_second_legacy_sampling():
            legacy_sampling_calls.append(sample_legacy_training_batches())
            if len(legacy_sampling_calls) == 2:
                raise RuntimeError("injected failure")
            return legacy_sampling_calls[-1]

        legacy_client._sample_training_batches = fail_after_second_legacy_sampling
        with pytest.raises(RuntimeError, match="injected failure"):
            run_legacy_training_event(
                legacy_client=legacy_client,
                training_event=training_event,
                legacy_random_state=legacy_random_state,
            )
        # 新: 2回目の共同更新を、抽出の後・更新の前に失敗させる。
        perform_joint_model_parameter_update = (
            joint_training_iteration_module.perform_joint_model_parameter_update
        )
        joint_update_calls = []

        def fail_at_second_joint_update(**joint_update_arguments):
            joint_update_calls.append(joint_update_arguments)
            if len(joint_update_calls) == 2:
                raise RuntimeError("injected failure")
            return perform_joint_model_parameter_update(**joint_update_arguments)

        monkeypatch.setattr(
            joint_training_iteration_module,
            "perform_joint_model_parameter_update",
            fail_at_second_joint_update,
        )
        with pytest.raises(RuntimeError, match="injected failure"):
            TRAINING_FUNCTIONS[training_event](**training_arguments)
        # 完了した1回ぶんのparameter・optimizer・計数が実旧と一致し、保留は実旧と同じ件数のまま残る。
        assert len(joint_update_calls) == len(legacy_sampling_calls) == 2
        assert schedule.pending_training_request_count == legacy_client._pending_updates
        assert schedule.pending_training_request_count == (2 if training_event == "request" else 1)
        assert_absorption_matches_legacy(
            absorption_arguments=absorption_arguments, legacy_client=legacy_client
        )
        assert_held_model_states_match_legacy(
            registry=training_arguments["held_model_training_state_registry"],
            shared_optimizer_owners=shared_optimizer_owners[:-1],
            legacy_client=legacy_client,
            input_features=adoption_arguments["initial_statistics_input_features"],
        )
        counts_after = training_arguments[
            "model_training_and_assignment_counts_store"
        ].snapshot_model_training_and_assignment_counts()
        assert counts_after.parameter_update_step_counts_by_model_id == {
            model_id: step_count + 1
            for model_id, step_count in counts_before.parameter_update_step_counts_by_model_id.items()
        }


@pytest.mark.parametrize("training_event", ("request", "flush"))
def test_training_inputs_are_not_read_when_iteration_budget_is_zero(training_event, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(541)
        training_arguments, absorption_arguments, _, shared_optimizer_owners, _ = (
            build_training_request_oracle(
                class_count=2,
                monkeypatch=monkeypatch,
                update_interval=2,
                iterations_per_request=0,
                batch_sample_count=3,
            )
        )
        schedule = training_arguments["local_training_request_schedule"]
        schedule.record_training_request()
        joint_training_calls = spy_on_joint_training_iterations(monkeypatch)
        state_snapshot = snapshot_training_request_state(
            training_arguments=training_arguments,
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        # 回数が0なら、間隔に達していても学習へ渡すだけの値を読まず、保留だけを消化する。
        unread_arguments = training_arguments | dict(
            batch_sample_count=0,
            python_random_generator=object(),
            local_training_settings=object(),
            shared_feature_extractor=object(),
            shared_parameter_optimizer=object(),
        )
        assert TRAINING_FUNCTIONS[training_event](**unread_arguments) == ()
        assert joint_training_calls == []
        assert schedule.pending_training_request_count == 0
        assert_training_request_state_unchanged(
            state_snapshot=state_snapshot,
            training_arguments=training_arguments,
            absorption_arguments=absorption_arguments,
        )


def test_counts_are_not_recorded_for_iterations_without_completed_update(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(541)
        training_arguments, absorption_arguments, _, shared_optimizer_owners, _ = (
            build_training_request_oracle(
                class_count=2,
                monkeypatch=monkeypatch,
                update_interval=1,
                iterations_per_request=2,
                batch_sample_count=3,
            )
        )
        state_snapshot = snapshot_training_request_state(
            training_arguments=training_arguments,
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        # 共同学習の反復が「完了した更新なし」を返したら、標本の足りる保有モデルがあっても計数しない。
        joint_training_calls = []
        monkeypatch.setattr(
            training_request_module,
            "perform_held_model_joint_training_iterations",
            lambda **joint_training_arguments: (
                joint_training_calls.append(joint_training_arguments) or ()
            ),
        )
        assert record_training_request_and_train_held_models_when_due(**training_arguments) == ()
        assert len(joint_training_calls) == 2
        assert (
            training_arguments["local_training_request_schedule"].pending_training_request_count
            == 0
        )
        assert_training_request_state_unchanged(
            state_snapshot=state_snapshot,
            training_arguments=training_arguments,
            absorption_arguments=absorption_arguments,
        )
