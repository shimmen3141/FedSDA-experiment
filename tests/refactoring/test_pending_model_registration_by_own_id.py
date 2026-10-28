"""送信保留のモデルを、それ自身の一時IDから正式IDへ付け替える方針（LEGACY-016の修正）。

実旧には、この方針の正解が無い（旧は、現行モデルを付け替える）。現行モデルが送信保留のモデルである場合は、
実旧と照合済みの、現行を付け替える方針の結果と比べ、それ以外は、状態の遷移を直接確かめる。
"""

from copy import deepcopy
from dataclasses import replace

import behavior_golden_cases as golden
import pytest
import test_server_model_registration_and_aggregation as server_test_module
import torch
from test_held_model_registration_confirmation import (
    build_registration_confirmation_oracle,
    snapshot_confirmation_owners,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

import federated_learning_experiments.runtime.feddrift_run_server as feddrift_server_module
import federated_learning_experiments.runtime.server_model_registration_and_aggregation as registration_module
import federated_learning_experiments.runtime.server_round_synchronization as synchronization_module
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (
    KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
    REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
    PendingModelRegistrationSettings,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.held_model_registration_confirmation import (
    confirm_held_model_registration,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    register_ready_client_models,
)

BUILD_SEED = 20261013


def build_own_id_confirmation_arguments(
    *, initial_model_ids=(4, -7, 9), ready_pending_model_ids=(-7,), current_model_id=-7
):
    """確認の引数を、既存のoracleの組立てで作り、送信保留を、全部保持する方針のownerへ替える。"""
    torch.manual_seed(BUILD_SEED)
    confirmation_arguments, _ = build_registration_confirmation_oracle(
        initial_model_ids=initial_model_ids
    )
    registry = confirmation_arguments["held_model_training_state_registry"]
    pending_upload_state = PendingModelUploadState(
        pending_model_registration_policy=REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME
    )
    held_classifier = registry.snapshot_ordered_held_model_training_states()[0].classifier
    for pending_model_id in ready_pending_model_ids:
        pending_upload_state.queue_ready_model_upload(
            model_id=pending_model_id,
            parameter_snapshot=snapshot_classifier_parameters(classifier=held_classifier),
        )
    confirmation_arguments["pending_model_upload_state"] = pending_upload_state
    if current_model_id != -7:
        confirmation_arguments["current_training_model_assignment"].assign_model_for_training(
            model_id=current_model_id
        )
    return confirmation_arguments


def describe_owners_by_model_id(confirmation_arguments):
    """ownerの中身を、モデルIDごとに、objectの同一性に依らない値で表す（並びも含む）。"""
    (
        held_states,
        loss_statistics_snapshot,
        training_sample_collections,
        evaluation_sample_collections,
        counts_snapshot,
        current_model_id,
        _,
        _,
    ) = snapshot_confirmation_owners(confirmation_arguments)
    return dict(
        held_model_ids=tuple(state.model_id for state in held_states),
        held_parameters={
            state.model_id: {
                name: values.clone()
                for name, values in snapshot_classifier_parameters(
                    classifier=state.classifier
                ).items()
            }
            for state in held_states
        },
        loss_statistics=loss_statistics_snapshot,
        training_samples=tuple(
            (
                collection.model_id,
                tuple(
                    (sample.input_features.tolist(), sample.observed_class_labels.tolist())
                    for sample in collection.training_samples
                ),
            )
            for collection in training_sample_collections
        ),
        evaluation_samples=tuple(
            (
                collection.model_id,
                tuple(
                    (sample.input_features.tolist(), sample.observed_class_labels.tolist())
                    for sample in collection.evaluation_samples
                ),
            )
            for collection in evaluation_sample_collections
        ),
        counts=counts_snapshot,
        current_model_id=current_model_id,
    )


def assert_owner_descriptions_equal(actual, expected, *, ignored_keys=()):
    assert set(actual) == set(expected)
    for key in actual:
        if key in ignored_keys:
            continue
        if key == "held_parameters":
            assert tuple(actual[key]) == tuple(expected[key])
            for model_id, parameters in actual[key].items():
                assert tuple(parameters) == tuple(expected[key][model_id])
                for name, values in parameters.items():
                    assert torch.equal(values, expected[key][model_id][name]), (model_id, name)
        else:
            assert actual[key] == expected[key], key


def describe_pending_uploads(confirmation_arguments):
    pending_upload_state = confirmation_arguments["pending_model_upload_state"]
    return (
        tuple(
            pending_model_upload.model_id
            for pending_model_upload in pending_upload_state.get_all_pending_model_uploads()
        ),
        pending_upload_state.remaining_upload_delay_round_count,
        pending_upload_state.has_ready_model_upload(),
    )


def confirm_with_keep_latest_policy(*, initial_model_ids, registered_global_model_id):
    """現行（-7）を付け替える方針の結果（実旧との一致は、既存のtestが確かめている）。"""
    torch.manual_seed(BUILD_SEED)
    confirmation_arguments, _ = build_registration_confirmation_oracle(
        initial_model_ids=initial_model_ids, receiving_model_id=registered_global_model_id
    )
    confirm_held_model_registration(**confirmation_arguments)
    return describe_owners_by_model_id(confirmation_arguments)


@pytest.mark.parametrize("initial_model_ids", [(-7,), (4, -7, 9), (-7, 4, 9), (4, 9, -7)])
def test_current_pending_model_is_reassigned_like_keep_latest_policy(initial_model_ids):
    """現行が送信保留のモデル自身: 現行を付け替える方針と、全ownerの結果が同じ。"""
    with torch.random.fork_rng(devices=[]):
        expected = confirm_with_keep_latest_policy(
            initial_model_ids=initial_model_ids, registered_global_model_id=12
        )
        confirmation_arguments = build_own_id_confirmation_arguments(
            initial_model_ids=initial_model_ids
        )
        confirmation_arguments["registered_global_model_id"] = 12
        assignment_change = confirm_held_model_registration(**confirmation_arguments)
        assert (assignment_change.previous_model_id, assignment_change.current_model_id) == (-7, 12)
        assert_owner_descriptions_equal(
            describe_owners_by_model_id(confirmation_arguments), expected
        )
        assert describe_pending_uploads(confirmation_arguments) == ((), 0, False)


@pytest.mark.parametrize(
    ("initial_model_ids", "current_model_id"),
    [((4, -7, 9), 4), ((4, -7, 9), 9), ((-7, 4, 9), 4), ((4, -7, -9), -9)],
)
def test_pending_model_is_reassigned_while_another_model_is_current(
    initial_model_ids, current_model_id
):
    """現行が別のモデル（非負、または別の一時ID）: 送信保留のモデルを付け替え、現行は変えない。

    旧と同じ方針は、現行が非負なら送信保留を外すだけ（LEGACY-016）。この方針は、その短絡を通らない。
    """
    with torch.random.fork_rng(devices=[]):
        expected = confirm_with_keep_latest_policy(
            initial_model_ids=initial_model_ids, registered_global_model_id=12
        )
        confirmation_arguments = build_own_id_confirmation_arguments(
            initial_model_ids=initial_model_ids, current_model_id=current_model_id
        )
        confirmation_arguments["registered_global_model_id"] = 12
        assert confirm_held_model_registration(**confirmation_arguments) is None
        actual = describe_owners_by_model_id(confirmation_arguments)
        # 現行のほかは、現行を付け替える方針で-7を付け替えた結果と同じ（全ownerで、-7が12になる）。
        assert_owner_descriptions_equal(actual, expected, ignored_keys=("current_model_id",))
        assert actual["current_model_id"] == current_model_id
        assert -7 not in actual["held_model_ids"] and 12 in actual["held_model_ids"]
        assert describe_pending_uploads(confirmation_arguments) == ((), 0, False)


def test_each_ready_pending_model_is_confirmed_in_adoption_order():
    """送信保留が2つ: 確認のたびに、古い順に1つずつ付け替え、その1つだけを外す。"""
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments = build_own_id_confirmation_arguments(
            initial_model_ids=(4, -7, -9), ready_pending_model_ids=(-7, -9), current_model_id=-9
        )
        confirmation_arguments["registered_global_model_id"] = 12
        assert confirm_held_model_registration(**confirmation_arguments) is None
        assert describe_owners_by_model_id(confirmation_arguments)["held_model_ids"] == (4, -9, 12)
        assert describe_pending_uploads(confirmation_arguments) == ((-9,), 0, True)
        confirmation_arguments["registered_global_model_id"] = 13
        assignment_change = confirm_held_model_registration(**confirmation_arguments)
        assert (assignment_change.previous_model_id, assignment_change.current_model_id) == (-9, 13)
        actual = describe_owners_by_model_id(confirmation_arguments)
        assert actual["held_model_ids"] == (4, 12, 13)
        assert actual["current_model_id"] == 13
        assert tuple(model_id for model_id, _ in actual["training_samples"]) == (4, 12, 13)
        assert describe_pending_uploads(confirmation_arguments) == ((), 0, False)


def test_waiting_pending_model_is_skipped_for_a_ready_one():
    """待ちの残る送信保留が先頭にあっても、待ちの済んだものを付け替え、待ちの残るものは残す。"""
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments = build_own_id_confirmation_arguments(
            initial_model_ids=(4, -7, -9), ready_pending_model_ids=(), current_model_id=4
        )
        pending_upload_state = confirmation_arguments["pending_model_upload_state"]
        registry = confirmation_arguments["held_model_training_state_registry"]
        parameter_snapshot = snapshot_classifier_parameters(
            classifier=registry.snapshot_ordered_held_model_training_states()[0].classifier
        )
        pending_upload_state.queue_model_upload(
            model_id=-7, parameter_snapshot=parameter_snapshot, upload_delay_round_count=2
        )
        pending_upload_state.queue_ready_model_upload(
            model_id=-9, parameter_snapshot=parameter_snapshot
        )
        confirmation_arguments["registered_global_model_id"] = 12
        assert confirm_held_model_registration(**confirmation_arguments) is None
        assert describe_owners_by_model_id(confirmation_arguments)["held_model_ids"] == (4, -7, 12)
        assert describe_pending_uploads(confirmation_arguments) == ((-7,), 2, False)


@pytest.mark.parametrize(
    "rejection_case",
    [
        "no_pending_upload",
        "pending_upload_still_waiting",
        "pending_model_not_held",
        "pending_model_id_nonnegative",
    ],
)
@pytest.mark.parametrize("current_model_id", [-7, 4])
def test_own_id_confirmation_rejects_before_any_update(rejection_case, current_model_id):
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments = build_own_id_confirmation_arguments(
            ready_pending_model_ids=(), current_model_id=current_model_id
        )
        pending_upload_state = confirmation_arguments["pending_model_upload_state"]
        registry = confirmation_arguments["held_model_training_state_registry"]
        parameter_snapshot = snapshot_classifier_parameters(
            classifier=registry.snapshot_ordered_held_model_training_states()[0].classifier
        )
        if rejection_case == "pending_upload_still_waiting":
            pending_upload_state.queue_model_upload(
                model_id=-7, parameter_snapshot=parameter_snapshot, upload_delay_round_count=1
            )
        elif rejection_case == "pending_model_not_held":
            pending_upload_state.queue_ready_model_upload(
                model_id=-55, parameter_snapshot=parameter_snapshot
            )
        elif rejection_case == "pending_model_id_nonnegative":
            pending_upload_state.queue_ready_model_upload(
                model_id=4, parameter_snapshot=parameter_snapshot
            )
        confirmation_arguments["registered_global_model_id"] = 12
        owners_before = describe_owners_by_model_id(confirmation_arguments)
        pending_before = describe_pending_uploads(confirmation_arguments)
        # 保有していないIDは、保有モデルの一覧の検査が、KeyErrorで拒否する（現行を付け替える方針と同じ）。
        with pytest.raises((KeyError, ValueError)):
            confirm_held_model_registration(**confirmation_arguments)
        assert_owner_descriptions_equal(
            describe_owners_by_model_id(confirmation_arguments), owners_before
        )
        assert describe_pending_uploads(confirmation_arguments) == pending_before


# ---- サーバの登録: 実体のclientで、1つのclientの複数の送信保留を、採用の順に登録する ----


def add_ready_temporary_model(run_client, temporary_model_id):
    """保有モデル0の写しを、一時IDの保有モデルとして足し、待ちの済んだ送信保留にする（公開の操作だけを使う）。"""
    owners = run_client.owners
    source_state = owners.held_model_training_state_registry.get_held_model_training_state(
        model_id=0
    )
    classifier, optimizer_state = deepcopy(
        (source_state.classifier, source_state.concept_specific_parameter_optimizer_state)
    )
    owners.held_model_training_state_registry.register_held_model_training_state(
        model_id=temporary_model_id,
        classifier=classifier,
        concept_specific_parameter_optimizer_state=optimizer_state,
    )
    owners.loss_statistics_store.set_model_loss_statistics(
        model_id=temporary_model_id,
        loss_statistics=owners.loss_statistics_store.get_model_loss_statistics(model_id=0),
    )
    owners.pending_model_upload_state.queue_ready_model_upload(
        model_id=temporary_model_id,
        parameter_snapshot=snapshot_classifier_parameters(classifier=classifier),
    )


def test_server_registers_every_ready_pending_model_in_client_then_adoption_order(
    monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        real_make_run_client_settings = server_test_module.make_run_client_settings
        monkeypatch.setattr(
            server_test_module,
            "make_run_client_settings",
            lambda *arguments, **keyword_arguments: replace(
                real_make_run_client_settings(*arguments, **keyword_arguments),
                pending_model_registration_settings=PendingModelRegistrationSettings(
                    pending_model_registration_policy=(
                        REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME
                    )
                ),
            ),
        )
        server_round_oracle = server_test_module.build_server_round_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=2,
        )
        run_clients = server_round_oracle["run_clients"]
        global_model_repository = server_round_oracle["global_model_repository"]
        assert len(run_clients) >= 3
        # client 1は、送信保留が2つ（採用の順は-100、-101）で、現行はモデル0のまま。client 2は、1つで、それが現行。
        add_ready_temporary_model(run_clients[1], -100)
        add_ready_temporary_model(run_clients[1], -101)
        add_ready_temporary_model(run_clients[2], -100)
        run_clients[2].owners.current_training_model_assignment.assign_model_for_training(
            model_id=-100
        )
        first_registered_model_id = global_model_repository.next_global_model_id
        registered_client_models = register_ready_client_models(
            run_clients=run_clients, global_model_repository=global_model_repository, round_index=3
        )
    assert [
        (registered.client_id, registered.registered_global_model_id)
        for registered in registered_client_models
    ] == [
        (run_clients[1].client_id, first_registered_model_id),
        (run_clients[1].client_id, first_registered_model_id + 1),
        (run_clients[2].client_id, first_registered_model_id + 2),
    ]
    assert [registered.training_assignment_change for registered in registered_client_models][
        :2
    ] == [None, None]
    last_assignment_change = registered_client_models[2].training_assignment_change
    assert (last_assignment_change.previous_model_id, last_assignment_change.current_model_id) == (
        -100,
        first_registered_model_id + 2,
    )
    assert [
        (record.model_id, record.registered_round_index, record.registering_client_id)
        for record in global_model_repository.snapshot_model_registration_records()[-3:]
    ] == [
        (first_registered_model_id, 3, run_clients[1].client_id),
        (first_registered_model_id + 1, 3, run_clients[1].client_id),
        (first_registered_model_id + 2, 3, run_clients[2].client_id),
    ]
    held_model_ids_by_client = [
        tuple(
            held_state.model_id
            for held_state in run_client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        )
        for run_client in run_clients
    ]
    # -100が先の正式IDへ、-101が次の正式IDへ付け替わる（採用の順）。
    assert held_model_ids_by_client[1] == (
        0,
        first_registered_model_id,
        first_registered_model_id + 1,
    )
    assert held_model_ids_by_client[2] == (0, first_registered_model_id + 2)
    assert run_clients[1].owners.current_training_model_assignment.current_training_model_id == 0
    for run_client in run_clients:
        assert not run_client.has_model_ready_for_server_registration()
        assert run_client.owners.pending_model_upload_state.get_all_pending_model_uploads() == ()


# ---- 全体run: 採用したモデルが、全部登録される ----

SINE2_EVENT_CASE_NAME = "fedsda-sine2-events"


def execute_sine2_event_run_with_observation(pending_model_registration_policy):
    """基準版のケース`fedsda-sine2-events`の設定で、登録の扱いだけを替えて実行し、登録と採用の様子を観測する。"""
    run_settings = golden.build_case_settings(SINE2_EVENT_CASE_NAME, 1)
    run_participant_settings = run_settings.run_participant_settings
    run_settings = replace(
        run_settings,
        run_participant_settings=replace(
            run_participant_settings,
            run_client_settings=replace(
                run_participant_settings.run_client_settings,
                pending_model_registration_settings=PendingModelRegistrationSettings(
                    pending_model_registration_policy=pending_model_registration_policy
                ),
            ),
        ),
    )
    observation = dict(
        confirmation_count=0,
        confirmation_of_non_current_pending_model_count=0,
        adoption_while_another_model_is_pending_count=0,
        ready_pending_model_count_after_registration=[],
        registered_model_missing_from_client_count=0,
    )
    real_confirmation = registration_module.confirm_held_model_registration
    real_registration = synchronization_module.register_ready_client_models
    real_queue_model_upload = PendingModelUploadState.queue_model_upload

    def observe_confirmation(**confirmation_arguments):
        pending_model_upload = confirmation_arguments[
            "pending_model_upload_state"
        ].get_pending_model_upload()
        observation["confirmation_count"] += 1
        observation["confirmation_of_non_current_pending_model_count"] += (
            pending_model_upload is not None
            and pending_model_upload.model_id
            != confirmation_arguments["current_training_model_assignment"].current_training_model_id
        )
        return real_confirmation(**confirmation_arguments)

    def observe_registration(**registration_arguments):
        registered_client_models = real_registration(**registration_arguments)
        run_clients_by_id = {
            run_client.client_id: run_client for run_client in registration_arguments["run_clients"]
        }
        # 登録の処理が戻った時点（待ちを進める前）。
        observation["ready_pending_model_count_after_registration"].append(
            sum(
                run_client.count_models_ready_for_server_registration()
                for run_client in registration_arguments["run_clients"]
            )
        )
        for registered_client_model in registered_client_models:
            held_model_ids = {
                held_state.model_id
                for held_state in run_clients_by_id[
                    registered_client_model.client_id
                ].owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
            }
            observation["registered_model_missing_from_client_count"] += (
                registered_client_model.registered_global_model_id not in held_model_ids
            )
        return registered_client_models

    def observe_queue_model_upload(pending_model_upload_state, **queue_arguments):
        observation["adoption_while_another_model_is_pending_count"] += (
            pending_model_upload_state.get_pending_model_upload() is not None
        )
        return real_queue_model_upload(pending_model_upload_state, **queue_arguments)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            registration_module, "confirm_held_model_registration", observe_confirmation
        )
        monkeypatch.setattr(
            synchronization_module, "register_ready_client_models", observe_registration
        )
        monkeypatch.setattr(
            PendingModelUploadState, "queue_model_upload", observe_queue_model_upload
        )
        with torch.random.fork_rng(devices=[]):
            experiment_run = execute_fedsda_experiment_run(run_settings=run_settings)
    return experiment_run, observation


@pytest.fixture(scope="module")
def sine2_event_runs_by_policy():
    return {
        policy_name: execute_sine2_event_run_with_observation(policy_name)
        # このspecの2つの方針（現行モデルとして待った後に登録する方針は、別のspecのtestが扱う）。
        for policy_name in (
            REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
            KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
        )
    }


@pytest.mark.slow
def test_whole_run_registers_every_adopted_model_by_its_own_id(sine2_event_runs_by_policy):
    """既定の扱い: 採番した正式IDの全部が、登録したclientに保有され、登録されないままの採用が残らない。"""
    experiment_run, observation = sine2_event_runs_by_policy[
        REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME
    ]
    assert golden.count_unregistered_adopted_models(experiment_run) == 0
    assert observation["confirmation_count"] > 0
    assert observation["registered_model_missing_from_client_count"] == 0
    # 同期のたびに、登録の処理が戻った時点で、待ちの済んだ送信保留は残らない。
    assert observation["ready_pending_model_count_after_registration"]
    assert set(observation["ready_pending_model_count_after_registration"]) == {0}
    # この条件は、旧と同じ扱いでは登録されない経路（現行でないモデルの登録、または、送信保留がある間の次の採用）を通る。
    assert (
        observation["confirmation_of_non_current_pending_model_count"]
        + observation["adoption_while_another_model_is_pending_count"]
        > 0
    )


def test_whole_run_with_keep_latest_policy_leaves_adopted_models_unregistered(
    sine2_event_runs_by_policy,
):
    """旧と同じ扱い: 同じ条件で、登録されないままの採用が残る（この条件が、LEGACY-016の経路を通ることの確認）。"""
    experiment_run, observation = sine2_event_runs_by_policy[
        KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
    ]
    # 候補検証を消したので、この条件のrunは検証なしで流れる（前は候補検証ありで2件。基準版15のspec）。
    assert golden.count_unregistered_adopted_models(experiment_run) == 4
    # 採番はされるが、モデルは登録されない（欠番）か、送信保留が上書きされる。
    assert (
        observation["registered_model_missing_from_client_count"]
        + observation["adoption_while_another_model_is_pending_count"]
        > 0
    )


def test_whole_runs_of_both_policies_produce_different_metrics(
    sine2_event_runs_by_policy,
):
    """2つの扱いの結果は、この条件では違う（修正が、実際に結果を変える）。"""
    each_run, _ = sine2_event_runs_by_policy[REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME]
    keep_run, _ = sine2_event_runs_by_policy[
        KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
    ]
    assert each_run.run_metrics != keep_run.run_metrics


# ---- FedDrift ----（構成の既定のtestは、既定が変わったので、`test_pending_model_registration_after_waiting_as_current_model.py`へ移した）


def test_feddrift_confirms_only_the_current_pending_model():
    """FedDrift: 正式IDの確認のたびに、送信保留のモデルは現行である（1つだけ保持する扱いで、登録されないモデルを作らない）。"""
    confirmations = []
    real_confirmation = feddrift_server_module.confirm_held_model_registration

    def observe_confirmation(**confirmation_arguments):
        pending_upload_state = confirmation_arguments["pending_model_upload_state"]
        pending_model_upload = pending_upload_state.get_pending_model_upload()
        confirmations.append(
            (
                pending_upload_state.pending_model_registration_policy,
                None if pending_model_upload is None else pending_model_upload.model_id,
                confirmation_arguments[
                    "current_training_model_assignment"
                ].current_training_model_id,
                len(pending_upload_state.get_all_pending_model_uploads()),
            )
        )
        return real_confirmation(**confirmation_arguments)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            feddrift_server_module, "confirm_held_model_registration", observe_confirmation
        )
        with torch.random.fork_rng(devices=[]):
            golden.capture_case("feddrift-drift-tail", golden.BASELINE_VERSION)
    confirmations_with_pending_model = [
        confirmation for confirmation in confirmations if confirmation[1] is not None
    ]
    assert confirmations_with_pending_model
    for policy_name, pending_model_id, current_model_id, pending_count in confirmations:
        assert policy_name == KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
        assert pending_count <= 1
        if pending_model_id is not None:
            assert pending_model_id == current_model_id
