"""送信保留のモデルを、現行モデルとして待った後に登録する方針（LEGACY-016の扱いの見直し）。

実旧には、この方針の正解が無い。送信保留が1つだけで現行の場合は、実旧と照合済みの方針と照合済みの
「それ自身のID」の方針の結果と比べ、それ以外は、状態の遷移を直接確かめる。
"""

from dataclasses import replace

import pytest
import test_server_model_registration_and_aggregation as server_test_module
import torch
from test_pending_model_registration_by_own_id import (
    add_ready_temporary_model,
    assert_owner_descriptions_equal,
    build_own_id_confirmation_arguments,
)
from test_pending_model_registration_by_own_id import (
    describe_owners_by_model_id as describe_owners_including_pending_upload_reads,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (
    REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
    REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME,
    PendingModelRegistrationSettings,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.runtime.held_model_registration_confirmation import (
    confirm_held_model_registration,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    register_ready_client_models,
)


def describe_pending_upload_waits(pending_upload_state):
    """（送信保留のID、残りの待ち）の列。この方針では、残りの待ちを読む公開の操作が無いので、内部を読む。"""
    return tuple(
        (pending_model_upload.model_id, remaining_upload_delay_round_count)
        for pending_model_upload, remaining_upload_delay_round_count in (
            pending_upload_state._pending_model_uploads
        )
    )


def describe_owners_by_model_id(confirmation_arguments):
    """送信保留のほかのownerを、モデルIDごとに表す。

    前のspecのhelperは、送信保留の、現行を知らない操作を読む（その結果は使わない）。この方針では、その操作が
    拒否されるので、送信保留を、空の別のownerに替えた写しの引数で読む（送信保留は、別に比べる）。
    """
    return describe_owners_including_pending_upload_reads(
        {
            **confirmation_arguments,
            "pending_model_upload_state": PendingModelUploadState(
                pending_model_registration_policy=REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME
            ),
        }
    )


def build_waiting_as_current_confirmation_arguments(
    *, initial_model_ids=(4, -7, 9), pending_model_waits=((-7, 0),), current_model_id=-7
):
    """確認の引数を、既存のoracleの組立てで作り、送信保留を、この方針のownerへ替える。"""
    confirmation_arguments = build_own_id_confirmation_arguments(
        initial_model_ids=initial_model_ids,
        ready_pending_model_ids=(),
        current_model_id=current_model_id,
    )
    pending_upload_state = PendingModelUploadState(
        pending_model_registration_policy=(
            REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME
        )
    )
    registry = confirmation_arguments["held_model_training_state_registry"]
    parameter_snapshot = snapshot_classifier_parameters(
        classifier=registry.snapshot_ordered_held_model_training_states()[0].classifier
    )
    for pending_model_id, remaining_wait in pending_model_waits:
        if remaining_wait == 0:
            pending_upload_state.queue_ready_model_upload(
                model_id=pending_model_id, parameter_snapshot=parameter_snapshot
            )
        else:
            pending_upload_state.queue_model_upload(
                model_id=pending_model_id,
                parameter_snapshot=parameter_snapshot,
                upload_delay_round_count=remaining_wait,
            )
    confirmation_arguments["pending_model_upload_state"] = pending_upload_state
    return confirmation_arguments


@pytest.mark.parametrize("initial_model_ids", [(-7,), (4, -7, 9), (-7, 4, 9), (4, 9, -7)])
def test_only_current_pending_model_is_reassigned_like_own_id_policy(initial_model_ids):
    """送信保留が1つだけで現行: 「それ自身のID」の方針と、全ownerの結果と、現行の変更が同じ。"""
    with torch.random.fork_rng(devices=[]):
        own_id_arguments = build_own_id_confirmation_arguments(initial_model_ids=initial_model_ids)
        own_id_arguments["registered_global_model_id"] = 12
        own_id_change = confirm_held_model_registration(**own_id_arguments)
        expected = describe_owners_by_model_id(own_id_arguments)
        confirmation_arguments = build_waiting_as_current_confirmation_arguments(
            initial_model_ids=initial_model_ids
        )
        confirmation_arguments["registered_global_model_id"] = 12
        assignment_change = confirm_held_model_registration(**confirmation_arguments)
    assert (
        (assignment_change.previous_model_id, assignment_change.current_model_id)
        == (
            own_id_change.previous_model_id,
            own_id_change.current_model_id,
        )
        == (-7, 12)
    )
    assert_owner_descriptions_equal(describe_owners_by_model_id(confirmation_arguments), expected)
    assert describe_pending_upload_waits(confirmation_arguments["pending_model_upload_state"]) == ()


@pytest.mark.parametrize(
    ("initial_model_ids", "pending_model_waits", "current_model_id"),
    [
        # 待ちが0でも、現行でない送信保留は、対象にならない。
        ((4, -7, 9), ((-7, 0),), 4),
        # 現行の一時ID（-9）は、送信保留でないので、付け替えない。
        ((4, -7, -9), ((-7, 0),), -9),
        # 現行の送信保留でも、待ちが残っていれば、対象にならない。
        ((4, -7, 9), ((-7, 1),), -7),
        # 送信保留が無い。
        ((4, -7, 9), (), -7),
        # 現行の送信保留が、保有されていない。
        ((4, -7, 9), ((-55, 0),), -55),
    ],
)
def test_confirmation_without_current_ready_pending_model_rejects_before_any_update(
    initial_model_ids, pending_model_waits, current_model_id
):
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments = build_waiting_as_current_confirmation_arguments(
            initial_model_ids=initial_model_ids,
            pending_model_waits=pending_model_waits,
            current_model_id=current_model_id,
        )
        confirmation_arguments["registered_global_model_id"] = 12
        owners_before = describe_owners_by_model_id(confirmation_arguments)
        pending_upload_state = confirmation_arguments["pending_model_upload_state"]
        pending_before = describe_pending_upload_waits(pending_upload_state)
        # 保有していないIDは、保有モデルの一覧の検査が、KeyErrorで拒否する（ほかの方針と同じ）。
        with pytest.raises((KeyError, ValueError)):
            confirm_held_model_registration(**confirmation_arguments)
    assert_owner_descriptions_equal(
        describe_owners_by_model_id(confirmation_arguments), owners_before
    )
    assert describe_pending_upload_waits(pending_upload_state) == pending_before


def test_current_pending_model_is_registered_before_an_older_one_that_is_not_current():
    """待ち0で現行でないQ（-7）と、後に採用した現行のP（-9）。

    「それ自身のID」は、Qを先に登録し、Pを次の正式IDで登録する。この方針は、Qを保持し、Pを先の正式IDで登録する
    （同じclientで先に登録しなかった送信保留があると、正式IDがずれる。要求1.8）。
    """
    with torch.random.fork_rng(devices=[]):
        own_id_arguments = build_own_id_confirmation_arguments(
            initial_model_ids=(4, -7, -9), ready_pending_model_ids=(-7, -9), current_model_id=-9
        )
        own_id_arguments["registered_global_model_id"] = 12
        assert confirm_held_model_registration(**own_id_arguments) is None
        assert describe_owners_by_model_id(own_id_arguments)["held_model_ids"] == (4, -9, 12)
        confirmation_arguments = build_waiting_as_current_confirmation_arguments(
            initial_model_ids=(4, -7, -9),
            pending_model_waits=((-7, 0), (-9, 0)),
            current_model_id=-9,
        )
        confirmation_arguments["registered_global_model_id"] = 12
        assignment_change = confirm_held_model_registration(**confirmation_arguments)
    assert (assignment_change.previous_model_id, assignment_change.current_model_id) == (-9, 12)
    actual = describe_owners_by_model_id(confirmation_arguments)
    assert actual["held_model_ids"] == (4, -7, 12)
    assert actual["current_model_id"] == 12
    # Qは、待ち0のまま、送信保留として残る。
    assert describe_pending_upload_waits(confirmation_arguments["pending_model_upload_state"]) == (
        (-7, 0),
    )


# ---- 実体のclient: 件数、待ちの進行、外れて戻ったモデルの登録 ----


@pytest.fixture
def waiting_as_current_server_round(monkeypatch, valid_run_settings_mapping):
    """サーバの登録のtestの組立てを、この方針のclientの設定で使う。"""
    real_make_run_client_settings = server_test_module.make_run_client_settings
    monkeypatch.setattr(
        server_test_module,
        "make_run_client_settings",
        lambda *arguments, **keyword_arguments: replace(
            real_make_run_client_settings(*arguments, **keyword_arguments),
            pending_model_registration_settings=PendingModelRegistrationSettings(
                pending_model_registration_policy=(
                    REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME
                )
            ),
        ),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        yield server_test_module.build_server_round_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=2,
        )


def held_model_ids(run_client):
    return tuple(
        held_state.model_id
        for held_state in run_client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )


def test_client_advances_wait_and_counts_only_the_current_pending_model(
    waiting_as_current_server_round,
):
    run_client = waiting_as_current_server_round["run_clients"][1]
    owners = run_client.owners
    assert (
        owners.pending_model_upload_state.pending_model_registration_policy
        == REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME
    )
    # 待ちの残る送信保留を2つ（-100と-101）。-101が現行。
    add_ready_temporary_model(run_client, -100)
    add_ready_temporary_model(run_client, -101)
    pending_upload_state = owners.pending_model_upload_state
    parameter_snapshots = {
        upload.model_id: upload.parameter_snapshot
        for upload in pending_upload_state.get_all_pending_model_uploads()
    }
    pending_upload_state.clear_pending_model_upload()
    for model_id in (-100, -101):
        pending_upload_state.queue_model_upload(
            model_id=model_id,
            parameter_snapshot=parameter_snapshots[model_id],
            upload_delay_round_count=1,
        )
    owners.current_training_model_assignment.assign_model_for_training(model_id=-101)
    assert not run_client.has_model_ready_for_server_registration()
    assert run_client.count_models_ready_for_server_registration() == 0
    run_client.advance_new_model_upload_wait_after_synchronization(round_index=0)
    assert describe_pending_upload_waits(pending_upload_state) == ((-100, 1), (-101, 0))
    assert run_client.has_model_ready_for_server_registration()
    assert run_client.count_models_ready_for_server_registration() == 1
    # 現行が非負のモデルへ移ると、待ちの済んだ-101も、登録の対象にならず、待ちも進まない。
    owners.current_training_model_assignment.assign_model_for_training(model_id=0)
    assert not run_client.has_model_ready_for_server_registration()
    assert run_client.count_models_ready_for_server_registration() == 0
    run_client.advance_new_model_upload_wait_after_synchronization(round_index=1)
    assert describe_pending_upload_waits(pending_upload_state) == ((-100, 1), (-101, 0))


def test_server_registers_a_returned_pending_model_only_while_it_is_current(
    waiting_as_current_server_round,
):
    run_clients = waiting_as_current_server_round["run_clients"]
    global_model_repository = waiting_as_current_server_round["global_model_repository"]
    run_client = run_clients[1]
    owners = run_client.owners
    # 待ちの済んだ送信保留が2つ（-100と-101）。現行は非負のモデル0。
    add_ready_temporary_model(run_client, -100)
    add_ready_temporary_model(run_client, -101)
    first_registered_model_id = global_model_repository.next_global_model_id
    with torch.random.fork_rng(devices=[]):
        assert (
            register_ready_client_models(
                run_clients=run_clients,
                global_model_repository=global_model_repository,
                round_index=3,
            )
            == ()
        )
    # 正式IDは採番されず、どちらも送信保留のまま残る。
    assert global_model_repository.next_global_model_id == first_registered_model_id
    assert held_model_ids(run_client) == (0, -100, -101)
    assert describe_pending_upload_waits(owners.pending_model_upload_state) == (
        (-100, 0),
        (-101, 0),
    )
    # -101が現行へ戻ると、その同期で、-101だけを登録する（1つのclientから、高々1件）。
    owners.current_training_model_assignment.assign_model_for_training(model_id=-101)
    with torch.random.fork_rng(devices=[]):
        registered_client_models = register_ready_client_models(
            run_clients=run_clients, global_model_repository=global_model_repository, round_index=4
        )
    assert [
        (registered.client_id, registered.registered_global_model_id)
        for registered in registered_client_models
    ] == [(run_client.client_id, first_registered_model_id)]
    assignment_change = registered_client_models[0].training_assignment_change
    assert (assignment_change.previous_model_id, assignment_change.current_model_id) == (
        -101,
        first_registered_model_id,
    )
    assert held_model_ids(run_client) == (0, -100, first_registered_model_id)
    assert describe_pending_upload_waits(owners.pending_model_upload_state) == ((-100, 0),)
    assert [
        (record.model_id, record.registered_round_index, record.registering_client_id)
        for record in global_model_repository.snapshot_model_registration_records()[-1:]
    ] == [(first_registered_model_id, 4, run_client.client_id)]
