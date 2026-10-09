"""サーバが持つ状態のowner: グローバルモデル（パラメータ、損失統計、採番、来歴）と、通信量の記録。"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
    CommunicationVolumeSnapshot,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRegistrationRecord,
    GlobalModelRepository,
)


def make_loss_statistics(*, mean_loss=0.25, observed_loss_count=4):
    return ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=observed_loss_count,
            mean_loss=mean_loss,
            sum_squared_loss_deviations=0.5,
        )
    )


def make_parameter_snapshot(*, offset=0.0):
    return {
        "feature_extractor.hidden_layers.0.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]) + offset,
        "classification_layer.bias": torch.tensor([0.5]) + offset,
    }


def make_repository():
    return GlobalModelRepository(
        initial_model_id=0,
        initial_parameter_snapshot=make_parameter_snapshot(),
        initial_loss_statistics=make_loss_statistics(),
    )


def assert_parameter_snapshots_equal(parameter_snapshot, expected_parameter_snapshot):
    assert list(parameter_snapshot) == list(expected_parameter_snapshot)
    for parameter_name, parameter_values in expected_parameter_snapshot.items():
        assert torch.equal(parameter_snapshot[parameter_name], parameter_values)


def snapshot_repository_state(global_model_repository):
    return (
        global_model_repository.next_global_model_id,
        global_model_repository.global_model_ids,
        tuple(
            tuple(
                (parameter_name, parameter_values.tolist())
                for parameter_name, parameter_values in global_model_repository.get_global_model_parameters(
                    model_id=model_id
                ).items()
            )
            for model_id in global_model_repository.global_model_ids
        ),
        tuple(
            global_model_repository.get_global_model_loss_statistics(model_id=model_id)
            for model_id in global_model_repository.global_model_ids
        ),
        global_model_repository.snapshot_model_registration_records(),
    )


def test_repository_starts_with_initial_model():
    initial_parameter_snapshot = make_parameter_snapshot()
    initial_loss_statistics = make_loss_statistics()
    global_model_repository = GlobalModelRepository(
        initial_model_id=0,
        initial_parameter_snapshot=initial_parameter_snapshot,
        initial_loss_statistics=initial_loss_statistics,
    )
    assert global_model_repository.global_model_ids == (0,)
    assert global_model_repository.next_global_model_id == 1
    assert_parameter_snapshots_equal(
        global_model_repository.get_global_model_parameters(model_id=0), make_parameter_snapshot()
    )
    assert (
        global_model_repository.get_global_model_loss_statistics(model_id=0)
        == initial_loss_statistics
    )
    assert global_model_repository.snapshot_model_registration_records() == (
        GlobalModelRegistrationRecord(
            model_id=0, registered_round_index=None, registering_client_id=None
        ),
    )
    # 渡したパラメータを後から変えても、内部は変わらない。
    with torch.no_grad():
        initial_parameter_snapshot["classification_layer.bias"].fill_(9.0)
    initial_parameter_snapshot["extra"] = torch.tensor([1.0])
    assert_parameter_snapshots_equal(
        global_model_repository.get_global_model_parameters(model_id=0), make_parameter_snapshot()
    )
    # 初期モデルのIDが0でなければ、次の正式IDは、その次。
    assert (
        GlobalModelRepository(
            initial_model_id=4,
            initial_parameter_snapshot=make_parameter_snapshot(),
            initial_loss_statistics=initial_loss_statistics,
        ).next_global_model_id
        == 5
    )


def test_repository_allocates_consecutive_global_model_ids():
    global_model_repository = make_repository()
    assert [global_model_repository.allocate_global_model_id() for _ in range(3)] == [1, 2, 3]
    assert global_model_repository.next_global_model_id == 4
    # 採番だけでは、パラメータも来歴も増えない。
    assert global_model_repository.global_model_ids == (0,)
    assert len(global_model_repository.snapshot_model_registration_records()) == 1


def test_repository_keeps_parameters_and_statistics_in_first_set_order_as_copies():
    global_model_repository = make_repository()
    global_model_repository.set_global_model_parameters(
        model_id=3, parameter_snapshot=make_parameter_snapshot(offset=1.0)
    )
    global_model_repository.set_global_model_parameters(
        model_id=1, parameter_snapshot=make_parameter_snapshot(offset=2.0)
    )
    # 既にあるIDの置換えは、最初に置いた位置のまま。
    global_model_repository.set_global_model_parameters(
        model_id=0, parameter_snapshot=make_parameter_snapshot(offset=5.0)
    )
    assert global_model_repository.global_model_ids == (0, 3, 1)
    assert_parameter_snapshots_equal(
        global_model_repository.get_global_model_parameters(model_id=0),
        make_parameter_snapshot(offset=5.0),
    )
    # 返った辞書とtensorを変えても、内部は変わらない。
    returned_parameter_snapshot = global_model_repository.get_global_model_parameters(model_id=3)
    with torch.no_grad():
        returned_parameter_snapshot["classification_layer.bias"].fill_(-7.0)
    returned_parameter_snapshot.clear()
    assert_parameter_snapshots_equal(
        global_model_repository.get_global_model_parameters(model_id=3),
        make_parameter_snapshot(offset=1.0),
    )
    # 統計は、持っていなければNone。置けば、その値を返す。
    assert global_model_repository.get_global_model_loss_statistics(model_id=3) is None
    replaced_loss_statistics = make_loss_statistics(mean_loss=0.75, observed_loss_count=9)
    global_model_repository.set_global_model_loss_statistics(
        model_id=3, loss_statistics=replaced_loss_statistics
    )
    assert (
        global_model_repository.get_global_model_loss_statistics(model_id=3)
        == replaced_loss_statistics
    )
    with pytest.raises(KeyError):
        global_model_repository.get_global_model_parameters(model_id=7)


def test_repository_returns_registration_records_in_model_id_order_and_replaces_same_id():
    global_model_repository = make_repository()
    global_model_repository.record_model_registration(
        model_id=2, registered_round_index=5, registering_client_id=1
    )
    global_model_repository.record_model_registration(
        model_id=1, registered_round_index=7, registering_client_id=0
    )
    global_model_repository.record_model_registration(
        model_id=2, registered_round_index=8, registering_client_id=3
    )
    registration_records = global_model_repository.snapshot_model_registration_records()
    assert registration_records == (
        GlobalModelRegistrationRecord(
            model_id=0, registered_round_index=None, registering_client_id=None
        ),
        GlobalModelRegistrationRecord(
            model_id=1, registered_round_index=7, registering_client_id=0
        ),
        GlobalModelRegistrationRecord(
            model_id=2, registered_round_index=8, registering_client_id=3
        ),
    )
    with pytest.raises(FrozenInstanceError):
        registration_records[0].model_id = 9


@pytest.mark.parametrize(
    "invalid_operation,expected_exception",
    [
        (
            lambda repository: repository.set_global_model_parameters(
                model_id=True, parameter_snapshot=make_parameter_snapshot()
            ),
            TypeError,
        ),
        (
            lambda repository: repository.set_global_model_parameters(
                model_id=-1, parameter_snapshot=make_parameter_snapshot()
            ),
            ValueError,
        ),
        (
            lambda repository: repository.set_global_model_parameters(
                model_id=1, parameter_snapshot={}
            ),
            ValueError,
        ),
        (
            lambda repository: repository.set_global_model_parameters(
                model_id=1, parameter_snapshot=[("a", torch.tensor([1.0]))]
            ),
            TypeError,
        ),
        (
            lambda repository: repository.set_global_model_parameters(
                model_id=1, parameter_snapshot={1: torch.tensor([1.0])}
            ),
            TypeError,
        ),
        (
            lambda repository: repository.set_global_model_parameters(
                model_id=1, parameter_snapshot={"a": [1.0]}
            ),
            TypeError,
        ),
        (
            lambda repository: repository.set_global_model_parameters(
                model_id=1, parameter_snapshot={"a": torch.nn.Parameter(torch.tensor([1.0]))}
            ),
            TypeError,
        ),
        (
            lambda repository: repository.set_global_model_loss_statistics(
                model_id=1, loss_statistics={"n": 1}
            ),
            TypeError,
        ),
        (
            lambda repository: repository.set_global_model_loss_statistics(
                model_id=1.0, loss_statistics=make_loss_statistics()
            ),
            TypeError,
        ),
        (
            lambda repository: repository.record_model_registration(
                model_id=-2, registered_round_index=0, registering_client_id=0
            ),
            ValueError,
        ),
        (
            lambda repository: repository.record_model_registration(
                model_id=1, registered_round_index=-1, registering_client_id=0
            ),
            ValueError,
        ),
        (
            lambda repository: repository.record_model_registration(
                model_id=1, registered_round_index=0, registering_client_id=None
            ),
            TypeError,
        ),
        (
            lambda repository: repository.record_model_registration(
                model_id=1, registered_round_index=np.int64(0), registering_client_id=0
            ),
            TypeError,
        ),
        (lambda repository: repository.get_global_model_parameters(model_id="0"), TypeError),
        (
            lambda repository: repository.get_global_model_loss_statistics(model_id=-1),
            ValueError,
        ),
    ],
)
def test_repository_rejects_invalid_input_without_change(invalid_operation, expected_exception):
    global_model_repository = make_repository()
    global_model_repository.allocate_global_model_id()
    repository_state = snapshot_repository_state(global_model_repository)
    with pytest.raises(expected_exception):
        invalid_operation(global_model_repository)
    assert snapshot_repository_state(global_model_repository) == repository_state


@pytest.mark.parametrize(
    "invalid_arguments,expected_exception",
    [
        (dict(initial_model_id=-1), ValueError),
        (dict(initial_model_id=False), TypeError),
        (dict(initial_parameter_snapshot={}), ValueError),
        (dict(initial_parameter_snapshot=None), TypeError),
        (dict(initial_loss_statistics=None), TypeError),
    ],
)
def test_repository_rejects_invalid_initial_model(invalid_arguments, expected_exception):
    with pytest.raises(expected_exception):
        GlobalModelRepository(
            **dict(
                initial_model_id=0,
                initial_parameter_snapshot=make_parameter_snapshot(),
                initial_loss_statistics=make_loss_statistics(),
            )
            | invalid_arguments
        )


def test_communication_volume_store_counts_each_direction_separately():
    communication_volume_record_store = CommunicationVolumeRecordStore()
    assert communication_volume_record_store.get_state_snapshot() == CommunicationVolumeSnapshot(
        uploaded_model_count=0,
        downloaded_model_count=0,
        uploaded_message_count=0,
        downloaded_message_count=0,
        uploaded_parameter_value_count=0,
        downloaded_parameter_value_count=0,
        uploaded_byte_count=0,
        downloaded_byte_count=0,
    )
    communication_volume_record_store.record_model_transfers(
        transfer_direction="upload", model_count=3
    )
    communication_volume_record_store.record_model_transfers(
        transfer_direction="download", model_count=5
    )
    communication_volume_record_store.record_messages(transfer_direction="upload", message_count=2)
    communication_volume_record_store.record_messages(
        transfer_direction="download", message_count=7
    )
    communication_volume_record_store.record_messages(transfer_direction="upload", message_count=0)
    # float32が6値（24 byte）と、float64が2値（16 byte）。
    parameter_snapshot = {
        "weight": torch.zeros((2, 3), dtype=torch.float32),
        "scale": torch.zeros(2, dtype=torch.float64),
    }
    communication_volume_record_store.record_parameter_transfer(
        transfer_direction="upload", parameter_snapshot=parameter_snapshot
    )
    communication_volume_record_store.record_parameter_transfer(
        transfer_direction="download", parameter_snapshot=parameter_snapshot, transfer_count=3
    )
    communication_volume_record_store.record_parameter_transfer(
        transfer_direction="download", parameter_snapshot=parameter_snapshot, transfer_count=0
    )
    state_snapshot = communication_volume_record_store.get_state_snapshot()
    assert state_snapshot == CommunicationVolumeSnapshot(
        uploaded_model_count=3,
        downloaded_model_count=5,
        uploaded_message_count=2,
        downloaded_message_count=7,
        uploaded_parameter_value_count=8,
        downloaded_parameter_value_count=24,
        uploaded_byte_count=40,
        downloaded_byte_count=120,
    )
    with pytest.raises(FrozenInstanceError):
        state_snapshot.uploaded_model_count = 0
    # パラメータの転送は、モデル転送数とメッセージ数を足さない。空のパラメータは0を足す。
    communication_volume_record_store.record_parameter_transfer(
        transfer_direction="upload", parameter_snapshot={}
    )
    assert communication_volume_record_store.get_state_snapshot() == state_snapshot


@pytest.mark.parametrize(
    "invalid_operation,expected_exception",
    [
        (
            lambda store: store.record_model_transfers(transfer_direction="up", model_count=1),
            ValueError,
        ),
        (
            lambda store: store.record_model_transfers(transfer_direction=None, model_count=1),
            TypeError,
        ),
        (
            lambda store: store.record_model_transfers(transfer_direction="upload", model_count=-1),
            ValueError,
        ),
        (
            lambda store: store.record_model_transfers(
                transfer_direction="upload", model_count=True
            ),
            TypeError,
        ),
        (
            lambda store: store.record_messages(transfer_direction="download", message_count=1.0),
            TypeError,
        ),
        (
            lambda store: store.record_messages(transfer_direction="sideways", message_count=1),
            ValueError,
        ),
        (
            lambda store: store.record_parameter_transfer(
                transfer_direction="upload", parameter_snapshot=[torch.zeros(2)]
            ),
            TypeError,
        ),
        (
            lambda store: store.record_parameter_transfer(
                transfer_direction="upload", parameter_snapshot={"a": np.zeros(2)}
            ),
            TypeError,
        ),
        (
            lambda store: store.record_parameter_transfer(
                transfer_direction="upload", parameter_snapshot={3: torch.zeros(2)}
            ),
            TypeError,
        ),
        (
            lambda store: store.record_parameter_transfer(
                transfer_direction="upload",
                parameter_snapshot={"a": torch.zeros(2)},
                transfer_count=-1,
            ),
            ValueError,
        ),
        (
            lambda store: store.record_parameter_transfer(
                transfer_direction="downward", parameter_snapshot={"a": torch.zeros(2)}
            ),
            ValueError,
        ),
    ],
)
def test_communication_volume_store_rejects_invalid_input_without_change(
    invalid_operation, expected_exception
):
    communication_volume_record_store = CommunicationVolumeRecordStore()
    communication_volume_record_store.record_messages(transfer_direction="upload", message_count=4)
    state_snapshot = communication_volume_record_store.get_state_snapshot()
    with pytest.raises(expected_exception):
        invalid_operation(communication_volume_record_store)
    assert communication_volume_record_store.get_state_snapshot() == state_snapshot
