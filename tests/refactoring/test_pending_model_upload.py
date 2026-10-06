"""一保留枠・ラウンド待機と実旧FedSDAの送信準備を比較する。"""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest
import torch
from test_joint_model_parameter_update import assert_nested_state_equal

from federated_drift_experiment.clients.fedsda import ClassConditionalESRFedSDAClient
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUpload,
    PendingModelUploadState,
)


def build_legacy_pending_upload_client():
    # 実旧メソッドだけを呼ぶoracle。detector/routerなど不要なconstructor処理を省く。
    legacy_client = ClassConditionalESRFedSDAClient.__new__(ClassConditionalESRFedSDAClient)
    legacy_client.pending_model_params = None
    legacy_client.pending_model_stats = None
    legacy_client.pending_model_ready = True
    legacy_client._pending_upload_rounds = 0
    legacy_client.verbose = False
    legacy_client.current_model_id = 0
    return legacy_client


def assert_pending_upload_state_matches_legacy(pending_upload_state, legacy_client):
    pending_model_upload = pending_upload_state.get_pending_model_upload()
    assert pending_upload_state.has_ready_model_upload() is legacy_client.has_pending_model()
    if legacy_client.pending_model_params is None:
        assert pending_model_upload is None
        # 旧空状態の私有counter残値は無効。次queueで上書きされる。
        assert pending_upload_state.remaining_upload_delay_round_count == 0
    else:
        assert pending_model_upload is not None
        assert pending_model_upload.model_id == legacy_client.current_model_id
        assert pending_model_upload.parameter_snapshot is legacy_client.get_pending_model_info()[0]
        assert (
            pending_upload_state.remaining_upload_delay_round_count
            == legacy_client._pending_upload_rounds
        )


@pytest.mark.parametrize("upload_delay_round_count", [1, 2, 4])
@pytest.mark.parametrize("model_id", [-7, 0, 5])
def test_pending_upload_transition_sequence_matches_legacy(upload_delay_round_count, model_id):
    pending_upload_state = PendingModelUploadState()
    legacy_client = build_legacy_pending_upload_client()
    assert_pending_upload_state_matches_legacy(pending_upload_state, legacy_client)
    pending_upload_state.advance_upload_readiness_at_round_boundary()
    legacy_client.promote_pending_to_ready()
    assert_pending_upload_state_matches_legacy(pending_upload_state, legacy_client)
    initial_parameter_snapshot = {"weight": torch.tensor([0.5, -0.25])}
    for _ in range(2):
        pending_upload_state.queue_model_upload(
            model_id=model_id,
            parameter_snapshot=initial_parameter_snapshot,
            upload_delay_round_count=upload_delay_round_count,
        )
        legacy_client.current_model_id = model_id
        legacy_client.pending_model_params = initial_parameter_snapshot
        legacy_client.pending_model_ready = False
        legacy_client._pending_upload_rounds = upload_delay_round_count
        assert_pending_upload_state_matches_legacy(pending_upload_state, legacy_client)
        for _ in range(upload_delay_round_count + 2):
            pending_upload_state.advance_upload_readiness_at_round_boundary()
            legacy_client.promote_pending_to_ready()
            assert_pending_upload_state_matches_legacy(pending_upload_state, legacy_client)
        # waiting中の置換と途中解除も確認する。
        pending_upload_state.queue_model_upload(
            model_id=model_id,
            parameter_snapshot=initial_parameter_snapshot,
            upload_delay_round_count=2,
        )
        legacy_client.pending_model_ready = False
        legacy_client._pending_upload_rounds = 2
        pending_upload_state.advance_upload_readiness_at_round_boundary()
        legacy_client.promote_pending_to_ready()
        assert_pending_upload_state_matches_legacy(pending_upload_state, legacy_client)
        pending_upload_state.clear_pending_model_upload()
        # ID付替えは範囲外。旧confirmの保留消去分岐だけを使用する。
        legacy_client.current_model_id = 0
        legacy_client.confirm_model_registration(99)
        assert_pending_upload_state_matches_legacy(pending_upload_state, legacy_client)
        pending_upload_state.clear_pending_model_upload()
        pending_upload_state.advance_upload_readiness_at_round_boundary()
        legacy_client.promote_pending_to_ready()
        assert_pending_upload_state_matches_legacy(pending_upload_state, legacy_client)


def test_pending_upload_retains_borrowed_snapshot_and_nonconsuming_record():
    initial_parameter_snapshot = {"second": torch.tensor([0.5]), "first": torch.tensor([-0.25])}
    previous_parameter_snapshot = deepcopy(initial_parameter_snapshot)
    pending_upload_state = PendingModelUploadState()
    pending_upload_state.queue_model_upload(
        model_id=-7, parameter_snapshot=initial_parameter_snapshot, upload_delay_round_count=1
    )
    previous_pending_model_upload = pending_upload_state.get_pending_model_upload()
    assert previous_pending_model_upload.parameter_snapshot is initial_parameter_snapshot
    for parameter_name, parameter_values in initial_parameter_snapshot.items():
        assert previous_pending_model_upload.parameter_snapshot[parameter_name] is parameter_values
    pending_upload_state.advance_upload_readiness_at_round_boundary()
    assert pending_upload_state.get_pending_model_upload() is previous_pending_model_upload
    assert_nested_state_equal(initial_parameter_snapshot, previous_parameter_snapshot)
    with pytest.raises(FrozenInstanceError):
        previous_pending_model_upload.model_id = 12
    with pytest.raises(AttributeError):
        pending_upload_state.remaining_upload_delay_round_count = 10
    replacement_parameter_snapshot = {"weight": torch.tensor([1.0])}
    pending_upload_state.queue_model_upload(
        model_id=4, parameter_snapshot=replacement_parameter_snapshot, upload_delay_round_count=2
    )
    pending_model_upload = pending_upload_state.get_pending_model_upload()
    assert pending_model_upload is not previous_pending_model_upload
    assert pending_model_upload.model_id == 4
    assert pending_model_upload.parameter_snapshot is replacement_parameter_snapshot
    assert previous_pending_model_upload.model_id == -7
    assert_nested_state_equal(
        previous_pending_model_upload.parameter_snapshot, previous_parameter_snapshot
    )


@pytest.mark.parametrize("state_variant", ["empty", "waiting", "ready"])
@pytest.mark.parametrize(
    "invalid_input_kind",
    [
        "id_bool",
        "id_float",
        "delay_bool",
        "delay_float",
        "delay_zero",
        "delay_negative",
        "snapshot_none",
        "snapshot_empty",
        "key_empty",
        "key_bool",
        "value_none",
        "value_float64",
        "value_meta",
        "value_sparse",
        "value_nan",
        "value_inf",
        "value_grad",
    ],
)
def test_pending_upload_rejects_invalid_inputs_without_mutation(state_variant, invalid_input_kind):
    pending_upload_state = PendingModelUploadState()
    initial_parameter_snapshot = {"weight": torch.tensor([0.25])}
    if state_variant != "empty":
        pending_upload_state.queue_model_upload(
            model_id=-7, parameter_snapshot=initial_parameter_snapshot, upload_delay_round_count=1
        )
    if state_variant == "ready":
        pending_upload_state.advance_upload_readiness_at_round_boundary()
    previous_pending_model_upload = pending_upload_state.get_pending_model_upload()
    previous_remaining_round_count = pending_upload_state.remaining_upload_delay_round_count
    invalid_inputs = {
        "model_id": -7,
        "parameter_snapshot": {"weight": torch.tensor([0.25])},
        "upload_delay_round_count": 2,
    }
    if invalid_input_kind == "id_bool":
        invalid_inputs["model_id"] = True
    elif invalid_input_kind == "id_float":
        invalid_inputs["model_id"] = -7.0
    elif invalid_input_kind.startswith("delay"):
        invalid_inputs["upload_delay_round_count"] = {
            "delay_bool": True,
            "delay_float": 2.0,
            "delay_zero": 0,
            "delay_negative": -1,
        }[invalid_input_kind]
    elif invalid_input_kind == "snapshot_none":
        invalid_inputs["parameter_snapshot"] = None
    elif invalid_input_kind == "snapshot_empty":
        invalid_inputs["parameter_snapshot"] = {}
    elif invalid_input_kind == "key_empty":
        invalid_inputs["parameter_snapshot"] = {"": torch.zeros(1)}
    elif invalid_input_kind == "key_bool":
        invalid_inputs["parameter_snapshot"] = {True: torch.zeros(1)}
    elif invalid_input_kind == "value_none":
        invalid_inputs["parameter_snapshot"] = {"weight": None}
    else:
        parameter_values = torch.tensor([0.25])
        if invalid_input_kind == "value_float64":
            parameter_values = parameter_values.double()
        elif invalid_input_kind == "value_meta":
            parameter_values = parameter_values.to("meta")
        elif invalid_input_kind == "value_sparse":
            parameter_values = parameter_values.to_sparse()
        elif invalid_input_kind == "value_grad":
            parameter_values.requires_grad_(True)
        else:
            parameter_values[0] = float(invalid_input_kind.removeprefix("value_"))
        invalid_inputs["parameter_snapshot"] = {"weight": parameter_values}
    model_id = invalid_inputs["model_id"]
    upload_delay_round_count = invalid_inputs["upload_delay_round_count"]
    previous_parameter_snapshot = deepcopy(invalid_inputs["parameter_snapshot"])
    with pytest.raises(
        (TypeError, ValueError), match="model_id|parameter_snapshot|upload_delay_round_count"
    ):
        pending_upload_state.queue_model_upload(**invalid_inputs)
    assert pending_upload_state.get_pending_model_upload() is previous_pending_model_upload
    assert pending_upload_state.remaining_upload_delay_round_count == previous_remaining_round_count
    assert pending_upload_state.has_ready_model_upload() is (state_variant == "ready")
    assert torch.equal(initial_parameter_snapshot["weight"], torch.tensor([0.25]))
    assert invalid_inputs["model_id"] is model_id
    assert invalid_inputs["upload_delay_round_count"] is upload_delay_round_count
    if previous_parameter_snapshot is None:
        assert invalid_inputs["parameter_snapshot"] is None
    else:
        assert tuple(invalid_inputs["parameter_snapshot"]) == tuple(previous_parameter_snapshot)
        for parameter_name, parameter_values in invalid_inputs["parameter_snapshot"].items():
            expected_parameter_values = previous_parameter_snapshot[parameter_name]
            if parameter_values is None:
                assert expected_parameter_values is None
                continue
            assert parameter_values.shape == expected_parameter_values.shape
            assert parameter_values.dtype == expected_parameter_values.dtype
            assert parameter_values.device == expected_parameter_values.device
            assert parameter_values.layout == expected_parameter_values.layout
            assert parameter_values.requires_grad is expected_parameter_values.requires_grad
            assert parameter_values.grad_fn is None
            if parameter_values.device.type != "meta":
                torch.testing.assert_close(
                    parameter_values.to_dense(),
                    expected_parameter_values.to_dense(),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
    if not invalid_input_kind.startswith("delay"):
        invalid_inputs.pop("upload_delay_round_count")
        with pytest.raises((TypeError, ValueError), match="model_id|parameter_snapshot"):
            PendingModelUpload(**invalid_inputs)
