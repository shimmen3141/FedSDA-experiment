"""一保留枠・ラウンド待機と実旧FedSDAの送信準備を比較する。"""

import random
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from test_classifier_bounded_loss_evaluation import assert_initial_loss_statistics_match_legacy
from test_joint_model_parameter_update import (
    assert_nested_state_equal,
    build_joint_update_oracle_pair,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment.clients.base import BaseClient
from federated_drift_experiment.clients.fedsda import ClassConditionalESRFedSDAClient
from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import (
    initialize_model_and_class_loss_statistics_from_batch,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
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


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("upload_delay_round_count", [1, 2, 4])
def test_pending_upload_connects_fixed_parameters_to_current_statistics(
    class_count, upload_delay_round_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        training_batches, _, source_legacy_client = build_joint_update_oracle_pair(
            class_count=class_count,
            batch_sample_counts=(5,),
            optimizer_variant="standard",
            monkeypatch=monkeypatch,
        )
        classifier = training_batches[0].classifier
        input_features = training_batches[0].input_features
        observed_class_labels = training_batches[0].observed_class_labels
        legacy_client = build_legacy_pending_upload_client()
        legacy_client.current_model_id = -7
        legacy_client.models = {}
        legacy_client.model_stats = {}
        legacy_client._prepare_model_for_registration = lambda model: model
        legacy_client._record_model_compute = lambda *args: None
        BaseClient._register_trained_new_model(
            legacy_client,
            temp_id=-7,
            new_model=source_legacy_client.models[4],
            bx=input_features,
            by=observed_class_labels,
            pending_ready=False,
        )
        legacy_client._pending_upload_rounds = upload_delay_round_count
        previous_parameter_state = snapshot_parameter_values_and_gradients(classifier.parameters())
        previous_random_states = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
        )
        initial_parameter_snapshot = snapshot_classifier_parameters(classifier=classifier)
        per_sample_bounded_losses = evaluate_classifier_per_sample_bounded_losses(
            classifier=classifier,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
        )
        initial_loss_statistics = initialize_model_and_class_loss_statistics_from_batch(
            per_sample_bounded_losses=per_sample_bounded_losses,
            observed_class_labels=observed_class_labels,
            class_count=class_count,
        )
        loss_statistics_store = ModelAndClassLossStatisticsStore(
            initial_loss_statistics_by_model_id={-7: initial_loss_statistics}
        )
        pending_upload_state = PendingModelUploadState()
        pending_upload_state.queue_model_upload(
            model_id=-7,
            parameter_snapshot=initial_parameter_snapshot,
            upload_delay_round_count=upload_delay_round_count,
        )
        assert_parameter_values_and_gradients_unchanged(previous_parameter_state)
        expected_parameter_values = {
            parameter_name.replace("backbone.net.", "feature_extractor.hidden_layers.")
            .replace("adapter.down.", "residual_adapter.feature_compression.")
            .replace("adapter.up.", "residual_adapter.feature_expansion.")
            .replace("head.", "classification_layer."): parameter_values
            for parameter_name, parameter_values in legacy_client.pending_model_params.items()
        }
        assert_nested_state_equal(initial_parameter_snapshot, expected_parameter_values)
        assert_initial_loss_statistics_match_legacy(
            initial_loss_statistics, legacy_client.pending_model_stats
        )
        # 保留後もモデルと統計は更新される。snapshotは登録時の値を保持する。
        with torch.no_grad():
            classifier.classification_layer.weight.add_(2)
            source_legacy_client.models[4].head.weight.add_(2)
        for observed_loss, observed_class_id in ((0.25, 0), (0.75, 1), (0.5, class_count - 1)):
            loss_statistics_store.record_assigned_loss(
                model_id=-7, observed_loss=observed_loss, observed_class_id=observed_class_id
            )
            legacy_client._update_model_stats(-7, observed_loss, class_id=observed_class_id)
        for round_boundary_index in range(upload_delay_round_count + 2):
            pending_model_upload = pending_upload_state.get_pending_model_upload()
            assert pending_model_upload is not None
            assert pending_model_upload.parameter_snapshot is initial_parameter_snapshot
            assert_nested_state_equal(
                pending_model_upload.parameter_snapshot, expected_parameter_values
            )
            current_loss_statistics = loss_statistics_store.get_model_loss_statistics(
                model_id=pending_model_upload.model_id
            )
            assert current_loss_statistics is not None
            legacy_loss_statistics = legacy_client.get_pending_model_info()[1]
            assert_initial_loss_statistics_match_legacy(
                current_loss_statistics, legacy_loss_statistics
            )
            assert current_loss_statistics.overall_loss_moments.observed_loss_count == 8
            assert (
                pending_upload_state.has_ready_model_upload() is legacy_client.has_pending_model()
            )
            assert pending_upload_state.has_ready_model_upload() is (
                round_boundary_index >= upload_delay_round_count
            )
            assert (
                pending_upload_state.remaining_upload_delay_round_count
                == legacy_client._pending_upload_rounds
            )
            pending_upload_state.advance_upload_readiness_at_round_boundary()
            legacy_client.promote_pending_to_ready()
        assert not torch.equal(
            classifier.classification_layer.weight,
            initial_parameter_snapshot["classification_layer.weight"],
        )
        assert random.getstate() == previous_random_states[0]
        assert np.random.get_state()[0] == previous_random_states[1][0]
        assert np.array_equal(np.random.get_state()[1], previous_random_states[1][1])
        assert np.random.get_state()[2:] == previous_random_states[1][2:]
        assert torch.equal(torch.get_rng_state(), previous_random_states[2])
