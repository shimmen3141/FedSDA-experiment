"""正式登録の統計付替えを実旧処理と照合する。"""

import random

import numpy as np
import pytest
import torch
from test_classifier_bounded_loss_evaluation import assert_initial_loss_statistics_match_legacy
from test_joint_model_parameter_update import (
    assert_nested_state_equal,
    build_joint_update_oracle_pair,
)
from test_pending_model_upload import build_legacy_pending_upload_client
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import (
    initialize_model_and_class_loss_statistics_from_batch,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)


class DerivedModelId(int):
    """契約外のint派生型。"""


def build_legacy_registration_client(*, loss_statistics_store, original_model_id=-7):
    legacy_client = build_legacy_pending_upload_client()
    legacy_client.current_model_id = original_model_id
    legacy_client.models = {}
    legacy_client.train_data_store = {}
    legacy_client.stored_data = {}
    legacy_client.model_training_examples = {}
    legacy_client.model_optimizer_steps = {}
    legacy_client.model_concept_counts = {}
    legacy_client.model_stats = {
        model_id: {
            "n": loss_statistics.overall_loss_moments.observed_loss_count,
            "mean": loss_statistics.overall_loss_moments.mean_loss,
            "M2": loss_statistics.overall_loss_moments.sum_squared_loss_deviations,
            "class_stats": {
                observed_class_id: {
                    "n": class_loss_moments.observed_loss_count,
                    "mean": class_loss_moments.mean_loss,
                    "M2": class_loss_moments.sum_squared_loss_deviations,
                }
                for observed_class_id, class_loss_moments in (
                    loss_statistics.class_loss_moments_by_class_id
                )
            },
        }
        for model_id, loss_statistics in loss_statistics_store.get_state_snapshot()
    }
    return legacy_client


def assert_store_statistics_match_legacy(*, loss_statistics_store, legacy_client):
    assert tuple(model_id for model_id, _ in loss_statistics_store.get_state_snapshot()) == tuple(
        legacy_client.model_stats
    )
    for model_id, loss_statistics in loss_statistics_store.get_state_snapshot():
        assert_initial_loss_statistics_match_legacy(
            loss_statistics, legacy_client.model_stats[model_id]
        )


@pytest.mark.parametrize("initial_model_ids", [(), (4,), (-7,), (-7, 4, 9), (4, -7, 9), (4, 9, -7)])
@pytest.mark.parametrize("reassigned_model_id", [4, 8, -7])
def test_loss_statistics_id_reassignment_matches_legacy_registration(
    initial_model_ids, reassigned_model_id
):
    loss_statistics_store = ModelAndClassLossStatisticsStore()
    for model_id in initial_model_ids:
        # 元0件を件数の大きい先へ移しても、統計選択や合成をしない。
        loss_statistics_store.set_model_loss_statistics(
            model_id=model_id,
            loss_statistics=ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
                )
            ),
        )
        if model_id != -7:
            for observed_loss, observed_class_id in ((0.25, 3), (0.5, 1), (0.75, 3)):
                loss_statistics_store.record_assigned_loss(
                    model_id=model_id,
                    observed_loss=observed_loss,
                    observed_class_id=observed_class_id,
                )
    legacy_client = build_legacy_registration_client(loss_statistics_store=loss_statistics_store)
    assert (
        loss_statistics_store.reassign_model_loss_statistics_id(
            original_model_id=-7, reassigned_model_id=reassigned_model_id
        )
        is None
    )
    legacy_client.confirm_model_registration(reassigned_model_id)
    assert_store_statistics_match_legacy(
        loss_statistics_store=loss_statistics_store, legacy_client=legacy_client
    )


@pytest.mark.parametrize(
    "original_model_id,reassigned_model_id,expected_model_ids",
    [(0, -2, (19, -2)), (3, -4, (19, -4)), (3, 3, (19, 3)), (-5, -5, (19, -5))],
)
def test_loss_statistics_id_reassignment_preserves_signed_id_and_same_id_order(
    original_model_id, reassigned_model_id, expected_model_ids
):
    loss_statistics_store = ModelAndClassLossStatisticsStore()
    for model_id in (original_model_id, 19):
        loss_statistics_store.record_assigned_loss(
            model_id=model_id, observed_loss=0.25, observed_class_id=3
        )
        loss_statistics_store.record_assigned_loss(
            model_id=model_id, observed_loss=0.75, observed_class_id=1
        )
    previous_loss_statistics = loss_statistics_store.get_model_loss_statistics(
        model_id=original_model_id
    )
    loss_statistics_store.reassign_model_loss_statistics_id(
        original_model_id=original_model_id, reassigned_model_id=reassigned_model_id
    )
    assert tuple(model_id for model_id, _ in loss_statistics_store.get_state_snapshot()) == (
        expected_model_ids
    )
    assert (
        loss_statistics_store.get_model_loss_statistics(model_id=reassigned_model_id)
        == previous_loss_statistics
    )


@pytest.mark.parametrize("invalid_model_id", [True, 1.0, "4", None, DerivedModelId(4)])
@pytest.mark.parametrize("invalid_parameter_name", ["original_model_id", "reassigned_model_id"])
@pytest.mark.parametrize("source_present", [False, True])
@pytest.mark.parametrize("destination_present", [False, True])
def test_loss_statistics_id_reassignment_rejects_invalid_ids_without_mutation(
    invalid_model_id, invalid_parameter_name, source_present, destination_present
):
    loss_statistics_store = ModelAndClassLossStatisticsStore()
    for model_id in (*((-7,) if source_present else ()), *((4,) if destination_present else ())):
        loss_statistics_store.record_assigned_loss(model_id=model_id, observed_loss=0.5)
    previous_statistics_snapshot = loss_statistics_store.get_state_snapshot()
    with pytest.raises(TypeError, match=invalid_parameter_name):
        loss_statistics_store.reassign_model_loss_statistics_id(
            original_model_id=(
                invalid_model_id if invalid_parameter_name == "original_model_id" else -7
            ),
            reassigned_model_id=(
                invalid_model_id if invalid_parameter_name == "reassigned_model_id" else 4
            ),
        )
    assert loss_statistics_store.get_state_snapshot() == previous_statistics_snapshot


def test_reassigned_loss_statistics_remain_independent_and_updatable():
    loss_statistics_store = ModelAndClassLossStatisticsStore()
    for model_id, observed_loss, observed_class_id in ((-7, 0.25, 3), (-7, 0.75, 1), (9, 0.5, 0)):
        loss_statistics_store.record_assigned_loss(
            model_id=model_id, observed_loss=observed_loss, observed_class_id=observed_class_id
        )
    legacy_client = build_legacy_registration_client(loss_statistics_store=loss_statistics_store)
    previous_loss_statistics = loss_statistics_store.get_model_loss_statistics(model_id=-7)
    previous_statistics_snapshot = loss_statistics_store.get_state_snapshot()
    loss_statistics_store.reassign_model_loss_statistics_id(
        original_model_id=-7, reassigned_model_id=4
    )
    legacy_client.confirm_model_registration(4)
    reassigned_loss_statistics = loss_statistics_store.get_model_loss_statistics(model_id=4)
    assert reassigned_loss_statistics == previous_loss_statistics
    assert reassigned_loss_statistics is not previous_loss_statistics
    assert loss_statistics_store.get_model_loss_statistics(model_id=-7) is None
    loss_statistics_store.record_assigned_loss(model_id=4, observed_loss=0.5, observed_class_id=3)
    legacy_client._update_model_stats(4, 0.5, class_id=3)
    assert_store_statistics_match_legacy(
        loss_statistics_store=loss_statistics_store, legacy_client=legacy_client
    )
    assert previous_loss_statistics == previous_statistics_snapshot[0][1]
    assert previous_loss_statistics.overall_loss_moments.observed_loss_count == 2
    # 公開取得値の意図的改変も内部へ波及しない。
    object.__setattr__(reassigned_loss_statistics.overall_loss_moments, "mean_loss", 0.0)
    object.__setattr__(
        previous_loss_statistics.class_loss_moments_by_class_id[0][1], "mean_loss", 0.0
    )
    assert_store_statistics_match_legacy(
        loss_statistics_store=loss_statistics_store, legacy_client=legacy_client
    )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("upload_delay_round_count", [1, 2])
def test_loss_statistics_id_reassignment_connects_pending_upload_confirmation(
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
        loss_statistics_store = ModelAndClassLossStatisticsStore()
        legacy_client = build_legacy_registration_client(
            loss_statistics_store=loss_statistics_store
        )
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
        previous_random_states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        previous_numeric_environment = (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.is_grad_enabled(),
        )
        initial_parameter_snapshot = snapshot_classifier_parameters(classifier=classifier)
        expected_parameter_values = {
            parameter_name.replace("backbone.net.", "feature_extractor.hidden_layers.")
            .replace("adapter.down.", "residual_adapter.feature_compression.")
            .replace("adapter.up.", "residual_adapter.feature_expansion.")
            .replace("head.", "classification_layer."): parameter_values
            for parameter_name, parameter_values in legacy_client.pending_model_params.items()
        }
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
        loss_statistics_store.set_model_loss_statistics(
            model_id=-7, loss_statistics=initial_loss_statistics
        )
        pending_upload_state = PendingModelUploadState()
        pending_upload_state.queue_model_upload(
            model_id=-7,
            parameter_snapshot=initial_parameter_snapshot,
            upload_delay_round_count=upload_delay_round_count,
        )
        with torch.no_grad():
            classifier.classification_layer.weight.add_(2)
            source_legacy_client.models[4].head.weight.add_(2)
        for observed_loss, observed_class_id in ((0.25, 0), (0.75, class_count - 1)):
            loss_statistics_store.record_assigned_loss(
                model_id=-7, observed_loss=observed_loss, observed_class_id=observed_class_id
            )
            legacy_client._update_model_stats(-7, observed_loss, class_id=observed_class_id)
        for _ in range(upload_delay_round_count):
            pending_upload_state.advance_upload_readiness_at_round_boundary()
            legacy_client.promote_pending_to_ready()
        assert pending_upload_state.has_ready_model_upload() is legacy_client.has_pending_model()
        assert pending_upload_state.has_ready_model_upload()
        pending_model_upload = pending_upload_state.get_pending_model_upload()
        current_loss_statistics = loss_statistics_store.get_model_loss_statistics(
            model_id=pending_model_upload.model_id
        )
        assert_initial_loss_statistics_match_legacy(
            current_loss_statistics, legacy_client.get_pending_model_info()[1]
        )
        previous_parameter_state = snapshot_parameter_values_and_gradients(classifier.parameters())
        loss_statistics_store.reassign_model_loss_statistics_id(
            original_model_id=pending_model_upload.model_id, reassigned_model_id=12
        )
        # ID付替えだけでは保留を変更しない。上位が別途解除する。
        assert pending_upload_state.get_pending_model_upload() is pending_model_upload
        assert pending_model_upload.model_id == -7
        assert pending_model_upload.parameter_snapshot is initial_parameter_snapshot
        assert pending_upload_state.has_ready_model_upload()
        assert pending_upload_state.remaining_upload_delay_round_count == 0
        assert_nested_state_equal(initial_parameter_snapshot, expected_parameter_values)
        assert loss_statistics_store.get_model_loss_statistics(model_id=-7) is None
        assert (
            loss_statistics_store.get_model_loss_statistics(model_id=12) == current_loss_statistics
        )
        legacy_client.confirm_model_registration(12)
        pending_upload_state.clear_pending_model_upload()
        assert pending_upload_state.get_pending_model_upload() is None
        assert legacy_client.get_pending_model_info() == (None, None)
        assert not pending_upload_state.has_ready_model_upload()
        assert not legacy_client.has_pending_model()
        assert pending_upload_state.remaining_upload_delay_round_count == 0
        assert legacy_client.current_model_id == 12
        loss_statistics_store.record_assigned_loss(
            model_id=12, observed_loss=0.5, observed_class_id=class_count - 1
        )
        legacy_client._update_model_stats(12, 0.5, class_id=class_count - 1)
        assert_store_statistics_match_legacy(
            loss_statistics_store=loss_statistics_store, legacy_client=legacy_client
        )
        assert_nested_state_equal(initial_parameter_snapshot, expected_parameter_values)
        assert_parameter_values_and_gradients_unchanged(previous_parameter_state)
        assert random.getstate() == previous_random_states[0]
        assert np.random.get_state()[0] == previous_random_states[1][0]
        assert np.array_equal(np.random.get_state()[1], previous_random_states[1][1])
        assert np.random.get_state()[2:] == previous_random_states[1][2:]
        assert torch.equal(torch.get_rng_state(), previous_random_states[2])
        assert previous_numeric_environment == (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.is_grad_enabled(),
        )
