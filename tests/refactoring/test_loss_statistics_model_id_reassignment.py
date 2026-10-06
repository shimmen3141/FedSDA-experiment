"""正式登録の統計付替えを実旧処理と照合する。"""

import pytest
from test_classifier_bounded_loss_evaluation import assert_initial_loss_statistics_match_legacy
from test_pending_model_upload import build_legacy_pending_upload_client

from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
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
