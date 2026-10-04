"""モデル全体・クラス別の帰属統計を旧更新へ直接照合する。"""

from types import SimpleNamespace

import pytest

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)


@pytest.mark.parametrize("seed", [None, 1, 10])
@pytest.mark.parametrize("loss_sequence", [
    ((-100, 0.2, 0), (-100, 0.7, None), (-100, 0.5, 1),
     (5, 0.1, 0), (-100, 1.0, 0), (4, 0.0, 1)),
    ((5, 0.0, None), (-100, 1e-15, 1), (-100, 2e-15, 0),
     (-100, 0.0, None), (5, 1.0, 1), (5, 0.5, None)),
])
def test_model_class_statistics_matches_legacy_updates(seed, loss_sequence):
    legacy_client = SimpleNamespace(
        model_stats={}, _update_running_stats=BaseClient._update_running_stats)
    if seed is None:
        store = ModelAndClassLossStatisticsStore()
    else:
        legacy_client.model_stats[-100] = {
            "n": seed, "mean": 0.25, "M2": 0.1 if seed == 1 else 0.0,
            "class_stats": {0: {"n": 1, "mean": 0.25, "M2": 0.0}},
        }
        store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={
            -100: ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(observed_loss_count=seed,
                    mean_loss=0.25, sum_squared_loss_deviations=0.1 if seed == 1 else 0.0),
                class_loss_moments_by_class_id=((0, BoundedLossMoments(
                    observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.0)),)),
        })
    for model_id, observed_loss, observed_class_id in loss_sequence:
        assert store.record_assigned_loss(model_id=model_id, observed_loss=observed_loss,
                                         observed_class_id=observed_class_id) is None
        BaseClient._update_model_stats(legacy_client, model_id, observed_loss,
                                      class_id=observed_class_id)
        snapshot = store.get_state_snapshot()
        assert tuple(model_id for model_id, result in snapshot) == tuple(legacy_client.model_stats)
        for model_id, result in snapshot:
            legacy_stats = legacy_client.model_stats[model_id]
            assert (result.overall_loss_moments.observed_loss_count,
                    result.overall_loss_moments.mean_loss,
                    result.overall_loss_moments.sum_squared_loss_deviations) == (
                        legacy_stats["n"], legacy_stats["mean"], legacy_stats["M2"])
            assert tuple(class_id for class_id, class_loss_moments in
                         result.class_loss_moments_by_class_id) == tuple(
                             legacy_stats.get("class_stats", {}))
            for class_id, class_loss_moments in result.class_loss_moments_by_class_id:
                legacy_class_stats = legacy_stats["class_stats"][class_id]
                assert (class_loss_moments.observed_loss_count, class_loss_moments.mean_loss,
                        class_loss_moments.sum_squared_loss_deviations) == (
                            legacy_class_stats["n"], legacy_class_stats["mean"], legacy_class_stats["M2"])

def test_model_class_statistics_accepts_and_replaces_seeds():
    seed = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(observed_loss_count=1,
            mean_loss=0.25, sum_squared_loss_deviations=0.1),
        class_loss_moments_by_class_id=((0, BoundedLossMoments(observed_loss_count=1,
            mean_loss=0.25, sum_squared_loss_deviations=0.0)),))
    store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={
        -100: seed,
        4: ModelAndClassLossStatistics(overall_loss_moments=BoundedLossMoments(
            observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0)),
    })
    assert store.get_model_loss_statistics(model_id=99) is None
    assert store.get_model_loss_statistics(model_id=4).overall_loss_moments.observed_loss_count == 0
    assert store.get_model_loss_statistics(model_id=-100) == seed
    seed = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(observed_loss_count=10,
            mean_loss=0.3, sum_squared_loss_deviations=0.0),
        class_loss_moments_by_class_id=((1, BoundedLossMoments(observed_loss_count=20,
            mean_loss=0.4, sum_squared_loss_deviations=0.2)),))
    state_before_call = store.get_model_loss_statistics(model_id=4)
    assert store.set_model_loss_statistics(model_id=-100, loss_statistics=seed) is None
    assert store.get_model_loss_statistics(model_id=-100) == seed
    assert store.get_model_loss_statistics(model_id=4) == state_before_call
    assert store.set_model_loss_statistics(model_id=7, loss_statistics=seed) is None
    assert tuple(model_id for model_id, result in store.get_state_snapshot()) == (-100, 4, 7)
    store.record_assigned_loss(model_id=-100, observed_loss=0.5, observed_class_id=None)
    result = store.get_model_loss_statistics(model_id=-100)
    assert result.overall_loss_moments.observed_loss_count == 11
    assert result.class_loss_moments_by_class_id == seed.class_loss_moments_by_class_id


def test_model_class_statistics_preserves_snapshot_independence():
    seed = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(observed_loss_count=1,
            mean_loss=0.25, sum_squared_loss_deviations=0.1),
        class_loss_moments_by_class_id=((1, BoundedLossMoments(observed_loss_count=1,
            mean_loss=0.25, sum_squared_loss_deviations=0.0)),))
    store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={-100: seed})
    other_store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={-100: seed})
    result = store.get_model_loss_statistics(model_id=-100)
    snapshot = store.get_state_snapshot()
    assert result == snapshot[0][1] == seed
    assert result is not snapshot[0][1] and result is not seed
    assert result.overall_loss_moments is not seed.overall_loss_moments
    assert result.overall_loss_moments is not snapshot[0][1].overall_loss_moments
    assert result.class_loss_moments_by_class_id[0][1] is not seed.class_loss_moments_by_class_id[0][1]
    assert result.class_loss_moments_by_class_id[0][1] is not snapshot[0][1].class_loss_moments_by_class_id[0][1]
    store.record_assigned_loss(model_id=-100, observed_loss=0.75, observed_class_id=0)
    assert result == seed and snapshot == ((-100, seed),)
    assert other_store.get_state_snapshot() == snapshot
    assert tuple(class_id for class_id, class_loss_moments in
                 store.get_model_loss_statistics(model_id=-100).class_loss_moments_by_class_id) == (1, 0)
    assert ModelAndClassLossStatisticsStore().get_state_snapshot() == ()
