"""モデルID対応後の統計選択を旧処理の全値と順序へ直接照合する。"""

from types import SimpleNamespace

import pytest

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import (
    select_loss_statistics_after_model_id_mapping,
)


def build_model_and_class_loss_statistics_test_seed(
    *, observed_loss_count, mean_loss, sum_squared_loss_deviations=0.0,
    class_loss_moments_by_class_id=(),
):
    return ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=observed_loss_count, mean_loss=mean_loss,
            sum_squared_loss_deviations=sum_squared_loss_deviations),
        class_loss_moments_by_class_id=class_loss_moments_by_class_id,
    )


def convert_loss_statistics_snapshot_to_legacy_model_stats(*, loss_statistics_snapshot):
    return {model_id: {
        "n": loss_statistics.overall_loss_moments.observed_loss_count,
        "mean": loss_statistics.overall_loss_moments.mean_loss,
        "M2": loss_statistics.overall_loss_moments.sum_squared_loss_deviations,
        "class_stats": {class_id: {
            "n": loss_moments.observed_loss_count,
            "mean": loss_moments.mean_loss,
            "M2": loss_moments.sum_squared_loss_deviations,
        } for class_id, loss_moments in loss_statistics.class_loss_moments_by_class_id},
    } for model_id, loss_statistics in loss_statistics_snapshot}


def select_legacy_loss_statistics_after_model_id_mapping(
    *, local_model_loss_statistics, model_id_mapping, server_model_loss_statistics=None,
):
    legacy_client = SimpleNamespace(
        current_model_id=999,
        model_stats=convert_loss_statistics_snapshot_to_legacy_model_stats(
            loss_statistics_snapshot=local_model_loss_statistics),
        stored_data={}, train_data_store={}, model_training_examples={},
        model_optimizer_steps={}, model_concept_counts={}, models={},
        stored_data_limit=10, _after_models_rebuilt=lambda: None,
    )
    BaseClient.apply_server_mapping(
        legacy_client, model_id_mapping, {},
        None if server_model_loss_statistics is None else
        convert_loss_statistics_snapshot_to_legacy_model_stats(
            loss_statistics_snapshot=server_model_loss_statistics),
    )
    return legacy_client.model_stats


def assert_loss_statistics_snapshot_matches_legacy_model_stats(
    *, loss_statistics_snapshot, legacy_model_stats,
):
    assert type(loss_statistics_snapshot) is tuple
    assert tuple(model_id for model_id, loss_statistics in loss_statistics_snapshot) == tuple(legacy_model_stats)
    for model_id, loss_statistics in loss_statistics_snapshot:
        assert type(loss_statistics) is ModelAndClassLossStatistics
        legacy_stats = legacy_model_stats[model_id]
        assert (loss_statistics.overall_loss_moments.observed_loss_count,
                loss_statistics.overall_loss_moments.mean_loss,
                loss_statistics.overall_loss_moments.sum_squared_loss_deviations) == (
                    legacy_stats["n"], legacy_stats["mean"], legacy_stats["M2"])
        assert tuple(class_id for class_id, loss_moments in
                     loss_statistics.class_loss_moments_by_class_id) == tuple(legacy_stats["class_stats"])
        for class_id, loss_moments in loss_statistics.class_loss_moments_by_class_id:
            legacy_class_stats = legacy_stats["class_stats"][class_id]
            assert (loss_moments.observed_loss_count, loss_moments.mean_loss,
                    loss_moments.sum_squared_loss_deviations) == (
                        legacy_class_stats["n"], legacy_class_stats["mean"], legacy_class_stats["M2"])


@pytest.mark.parametrize("local_model_loss_statistics, model_id_mapping, server_model_loss_statistics", [
    pytest.param((), {}, None, id="empty-none"),
    pytest.param((), {}, (), id="empty-server-tuple"),
    pytest.param((), {-100: 7}, None, id="unused-mapping-no-statistics"),
    pytest.param((), {}, ((4, build_model_and_class_loss_statistics_test_seed(
        observed_loss_count=3, mean_loss=0.2)),), id="server-only"),
    pytest.param((), {}, ((4, build_model_and_class_loss_statistics_test_seed(
        observed_loss_count=0, mean_loss=0)),), id="zero-server-only"),
    pytest.param(((-100, build_model_and_class_loss_statistics_test_seed(observed_loss_count=1,
        mean_loss=0.25, sum_squared_loss_deviations=0.1)),), {}, None, id="negative-no-mapping"),
    pytest.param(((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2,
        mean_loss=0.2)),), {1: 1}, (), id="identity"),
    pytest.param(((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.2)),
                  (2, build_model_and_class_loss_statistics_test_seed(observed_loss_count=3, mean_loss=0.3))),
                 {1: 2, 2: 3}, None, id="one-hop-chain"),
    pytest.param(((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.2)),
                  (2, build_model_and_class_loss_statistics_test_seed(observed_loss_count=3, mean_loss=0.3))),
                 {1: 2, 2: 1}, None, id="simultaneous-cycle"),
    pytest.param(((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.2)),),
                 {-100: -200}, (), id="unused-negative-mapping"),
    pytest.param(((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=1, mean_loss=0.2)),
                  (3, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.3)),
                  (2, build_model_and_class_loss_statistics_test_seed(observed_loss_count=4, mean_loss=0.4,
                    sum_squared_loss_deviations=0.2, class_loss_moments_by_class_id=(
                      (3, BoundedLossMoments(observed_loss_count=2, mean_loss=0.8, sum_squared_loss_deviations=0.1)),
                      (1, BoundedLossMoments(observed_loss_count=1, mean_loss=0.2, sum_squared_loss_deviations=0.0)))))),
                 {1: 8, 2: 8}, None, id="larger-winner-keeps-target-position-and-class-order"),
    pytest.param(((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.2)),
                  (2, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.8))),
                 {1: 8, 2: 8}, None, id="tie-first"),
    pytest.param(((2, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.8)),
                  (1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.2))),
                 {1: 8, 2: 8}, None, id="tie-reversed-input"),
    pytest.param(((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.2)),),
                 {}, ((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=100, mean_loss=0.8)),),
                 id="positive-local-defeats-larger-server"),
    pytest.param(((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=0, mean_loss=0)),),
                 {}, ((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=5, mean_loss=0.7)),),
                 id="zero-local-replaced"),
    pytest.param(((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=0, mean_loss=0)),),
                 {}, ((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=0, mean_loss=0,
                    class_loss_moments_by_class_id=((2, BoundedLossMoments(observed_loss_count=1,
                        mean_loss=0.7, sum_squared_loss_deviations=0.0)),))),), id="zero-server-replaces-zero-local"),
    pytest.param(((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.2)),),
                 {1: 2, 2: 3}, ((1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=5,
                     mean_loss=0.7)),), id="server-id-not-remapped"),
    pytest.param(((3, build_model_and_class_loss_statistics_test_seed(observed_loss_count=0, mean_loss=0)),
                  (1, build_model_and_class_loss_statistics_test_seed(observed_loss_count=2, mean_loss=0.2))),
                 {}, ((7, build_model_and_class_loss_statistics_test_seed(observed_loss_count=0, mean_loss=0)),
                      (3, build_model_and_class_loss_statistics_test_seed(observed_loss_count=4, mean_loss=0.4)),
                      (-5, build_model_and_class_loss_statistics_test_seed(observed_loss_count=1, mean_loss=0.5))),
                 id="server-replacement-position-and-append-order"),
])
def test_model_id_mapped_statistics_match_legacy_selection(
    local_model_loss_statistics, model_id_mapping, server_model_loss_statistics,
):
    legacy_model_stats = select_legacy_loss_statistics_after_model_id_mapping(
        local_model_loss_statistics=local_model_loss_statistics,
        model_id_mapping=model_id_mapping, server_model_loss_statistics=server_model_loss_statistics)
    result = select_loss_statistics_after_model_id_mapping(
        local_model_loss_statistics=local_model_loss_statistics,
        model_id_mapping=model_id_mapping, server_model_loss_statistics=server_model_loss_statistics)
    assert_loss_statistics_snapshot_matches_legacy_model_stats(
        loss_statistics_snapshot=result, legacy_model_stats=legacy_model_stats)
    if server_model_loss_statistics is None:
        assert select_loss_statistics_after_model_id_mapping(
            local_model_loss_statistics=local_model_loss_statistics,
            model_id_mapping=model_id_mapping) == result
