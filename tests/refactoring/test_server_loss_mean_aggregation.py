"""参加済み損失平均を旧2サーバの直接呼出と照合する。"""

import copy
from types import SimpleNamespace

import pytest

from federated_drift_experiment.servers.base import BaseServer
from federated_drift_experiment.servers.shared_backbone import SharedBackboneFedSDANoCachedServer
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments
from federated_learning_experiments.learning.loss_statistics.server_loss_mean_aggregation import (
    aggregate_participating_client_loss_means,
)


def build_loss_moments_test_seed(
    *, observed_loss_count, mean_loss, sum_squared_loss_deviations=0.0,
):
    return BoundedLossMoments(
        observed_loss_count=observed_loss_count, mean_loss=mean_loss,
        sum_squared_loss_deviations=sum_squared_loss_deviations,
    )


def convert_loss_moments_to_legacy_statistics(*, loss_moments):
    return {"n": loss_moments.observed_loss_count, "mean": loss_moments.mean_loss,
            "M2": loss_moments.sum_squared_loss_deviations}


def run_legacy_server_loss_statistics_aggregation(
    *, clients, operation, global_stats=None, model_id=0,
):
    legacy_server = SimpleNamespace(
        clients=clients, global_models={},
        global_stats=copy.deepcopy({} if global_stats is None else global_stats),
        record_model_transfer=lambda *args, **kwargs: None,
        record_parameter_transfer=lambda *args, **kwargs: None,
        comm_models_up=0,
    )
    operation(legacy_server, [model_id])
    return legacy_server.global_stats


def assert_aggregated_loss_moments_match_legacy_statistics(
    *, aggregated_loss_moments, legacy_statistics, global_stats, model_id=0,
):
    if aggregated_loss_moments is None:
        assert legacy_statistics == global_stats
    else:
        assert type(aggregated_loss_moments) is BoundedLossMoments
        assert legacy_statistics == {model_id: convert_loss_moments_to_legacy_statistics(
            loss_moments=aggregated_loss_moments)}
        assert aggregated_loss_moments.sum_squared_loss_deviations == 0.0


@pytest.mark.parametrize("operation", [
    BaseServer.update_global_models,
    SharedBackboneFedSDANoCachedServer.update_global_models,
])
@pytest.mark.parametrize("clients", [
    pytest.param((), id="empty"),
    pytest.param(((True, 1, build_loss_moments_test_seed(
        observed_loss_count=0, mean_loss=0)),), id="zero"),
    pytest.param(((True, 3, build_loss_moments_test_seed(
        observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.1)),),
        id="singleton-nonzero-seed-M2"),
    pytest.param(((True, 1, build_loss_moments_test_seed(
        observed_loss_count=2, mean_loss=0.25, sum_squared_loss_deviations=0.3)),
                  (True, 9, build_loss_moments_test_seed(
        observed_loss_count=6, mean_loss=0.75, sum_squared_loss_deviations=0.7))),
        id="statistics-count-not-training-count"),
    pytest.param(((True, 1, build_loss_moments_test_seed(observed_loss_count=0, mean_loss=0)),
                  (True, 1, build_loss_moments_test_seed(observed_loss_count=3, mean_loss=1))),
        id="zero-and-upper-bound"),
    pytest.param(((True, 2, build_loss_moments_test_seed(observed_loss_count=4, mean_loss=0)),),
        id="positive-count-zero-mean"),
    pytest.param(((True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=1)),
                  (True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=2**-53)),
                  (True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=2**-53))),
        id="rounding-large-first"),
    pytest.param(((True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=2**-53)),
                  (True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=2**-53)),
                  (True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=1))),
        id="rounding-small-first"),
    pytest.param(((False, 1, build_loss_moments_test_seed(observed_loss_count=9, mean_loss=1)),
                  (True, 0, build_loss_moments_test_seed(observed_loss_count=9, mean_loss=1)),
                  (True, 4, None)), id="all-excluded-or-no-statistics"),
    pytest.param(((False, 1, build_loss_moments_test_seed(observed_loss_count=9, mean_loss=1)),
                  (True, 0, build_loss_moments_test_seed(observed_loss_count=9, mean_loss=1)),
                  (True, 4, None),
                  (True, 2, build_loss_moments_test_seed(observed_loss_count=2, mean_loss=0.25))),
        id="eligible-mixed-with-excluded"),
])
def test_server_loss_mean_aggregation_matches_legacy_servers(*, clients, operation):
    # 参加判定は上位に置き、同じクライアント順を旧呼出にも渡す。
    participating_client_loss_moments = tuple(
        loss_moments for models, sample_count, loss_moments in clients
        if models and sample_count > 0 and loss_moments is not None)
    clients = [SimpleNamespace(
        models={0: SimpleNamespace(get_params=lambda: {
            "backbone.weight": 1.0, "head.weight": 2.0})} if models else {},
        train_data_store={0: [None] * sample_count},
        model_stats={} if loss_moments is None else {
            0: convert_loss_moments_to_legacy_statistics(loss_moments=loss_moments)},
    ) for models, sample_count, loss_moments in clients]
    global_stats = {0: {"n": 5, "mean": 0.6, "M2": 0.2,
                        "class_stats": {1: {"n": 1, "mean": 0.6, "M2": 0.0}}}}
    legacy_statistics = run_legacy_server_loss_statistics_aggregation(
        clients=clients, operation=operation, global_stats=global_stats)
    aggregated_loss_moments = aggregate_participating_client_loss_means(
        participating_client_loss_moments=participating_client_loss_moments)
    assert_aggregated_loss_moments_match_legacy_statistics(
        aggregated_loss_moments=aggregated_loss_moments,
        legacy_statistics=legacy_statistics, global_stats=global_stats)
