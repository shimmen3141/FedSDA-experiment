"""参加済み損失平均を旧2サーバの直接呼出と照合する。"""

import copy
from types import SimpleNamespace

import pytest

from federated_drift_experiment.servers.base import BaseServer
from federated_drift_experiment.servers.shared_backbone import SharedBackboneFedSDANoCachedServer
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.server_loss_mean_aggregation import (
    aggregate_participating_client_loss_means,
)


def build_loss_moments_test_seed(
    *,
    observed_loss_count,
    mean_loss,
    sum_squared_loss_deviations=0.0,
):
    return BoundedLossMoments(
        observed_loss_count=observed_loss_count,
        mean_loss=mean_loss,
        sum_squared_loss_deviations=sum_squared_loss_deviations,
    )


def convert_loss_moments_to_legacy_statistics(*, loss_moments):
    return {
        "n": loss_moments.observed_loss_count,
        "mean": loss_moments.mean_loss,
        "M2": loss_moments.sum_squared_loss_deviations,
    }


def run_legacy_server_loss_statistics_aggregation(
    *,
    clients,
    operation,
    global_stats=None,
    model_id=0,
):
    legacy_server = SimpleNamespace(
        clients=clients,
        global_models={},
        global_stats=copy.deepcopy({} if global_stats is None else global_stats),
        record_model_transfer=lambda *args, **kwargs: None,
        record_parameter_transfer=lambda *args, **kwargs: None,
        comm_models_up=0,
    )
    operation(legacy_server, [model_id])
    return legacy_server.global_stats


def assert_aggregated_loss_moments_match_legacy_statistics(
    *,
    aggregated_loss_moments,
    legacy_statistics,
    global_stats,
    model_id=0,
):
    if aggregated_loss_moments is None:
        assert legacy_statistics == global_stats
    else:
        assert type(aggregated_loss_moments) is BoundedLossMoments
        assert legacy_statistics == {
            model_id: convert_loss_moments_to_legacy_statistics(
                loss_moments=aggregated_loss_moments
            )
        }
        assert aggregated_loss_moments.sum_squared_loss_deviations == 0.0


@pytest.mark.parametrize(
    "operation",
    [
        BaseServer.update_global_models,
        SharedBackboneFedSDANoCachedServer.update_global_models,
    ],
)
@pytest.mark.parametrize(
    "clients",
    [
        pytest.param((), id="empty"),
        pytest.param(
            ((True, 1, build_loss_moments_test_seed(observed_loss_count=0, mean_loss=0)),),
            id="zero",
        ),
        pytest.param(
            (
                (
                    True,
                    3,
                    build_loss_moments_test_seed(
                        observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.1
                    ),
                ),
            ),
            id="singleton-nonzero-seed-M2",
        ),
        pytest.param(
            (
                (
                    True,
                    1,
                    build_loss_moments_test_seed(
                        observed_loss_count=2, mean_loss=0.25, sum_squared_loss_deviations=0.3
                    ),
                ),
                (
                    True,
                    9,
                    build_loss_moments_test_seed(
                        observed_loss_count=6, mean_loss=0.75, sum_squared_loss_deviations=0.7
                    ),
                ),
            ),
            id="statistics-count-not-training-count",
        ),
        pytest.param(
            (
                (True, 1, build_loss_moments_test_seed(observed_loss_count=0, mean_loss=0)),
                (True, 1, build_loss_moments_test_seed(observed_loss_count=3, mean_loss=1)),
            ),
            id="zero-and-upper-bound",
        ),
        pytest.param(
            ((True, 2, build_loss_moments_test_seed(observed_loss_count=4, mean_loss=0)),),
            id="positive-count-zero-mean",
        ),
        pytest.param(
            (
                (True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=1)),
                (True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=2**-53)),
                (True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=2**-53)),
            ),
            id="rounding-large-first",
        ),
        pytest.param(
            (
                (True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=2**-53)),
                (True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=2**-53)),
                (True, 1, build_loss_moments_test_seed(observed_loss_count=1, mean_loss=1)),
            ),
            id="rounding-small-first",
        ),
        pytest.param(
            (
                (False, 1, build_loss_moments_test_seed(observed_loss_count=9, mean_loss=1)),
                (True, 0, build_loss_moments_test_seed(observed_loss_count=9, mean_loss=1)),
                (True, 4, None),
            ),
            id="all-excluded-or-no-statistics",
        ),
        pytest.param(
            (
                (False, 1, build_loss_moments_test_seed(observed_loss_count=9, mean_loss=1)),
                (True, 0, build_loss_moments_test_seed(observed_loss_count=9, mean_loss=1)),
                (True, 4, None),
                (True, 2, build_loss_moments_test_seed(observed_loss_count=2, mean_loss=0.25)),
            ),
            id="eligible-mixed-with-excluded",
        ),
    ],
)
def test_server_loss_mean_aggregation_matches_legacy_servers(*, clients, operation):
    # 参加判定は上位に置き、同じクライアント順を旧呼出にも渡す。
    participating_client_loss_moments = tuple(
        loss_moments
        for models, sample_count, loss_moments in clients
        if models and sample_count > 0 and loss_moments is not None
    )
    clients = [
        SimpleNamespace(
            models={
                0: SimpleNamespace(get_params=lambda: {"backbone.weight": 1.0, "head.weight": 2.0})
            }
            if models
            else {},
            train_data_store={0: [None] * sample_count},
            model_stats={}
            if loss_moments is None
            else {0: convert_loss_moments_to_legacy_statistics(loss_moments=loss_moments)},
        )
        for models, sample_count, loss_moments in clients
    ]
    global_stats = {
        0: {"n": 5, "mean": 0.6, "M2": 0.2, "class_stats": {1: {"n": 1, "mean": 0.6, "M2": 0.0}}}
    }
    legacy_statistics = run_legacy_server_loss_statistics_aggregation(
        clients=clients, operation=operation, global_stats=global_stats
    )
    aggregated_loss_moments = aggregate_participating_client_loss_means(
        participating_client_loss_moments=participating_client_loss_moments
    )
    assert_aggregated_loss_moments_match_legacy_statistics(
        aggregated_loss_moments=aggregated_loss_moments,
        legacy_statistics=legacy_statistics,
        global_stats=global_stats,
    )


def build_valid_server_loss_mean_aggregation_inputs():
    """異常値に置換する前の独立した正常集計を作る。"""
    return (
        BoundedLossMoments(observed_loss_count=3, mean_loss=0.25, sum_squared_loss_deviations=0.1),
        BoundedLossMoments(observed_loss_count=1, mean_loss=0.75, sum_squared_loss_deviations=0.1),
    )


@pytest.mark.parametrize("invalid_input", [None, [], {}, 1, (None,), (object(),)])
def test_server_loss_mean_aggregation_rejects_invalid_input_types(invalid_input):
    with pytest.raises(TypeError, match="participating_client_loss_moments"):
        aggregate_participating_client_loss_means(participating_client_loss_moments=invalid_input)


@pytest.mark.parametrize(
    "field,invalid_value",
    [
        ("observed_loss_count", True),
        ("observed_loss_count", -1),
        ("observed_loss_count", 1.5),
        ("mean_loss", float("nan")),
        ("mean_loss", float("inf")),
        ("mean_loss", -0.1),
        ("mean_loss", 1.1),
        ("sum_squared_loss_deviations", -0.1),
        ("sum_squared_loss_deviations", float("inf")),
        ("sum_squared_loss_deviations", False),
    ],
)
def test_server_loss_mean_aggregation_validates_late_forged_moments_without_mutation(
    field, invalid_value
):
    moments = build_valid_server_loss_mean_aggregation_inputs()
    object.__setattr__(moments[-1], field, invalid_value)
    before = repr(moments)
    with pytest.raises((TypeError, ValueError), match="participating_client_loss_moments"):
        aggregate_participating_client_loss_means(participating_client_loss_moments=moments)
    assert repr(moments) == before


def test_server_loss_mean_aggregation_rejects_exact_type_subclasses():
    class TupleSubclass(tuple):
        pass

    class MomentSubclass(BoundedLossMoments):
        pass

    with pytest.raises(TypeError, match="participating_client_loss_moments"):
        aggregate_participating_client_loss_means(
            participating_client_loss_moments=TupleSubclass(
                build_valid_server_loss_mean_aggregation_inputs()
            )
        )
    with pytest.raises(TypeError, match="participating_client_loss_moments"):
        aggregate_participating_client_loss_means(
            participating_client_loss_moments=(
                MomentSubclass(
                    observed_loss_count=1, mean_loss=0.2, sum_squared_loss_deviations=0.0
                ),
            )
        )


@pytest.mark.parametrize("invalid_zero_field", ["mean_loss", "sum_squared_loss_deviations"])
def test_server_loss_mean_aggregation_revalidates_zero_count_records(invalid_zero_field):
    record = BoundedLossMoments(
        observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
    )
    object.__setattr__(record, invalid_zero_field, 0.1)
    before = repr(record)
    with pytest.raises(ValueError, match="participating_client_loss_moments"):
        aggregate_participating_client_loss_means(participating_client_loss_moments=(record,))
    assert repr(record) == before


def test_server_loss_mean_aggregation_returns_independent_frozen_values():
    from dataclasses import FrozenInstanceError

    moments = build_valid_server_loss_mean_aggregation_inputs()
    first = aggregate_participating_client_loss_means(participating_client_loss_moments=moments)
    second = aggregate_participating_client_loss_means(participating_client_loss_moments=moments)
    assert first == second and first is not second and all(first is not m for m in moments)
    assert (first.observed_loss_count, first.mean_loss, first.sum_squared_loss_deviations) == (
        4,
        0.375,
        0.0,
    )
    with pytest.raises(FrozenInstanceError):
        first.mean_loss = 0.99
    object.__setattr__(moments[0], "mean_loss", 0.99)
    assert first.mean_loss == second.mean_loss == 0.375
    object.__setattr__(first, "mean_loss", 0.01)
    assert second.mean_loss == 0.375


def test_server_loss_mean_aggregation_preserves_shared_environment_and_keyword_contract():
    import random

    import numpy as np
    import torch

    moments = build_valid_server_loss_mean_aggregation_inputs()
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    dtype, grad = torch.get_default_dtype(), torch.is_grad_enabled()
    with torch.device("meta"):
        aggregate_participating_client_loss_means(participating_client_loss_moments=moments)
        assert torch.empty(0).device.type == "meta"
        with pytest.raises(TypeError):
            aggregate_participating_client_loss_means(moments)
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert (
        np_state[0] == after[0]
        and np.array_equal(np_state[1], after[1])
        and np_state[2:] == after[2:]
    )
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert torch.get_default_dtype() == dtype and torch.is_grad_enabled() == grad


@pytest.mark.parametrize(
    "counts,legacy_exception",
    [
        ((2**53, 3, 3), None),
        ((10**308, 10**308), OverflowError),
    ],
)
def test_server_loss_mean_aggregation_extreme_counts_reproduce_legacy008(counts, legacy_exception):
    from types import SimpleNamespace

    from federated_drift_experiment.servers.base import BaseServer

    moments = tuple(
        BoundedLossMoments(observed_loss_count=n, mean_loss=1.0, sum_squared_loss_deviations=0.0)
        for n in counts
    )
    clients = [
        SimpleNamespace(
            models={1: SimpleNamespace(get_params=lambda: {"dummy": 1.0})},
            train_data_store={1: [None]},
            model_stats={1: {"n": n, "mean": 1.0, "M2": 0.0}},
        )
        for n in counts
    ]
    old = SimpleNamespace(
        clients=clients,
        global_models={1: {"dummy": -1.0}},
        global_stats={1: {"n": 1, "mean": 0.25, "M2": 0.1}},
        record_model_transfer=lambda *args, **kwargs: None,
    )
    if legacy_exception is None:
        BaseServer.update_global_models(old, [1])
        assert old.global_stats[1]["mean"] == 1.0000000000000002
        assert old.global_stats[1]["n"] == 9007199254740998
    else:
        prior = old.global_stats[1].copy()
        with pytest.raises(legacy_exception):
            BaseServer.update_global_models(old, [1])
        assert old.global_models[1]["dummy"] == 1.0
        assert old.global_stats[1] == prior
    before = repr(moments)
    with pytest.raises(ValueError):
        aggregate_participating_client_loss_means(participating_client_loss_moments=moments)
    assert repr(moments) == before


@pytest.mark.parametrize("participating_counts", [(), (0, 0), (1, 3)])
def test_server_loss_mean_aggregation_connects_whole_server_update_and_id_supplement(
    participating_counts,
):
    from types import SimpleNamespace

    from federated_drift_experiment.clients.base import BaseClient
    from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
        ModelAndClassLossStatistics,
        ModelAndClassLossStatisticsStore,
    )
    from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import (
        select_loss_statistics_after_model_id_mapping,
    )
    from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
        select_alarm_interval_reuse_baseline_mean_loss,
        select_loss_monitoring_baseline_mean_loss,
        select_post_alarm_reference_historical_mean_loss,
    )

    prior = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=2, mean_loss=0.25, sum_squared_loss_deviations=0.1
        ),
        class_loss_moments_by_class_id=(
            (
                2,
                BoundedLossMoments(
                    observed_loss_count=1, mean_loss=0.9, sum_squared_loss_deviations=0.0
                ),
            ),
        ),
    )
    server_store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={4: prior})
    moments = tuple(
        BoundedLossMoments(
            observed_loss_count=n,
            mean_loss=0.5 if n else 0.0,
            sum_squared_loss_deviations=0.1 if n else 0.0,
        )
        for n in participating_counts
    )
    result = aggregate_participating_client_loss_means(participating_client_loss_moments=moments)
    if result is not None:
        server_store.set_model_loss_statistics(
            model_id=4,
            loss_statistics=ModelAndClassLossStatistics(
                overall_loss_moments=result, class_loss_moments_by_class_id=()
            ),
        )
        assert (
            server_store.get_model_loss_statistics(model_id=4).class_loss_moments_by_class_id == ()
        )
    else:
        assert server_store.get_model_loss_statistics(model_id=4) == prior
    zero = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
        )
    )
    mapped = select_loss_statistics_after_model_id_mapping(
        local_model_loss_statistics=((-1, zero), (9, prior)),
        model_id_mapping={-1: 4},
        server_model_loss_statistics=server_store.get_state_snapshot(),
    )
    store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id=dict(mapped))
    assert store.get_model_loss_statistics(model_id=-1) is None
    assert store.get_model_loss_statistics(model_id=9) == prior
    assigned = store.get_model_loss_statistics(model_id=4)
    old = SimpleNamespace(
        model_stats={
            4: {
                "n": assigned.overall_loss_moments.observed_loss_count,
                "mean": assigned.overall_loss_moments.mean_loss,
                "M2": assigned.overall_loss_moments.sum_squared_loss_deviations,
                "class_stats": {
                    cid: {
                        "n": m.observed_loss_count,
                        "mean": m.mean_loss,
                        "M2": m.sum_squared_loss_deviations,
                    }
                    for cid, m in assigned.class_loss_moments_by_class_id
                },
            }
        },
        _update_running_stats=BaseClient._update_running_stats,
    )
    store.record_assigned_loss(model_id=4, observed_loss=0.8, observed_class_id=2)
    BaseClient._update_model_stats(old, 4, 0.8, class_id=2)
    updated = store.get_model_loss_statistics(model_id=4)
    assert (
        updated.overall_loss_moments.observed_loss_count,
        updated.overall_loss_moments.mean_loss,
        updated.overall_loss_moments.sum_squared_loss_deviations,
    ) == (old.model_stats[4]["n"], old.model_stats[4]["mean"], old.model_stats[4]["M2"])
    for cid, m in updated.class_loss_moments_by_class_id:
        expected = old.model_stats[4]["class_stats"][cid]
        assert (m.observed_loss_count, m.mean_loss, m.sum_squared_loss_deviations) == (
            expected["n"],
            expected["mean"],
            expected["M2"],
        )
    for selector in (
        select_loss_monitoring_baseline_mean_loss,
        select_alarm_interval_reuse_baseline_mean_loss,
        select_post_alarm_reference_historical_mean_loss,
    ):
        assert (
            selector(loss_moments=updated.overall_loss_moments)
            == updated.overall_loss_moments.mean_loss
        )
