"""保存済みモデルの統合計算を実旧server/clientへ照合する。"""

from copy import deepcopy

import pytest
import torch

from federated_drift_experiment.servers.feddrift import FedDriftServer
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.methods.feddrift.cached_model_cluster_consolidation import (
    average_feddrift_cached_cluster_parameters,
    consolidate_feddrift_cached_model_clusters,
)
from tests.refactoring.test_feddrift_run_client import (
    assert_rng_equal,
    build_assembly_inputs,
    rng_snapshot,
)


def parameter_inputs():
    # 非昇順・2群・キー順の違い。float32の桁落ちが順序変更を検出する。
    return {
        8: {"weight": torch.tensor([1e8, 0.125]), "bias": torch.tensor([0.7])},
        2: {"bias": torch.tensor([-0.1]), "weight": torch.tensor([-1e8, 0.25])},
        5: {"weight": torch.tensor([1.0, -0.875]), "bias": torch.tensor([0.2])},
        1: {"weight": torch.tensor([0.75, -0.25]), "bias": torch.tensor([0.4])},
    }


def assert_parameters_equal(actual, expected):
    assert tuple(actual) == tuple(expected)
    for model_id in actual:
        assert tuple(actual[model_id]) == tuple(expected[model_id])
        for name in actual[model_id]:
            assert torch.equal(actual[model_id][name], expected[model_id][name])


@pytest.mark.parametrize("weights", ({8: 1, 2: 1, 5: 1}, {8: 0, 2: 7, 5: 2}, {8: 0, 2: 0, 5: 0}))
@pytest.mark.parametrize("grad_enabled", (False, True))
def test_average_matches_real_server_and_preserves_inputs(weights, grad_enabled):
    parameters = parameter_inputs()
    for params in parameters.values():
        for values in params.values():
            values.requires_grad_()
            values.grad = torch.ones_like(values)
    before_values = deepcopy(parameters)
    before_grads = {
        mid: {name: v.grad.clone() for name, v in p.items()} for mid, p in parameters.items()
    }
    server = FedDriftServer(verbose=False)
    server.global_models = {
        mid: {n: v.detach().clone() for n, v in p.items()} for mid, p in parameters.items()
    }
    used_weights = weights if sum(weights.values()) else {mid: 1 for mid in weights}
    expected = {
        2: server._weighted_model((8, 2, 5), used_weights),
        1: server._weighted_model((1,), {1: 4}),
    }
    rng_before = rng_snapshot()
    with torch.set_grad_enabled(grad_enabled):
        result = average_feddrift_cached_cluster_parameters(
            model_clusters=((8, 2, 5), (1,)),
            model_parameters_by_model_id=parameters,
            cluster_weights_by_representative_id={2: weights, 1: {1: 4}},
        )
        assert torch.is_grad_enabled() == grad_enabled
    assert_parameters_equal(result, expected)
    assert_parameters_equal(parameters, before_values)
    assert_rng_equal(rng_before)
    for mid, params in parameters.items():
        for name, value in params.items():
            assert torch.equal(value.grad, before_grads[mid][name])
    for params in result.values():
        for value in params.values():
            assert not value.requires_grad and value.grad is None
            assert all(
                value.data_ptr() != x.data_ptr() for p in parameters.values() for x in p.values()
            )
    result[1]["weight"].add_(1)
    assert_parameters_equal(parameters, before_values)


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize("zero_weights", (False, True))
def test_average_matches_real_client_cached_merge(
    monkeypatch, class_count, optimizer_name, zero_weights
):
    client, _ = build_assembly_inputs(
        monkeypatch=monkeypatch, class_count=class_count, optimizer_name=optimizer_name
    )
    for mid in (8, 2, 5):
        model = deepcopy(client.models[0])
        with torch.no_grad():
            for p in model.parameters():
                p.add_(mid / 10.0)
        client.models[mid] = model
    client.models.pop(0)
    client.current_model_id = 2
    parameters = {mid: dict(model.get_params()) for mid, model in client.models.items()}
    weights = {2: {8: 0, 2: 0, 5: 0} if zero_weights else {8: 3, 2: 1, 5: 0}}
    client.apply_cached_merge([[8, 2, 5]], weights, {2: {"n": 0, "mean": 0.0, "M2": 0.0}})
    before = rng_snapshot()
    result = average_feddrift_cached_cluster_parameters(
        model_clusters=((8, 2, 5),),
        model_parameters_by_model_id=parameters,
        cluster_weights_by_representative_id=weights,
    )
    assert_rng_equal(before)
    assert_parameters_equal(result, {2: client.models[2].get_params()})


@pytest.mark.parametrize(
    "case",
    (
        "list_clusters",
        "list_cluster",
        "empty_cluster",
        "duplicate",
        "missing",
        "extra",
        "bool_id",
        "negative_id",
        "list_parameters",
        "empty_parameters",
        "bad_name",
        "not_tensor",
        "float64",
        "nan",
        "inf",
        "empty_tensor",
        "shape",
        "keys",
        "sparse",
        "bool_weight",
        "negative_weight",
        "float_weight",
        "missing_weight",
        "extra_weight",
        "wrong_representative",
        "list_weights",
    ),
)
def test_average_rejects_invalid_inputs_without_changes(case):
    clusters = ((8, 2, 5), (1,))
    parameters = parameter_inputs()
    weights = {2: {8: 3, 2: 1, 5: 0}, 1: {1: 4}}
    if case == "list_clusters":
        clusters = list(clusters)
    elif case == "list_cluster":
        clusters = ([8, 2, 5], (1,))
    elif case == "empty_cluster":
        clusters = ((), (8, 2, 5, 1))
    elif case == "duplicate":
        clusters = ((8, 2, 5), (1, 8))
    elif case == "missing":
        clusters = ((8, 2, 5),)
    elif case == "extra":
        clusters = ((8, 2, 5), (1, 9))
    elif case == "bool_id":
        clusters = ((8, 2, 5), (True,))
    elif case == "negative_id":
        clusters = ((8, 2, 5), (-1,))
    elif case == "list_parameters":
        parameters = list(parameters.items())
    elif case == "empty_parameters":
        parameters[5] = {}
    elif case == "bad_name":
        parameters[5][1] = parameters[5].pop("weight")
    elif case == "not_tensor":
        parameters[5]["weight"] = [1.0, 2.0]
    elif case == "float64":
        parameters[5]["weight"] = parameters[5]["weight"].double()
    elif case == "nan":
        parameters[5]["weight"][0] = float("nan")
    elif case == "inf":
        parameters[5]["weight"][0] = float("inf")
    elif case == "empty_tensor":
        parameters[5]["weight"] = torch.empty(0)
    elif case == "shape":
        parameters[5]["weight"] = torch.ones(3)
    elif case == "keys":
        parameters[5]["wrong"] = parameters[5].pop("weight")
    elif case == "sparse":
        parameters[5]["weight"] = parameters[5]["weight"].to_sparse()
    elif case == "bool_weight":
        weights[2][5] = True
    elif case == "negative_weight":
        weights[2][5] = -1
    elif case == "float_weight":
        weights[2][5] = 0.5
    elif case == "missing_weight":
        weights[2].pop(5)
    elif case == "extra_weight":
        weights[2][9] = 1
    elif case == "wrong_representative":
        weights[0] = weights.pop(2)
    elif case == "list_weights":
        weights = list(weights.items())
    # tensorのreprではなく内容を固定。NaNはequal_nanで照合する。
    before = deepcopy(parameters)
    weights_before = deepcopy(weights)
    rng_before = rng_snapshot()
    with pytest.raises((TypeError, ValueError)):
        average_feddrift_cached_cluster_parameters(
            model_clusters=clusters,
            model_parameters_by_model_id=parameters,
            cluster_weights_by_representative_id=weights,
        )
    assert_rng_equal(rng_before)
    assert weights == weights_before
    if type(parameters) is dict:
        for mid, params in parameters.items():
            assert tuple(params) == tuple(before[mid])
            for name, value in params.items():
                if isinstance(value, torch.Tensor):
                    actual = value.to_dense() if value.is_sparse else value
                    expected = (
                        before[mid][name].to_dense() if value.is_sparse else before[mid][name]
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                else:
                    assert value == before[mid][name]
    else:
        assert_parameters_equal(dict(parameters), dict(before))


def test_average_empty_input_is_empty_and_consumes_no_rng():
    before = rng_snapshot()
    assert (
        average_feddrift_cached_cluster_parameters(
            model_clusters=(),
            model_parameters_by_model_id={},
            cluster_weights_by_representative_id={},
        )
        == {}
    )
    assert_rng_equal(before)


def server_inputs():
    return dict(
        timestep_index=4,
        model_clusters=((8, 2, 5), (1,)),
        model_parameters_by_model_id=parameter_inputs(),
        historical_model_weights_by_model_id={8: 7, 2: 2, 5: 0, 1: 3},
        loss_moments_by_model_id={
            8: BoundedLossMoments(
                observed_loss_count=3, mean_loss=0.25, sum_squared_loss_deviations=0.125
            ),
            5: BoundedLossMoments(
                observed_loss_count=7, mean_loss=0.625, sum_squared_loss_deviations=0.75
            ),
            1: BoundedLossMoments(
                observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
            ),
        },
        isolation_until_by_model_id={8: 2, 2: 4, 5: 6, 1: 4},
    )


@pytest.mark.parametrize(
    "weights",
    (
        {8: 7, 2: 2, 5: 0, 1: 3},
        {8: -7, 2: 0, 5: 0},
        {},
        {8: -7, 2: 2, 5: 0, 1: 9},
        {8: 2**53 + 1, 2: 2**53, 5: 3, 1: 1},
    ),
)
@pytest.mark.parametrize("statistics_kind", ("mixed", "empty", "zero"))
@pytest.mark.parametrize("timestep", (4, 6))
def test_server_consolidation_matches_real_merge(weights, statistics_kind, timestep):
    inputs = server_inputs()
    inputs["historical_model_weights_by_model_id"] = weights
    inputs["timestep_index"] = timestep
    if statistics_kind == "empty":
        inputs["loss_moments_by_model_id"] = {}
    elif statistics_kind == "zero":
        inputs["loss_moments_by_model_id"] = {
            8: BoundedLossMoments(
                observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
            )
        }
    before_parameters = deepcopy(inputs["model_parameters_by_model_id"])
    before_other = deepcopy(
        {k: v for k, v in inputs.items() if k != "model_parameters_by_model_id"}
    )
    server = FedDriftServer(verbose=False)
    server.global_models = deepcopy(inputs["model_parameters_by_model_id"])
    server.model_weights = deepcopy(weights)
    server.isolated_until = deepcopy(inputs["isolation_until_by_model_id"])
    server.global_stats.update(
        {
            mid: {
                "n": m.observed_loss_count,
                "mean": m.mean_loss,
                "M2": m.sum_squared_loss_deviations,
                "class_stats": {0: {"n": 1, "mean": 0.2, "M2": 0.0}},
            }
            for mid, m in inputs["loss_moments_by_model_id"].items()
        }
    )
    before_rng = rng_snapshot()
    server._merge_cached_clusters(timestep, inputs["model_clusters"])
    assert_rng_equal(before_rng)
    result = consolidate_feddrift_cached_model_clusters(**inputs)
    assert_rng_equal(before_rng)
    assert_parameters_equal(result.model_parameters_by_representative_id, server.global_models)
    assert result.model_weights_by_representative_id == server.model_weights
    assert result.isolation_until_by_representative_id == server.isolated_until
    assert tuple(result.loss_moments_by_representative_id) == tuple(server.global_stats)
    for mid, moments in result.loss_moments_by_representative_id.items():
        assert moments.observed_loss_count == server.global_stats[mid]["n"]
        assert moments.mean_loss == server.global_stats[mid]["mean"]
        assert moments.sum_squared_loss_deviations == server.global_stats[mid]["M2"] == 0.0
    expected_weights = {}
    for cluster in inputs["model_clusters"]:
        member_weights = {mid: max(weights.get(mid, 0), 0) for mid in cluster}
        if sum(member_weights.values()) <= 0:
            member_weights = {mid: 1 for mid in cluster}
        expected_weights[min(cluster)] = member_weights
    assert result.cluster_weights_by_representative_id == expected_weights
    assert tuple(result.model_id_mapping.items()) == ((8, 2), (2, 2), (5, 2), (1, 1))
    assert_parameters_equal(inputs["model_parameters_by_model_id"], before_parameters)
    assert {k: v for k, v in inputs.items() if k != "model_parameters_by_model_id"} == before_other
    # server結果をclient自身の異なる値へ適用する。server値の配布に置換すると失敗する。
    own_parameters = {
        mid: {n: v + 0.375 for n, v in p.items()}
        for mid, p in inputs["model_parameters_by_model_id"].items()
    }
    client_result = average_feddrift_cached_cluster_parameters(
        model_clusters=inputs["model_clusters"],
        model_parameters_by_model_id=own_parameters,
        cluster_weights_by_representative_id=result.cluster_weights_by_representative_id,
    )
    assert any(
        not torch.equal(client_result[mid]["bias"], server.global_models[mid]["bias"])
        for mid in client_result
    )


@pytest.mark.parametrize(
    "case",
    (
        "bool_time",
        "negative_time",
        "float_time",
        "list_historical",
        "bool_historical",
        "float_historical",
        "extra_historical",
        "bool_historical_id",
        "list_stats",
        "bad_stats",
        "extra_stats",
        "broken_stats",
        "list_isolation",
        "bool_isolation",
        "float_isolation",
        "negative_isolation",
        "extra_isolation",
        "late_bad_parameter",
    ),
)
def test_server_rejects_invalid_inputs_before_any_arithmetic(case, monkeypatch):
    inputs = server_inputs()
    if case == "bool_time":
        inputs["timestep_index"] = True
    elif case == "negative_time":
        inputs["timestep_index"] = -1
    elif case == "float_time":
        inputs["timestep_index"] = 4.0
    elif case == "list_historical":
        inputs["historical_model_weights_by_model_id"] = []
    elif case == "bool_historical":
        inputs["historical_model_weights_by_model_id"][1] = True
    elif case == "float_historical":
        inputs["historical_model_weights_by_model_id"][1] = 1.5
    elif case == "extra_historical":
        inputs["historical_model_weights_by_model_id"][9] = 1
    elif case == "bool_historical_id":
        inputs["historical_model_weights_by_model_id"] = {True: 1}
    elif case == "list_stats":
        inputs["loss_moments_by_model_id"] = []
    elif case == "bad_stats":
        inputs["loss_moments_by_model_id"][1] = None
    elif case == "extra_stats":
        inputs["loss_moments_by_model_id"][9] = inputs["loss_moments_by_model_id"][8]
    elif case == "broken_stats":
        object.__setattr__(inputs["loss_moments_by_model_id"][1], "observed_loss_count", True)
    elif case == "list_isolation":
        inputs["isolation_until_by_model_id"] = []
    elif case == "bool_isolation":
        inputs["isolation_until_by_model_id"][1] = True
    elif case == "float_isolation":
        inputs["isolation_until_by_model_id"][1] = 4.5
    elif case == "negative_isolation":
        inputs["isolation_until_by_model_id"][1] = -1
    elif case == "extra_isolation":
        inputs["isolation_until_by_model_id"][9] = 5
    elif case == "late_bad_parameter":
        inputs["model_parameters_by_model_id"][1]["weight"] = torch.ones(2, dtype=torch.float64)
    before_parameters = deepcopy(inputs["model_parameters_by_model_id"])
    before_other = deepcopy(
        {k: v for k, v in inputs.items() if k != "model_parameters_by_model_id"}
    )
    before_rng = rng_snapshot()

    def arithmetic_forbidden(*args, **kwargs):
        pytest.fail("invalid inputs reached parameter arithmetic")

    monkeypatch.setattr(torch.Tensor, "__mul__", arithmetic_forbidden)
    with pytest.raises((TypeError, ValueError)):
        consolidate_feddrift_cached_model_clusters(**inputs)
    assert_parameters_equal(inputs["model_parameters_by_model_id"], before_parameters)
    assert {k: v for k, v in inputs.items() if k != "model_parameters_by_model_id"} == before_other
    assert_rng_equal(before_rng)


def test_server_empty_input_matches_real_merge():
    server = FedDriftServer(verbose=False)
    server._merge_cached_clusters(0, ())
    before = rng_snapshot()
    result = consolidate_feddrift_cached_model_clusters(
        timestep_index=0,
        model_clusters=(),
        model_parameters_by_model_id={},
        historical_model_weights_by_model_id={},
        loss_moments_by_model_id={},
        isolation_until_by_model_id={},
    )
    assert result.model_parameters_by_representative_id == server.global_models == {}
    assert result.loss_moments_by_representative_id == server.global_stats == {}
    assert result.model_weights_by_representative_id == server.model_weights == {}
    assert result.isolation_until_by_representative_id == server.isolated_until == {}
    assert result.model_id_mapping == result.cluster_weights_by_representative_id == {}
    assert_rng_equal(before)
