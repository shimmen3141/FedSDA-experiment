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


def build_valid_selection_inputs_for_rejection_tests():
    """異常検査用に独立した正常snapshotを作る。"""
    from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments
    from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics
    record = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=3, mean_loss=.25, sum_squared_loss_deviations=.1),
        class_loss_moments_by_class_id=((2, BoundedLossMoments(
            observed_loss_count=1, mean_loss=.75, sum_squared_loss_deviations=0.)),))
    server_record = ModelAndClassLossStatistics(
        overall_loss_moments=record.overall_loss_moments,
        class_loss_moments_by_class_id=record.class_loss_moments_by_class_id)
    return dict(local_model_loss_statistics=((-1, record),),
                model_id_mapping={-1:4}, server_model_loss_statistics=((4, server_record),))


@pytest.mark.parametrize("input_name,invalid_value", [
    ("local_model_loss_statistics", None),
    ("local_model_loss_statistics", []),
    ("local_model_loss_statistics", {}),
    ("local_model_loss_statistics", ([1, None],)),
    ("local_model_loss_statistics", ((1,),)),
    ("local_model_loss_statistics", ((1, None, None),)),
    ("local_model_loss_statistics", ((True, None),)),
    ("local_model_loss_statistics", ((1.0, None),)),
    ("local_model_loss_statistics", (("1", None),)),
    ("local_model_loss_statistics", ((1, object()),)),
    ("server_model_loss_statistics", []),
    ("server_model_loss_statistics", ((False, object()),)),
    ("server_model_loss_statistics", ((1, object()),)),
    ("model_id_mapping", None),
    ("model_id_mapping", []),
    ("model_id_mapping", {True:2}),
    ("model_id_mapping", {1:False}),
    ("model_id_mapping", {1.0:2}),
    ("model_id_mapping", {1:2.0}),
    ("model_id_mapping", {"1":2}),
    ("model_id_mapping", {1:"2"}),
])
def test_mapping_selection_rejects_invalid_shapes_and_types(input_name, invalid_value):
    from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import select_loss_statistics_after_model_id_mapping
    inputs = build_valid_selection_inputs_for_rejection_tests()
    inputs[input_name] = invalid_value
    with pytest.raises((TypeError, ValueError), match=input_name):
        select_loss_statistics_after_model_id_mapping(**inputs)


@pytest.mark.parametrize("input_name", ["local_model_loss_statistics", "server_model_loss_statistics"])
@pytest.mark.parametrize("invalid_kind", ["duplicate_id", "late_bad_id", "bad_mean", "bad_count", "bad_m2", "bad_class_id", "bad_class_moment", "bad_class_tuple"])
def test_mapping_selection_validates_unused_and_late_statistics(input_name, invalid_kind):
    """負ける候補や補完不要serverも全検査し、入力の値を変えない。"""
    from copy import deepcopy
    from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import select_loss_statistics_after_model_id_mapping
    inputs = build_valid_selection_inputs_for_rejection_tests()
    record = inputs[input_name][0][1]
    if invalid_kind == "duplicate_id":
        inputs[input_name] = ((4,record),(4,record))
    elif invalid_kind == "late_bad_id":
        inputs[input_name] = ((4,record),(False,record))
    else:
        if invalid_kind == "bad_mean":
            object.__setattr__(record.overall_loss_moments, "mean_loss", float("inf"))
        elif invalid_kind == "bad_count":
            object.__setattr__(record.overall_loss_moments, "observed_loss_count", -1)
        elif invalid_kind == "bad_m2":
            object.__setattr__(record.overall_loss_moments, "sum_squared_loss_deviations", -1.)
        elif invalid_kind == "bad_class_id":
            object.__setattr__(record, "class_loss_moments_by_class_id",
                               ((-1,record.class_loss_moments_by_class_id[0][1]),))
        elif invalid_kind == "bad_class_moment":
            object.__setattr__(record.class_loss_moments_by_class_id[0][1], "mean_loss", 1.1)
        else:
            object.__setattr__(record, "class_loss_moments_by_class_id", [])
    before = deepcopy(inputs)
    with pytest.raises((TypeError, ValueError), match=input_name):
        select_loss_statistics_after_model_id_mapping(**inputs)
    assert inputs == before


def test_mapping_selection_validates_unused_mapping_entries():
    from copy import deepcopy
    from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import select_loss_statistics_after_model_id_mapping
    inputs = build_valid_selection_inputs_for_rejection_tests()
    inputs["model_id_mapping"][999] = False
    before = deepcopy(inputs)
    with pytest.raises(TypeError, match="model_id_mapping"):
        select_loss_statistics_after_model_id_mapping(**inputs)
    assert inputs == before


def test_mapping_selection_rejects_snapshot_and_mapping_subclasses():
    from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import select_loss_statistics_after_model_id_mapping
    class TupleSubclass(tuple):
        pass
    class DictSubclass(dict):
        pass
    inputs = build_valid_selection_inputs_for_rejection_tests()
    with pytest.raises(TypeError, match="local_model_loss_statistics"):
        select_loss_statistics_after_model_id_mapping(**{**inputs,
            "local_model_loss_statistics":TupleSubclass(inputs["local_model_loss_statistics"])})
    with pytest.raises(TypeError, match="model_id_mapping"):
        select_loss_statistics_after_model_id_mapping(**{**inputs,
            "model_id_mapping":DictSubclass(inputs["model_id_mapping"])})


def test_mapping_selection_returns_deeply_independent_frozen_results():
    from dataclasses import FrozenInstanceError
    from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import select_loss_statistics_after_model_id_mapping
    inputs = build_valid_selection_inputs_for_rejection_tests()
    # サーバのみのIDも返却する。両入力が同じrecordでも結果は独立する。
    source = inputs["local_model_loss_statistics"][0][1]
    inputs["server_model_loss_statistics"] = ((8,source),)
    first = select_loss_statistics_after_model_id_mapping(**inputs)
    second = select_loss_statistics_after_model_id_mapping(**inputs)
    assert first == second and tuple(mid for mid, record in first) == (4,8)
    for (_, record),(_, other) in zip(first, second):
        assert record is not source and record is not other
        assert record.overall_loss_moments is not source.overall_loss_moments
        assert record.overall_loss_moments is not other.overall_loss_moments
        assert record.class_loss_moments_by_class_id[0][1] is not source.class_loss_moments_by_class_id[0][1]
    with pytest.raises(FrozenInstanceError):
        first[0][1].overall_loss_moments.mean_loss = .99
    with pytest.raises(TypeError):
        first[0] = (99,source)
    object.__setattr__(source.overall_loss_moments,"mean_loss",.99)
    object.__setattr__(first[0][1].class_loss_moments_by_class_id[0][1],"mean_loss",.01)
    inputs["model_id_mapping"].clear()
    assert first[0][1].overall_loss_moments.mean_loss == .25
    assert first[1][1].class_loss_moments_by_class_id[0][1].mean_loss == .75
    assert second[0][1].class_loss_moments_by_class_id[0][1].mean_loss == .75


def test_mapping_selection_preserves_shared_environment_and_keyword_contract():
    import random
    import numpy as np
    import torch
    from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import select_loss_statistics_after_model_id_mapping
    inputs = build_valid_selection_inputs_for_rejection_tests()
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    grad = torch.is_grad_enabled()
    with torch.device("meta"):
        device = torch.empty(0).device
        select_loss_statistics_after_model_id_mapping(**inputs)
        assert torch.empty(0).device == device
        with pytest.raises(TypeError):
            select_loss_statistics_after_model_id_mapping(
                inputs["local_model_loss_statistics"], inputs["model_id_mapping"])
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1],after[1]) and np_state[2:] == after[2:]
    assert torch.equal(torch.random.get_rng_state(),torch_state)
    assert torch.get_default_dtype() == dtype and torch.is_grad_enabled() == grad


def test_mapping_selection_connects_store_updates_and_baseline_selection():
    from types import SimpleNamespace
    from federated_drift_experiment.clients.base import BaseClient
    from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore
    from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import select_loss_statistics_after_model_id_mapping
    from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
        select_loss_monitoring_baseline_mean_loss,
        select_alarm_interval_reuse_baseline_mean_loss,
        select_post_alarm_reference_historical_mean_loss)
    inputs = build_valid_selection_inputs_for_rejection_tests()
    source_store = ModelAndClassLossStatisticsStore(
        initial_loss_statistics_by_model_id=dict(inputs["local_model_loss_statistics"]))
    before = source_store.get_state_snapshot()
    result = select_loss_statistics_after_model_id_mapping(
        local_model_loss_statistics=before, model_id_mapping=inputs["model_id_mapping"],
        server_model_loss_statistics=inputs["server_model_loss_statistics"])
    mapped_store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id=dict(result))
    assert mapped_store.get_model_loss_statistics(model_id=-1) is None
    assert tuple(mid for mid,record in mapped_store.get_state_snapshot()) == (4,)
    old = SimpleNamespace(model_stats={4:{"n":3,"mean":.25,"M2":.1,
        "class_stats":{2:{"n":1,"mean":.75,"M2":0.}}}},
        _update_running_stats=BaseClient._update_running_stats)
    for loss,class_id in [(.8,2),(.1,0),(.3,None)]:
        mapped_store.record_assigned_loss(model_id=4,observed_loss=loss,observed_class_id=class_id)
        BaseClient._update_model_stats(old,4,loss,class_id=class_id)
        record = mapped_store.get_model_loss_statistics(model_id=4)
        assert (record.overall_loss_moments.observed_loss_count,
                record.overall_loss_moments.mean_loss,
                record.overall_loss_moments.sum_squared_loss_deviations) == (
                    old.model_stats[4]["n"],old.model_stats[4]["mean"],old.model_stats[4]["M2"])
        assert tuple(cid for cid,m in record.class_loss_moments_by_class_id) == tuple(old.model_stats[4]["class_stats"])
        for cid,moments in record.class_loss_moments_by_class_id:
            legacy = old.model_stats[4]["class_stats"][cid]
            assert (moments.observed_loss_count,moments.mean_loss,moments.sum_squared_loss_deviations) == (
                legacy["n"],legacy["mean"],legacy["M2"])
    for selector in [select_loss_monitoring_baseline_mean_loss,
                     select_alarm_interval_reuse_baseline_mean_loss,
                     select_post_alarm_reference_historical_mean_loss]:
        assert selector(loss_moments=record.overall_loss_moments) == record.overall_loss_moments.mean_loss
    assert source_store.get_state_snapshot() == before
