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
@pytest.mark.parametrize(
    "loss_sequence",
    [
        (
            (-100, 0.2, 0),
            (-100, 0.7, None),
            (-100, 0.5, 1),
            (5, 0.1, 0),
            (-100, 1.0, 0),
            (4, 0.0, 1),
        ),
        (
            (5, 0.0, None),
            (-100, 1e-15, 1),
            (-100, 2e-15, 0),
            (-100, 0.0, None),
            (5, 1.0, 1),
            (5, 0.5, None),
        ),
    ],
)
def test_model_class_statistics_matches_legacy_updates(seed, loss_sequence):
    legacy_client = SimpleNamespace(
        model_stats={}, _update_running_stats=BaseClient._update_running_stats
    )
    if seed is None:
        store = ModelAndClassLossStatisticsStore()
    else:
        legacy_client.model_stats[-100] = {
            "n": seed,
            "mean": 0.25,
            "M2": 0.1 if seed == 1 else 0.0,
            "class_stats": {0: {"n": 1, "mean": 0.25, "M2": 0.0}},
        }
        store = ModelAndClassLossStatisticsStore(
            initial_loss_statistics_by_model_id={
                -100: ModelAndClassLossStatistics(
                    overall_loss_moments=BoundedLossMoments(
                        observed_loss_count=seed,
                        mean_loss=0.25,
                        sum_squared_loss_deviations=0.1 if seed == 1 else 0.0,
                    ),
                    class_loss_moments_by_class_id=(
                        (
                            0,
                            BoundedLossMoments(
                                observed_loss_count=1,
                                mean_loss=0.25,
                                sum_squared_loss_deviations=0.0,
                            ),
                        ),
                    ),
                ),
            }
        )
    for model_id, observed_loss, observed_class_id in loss_sequence:
        assert (
            store.record_assigned_loss(
                model_id=model_id, observed_loss=observed_loss, observed_class_id=observed_class_id
            )
            is None
        )
        BaseClient._update_model_stats(
            legacy_client, model_id, observed_loss, class_id=observed_class_id
        )
        snapshot = store.get_state_snapshot()
        assert tuple(model_id for model_id, result in snapshot) == tuple(legacy_client.model_stats)
        for model_id, result in snapshot:
            legacy_stats = legacy_client.model_stats[model_id]
            assert (
                result.overall_loss_moments.observed_loss_count,
                result.overall_loss_moments.mean_loss,
                result.overall_loss_moments.sum_squared_loss_deviations,
            ) == (legacy_stats["n"], legacy_stats["mean"], legacy_stats["M2"])
            assert tuple(
                class_id for class_id, class_loss_moments in result.class_loss_moments_by_class_id
            ) == tuple(legacy_stats.get("class_stats", {}))
            for class_id, class_loss_moments in result.class_loss_moments_by_class_id:
                legacy_class_stats = legacy_stats["class_stats"][class_id]
                assert (
                    class_loss_moments.observed_loss_count,
                    class_loss_moments.mean_loss,
                    class_loss_moments.sum_squared_loss_deviations,
                ) == (legacy_class_stats["n"], legacy_class_stats["mean"], legacy_class_stats["M2"])


def test_model_class_statistics_accepts_and_replaces_seeds():
    seed = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.1
        ),
        class_loss_moments_by_class_id=(
            (
                0,
                BoundedLossMoments(
                    observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.0
                ),
            ),
        ),
    )
    store = ModelAndClassLossStatisticsStore(
        initial_loss_statistics_by_model_id={
            -100: seed,
            4: ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
                )
            ),
        }
    )
    assert store.get_model_loss_statistics(model_id=99) is None
    assert store.get_model_loss_statistics(model_id=4).overall_loss_moments.observed_loss_count == 0
    assert store.get_model_loss_statistics(model_id=-100) == seed
    seed = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=10, mean_loss=0.3, sum_squared_loss_deviations=0.0
        ),
        class_loss_moments_by_class_id=(
            (
                1,
                BoundedLossMoments(
                    observed_loss_count=20, mean_loss=0.4, sum_squared_loss_deviations=0.2
                ),
            ),
        ),
    )
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
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.1
        ),
        class_loss_moments_by_class_id=(
            (
                1,
                BoundedLossMoments(
                    observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.0
                ),
            ),
        ),
    )
    store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={-100: seed})
    other_store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={-100: seed})
    result = store.get_model_loss_statistics(model_id=-100)
    snapshot = store.get_state_snapshot()
    assert result == snapshot[0][1] == seed
    assert result is not snapshot[0][1] and result is not seed
    assert result.overall_loss_moments is not seed.overall_loss_moments
    assert result.overall_loss_moments is not snapshot[0][1].overall_loss_moments
    assert (
        result.class_loss_moments_by_class_id[0][1] is not seed.class_loss_moments_by_class_id[0][1]
    )
    assert (
        result.class_loss_moments_by_class_id[0][1]
        is not snapshot[0][1].class_loss_moments_by_class_id[0][1]
    )
    store.record_assigned_loss(model_id=-100, observed_loss=0.75, observed_class_id=0)
    assert result == seed and snapshot == ((-100, seed),)
    assert other_store.get_state_snapshot() == snapshot
    assert tuple(
        class_id
        for class_id, class_loss_moments in store.get_model_loss_statistics(
            model_id=-100
        ).class_loss_moments_by_class_id
    ) == (1, 0)
    assert ModelAndClassLossStatisticsStore().get_state_snapshot() == ()
    # 入力・返却値の改変も所有者間で伝播しない。
    from dataclasses import FrozenInstanceError

    seed = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.1
        ),
        class_loss_moments_by_class_id=(
            (
                0,
                BoundedLossMoments(
                    observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0
                ),
            ),
        ),
    )
    loss_statistics = {7: seed}
    store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id=loss_statistics)
    other_store = ModelAndClassLossStatisticsStore(
        initial_loss_statistics_by_model_id=loss_statistics
    )
    state_before_call = store.get_state_snapshot()
    loss_statistics.clear()
    object.__setattr__(seed.overall_loss_moments, "mean_loss", 0.9)
    object.__setattr__(seed.class_loss_moments_by_class_id[0][1], "mean_loss", 0.8)
    assert store.get_state_snapshot() == state_before_call
    assert other_store.get_state_snapshot() == state_before_call
    snapshot = store.get_model_loss_statistics(model_id=7)
    with pytest.raises(FrozenInstanceError):
        snapshot.overall_loss_moments = None
    with pytest.raises(FrozenInstanceError):
        snapshot.overall_loss_moments.mean_loss = 0.8
    object.__setattr__(snapshot.overall_loss_moments, "mean_loss", 0.7)
    object.__setattr__(snapshot.class_loss_moments_by_class_id[0][1], "mean_loss", 0.6)
    snapshot = store.get_state_snapshot()
    object.__setattr__(snapshot[0][1].overall_loss_moments, "mean_loss", 0.4)
    object.__setattr__(snapshot[0][1].class_loss_moments_by_class_id[0][1], "mean_loss", 0.3)
    assert store.get_state_snapshot() == state_before_call
    store.record_assigned_loss(model_id=7, observed_loss=0.5, observed_class_id=0)
    assert other_store.get_state_snapshot() == state_before_call


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("model_id", True),
        ("model_id", 1.0),
        ("model_id", "7"),
        ("observed_class_id", True),
        ("observed_class_id", -1),
        ("observed_class_id", 0.5),
        ("observed_class_id", object()),
        ("observed_loss", True),
        ("observed_loss", -0.1),
        ("observed_loss", 1.1),
        ("observed_loss", float("nan")),
        ("observed_loss", float("inf")),
        ("observed_loss", 10**400),
        ("observed_loss", []),
        ("overall_loss_moments", None),
        ("class_loss_moments_by_class_id", []),
        ("class_loss_moments_by_class_id", ((-1, None),)),
        ("class_loss_moments_by_class_id", ((),)),
        ("class_loss_moments_by_class_id", ([0, None],)),
        ("class_loss_moments_by_class_id", ((0, None),)),
        (
            "class_loss_moments_by_class_id",
            (
                (
                    0,
                    BoundedLossMoments(
                        observed_loss_count=0, mean_loss=0, sum_squared_loss_deviations=0
                    ),
                ),
                (
                    0,
                    BoundedLossMoments(
                        observed_loss_count=0, mean_loss=0, sum_squared_loss_deviations=0
                    ),
                ),
            ),
        ),
        ("observed_loss_count", True),
        ("observed_loss_count", -1),
        ("mean_loss", float("nan")),
        ("sum_squared_loss_deviations", -0.1),
        ("initial_loss_statistics_by_model_id", []),
        ("initial_loss_statistics_by_model_id", {True: None}),
        ("loss_statistics", None),
        ("overall_count_overflow", None),
        ("class_count_overflow", None),
    ],
)
def test_model_class_statistics_rejects_invalid_input_atomically(field_name, invalid_value):
    """空store・既存store・seed拒否と後段class失敗で全状態を維持する。"""
    if field_name in ("overall_count_overflow", "class_count_overflow"):
        observed_loss_count = int(float.fromhex("0x1.fffffffffffffp+1023")) + 2**970 - 1
        seed = ModelAndClassLossStatistics(
            overall_loss_moments=BoundedLossMoments(
                observed_loss_count=observed_loss_count
                if field_name == "overall_count_overflow"
                else 1,
                mean_loss=0.25,
                sum_squared_loss_deviations=0.1,
            ),
            class_loss_moments_by_class_id=(
                (
                    0,
                    BoundedLossMoments(
                        observed_loss_count=observed_loss_count
                        if field_name == "class_count_overflow"
                        else 1,
                        mean_loss=0.25,
                        sum_squared_loss_deviations=0.0,
                    ),
                ),
            ),
        )
        store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={7: seed})
        state_before_call = store.get_state_snapshot()
        with pytest.raises(ValueError, match="observed_loss_count"):
            store.record_assigned_loss(model_id=7, observed_loss=0.5, observed_class_id=0)
        assert store.get_state_snapshot() == state_before_call
        return
    for seed in (
        None,
        {
            7: ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.1
                )
            )
        },
    ):
        store = ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id=seed)
        state_before_call = store.get_state_snapshot()
        if field_name in ("model_id", "observed_loss", "observed_class_id"):
            with pytest.raises((TypeError, ValueError), match=field_name):
                store.record_assigned_loss(
                    **(
                        dict(model_id=7, observed_loss=0.5, observed_class_id=0)
                        | {field_name: invalid_value}
                    )
                )
            if field_name == "model_id":
                with pytest.raises((TypeError, ValueError), match="model_id"):
                    store.get_model_loss_statistics(model_id=invalid_value)
                with pytest.raises((TypeError, ValueError), match="model_id"):
                    store.set_model_loss_statistics(
                        model_id=invalid_value,
                        loss_statistics=ModelAndClassLossStatistics(
                            overall_loss_moments=BoundedLossMoments(
                                observed_loss_count=0, mean_loss=0, sum_squared_loss_deviations=0
                            )
                        ),
                    )
        elif field_name == "initial_loss_statistics_by_model_id":
            with pytest.raises((TypeError, ValueError)):
                ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id=invalid_value)
        else:
            loss_statistics = ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.1
                )
            )
            if field_name == "loss_statistics":
                loss_statistics = invalid_value
            elif field_name in ("observed_loss_count", "mean_loss", "sum_squared_loss_deviations"):
                object.__setattr__(loss_statistics.overall_loss_moments, field_name, invalid_value)
            else:
                object.__setattr__(loss_statistics, field_name, invalid_value)
            with pytest.raises((TypeError, ValueError)):
                store.set_model_loss_statistics(model_id=9, loss_statistics=loss_statistics)
            with pytest.raises((TypeError, ValueError)):
                ModelAndClassLossStatisticsStore(
                    initial_loss_statistics_by_model_id={9: loss_statistics}
                )
        assert store.get_state_snapshot() == state_before_call
    if field_name == "observed_class_id" and type(invalid_value) is object:
        # LEGACY-006では旧全体だけが更新済みになる。同入力の新storeは上記で非変更。
        legacy_client = SimpleNamespace(
            model_stats={}, _update_running_stats=BaseClient._update_running_stats
        )
        with pytest.raises(TypeError):
            BaseClient._update_model_stats(legacy_client, 7, 0.5, class_id=invalid_value)
        assert legacy_client.model_stats[7]["n"] == 1
        assert legacy_client.model_stats[7]["mean"] == 0.5


def test_model_class_statistics_connects_to_baseline_and_monitor():
    """全体とclassの平均が異なっても全体を明示して監視・参照に接続する。"""
    from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
        select_available_reference_within_historical_loss_tolerance,
    )
    from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
        LossChangeDetectionSettings,
    )
    from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
        OverallAndTrueClassLossMonitor,
    )
    from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
        select_loss_monitoring_baseline_mean_loss,
        select_post_alarm_reference_historical_mean_loss,
    )

    store = ModelAndClassLossStatisticsStore()
    store.record_assigned_loss(model_id=7, observed_loss=0.2, observed_class_id=None)
    store.record_assigned_loss(model_id=7, observed_loss=0.8, observed_class_id=0)
    snapshot = store.get_model_loss_statistics(model_id=7)
    assert snapshot.overall_loss_moments.mean_loss == 0.5
    assert snapshot.class_loss_moments_by_class_id[0][1].mean_loss == 0.8
    baseline_mean_loss = select_loss_monitoring_baseline_mean_loss(
        loss_moments=snapshot.overall_loss_moments
    )
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=LossChangeDetectionSettings(
            drift_detector_name="e_sr",
            loss_monitoring_scope="overall_and_true_class_losses",
            e_sr_false_alarm_control_alpha=0.01,
        ),
        class_count=2,
        initial_baseline_loss_mean=baseline_mean_loss,
        maximum_retained_candidate_count=5,
        betting_fractions=(0.1, 0.4),
    )
    observation = monitor.observe_loss_after_label_observation(
        observed_loss=0.5,
        observed_class_id=0,
        sample_index=0,
        current_model_baseline_loss_mean=baseline_mean_loss,
    )
    assert observation.component_update_count == 2
    assert monitor.get_state_snapshot().class_esr_states_by_class_id[0][1].baseline_loss_mean == 0.5
    state_before_call = store.get_state_snapshot()
    reference_historical_mean_losses_by_model_id = {
        7: select_post_alarm_reference_historical_mean_loss(
            loss_moments=snapshot.overall_loss_moments
        )
    }
    reference_losses_by_model_id = {7: (0.4, 0.5)}
    assert (
        select_available_reference_within_historical_loss_tolerance(
            reference_losses_by_model_id=reference_losses_by_model_id,
            reference_historical_mean_losses_by_model_id=reference_historical_mean_losses_by_model_id,
            available_reference_model_ids=(7,),
            current_training_model_id=7,
            maximum_reference_mean_loss_increase=0.1,
        )
        == 7
    )
    assert store.get_state_snapshot() == state_before_call


def test_model_class_statistics_preserves_shared_state():
    """統計保存・更新・参照で共有乱数・既定dtype/deviceを変更しない。"""
    import random

    import numpy as np
    import torch

    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    global_torch_random_state = torch.get_rng_state().clone()
    global_default_dtype = torch.get_default_dtype()
    global_default_device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            store = ModelAndClassLossStatisticsStore()
            store.set_model_loss_statistics(
                model_id=-100,
                loss_statistics=ModelAndClassLossStatistics(
                    overall_loss_moments=BoundedLossMoments(
                        observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.1
                    )
                ),
            )
            store.record_assigned_loss(model_id=-100, observed_loss=0.75, observed_class_id=0)
            assert (
                store.get_model_loss_statistics(
                    model_id=-100
                ).overall_loss_moments.sum_squared_loss_deviations
                == 0.225
            )
            assert store.get_state_snapshot()
            assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_device() == torch.device("meta")
    finally:
        torch.set_default_dtype(global_default_dtype)
    assert random.getstate() == global_python_random_state
    assert np.random.get_state()[0] == global_numpy_random_state[0]
    assert np.array_equal(np.random.get_state()[1], global_numpy_random_state[1])
    assert np.random.get_state()[2:] == global_numpy_random_state[2:]
    assert torch.equal(torch.get_rng_state(), global_torch_random_state)
    assert torch.get_default_dtype() == global_default_dtype
    assert torch.get_default_device() == global_default_device


def test_model_class_statistics_uses_keyword_arguments():
    """ID・クラス・損失とseedを位置引数で混同できない。"""
    store = ModelAndClassLossStatisticsStore()
    loss_statistics = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=0, mean_loss=0, sum_squared_loss_deviations=0
        )
    )
    with pytest.raises(TypeError):
        ModelAndClassLossStatistics(loss_statistics.overall_loss_moments)
    with pytest.raises(TypeError):
        ModelAndClassLossStatisticsStore({})
    with pytest.raises(TypeError):
        store.set_model_loss_statistics(7, loss_statistics)
    with pytest.raises(TypeError):
        store.record_assigned_loss(7, 0.2, 0)
    with pytest.raises(TypeError):
        store.get_model_loss_statistics(7)
