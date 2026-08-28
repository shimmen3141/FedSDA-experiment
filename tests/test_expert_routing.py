import math

import pytest
from federated_drift_experiment.expert_routing import (
    AdaHedgeRouter,
    PeriodicForwardProbeActiveSet,
    SwitchingExpertRouter,
)


def test_periodic_forward_probe_active_set_reactivates_every_cycle():
    active_set = PeriodicForwardProbeActiveSet(probe_samples=2)

    assert active_set.select([0, 1, 2], 0, 0) == ((0, 1, 2), True)
    active_set.observe({
        0: {"bounded_delta": 0.1, "zero_one_delta": 0.0},
        1: {"bounded_delta": -0.1, "zero_one_delta": 0.0},
        2: {"bounded_delta": 0.0, "zero_one_delta": 0.0},
    }, 0)
    assert active_set.select([0, 1, 2], 0, 1) == ((0, 1, 2), True)
    active_set.observe({
        0: {"bounded_delta": 0.1, "zero_one_delta": 0.0},
        1: {"bounded_delta": -0.1, "zero_one_delta": 0.0},
        2: {"bounded_delta": 0.0, "zero_one_delta": 0.0},
    }, 1)

    assert active_set.select([0, 1, 2], 0, 2) == ((0,), False)
    assert active_set.select([0, 1, 2], 0, 3) == ((0,), False)
    assert active_set.select([0, 1, 2], 0, 4) == ((0, 1, 2), True)
    assert active_set.apply_retained_global_model_count_sum == 2
    assert active_set.apply_global_model_count_sum == 6


def test_periodic_forward_probe_active_set_keeps_new_current_model():
    active_set = PeriodicForwardProbeActiveSet(probe_samples=1)
    active_set.select([0, 1], 0, 0)
    active_set.observe({
        0: {"bounded_delta": 0.0, "zero_one_delta": 0.0},
        1: {"bounded_delta": 0.0, "zero_one_delta": 0.0},
    }, 0)

    selected, is_probe = active_set.select([0, 1], 1, 1)

    assert selected == (1,)
    assert not is_probe


def test_periodic_forward_probe_active_set_restarts_probe_for_concept_change():
    active_set = PeriodicForwardProbeActiveSet(probe_samples=2)
    active_set.select([0, 1], 0, 0)
    active_set.select([0, 1], 0, 1)
    assert active_set.select([0, 1], 0, 2)[1] is False

    active_set.restart_for_concept([0, 1], sample_index=3)

    assert active_set.select([0, 1], 1, 3) == ((0, 1), True)
    assert active_set.select([0, 1], 1, 4) == ((0, 1), True)


def test_periodic_forward_probe_restarts_after_active_failure():
    active_set = PeriodicForwardProbeActiveSet(probe_samples=2)
    active_set.select([0, 1], 0, 0)
    active_set.observe({
        0: {"bounded_delta": 0.1, "zero_one_delta": 0.0},
        1: {"bounded_delta": -0.1, "zero_one_delta": 0.0},
    }, 0)
    active_set.select([0, 1], 0, 1)
    active_set.observe({
        0: {"bounded_delta": 0.1, "zero_one_delta": 0.0},
        1: {"bounded_delta": -0.1, "zero_one_delta": 0.0},
    }, 1)
    assert active_set.select([0, 1], 0, 2) == ((0,), False)

    assert active_set.restart_after_active_failure([0, 1], 3) is True
    assert active_set.select([0, 1], 0, 3) == ((0, 1), True)
    assert active_set.failure_probe_count == 1

    assert active_set.restart_after_active_failure([0, 1], 4) is False
    assert active_set.failure_probe_count == 1


def test_adahedge_starts_uniform_and_concentrates_on_better_expert():
    router = AdaHedgeRouter()

    initial = router.probabilities([0, 1])
    assert initial == {0: 0.5, 1: 0.5}

    for _ in range(8):
        probabilities = router.probabilities([0, 1])
        router.update({0: 0.1, 1: 0.9}, probabilities)

    learned = router.probabilities([0, 1])
    assert learned[0] > learned[1]
    assert math.isclose(sum(learned.values()), 1.0)


def test_adahedge_resets_when_expert_pool_changes():
    router = AdaHedgeRouter()
    probabilities = router.probabilities([0, 1])
    router.update({0: 0.0, 1: 1.0}, probabilities)

    reset = router.probabilities([0, 1, 2])

    assert reset == pytest.approx({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
    assert router.pool_reset_count == 1
    assert router.mixability_gap == 0.0


def test_effective_expert_count_measures_weight_concentration():
    assert AdaHedgeRouter.effective_expert_count({0: 1.0}) == 1.0
    assert AdaHedgeRouter.effective_expert_count(
        {0: 0.5, 1: 0.5}
    ) == 2.0


def test_adahedge_update_is_stable_for_large_learning_rate():
    router = AdaHedgeRouter()
    router.cumulative_losses = {0: 0.0, 1: 1e-12}
    router.mixability_gap = 1e-12
    probabilities = router.probabilities([0, 1])

    router.update({0: 1.0, 1: 1.0}, probabilities)

    assert math.isfinite(router.mixability_gap)


def test_adahedge_concept_restart_discards_old_losses():
    router = AdaHedgeRouter()
    probabilities = router.probabilities([0, 1])
    router.update({0: 0.0, 1: 1.0}, probabilities)

    router.restart_for_concept()

    assert router.probabilities([0, 1]) == {0: 0.5, 1: 0.5}
    assert router.concept_restart_count == 1
    assert router.pool_reset_count == 0


def test_adahedge_aggregation_restart_has_separate_counter():
    router = AdaHedgeRouter()
    probabilities = router.probabilities([0, 1])
    router.update({0: 0.0, 1: 1.0}, probabilities)

    router.restart_after_aggregation()

    assert router.probabilities([0, 1]) == {0: 0.5, 1: 0.5}
    assert router.aggregation_restart_count == 1
    assert router.aggregation_recalibration_count == 1
    assert router.concept_restart_count == 0


def test_adahedge_replays_recent_losses_after_aggregation():
    router = AdaHedgeRouter()
    probabilities = router.probabilities([0, 1])
    router.update({0: 0.0, 1: 1.0}, probabilities)

    router.replay_after_aggregation([
        {0: 0.9, 1: 0.1},
        {0: 0.8, 1: 0.2},
    ])

    replayed = router.probabilities([0, 1])
    assert replayed[1] > replayed[0]
    assert router.aggregation_restart_count == 0
    assert router.aggregation_recalibration_count == 1
    assert router.aggregation_recalibration_sample_count == 2


def test_adahedge_skips_replay_when_recent_leader_is_unchanged():
    router = AdaHedgeRouter()
    probabilities = router.probabilities([0, 1])
    router.update({0: 0.1, 1: 0.9}, probabilities)
    previous_losses = dict(router.cumulative_losses)
    previous_gap = router.mixability_gap

    replayed = router.replay_after_aggregation_if_leader_changed([
        {0: 0.2, 1: 0.8},
        {0: 0.1, 1: 0.9},
    ])

    assert replayed is False
    assert router.cumulative_losses == previous_losses
    assert router.mixability_gap == previous_gap
    assert router.aggregation_recalibration_check_count == 1
    assert router.aggregation_recalibration_skip_count == 1
    assert router.aggregation_recalibration_count == 0


def test_adahedge_replays_when_recent_leader_changes():
    router = AdaHedgeRouter()
    probabilities = router.probabilities([0, 1])
    router.update({0: 0.1, 1: 0.9}, probabilities)

    replayed = router.replay_after_aggregation_if_leader_changed([
        {0: 0.9, 1: 0.1},
        {0: 0.8, 1: 0.2},
    ])

    assert replayed is True
    assert router.probabilities([0, 1])[1] > router.probabilities([0, 1])[0]
    assert router.aggregation_recalibration_check_count == 1
    assert router.aggregation_recalibration_skip_count == 0
    assert router.aggregation_recalibration_count == 1
    assert router.aggregation_recalibration_sample_count == 2


def test_adahedge_replays_when_expert_pool_changed():
    router = AdaHedgeRouter()
    probabilities = router.probabilities([0, 1])
    router.update({0: 0.1, 1: 0.9}, probabilities)

    replayed = router.replay_after_aggregation_if_leader_changed([
        {0: 0.1, 1: 0.8, 2: 0.9},
    ])

    assert replayed is True
    assert set(router.cumulative_losses) == {0, 1, 2}
    assert router.aggregation_recalibration_check_count == 1
    assert router.aggregation_recalibration_count == 1


def test_adahedge_replays_when_challenger_wins_both_fifo_halves():
    router = AdaHedgeRouter()
    probabilities = router.probabilities([0, 1])
    router.update({0: 0.1, 1: 0.9}, probabilities)

    replayed = router.replay_after_aggregation_if_leader_persists([
        {0: 0.8, 1: 0.2},
        {0: 0.7, 1: 0.3},
        {0: 0.9, 1: 0.1},
        {0: 0.6, 1: 0.4},
    ])

    assert replayed is True
    assert router.probabilities([0, 1])[1] > router.probabilities([0, 1])[0]
    assert router.aggregation_recalibration_check_count == 1
    assert router.aggregation_recalibration_skip_count == 0
    assert router.aggregation_recalibration_count == 1


def test_adahedge_skips_nonpersistent_fifo_challenger():
    router = AdaHedgeRouter()
    probabilities = router.probabilities([0, 1])
    router.update({0: 0.1, 1: 0.9}, probabilities)
    previous_losses = dict(router.cumulative_losses)
    # 全体ではモデル1が良いが、前半では旧leaderのモデル0が良い。
    loss_sequence = [
        {0: 0.1, 1: 0.2},
        {0: 0.1, 1: 0.2},
        {0: 0.9, 1: 0.0},
        {0: 0.9, 1: 0.0},
    ]

    replayed = router.replay_after_aggregation_if_leader_persists(loss_sequence)

    assert replayed is False
    assert router.cumulative_losses == previous_losses
    assert router.aggregation_recalibration_check_count == 1
    assert router.aggregation_recalibration_skip_count == 1
    assert router.aggregation_recalibration_count == 0


def test_adahedge_persistent_replay_requires_two_nonempty_halves():
    router = AdaHedgeRouter()
    probabilities = router.probabilities([0, 1])
    router.update({0: 0.1, 1: 0.9}, probabilities)

    replayed = router.replay_after_aggregation_if_leader_persists([
        {0: 0.9, 1: 0.1},
    ])

    assert replayed is False
    assert router.aggregation_recalibration_skip_count == 1


def test_switching_router_tracks_a_changed_best_expert():
    router = SwitchingExpertRouter(share_horizon=30)

    for _ in range(40):
        probabilities = router.probabilities([0, 1])
        router.update({0: 0.0, 1: 1.0}, probabilities)
    assert router.probabilities([0, 1])[0] > 0.9

    for _ in range(40):
        probabilities = router.probabilities([0, 1])
        router.update({0: 1.0, 1: 0.0}, probabilities)
    assert router.probabilities([0, 1])[1] > 0.9
    assert router.leader_switch_count >= 1


def test_switching_router_leader_prefers_requested_expert_on_tie():
    probabilities = {"meta": 0.5, "switching": 0.5}

    assert SwitchingExpertRouter.leader(
        probabilities, preferred_id="meta"
    ) == "meta"


def test_switching_router_replays_fifo_after_aggregation():
    router = SwitchingExpertRouter(share_horizon=30)
    losses = ({0: 1.0, 1: 0.0},) * 20

    router.replay_after_aggregation(losses)

    assert router.probabilities([0, 1])[1] > 0.9
    assert router.aggregation_recalibration_count == 1
    assert router.aggregation_recalibration_sample_count == 20
