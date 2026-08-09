import math

import pytest

from federated_drift_experiment.expert_routing import AdaHedgeRouter


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
    assert router.concept_restart_count == 0
