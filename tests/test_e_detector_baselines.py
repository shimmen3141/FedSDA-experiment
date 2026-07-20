from federated_drift_experiment.e_detector_baselines import (
    EmpiricalBernsteinUCB,
    HistoricalMeanBaseline,
    make_baseline_estimator,
)
from federated_drift_experiment.experiment import MODE_SPECS


def test_historical_mean_strategy_preserves_existing_baseline():
    estimator = HistoricalMeanBaseline()
    assert estimator.estimate({"n": 10, "mean": 0.2, "M2": 0.5}) == 0.2


def test_empirical_bernstein_ucb_is_above_mean_and_shrinks_with_n():
    estimator = EmpiricalBernsteinUCB(beta=0.05)
    small = estimator.estimate({"n": 20, "mean": 0.2, "M2": 0.8})
    large = estimator.estimate({"n": 200, "mean": 0.2, "M2": 8.0})
    assert 0.2 < large < small <= 1.0
    assert estimator.estimate({"n": 1, "mean": 0.2, "M2": 0.0}) == 1.0 - 1e-6


def test_baseline_strategy_factory_rejects_unknown_name():
    try:
        make_baseline_estimator("unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("未知の戦略名を受理してはいけません")


def test_ucb_modes_inject_strategy_without_changing_mean_modes():
    assert MODE_SPECS["FedSDA_NoCached_ESR"].client_kwargs == {}
    assert MODE_SPECS["FedSDA_Cached_ClassESR"].client_kwargs == {}
    for mode in (
        "FedSDA_NoCached_ESR_UCB", "FedSDA_NoCached_ClassESR_UCB",
        "FedSDA_Cached_ESR_UCB", "FedSDA_Cached_ClassESR_UCB",
    ):
        assert MODE_SPECS[mode].client_kwargs == {
            "baseline_strategy": "empirical_bernstein_ucb"
        }
