from federated_drift_experiment import config
from federated_drift_experiment.clustering import standardized_mean_increase
from federated_drift_experiment.servers import (
    FedDriftServer,
    FedSDANoCachedServer,
)


def _stats(values):
    return len(values), sum(values), sum(value * value for value in values)


def _two_model_stats(reference, shifted):
    return {
        0: {0: _stats(reference), 1: _stats(shifted)},
        1: {0: _stats(shifted), 1: _stats(reference)},
    }


def test_standardized_mean_increase_detects_clear_loss_increase():
    reference = _stats([0.1] * 10)
    shifted = _stats([0.3] * 10)

    assert standardized_mean_increase(shifted, reference) > 1.645
    assert standardized_mean_increase(reference, shifted) < 0.0


def test_confidence_decision_merges_uncertain_pair_ignored_by_distance(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_CLUSTERING_DECISION", "confidence")
    # 平均差はγを超えるが分散が大きく、差を十分に識別できない例。
    stats = _two_model_stats(
        [0.0, 0.2, 0.4, 0.6, 0.8] * 2,
        [0.1, 0.3, 0.5, 0.7, 0.9] * 2,
    )
    confidence_server = FedSDANoCachedServer(
        distance_threshold=0.05,
        linkage="complete",
        verbose=False,
    )
    distance_server = FedSDANoCachedServer(
        distance_threshold=0.05,
        linkage="complete",
        clustering_decision="distance",
        verbose=False,
    )

    assert confidence_server.perform_hierarchical_clustering([0, 1], stats) == [[0, 1]]
    assert distance_server.perform_hierarchical_clustering([0, 1], stats) == [[0], [1]]


def test_confidence_decision_keeps_clearly_distinct_pair(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_CLUSTERING_DECISION", "confidence")
    stats = _two_model_stats([0.1] * 10, [0.3] * 10)
    server = FedSDANoCachedServer(linkage="complete", verbose=False)

    assert server.perform_hierarchical_clustering([0, 1], stats) == [[0], [1]]


def test_feddrift_always_uses_paper_distance_decision(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_CLUSTERING_DECISION", "confidence")

    server = FedDriftServer(
        clustering_decision="confidence",
        distance_threshold=0.1,
        verbose=False,
    )

    assert server.clustering_decision == "distance"
