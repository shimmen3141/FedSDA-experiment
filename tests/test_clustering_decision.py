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


def test_confidence_margin_ignores_precise_but_practically_small_difference(
    monkeypatch,
):
    monkeypatch.setattr(
        config, "FEDSDA_CLUSTERING_DECISION", "confidence_margin"
    )
    # 低分散なのでゼロとの差は明確だが、平均差は実用許容幅γ未満。
    stats = _two_model_stats([0.10] * 20, [0.12] * 20)
    margin_server = FedSDANoCachedServer(
        distance_threshold=0.05,
        linkage="complete",
        verbose=False,
    )
    confidence_server = FedSDANoCachedServer(
        distance_threshold=0.05,
        linkage="complete",
        clustering_decision="confidence",
        verbose=False,
    )

    assert margin_server.perform_hierarchical_clustering([0, 1], stats) == [[0, 1]]
    assert confidence_server.perform_hierarchical_clustering([0, 1], stats) == [[0], [1]]


def test_confidence_margin_keeps_difference_clearly_above_margin(monkeypatch):
    monkeypatch.setattr(
        config, "FEDSDA_CLUSTERING_DECISION", "confidence_margin"
    )
    stats = _two_model_stats([0.1] * 10, [0.3] * 10)
    server = FedSDANoCachedServer(
        distance_threshold=0.1,
        linkage="complete",
        verbose=False,
    )

    assert server.perform_hierarchical_clustering([0, 1], stats) == [[0], [1]]


def test_feddrift_always_uses_paper_distance_decision(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_CLUSTERING_DECISION", "confidence")

    server = FedDriftServer(
        clustering_decision="confidence",
        distance_threshold=0.1,
        verbose=False,
    )

    assert server.clustering_decision == "distance"


def test_fedsda_collects_pair_prediction_complementarity():
    class DiagnosticClient:
        def get_held_model_ids(self):
            return [0, 1]

        def evaluate_model(self, params, target_model_id):
            return 10, 1.0, 0.1

        def evaluate_model_diagnostics(self, params, target_model_id):
            return (10, 1.0, 0.1), {
                "n": 10,
                "candidate_only_correct": 2,
                "target_only_correct": 1,
                "both_correct": 6,
                "both_wrong": 1,
            }

    server = FedSDANoCachedServer(verbose=False)
    server.global_models = {0: object(), 1: object()}
    server.clients = [DiagnosticClient()]

    server._cross_evaluate([0, 1], send_model_params=False)
    summary = server.pair_diagnostic_summary()

    assert summary["model_pair_evaluation_count"] == 2
    assert summary["model_pair_sample_count"] == 20
    assert summary["model_pair_correctness_disagreement_rate"] == 0.3
    assert summary["model_pair_oracle_gain_rate"] == 0.1
    assert summary["model_pair_both_correct_rate"] == 0.6
    assert len(server.cross_evaluation_diagnostics) == 4
    cross = server.cross_evaluation_diagnostics[1]
    assert cross["round_index"] == -1
    assert cross["candidate_model_id"] == 0
    assert cross["target_model_id"] == 1
    assert cross["candidate_only_correct"] == 2
