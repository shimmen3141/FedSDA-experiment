from federated_drift_experiment import config
import math

import torch

from federated_drift_experiment.clustering import (
    paired_mean_upper_bound,
    standardized_mean_increase,
)
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


def test_paired_mean_upper_bound_uses_loss_difference_variance():
    assert math.isclose(
        paired_mean_upper_bound(_stats([0.01] * 10), 0.95), 0.01
    )
    assert paired_mean_upper_bound(_stats([-0.02] * 10), 0.95) < 0.0


def test_noninferiority_merge_rejects_cluster_harmful_to_one_member(monkeypatch):
    monkeypatch.setattr(
        config, "FEDSDA_CLUSTERING_CONSOLIDATION", "noninferiority_merge"
    )
    monkeypatch.setattr(config, "FEDSDA_MERGE_NONINFERIORITY_MARGIN", 0.0)

    class PairedClient:
        def get_held_model_ids(self):
            return {0, 1}

        def evaluate_model_loss_difference(
            self, candidate_params, reference_params, target_model_id
        ):
            differences = [0.0] * 10 if target_model_id == 0 else [0.2] * 10
            return _stats(differences)

    server = FedSDANoCachedServer(verbose=False)
    server.clients = [PairedClient()]
    server.global_models = {
        0: {"weight": torch.tensor([0.0])},
        1: {"weight": torch.tensor([1.0])},
    }

    clusters, consolidation_params = server._validate_noninferiority_clusters(
        50, [[0, 1]], _two_model_stats([0.1] * 10, [0.3] * 10)
    )

    assert clusters == [[0], [1]]
    assert consolidation_params == {}
    assert server.noninferiority_summary() == {
        "clustering_noninferiority_candidate_count": 1,
        "clustering_noninferiority_accepted_count": 0,
        "clustering_noninferiority_rejected_count": 1,
        "clustering_noninferiority_comparison_count": 2,
        "clustering_noninferiority_sample_count": 20,
        "clustering_noninferiority_acceptance_rate": 0.0,
    }
    assert server.comm_models_down == 2


def test_noninferiority_merge_accepts_cluster_safe_for_every_member(monkeypatch):
    monkeypatch.setattr(
        config, "FEDSDA_CLUSTERING_CONSOLIDATION", "noninferiority_merge"
    )
    monkeypatch.setattr(config, "FEDSDA_MERGE_NONINFERIORITY_MARGIN", 0.01)

    class PairedClient:
        def get_held_model_ids(self):
            return {0, 1}

        def evaluate_model_loss_difference(
            self, candidate_params, reference_params, target_model_id
        ):
            return _stats([0.005] * 10)

    server = FedSDANoCachedServer(verbose=False)
    server.clients = [PairedClient()]
    server.global_models = {
        0: {"weight": torch.tensor([0.0])},
        1: {"weight": torch.tensor([1.0])},
    }

    clusters, consolidation_params = server._validate_noninferiority_clusters(
        50, [[0, 1]], _two_model_stats([0.1] * 10, [0.1] * 10)
    )
    assert clusters == [[0, 1]]
    assert torch.equal(consolidation_params[0]["weight"], torch.tensor([0.0]))
    assert server.noninferiority_summary()[
        "clustering_noninferiority_acceptance_rate"
    ] == 1.0


def test_noninferiority_merge_reuses_cross_evaluation_statistics(monkeypatch):
    monkeypatch.setattr(
        config, "FEDSDA_CLUSTERING_CONSOLIDATION", "noninferiority_merge"
    )
    monkeypatch.setattr(config, "FEDSDA_MERGE_NONINFERIORITY_MARGIN", 0.01)

    class CachedClient:
        def get_held_model_ids(self):
            return {0, 1}

        def evaluate_model_loss_difference(self, *args, **kwargs):
            raise AssertionError("クロス評価済みのモデルを再送してはならない")

    server = FedSDANoCachedServer(verbose=False)
    server.clients = [CachedClient()]
    server.global_models = {
        0: {"weight": torch.tensor([0.0])},
        1: {"weight": torch.tensor([1.0])},
    }
    server._last_paired_loss_difference_stats = {
        (0, 0): _stats([0.0] * 10),
        (0, 1): _stats([0.005] * 10),
    }

    clusters, _ = server._validate_noninferiority_clusters(
        50, [[0, 1]], _two_model_stats([0.1] * 10, [0.1] * 10)
    )

    assert clusters == [[0, 1]]
    assert server.comm_models_down == 0
    assert server.comm_messages_down == 0


def test_noninferiority_representative_minimizes_worst_mean_loss_increase():
    stats = {
        0: {0: _stats([0.1] * 10), 1: _stats([0.3] * 10)},
        1: {0: _stats([0.12] * 10), 1: _stats([0.1] * 10)},
    }

    assert FedSDANoCachedServer._select_minimax_representative(
        [0, 1], stats
    ) == 1


def test_noninferiority_merge_partitions_rejected_large_cluster(monkeypatch):
    monkeypatch.setattr(
        config, "FEDSDA_CLUSTERING_CONSOLIDATION", "noninferiority_merge"
    )
    monkeypatch.setattr(config, "FEDSDA_MERGE_NONINFERIORITY_MARGIN", 0.01)

    class PairedClient:
        def get_held_model_ids(self):
            return {0, 1, 2}

        def evaluate_model_loss_difference(
            self, candidate_params, reference_params, target_model_id
        ):
            candidate_id = int(candidate_params["weight"].item())
            difference = (
                0.005
                if candidate_id == 0 and target_model_id in {0, 1}
                else 0.2
            )
            return _stats([difference] * 10)

    server = FedSDANoCachedServer(verbose=False)
    server.clients = [PairedClient()]
    server.global_models = {
        model_id: {"weight": torch.tensor([float(model_id)])}
        for model_id in range(3)
    }
    stats = {
        0: {0: _stats([0.1] * 10), 1: _stats([0.11] * 10),
            2: _stats([0.4] * 10)},
        1: {0: _stats([0.2] * 10), 1: _stats([0.1] * 10),
            2: _stats([0.4] * 10)},
        2: {0: _stats([0.4] * 10), 1: _stats([0.4] * 10),
            2: _stats([0.1] * 10)},
    }

    clusters, consolidation_params = server._validate_noninferiority_clusters(
        50, [[0, 1, 2]], stats
    )

    assert clusters == [[0, 1], [2]]
    assert torch.equal(consolidation_params[0]["weight"], torch.tensor([0.0]))
    summary = server.noninferiority_summary()
    assert summary["clustering_noninferiority_candidate_count"] == 1
    assert summary["clustering_noninferiority_accepted_count"] == 1


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


def test_oracle_concept_merges_only_models_with_same_majority_concept():
    class ConceptClient:
        concept_counts = {
            0: {2: 12, 1: 1},
            1: {2: 8},
            2: {3: 10},
        }

        def get_model_concept_counts(self, model_id):
            return self.concept_counts.get(model_id, {})

    server = FedSDANoCachedServer(
        clustering_decision="oracle_concept",
        linkage="complete",
        verbose=False,
    )
    server.clients = [ConceptClient()]

    assert server.perform_hierarchical_clustering(
        [0, 1, 2], stats_matrix={0: {}, 1: {}, 2: {}}
    ) == [[0, 1], [2]]


def test_oracle_concept_does_not_merge_tied_or_unobserved_model():
    class ConceptClient:
        def get_model_concept_counts(self, model_id):
            return {0: {1: 5, 2: 5}, 1: {1: 10}}.get(model_id, {})

    server = FedSDANoCachedServer(
        clustering_decision="oracle_concept",
        linkage="connected",
        verbose=False,
    )
    server.clients = [ConceptClient()]

    assert server.perform_hierarchical_clustering(
        [0, 1, 2], stats_matrix={0: {}, 1: {}, 2: {}}
    ) == [[0], [1], [2]]


def test_oracle_clustering_diagnostics_measure_decisions_and_parameter_auc():
    class ConceptClient:
        concept_counts = {
            0: {1: 10},
            1: {1: 10},
            2: {2: 10},
        }

        def get_model_concept_counts(self, model_id):
            return self.concept_counts.get(model_id, {})

    def model_params(personalized_value):
        return {
            "backbone.weight": torch.tensor([100.0]),
            "adapter.weight": torch.tensor([personalized_value]),
        }

    diagonal = _stats([0.1] * 10)
    close = _stats([0.12] * 10)
    far = _stats([0.4] * 10)
    stats = {
        0: {0: diagonal, 1: close, 2: far},
        1: {0: close, 1: diagonal, 2: far},
        2: {0: far, 1: far, 2: diagonal},
    }
    server = FedSDANoCachedServer(
        clustering_decision="oracle_concept",
        linkage="complete",
        verbose=False,
    )
    server.clients = [ConceptClient()]
    server.global_models = {
        0: model_params(1.0),
        1: model_params(1.1),
        2: model_params(-1.0),
    }

    clusters = server.perform_hierarchical_clustering([0, 1, 2], stats)
    server.record_clustering_diagnostics(10, [0, 1, 2], clusters)
    summary = server.clustering_oracle_diagnostic_summary()

    assert clusters == [[0, 1], [2]]
    assert summary["clustering_oracle_pair_count"] == 3
    assert summary["clustering_oracle_merge_tp"] == 1
    assert summary["clustering_oracle_merge_tn"] == 2
    assert summary["clustering_oracle_merge_f1"] == 1.0
    assert summary["clustering_oracle_loss_distance_auc"] == 1.0
    assert summary["clustering_oracle_parameter_distance_auc"] == 1.0


def test_feddrift_always_uses_paper_distance_decision(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_CLUSTERING_DECISION", "confidence")

    server = FedDriftServer(
        clustering_decision="confidence",
        distance_threshold=0.1,
        verbose=False,
    )

    assert server.clustering_decision == "distance"


def test_servers_use_method_specific_default_linkages(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_CLUSTER_LINKAGE", "connected")
    monkeypatch.setattr(config, "FEDDRIFT_CLUSTER_LINKAGE", "complete")

    assert FedSDANoCachedServer(verbose=False).linkage == "connected"
    assert FedDriftServer(verbose=False).linkage == "complete"


def test_fedsda_linkage_can_be_overridden_explicitly():
    server = FedSDANoCachedServer(linkage="complete", verbose=False)

    assert server.linkage == "complete"


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
