import numpy as np

from tools.experiments.clustering_functional_diagnostics import (
    aggregate,
    summarize_raw,
)


def test_clustering_functional_diagnostics_separates_merged_and_retained(tmp_path):
    path = tmp_path / "raw.npz"
    np.savez_compressed(
        path,
        dataset="circle2",
        mode="FedSDA_NoCached_ClassESR",
        seed=0,
        aggregation_interval=50,
        clustering_decision="distance",
        clustering_pair_rounds=np.asarray([50, 50]),
        clustering_pair_left_model_ids=np.asarray([0, 0]),
        clustering_pair_right_model_ids=np.asarray([1, 2]),
        clustering_pair_distances=np.asarray([0.05, 0.20]),
        clustering_pair_decision_scores=np.asarray([0.05, 0.20]),
        clustering_pair_same_cluster=np.asarray([True, False]),
        cross_evaluation_round_index=np.asarray([50, 50, 50, 50]),
        cross_evaluation_candidate_model_id=np.asarray([0, 1, 0, 2]),
        cross_evaluation_target_model_id=np.asarray([1, 0, 2, 0]),
        cross_evaluation_n=np.asarray([10, 10, 10, 10]),
        cross_evaluation_sum=np.asarray([1.0, 1.0, 2.0, 2.0]),
        cross_evaluation_sum_sq=np.asarray([0.1, 0.1, 0.4, 0.4]),
        cross_evaluation_candidate_only_correct=np.asarray([1, 0, 3, 1]),
        cross_evaluation_target_only_correct=np.asarray([0, 1, 1, 3]),
        cross_evaluation_both_correct=np.asarray([8, 8, 5, 5]),
        cross_evaluation_both_wrong=np.asarray([1, 1, 1, 1]),
    )

    summaries = aggregate(summarize_raw(path))
    by_outcome = {item["outcome"]: item for item in summaries}

    assert by_outcome["merged"]["correctness_disagreement_rate"] == 0.1
    assert by_outcome["merged"]["oracle_gain_rate"] == 0.0
    assert by_outcome["retained"]["correctness_disagreement_rate"] == 0.4
    assert by_outcome["retained"]["oracle_gain_rate"] == 0.1
