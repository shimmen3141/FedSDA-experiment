import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federated_drift_experiment.model_lineage import ModelLineageRecorder


def test_model_lineage_records_origin_and_clustering_outcome():
    recorder = ModelLineageRecorder()
    recorder.ensure_model(0)
    recorder.register_model(1, round_index=3, client_id=7)
    recorder.register_model(2, round_index=3, client_id=8)

    recorder.record_clustering(
        round_index=4,
        model_ids=[0, 1, 2],
        pair_distances={(0, 1): 0.08, (0, 2): 0.20, (1, 2): 0.12},
        clusters=[[0, 1], [2]],
        pair_decision_scores={(0, 1): 0.7, (0, 2): 1.8, (1, 2): 1.2},
    )

    assert [(item.model_id, item.round_index, item.client_id)
            for item in recorder.registrations] == [
        (0, -1, -1),
        (1, 3, 7),
        (2, 3, 8),
    ]

    by_model = {
        observation.model_id: observation
        for observation in recorder.clustering_observations
    }
    assert by_model[0].nearest_model_id == 1
    assert by_model[0].nearest_distance == 0.08
    assert by_model[0].representative_model_id == 0
    assert by_model[0].cluster_size == 2
    assert by_model[0].cluster_max_distance == 0.08
    assert by_model[0].participated_in_merge is True
    assert by_model[0].absorbed is False

    assert by_model[1].representative_model_id == 0
    assert by_model[1].participated_in_merge is True
    assert by_model[1].absorbed is True
    assert by_model[2].representative_model_id == 2
    assert by_model[2].cluster_size == 1
    assert by_model[2].participated_in_merge is False
    assert by_model[2].absorbed is False
    assert math.isnan(by_model[2].cluster_max_distance)

    by_pair = {
        (item.left_model_id, item.right_model_id): item
        for item in recorder.clustering_pair_observations
    }
    assert by_pair[(0, 1)].round_index == 4
    assert by_pair[(0, 1)].distance == 0.08
    assert by_pair[(0, 1)].decision_score == 0.7
    assert by_pair[(0, 1)].same_cluster is True
    assert by_pair[(0, 2)].same_cluster is False


def test_model_lineage_marks_missing_distances_explicitly():
    recorder = ModelLineageRecorder()
    recorder.record_clustering(
        round_index=2,
        model_ids=[4, 5],
        pair_distances={},
        clusters=[[4], [5]],
    )

    observation = recorder.clustering_observations[0]
    assert observation.nearest_model_id == -1
    assert math.isnan(observation.nearest_distance)
    assert observation.cluster_evaluated_pairs == 0
    assert observation.cluster_possible_pairs == 0
