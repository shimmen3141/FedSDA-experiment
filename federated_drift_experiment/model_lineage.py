"""モデル登録とクラスタリング結果を記録する診断用データ構造。"""

from dataclasses import dataclass
from math import nan


@dataclass(frozen=True)
class ModelRegistration:
    """サーバが新規モデルへグローバルIDを割り当てた事実。"""

    model_id: int
    round_index: int
    client_id: int


@dataclass(frozen=True)
class ClusteringObservation:
    """1回のクラスタリングにおける1モデル分の診断値。"""

    round_index: int
    model_id: int
    nearest_model_id: int
    nearest_distance: float
    representative_model_id: int
    cluster_size: int
    cluster_max_distance: float
    cluster_evaluated_pairs: int
    cluster_possible_pairs: int
    participated_in_merge: bool
    absorbed: bool


@dataclass(frozen=True)
class ClusteringPairObservation:
    """1回のクラスタリングにおけるモデル対の判定結果。"""

    round_index: int
    left_model_id: int
    right_model_id: int
    distance: float
    decision_score: float
    same_cluster: bool
    oracle_same_concept: bool | None
    personalized_parameter_distance: float


class ModelLineageRecorder:
    """アルゴリズムへ影響を与えず、モデルの生成・統合履歴だけを保持する。"""

    def __init__(self):
        self._registrations = {}
        self.clustering_observations = []
        self.clustering_pair_observations = []

    @property
    def registrations(self):
        return [self._registrations[key] for key in sorted(self._registrations)]

    def register_model(self, model_id, round_index=-1, client_id=-1):
        """モデルの作成元を記録する。明示的な登録は初期値を上書きする。"""
        self._registrations[model_id] = ModelRegistration(
            model_id=int(model_id),
            round_index=int(round_index),
            client_id=int(client_id),
        )

    def ensure_model(self, model_id):
        """作成元が不明な初期・外部登録モデルを一度だけ記録する。"""
        self._registrations.setdefault(
            model_id,
            ModelRegistration(
                model_id=int(model_id),
                round_index=-1,
                client_id=-1,
            ),
        )

    def record_clustering(
        self,
        round_index,
        model_ids,
        pair_distances,
        clusters,
        pair_decision_scores=None,
        pair_oracle_same_concept=None,
        pair_personalized_parameter_distances=None,
    ):
        """距離とクラスタ割当をモデル単位の観測へ正規化して記録する。"""
        normalized_distances = {
            tuple(sorted((int(left), int(right)))): float(distance)
            for (left, right), distance in pair_distances.items()
        }
        cluster_by_model = {
            int(model_id): [int(member) for member in cluster]
            for cluster in clusters
            for model_id in cluster
        }
        normalized_scores = {
            tuple(sorted((int(left), int(right)))): float(score)
            for (left, right), score in (pair_decision_scores or {}).items()
        }
        normalized_oracle_labels = {
            tuple(sorted((int(left), int(right)))): bool(same_concept)
            for (left, right), same_concept in (
                pair_oracle_same_concept or {}
            ).items()
        }
        normalized_parameter_distances = {
            tuple(sorted((int(left), int(right)))): float(distance)
            for (left, right), distance in (
                pair_personalized_parameter_distances or {}
            ).items()
        }

        for (left, right), distance in sorted(normalized_distances.items()):
            self.clustering_pair_observations.append(
                ClusteringPairObservation(
                    round_index=int(round_index),
                    left_model_id=left,
                    right_model_id=right,
                    distance=distance,
                    decision_score=normalized_scores.get((left, right), distance),
                    same_cluster=(
                        cluster_by_model.get(left, [left])
                        == cluster_by_model.get(right, [right])
                    ),
                    oracle_same_concept=normalized_oracle_labels.get(
                        (left, right)
                    ),
                    personalized_parameter_distance=(
                        normalized_parameter_distances.get((left, right), nan)
                    ),
                )
            )

        for model_id in model_ids:
            model_id = int(model_id)
            distances = [
                (other_id, distance)
                for pair, distance in normalized_distances.items()
                if model_id in pair
                for other_id in pair
                if other_id != model_id
            ]
            if distances:
                nearest_model_id, nearest_distance = min(
                    distances, key=lambda item: (item[1], item[0])
                )
            else:
                nearest_model_id, nearest_distance = -1, nan

            cluster = cluster_by_model.get(model_id, [model_id])
            representative = min(cluster)
            possible_pairs = len(cluster) * (len(cluster) - 1) // 2
            within_cluster = [
                normalized_distances[pair]
                for pair in normalized_distances
                if pair[0] in cluster and pair[1] in cluster
            ]
            cluster_max_distance = max(within_cluster) if within_cluster else nan

            self.clustering_observations.append(
                ClusteringObservation(
                    round_index=int(round_index),
                    model_id=model_id,
                    nearest_model_id=nearest_model_id,
                    nearest_distance=nearest_distance,
                    representative_model_id=representative,
                    cluster_size=len(cluster),
                    cluster_max_distance=cluster_max_distance,
                    cluster_evaluated_pairs=len(within_cluster),
                    cluster_possible_pairs=possible_pairs,
                    participated_in_merge=len(cluster) > 1,
                    absorbed=model_id != representative,
                )
            )
