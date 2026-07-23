"""FedDrift固有のサーバ実装。"""

from collections import defaultdict

from .. import config
from .clustering import CrossEvaluationClusteringServer


class FedDriftServer(CrossEvaluationClusteringServer):
    """新規モデル隔離と正確なR回のFedAvgを行うFedDrift時刻プロトコル。"""

    def __init__(self, *args, linkage=None, isolation_timesteps=None, **kwargs):
        if kwargs.get("distance_threshold") is None:
            kwargs["distance_threshold"] = config.FEDDRIFT_DISTANCE_THRESHOLD
        super().__init__(
            *args,
            linkage=linkage or config.CLUSTER_LINKAGE,
            **kwargs,
        )
        self.isolation_timesteps = (
            config.FEDDRIFT_ISOLATION_TIMESTEPS
            if isolation_timesteps is None else isolation_timesteps
        )
        if self.isolation_timesteps < 1:
            raise ValueError("FEDDRIFT_ISOLATION_TIMESTEPSは1以上である必要があります")
        self.model_weights = {}
        self.isolated_until = {}

    def register_model_params(self, model_id, params):
        super().register_model_params(model_id, params)
        self.model_weights.setdefault(model_id, config.PRETRAIN_SAMPLES)

    def prepare_timestep(self, t):
        """隔離解除済みモデルをクラスタリングし、新規隔離モデルへIDを割り当てる。"""
        self.record_client_state_summaries()
        self._cluster_mature_models(t)
        self._register_isolated_models(t)

    def run_training_round(self, t):
        """モデル送信・FedAvg・ブロードキャストを1往復実行する。"""
        active_ids = sorted({
            mid for client in self.clients for mid in client.models if mid >= 0
        })
        round_weights = self.update_global_models(active_ids)
        for mid, weight in round_weights.items():
            if weight > 0:
                self.model_weights[mid] = weight
        self.broadcast_models()

    def _register_isolated_models(self, t):
        for client in self.clients:
            if not client.has_pending_model():
                continue
            model_id = self.request_new_model_id()
            client.confirm_model_registration(model_id)
            self.isolated_until[model_id] = t + self.isolation_timesteps
            self.comm_messages_down += 1  # モデルIDの割当

    def _cluster_mature_models(self, t):
        mature_ids = self.mature_model_ids(t)
        if len(mature_ids) <= 1:
            return

        stats_matrix = self._cross_evaluate(mature_ids, send_model_params=False)
        mature_clusters = self.perform_hierarchical_clustering(mature_ids, stats_matrix)
        if len(mature_clusters) == len(mature_ids):
            return

        mature_set = set(mature_ids)
        all_clusters = list(mature_clusters)
        all_clusters.extend(
            [mid] for mid in sorted(self.global_models) if mid not in mature_set
        )
        self._merge_cached_clusters(t, all_clusters)

    def mature_model_ids(self, t):
        """時刻``t``でクロス評価の対象にできるモデルIDを返す。"""
        return sorted(
            mid for mid in self.global_models
            if t >= self.isolated_until.get(mid, 0)
        )

    def _merge_cached_clusters(self, t, clusters):
        if self.verbose:
            print(f"\nServer [t={t}]: MERGE EXECUTED (FedDrift, {self.linkage})")
            print(f"  - Clusters: {clusters}")

        cluster_weights = {}
        new_models = {}
        new_stats = {}
        new_model_weights = {}
        new_isolated_until = {}

        for cluster in clusters:
            representative = min(cluster)
            weights = {
                mid: max(self.model_weights.get(mid, 0), 0) for mid in cluster
            }
            if sum(weights.values()) <= 0:
                weights = {mid: 1 for mid in cluster}
            cluster_weights[representative] = weights
            new_models[representative] = self._weighted_model(cluster, weights)
            new_stats[representative] = self._combined_stats(cluster)
            new_model_weights[representative] = sum(weights.values())
            isolated_until = max(self.isolated_until.get(mid, 0) for mid in cluster)
            if isolated_until > t:
                new_isolated_until[representative] = isolated_until

        for client in self.clients:
            client.apply_cached_merge(clusters, cluster_weights, new_stats)
        self.comm_messages_down += len(self.clients)  # キャッシュマージの構成と重み

        self.global_models = new_models
        self.global_stats = defaultdict(
            lambda: {'n': 0, 'mean': 0.0, 'M2': 0.0}, new_stats
        )
        self.model_weights = new_model_weights
        self.isolated_until = new_isolated_until

    def _weighted_model(self, cluster, weights):
        total = sum(weights.values())
        averaged = None
        for mid in cluster:
            weight = weights[mid] / total
            params = self.global_models[mid]
            if averaged is None:
                averaged = {name: value * weight for name, value in params.items()}
            else:
                for name, value in params.items():
                    averaged[name] = averaged[name] + value * weight
        return averaged

    def _combined_stats(self, cluster):
        members = [mid for mid in cluster if mid in self.global_stats]
        total_n = sum(self.global_stats[mid]['n'] for mid in members)
        if total_n == 0:
            return {'n': 0, 'mean': 0.0, 'M2': 0.0}
        mean = sum(
            self.global_stats[mid]['mean'] * self.global_stats[mid]['n']
            for mid in members
        ) / total_n
        return {'n': total_n, 'mean': mean, 'M2': 0.0}
