"""FedSDA v1とFedDrift v1で共有するクラスタリングサーバ。"""

import random
from collections import defaultdict

from .. import config
from ..clustering import cluster_models
from .base import BaseServer


class ClusteringServer(BaseServer):
    """FedDrift 式サーバ: クロス評価による損失距離行列 → 階層的クラスタリング → マージ。"""

    def __init__(self, *args, linkage="connected", **kwargs):
        super().__init__(*args, **kwargs)
        self.linkage = linkage

    def _maybe_cluster(self, t, active_ids):
        M = len(active_ids)
        if M <= 1:
            return active_ids

        stats_matrix = self._cross_evaluate(active_ids)
        clusters = self.perform_hierarchical_clustering(active_ids, stats_matrix)

        if len(clusters) < M:
            active_ids = self._merge_clusters(t, active_ids, clusters)

        return active_ids

    def _merge_clusters(self, t, active_ids, clusters):
        """各クラスタを代表ID(最小ID)に統合し、クライアント・グローバル状態を更新する。"""
        if self.verbose:
            print(f"\nServer [t={t}]: MERGE EXECUTED")
            print(f"  - Before: {active_ids}")
            print(f"  - Clusters: {clusters}")

        id_mapping = {}
        new_ids = []
        for cluster in clusters:
            rep_id = min(cluster)
            new_ids.append(rep_id)
            for old_id in cluster:
                id_mapping[old_id] = rep_id

        # マージ後モデルを全クライアントへ再配布(ダウンロード)
        self.comm_models_down += len(self.global_models) * len(self.clients)
        self.comm_messages_down += len(self.clients)  # 統合後のモデルID対応
        for c in self.clients:
            c.apply_server_mapping(id_mapping, self.global_models, self.global_stats)

        for old_id in active_ids:
            if old_id not in new_ids:
                if old_id in self.global_models:
                    del self.global_models[old_id]
                if old_id in self.global_stats:
                    del self.global_stats[old_id]

        if self.verbose:
            print(f"  - After IDs: {sorted(list(self.global_models.keys()))}\n")

        return sorted(list(self.global_models.keys()))

    def _cross_evaluate(self, model_ids, send_model_params=True):
        holders = defaultdict(list)
        for c in self.clients:
            held_ids = c.get_held_model_ids()
            for mid in held_ids:
                holders[mid].append(c)

        stats_matrix = defaultdict(dict)

        for id_i in model_ids:
            params_i = self.global_models[id_i]
            for id_j in model_ids:
                target_clients = holders.get(id_j, [])
                if len(target_clients) > config.CROSS_EVAL_MAX_CLIENTS:
                    target_clients = random.sample(target_clients, config.CROSS_EVAL_MAX_CLIENTS)

                # 評価依頼と評価統計は、モデル転送とは別の軽量メッセージとして全方式で数える。
                self.comm_messages_down += len(target_clients)
                self.comm_messages_up += len(target_clients)
                if send_model_params:
                    # v1: 評価のためモデル id_i を各対象クライアントへ再送する。
                    self.comm_models_down += len(target_clients)

                total_n, total_S, total_SS = 0, 0.0, 0.0
                for c in target_clients:
                    n, S, SS = c.evaluate_model(params_i, target_model_id=id_j)
                    total_n += n; total_S += S; total_SS += SS

                stats_matrix[id_i][id_j] = (total_n, total_S, total_SS)
        return stats_matrix

    def perform_hierarchical_clustering(self, model_ids, stats_matrix):
        """損失ベースの距離が閾値以下のモデル対を辺とみなし、連結成分をクラスタとして返す。

        距離 dist(i,j) = max(「モデルiをjのデータで評価した際の損失悪化量」, その逆向き)。
        評価サンプル数が CLUSTER_MIN_EVAL_N 未満の対は判定しない。
        """
        if self.verbose:
            print(f"Server: Clustering models (Threshold={self.distance_threshold})...")

        pair_distances = {}
        M = len(model_ids)

        for i in range(M):
            for j in range(i + 1, M):
                id_i, id_j = model_ids[i], model_ids[j]

                stats_ii = stats_matrix[id_i].get(id_i, (0, 0, 0))
                stats_ij = stats_matrix[id_i].get(id_j, (0, 0, 0))
                stats_jj = stats_matrix[id_j].get(id_j, (0, 0, 0))
                stats_ji = stats_matrix[id_j].get(id_i, (0, 0, 0))

                min_n = config.CLUSTER_MIN_EVAL_N
                if stats_ii[0] < min_n or stats_ij[0] < min_n or stats_jj[0] < min_n or stats_ji[0] < min_n:
                    continue

                mu_ii = stats_ii[1] / stats_ii[0]
                mu_ij = stats_ij[1] / stats_ij[0]
                mu_jj = stats_jj[1] / stats_jj[0]
                mu_ji = stats_ji[1] / stats_ji[0]

                diff_i_to_j = mu_ij - mu_ii
                diff_j_to_i = mu_ji - mu_jj
                dist = max(diff_i_to_j, diff_j_to_i)
                pair_distances[(id_i, id_j)] = dist

                if dist <= self.distance_threshold:
                    if self.verbose and random.random() < 0.1:
                        print(f"  MERGE candidate: {id_i}-{id_j} (Dist={dist:.3f})")

        return cluster_models(model_ids, pair_distances,
                              self.distance_threshold, self.linkage)
