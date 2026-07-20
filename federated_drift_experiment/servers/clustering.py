"""クロス評価と階層クラスタリングを提供する共通サーバ。"""

import random
from collections import defaultdict

from .. import config
from ..clustering import cluster_models
from .base import BaseServer


class CrossEvaluationClusteringServer(BaseServer):
    """クロス評価と、その統計に基づく階層クラスタリングを共有する基底サーバ。"""

    def __init__(self, *args, linkage="connected", **kwargs):
        super().__init__(*args, **kwargs)
        self.linkage = linkage

    def _cross_evaluate(self, model_ids, send_model_params=True, use_client_cache=False):
        """モデル対をクライアントで評価し、集約統計を返す。

        use_client_cache=True は、事前に全対象モデルが配布済みであるプロトコル専用。
        この場合はモデル本体を再送せず、クライアントの不変キャッシュを評価する。
        """
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
                    # キャッシュを使わない評価ではモデルを各対象クライアントへ送る。
                    self.comm_models_down += len(target_clients)

                total_n, total_S, total_SS = 0, 0.0, 0.0
                for c in target_clients:
                    if use_client_cache:
                        n, S, SS = c.evaluate_cached_model(id_i, target_model_id=id_j)
                    else:
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
