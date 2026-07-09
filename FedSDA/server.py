"""中央サーバ。

- BaseServer: 新規モデルの回収・モデルIDごとの加重平均(FedAvg)・ブロードキャストの
  共通土台。クラスタリングを持たない手法(単一モデル・Oblivious 等)はこれで十分。
- ClusteringServer: FedDrift 式のクロス評価 + 階層的クラスタリングによるモデルマージを
  追加する(FedSDA / FedDrift が使用)。

新しいサーバ手法(IFCA / CFL / アンサンブル等)を追加する場合は BaseServer を継承し、
experiment.py の MODE_SPECS の server_cls に登録する。
"""
import copy
import random
from collections import defaultdict

from . import config


class BaseServer:
    """回収・FedAvg・ブロードキャストのみを行う基底サーバ(クラスタリングなし)。"""

    def __init__(self, distance_threshold=None, verbose=True):
        self.global_models = {}
        self.next_model_id = 1
        self.clients = []
        self.distance_threshold = (distance_threshold if distance_threshold is not None
                                   else config.DISTANCE_THRESHOLD)
        self.verbose = verbose
        self.global_stats = defaultdict(lambda: {'n': 0, 'mean': 0.0, 'M2': 0.0})

        # 通信量カウンタ(1単位 = 1モデルのパラメータを1回転送。全モデル同一サイズ)
        self.comm_up = 0    # クライアント→サーバ(新規モデル回収・FedAvg のアップロード)
        self.comm_down = 0  # サーバ→クライアント(ブロードキャスト・クロス評価のモデル送信)

    def register_client(self, client):
        self.clients.append(client)

    def request_new_model_id(self):
        new_id = self.next_model_id
        self.next_model_id += 1
        return new_id

    def register_model_params(self, model_id, params):
        self.global_models[model_id] = copy.deepcopy(params)

    def register_model_stats(self, model_id, stats):
        self.global_stats[model_id] = copy.deepcopy(stats)

    def run_aggregation_and_merge(self, t, clustering_enabled=True):
        """1回のサーバ処理: 新規モデル回収 → (任意でクラスタリング) → FedAvg → 配布。

        clustering_enabled は _maybe_cluster に渡すフラグ。BaseServer はクラスタリングを
        持たないため無視される(サブクラスが利用する)。
        """
        self._collect_pending_models(t)

        active_ids = sorted(list(self.global_models.keys()))
        if clustering_enabled:
            active_ids = self._maybe_cluster(t, active_ids)

        self.update_global_models(active_ids)
        self.broadcast_models()

    def _collect_pending_models(self, t):
        """各クライアントの pending(新規作成)モデルを回収しグローバルIDを発行する。"""
        new_registrations = 0
        for c in self.clients:
            if c.has_pending_model():
                params, stats = c.get_pending_model_info()
                self.comm_up += 1  # クライアントが新規モデルをアップロード
                new_global_id = self.request_new_model_id()
                self.register_model_params(new_global_id, params)
                self.register_model_stats(new_global_id, stats)
                c.confirm_model_registration(new_global_id)
                new_registrations += 1

        if new_registrations > 0 and self.verbose:
            print(f"Server [t={t}]: Collected {new_registrations} new models.")

    def _maybe_cluster(self, t, active_ids):
        """クラスタリング/マージのフック。BaseServer では何もしない。"""
        return active_ids

    def update_global_models(self, active_ids):
        """モデルIDごとに、参加クライアントのパラメータをデータ量で加重平均(FedAvg)。"""
        for mid in active_ids:
            participant_clients = []
            for c in self.clients:
                if mid in c.models:
                    participant_clients.append(c)

            if not participant_clients:
                continue

            total_weight = 0
            new_params = None
            total_n_stat = 0
            weighted_mean_sum = 0.0

            for c in participant_clients:
                n_data = len(c.train_data_store.get(mid, []))
                if n_data == 0:
                    continue

                self.comm_up += 1  # クライアントが mid のパラメータをアップロード(FedAvg)
                params = c.models[mid].get_params()
                if new_params is None:
                    new_params = copy.deepcopy(params)
                    for k in new_params:
                        new_params[k] = new_params[k] * n_data
                else:
                    for k in new_params:
                        new_params[k] = new_params[k] + params[k] * n_data

                total_weight += n_data

                if mid in c.model_stats:
                    s = c.model_stats[mid]
                    weighted_mean_sum += s['mean'] * s['n']
                    total_n_stat += s['n']

            if total_weight > 0 and new_params is not None:
                for k in new_params:
                    new_params[k] = new_params[k] / total_weight
                self.global_models[mid] = new_params

            if total_n_stat > 0:
                avg_mean = weighted_mean_sum / total_n_stat
                self.global_stats[mid] = {'n': total_n_stat, 'mean': avg_mean, 'M2': 0.0}

    def broadcast_models(self):
        # 全グローバルモデルを全クライアントへ配布(ダウンロード)
        self.comm_down += len(self.global_models) * len(self.clients)
        for c in self.clients:
            c.apply_server_mapping({}, self.global_models, self.global_stats)


class ClusteringServer(BaseServer):
    """FedDrift 式サーバ: クロス評価による損失距離行列 → 階層的クラスタリング → マージ。"""

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
        self.comm_down += len(self.global_models) * len(self.clients)
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

    def _cross_evaluate(self, model_ids):
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

                # 評価のためモデル id_i を各対象クライアントへ送信(ダウンロード)
                self.comm_down += len(target_clients)

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

        adj = {mid: set() for mid in model_ids}
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

                if dist <= self.distance_threshold:
                    adj[id_i].add(id_j)
                    adj[id_j].add(id_i)
                    if self.verbose and random.random() < 0.1:
                        print(f"  MERGE candidate: {id_i}-{id_j} (Dist={dist:.3f})")

        visited = set()
        clusters = []
        for mid in model_ids:
            if mid not in visited:
                component = []
                stack = [mid]
                visited.add(mid)
                while stack:
                    curr = stack.pop()
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                clusters.append(sorted(component))
        return clusters
