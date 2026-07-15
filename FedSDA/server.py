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
from .clustering import cluster_models


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
        self.control_up = 0
        self.control_down = 0

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

    def run_round(self, t, clustering_enabled=True):
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
        """モデルIDごとに、参加クライアントのパラメータをデータ量で加重平均(FedAvg)。

        戻り値: {モデルID: 総データ量}(FedAvg の重みの合計)。マージ時の加重平均など
        呼び出し側で流用できる(v1 経路は無視してよい)。
        """
        agg_weights = {}
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

            agg_weights[mid] = total_weight

        return agg_weights

    def broadcast_models(self, id_mapping=None):
        """全グローバルモデルを全クライアントへ配布(ダウンロード)。

        id_mapping を渡すと、マージ等によるモデルIDの付け替えを配布と同時に適用する
        (省略時は付け替えなし=従来挙動)。
        """
        self.comm_down += len(self.global_models) * len(self.clients)
        for c in self.clients:
            c.apply_server_mapping(id_mapping or {}, self.global_models, self.global_stats)


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

                if send_model_params:
                    # v1: 評価のためモデル id_i を各対象クライアントへ再送する。
                    self.comm_down += len(target_clients)
                else:
                    # v2: 配布済みモデルを再利用し、依頼と統計だけを軽量通信として数える。
                    self.control_down += len(target_clients)
                    self.control_up += len(target_clients)

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


class ClusteringServerV2(ClusteringServer):
    """FedSDA v2 サーバ(docs/sequence-diagrams.md の設計版。モード 'FedSDA_v2')。

    v1(ClusteringServer)との違いはラウンド内の処理順序:
      v1: 回収 → クロス評価/クラスタリング(前ラウンド末のモデルで評価) → FedAvg → 配布
      v2: 回収 → FedAvg → クロス評価/クラスタリング(今ラウンドの学習を反映したモデルで評価) → 配布

    これにより
      (a) 距離評価の鮮度が揃う(v1 は既存モデルだけ1ラウンド古い非対称な比較になる)、
      (b) マージはサーバ側でメンバーの FedAvg 済みパラメータをデータ量加重平均して統合できる
          (v1 のように非代表側のパラメータを破棄しない。追加通信なし)、
      (c) 配布はラウンド末の1回のみ(v1 のマージ発生ラウンドの二重配布を解消)、
      (d) マージ発生ラウンドでも当該ラウンドのローカル学習が FedAvg に反映される
          (v1 はマージ時の再配布がローカル学習を上書きする)。

    さらに新規モデルの回収を「グローバルID の採番だけ」にし、パラメータ送信を FedAvg の
    1回に集約する。v1 はクロス評価を FedAvg より前に行う都合で、回収時にもパラメータを
    送る必要があり新規モデルを二重送信するが、v2 はクロス評価が FedAvg 後なのでこの
    二重送信を解消できる(新規モデルは保持クライアント1台のみのため FedAvg は恒等)。
    クライアント側の挙動(ADWIN 検知・新規モデルの次ラウンド回収)は v1 と共通。
    """

    def run_round(self, t, clustering_enabled=True):
        """1回のサーバ処理(v2): 新規登録 → FedAvg → (任意でクラスタリング) → 配布。

        新規モデルは回収でグローバルID を採番するだけにし、パラメータ送信は次の FedAvg に
        1回集約する(v1 の二重送信を解消)。
        """
        self._register_new_models(t)

        # 全クライアントが保持するグローバルモデルID(既存 + 今ラウンド採番の新規)
        active_ids = sorted({mid for c in self.clients for mid in c.models if mid >= 0})

        # FedAvg: パラメータ送信はここ1回のみ。今ラウンドのローカル学習が反映される
        agg_weights = self.update_global_models(active_ids)

        id_mapping = {}
        if clustering_enabled:
            id_mapping = self._cluster_and_merge(t, active_ids, agg_weights)

        # 配布は1回のみ。マージの ID 付け替えも同時に適用する
        self.broadcast_models(id_mapping)

    def _register_new_models(self, t):
        """pending の新規モデルにグローバルID を採番する(パラメータ送信なし)。

        パラメータは後段の update_global_models(FedAvg)で1回だけ送るため、回収時に
        パラメータを送る _collect_pending_models(v1 が使う)は用いない。採番順は
        _collect_pending_models と同一(クライアント走査順)なので ID の付き方は変わらない。
        """
        n_new = 0
        for c in self.clients:
            if c.has_pending_model():
                c.confirm_model_registration(self.request_new_model_id())
                n_new += 1
        if n_new > 0 and self.verbose:
            print(f"Server [t={t}]: Registered {n_new} new models (params sent once in FedAvg).")

    def _cluster_and_merge(self, t, active_ids, agg_weights):
        """FedAvg 済みモデルでクロス評価・クラスタリングし、マージは加重平均で統合する。

        v1 の _merge_clusters と異なり再配布は行わず、id_mapping を返して
        run_round 末尾の broadcast_models に適用を委ねる。
        """
        M = len(active_ids)
        if M <= 1:
            return {}

        stats_matrix = self._cross_evaluate(active_ids)
        clusters = self.perform_hierarchical_clustering(active_ids, stats_matrix)
        if len(clusters) >= M:
            return {}

        if self.verbose:
            print(f"\nServer [t={t}]: MERGE EXECUTED (v2: weighted average)")
            print(f"  - Before: {active_ids}")
            print(f"  - Clusters: {clusters}")

        id_mapping = {}
        for cluster in clusters:
            rep_id = min(cluster)
            for old_id in cluster:
                id_mapping[old_id] = rep_id
            if len(cluster) > 1:
                self.global_models[rep_id] = self._weighted_average_params(cluster, agg_weights)
                self._merge_stats(rep_id, cluster)

        # 非代表IDのグローバル状態を削除(クライアント側の付け替えは broadcast で行う)
        for old_id in active_ids:
            if id_mapping.get(old_id, old_id) != old_id:
                if old_id in self.global_models:
                    del self.global_models[old_id]
                if old_id in self.global_stats:
                    del self.global_stats[old_id]

        if self.verbose:
            print(f"  - After IDs: {sorted(list(self.global_models.keys()))}\n")
        return id_mapping

    def _weighted_average_params(self, cluster, agg_weights):
        """クラスタメンバーの FedAvg 済みパラメータをデータ量で加重平均する。

        加重平均の結合則により「統合クラスタの全データでの加重平均」と同値になる。
        重みが全て 0 の場合は代表(最小ID)のパラメータを維持する。
        """
        weights = {m: max(agg_weights.get(m, 0), 0) for m in cluster}
        total = sum(weights.values())
        if total <= 0:
            return self.global_models[min(cluster)]

        avg = None
        for m in cluster:
            w = weights[m]
            if w == 0:
                continue
            params = self.global_models[m]
            if avg is None:
                avg = {k: v * w for k, v in params.items()}
            else:
                for k in avg:
                    avg[k] = avg[k] + params[k] * w
        for k in avg:
            avg[k] = avg[k] / total
        return avg

    def _merge_stats(self, rep_id, cluster):
        """クラスタメンバーの損失統計を n 加重平均で統合する(update_global_models と同じ簡易形)。"""
        members = [m for m in cluster if m in self.global_stats]
        total_n = sum(self.global_stats[m]['n'] for m in members)
        if total_n > 0:
            mean = sum(self.global_stats[m]['mean'] * self.global_stats[m]['n']
                       for m in members) / total_n
            self.global_stats[rep_id] = {'n': total_n, 'mean': mean, 'M2': 0.0}


class FedDriftV2Server(ClusteringServer):
    """新規モデル隔離と正確なR回のFedAvgを行うFedDrift時刻プロトコル。"""

    def __init__(self, *args, linkage=None, isolation_timesteps=None, **kwargs):
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
        self.control_up += len(self.clients)  # 割当・ドリフト要約
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
            self.control_down += 1  # モデルIDの割当

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
            print(f"\nServer [t={t}]: MERGE EXECUTED (FedDrift v2, {self.linkage})")
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
