"""FedSDA固有のサーバ実装。"""

from .clustering import ClusteringServer


class FedSDAV2Server(ClusteringServer):
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
    クライアント側の挙動(ADWIN 検知・設定された待機後の新規モデル回収)は v1 と共通。
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
