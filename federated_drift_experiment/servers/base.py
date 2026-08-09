"""モデル回収・FedAvg・配布を担う共通サーバ。"""

import copy
from collections import defaultdict

from .. import config
from ..model_lineage import ModelLineageRecorder
from ..models import parameter_payload_size


class BaseServer:
    """回収・FedAvg・ブロードキャストのみを行う基底サーバ(クラスタリングなし)。"""

    def __init__(self, distance_threshold=None, verbose=True):
        self.global_models = {}
        self.next_model_id = 1
        self.clients = []
        self.distance_threshold = distance_threshold
        self.verbose = verbose
        self.global_stats = defaultdict(lambda: {'n': 0, 'mean': 0.0, 'M2': 0.0})
        self.model_lineage = ModelLineageRecorder()

        # 論理モデル転送回数と、モデル構造を反映した実パラメータ量を併記する。
        self.comm_models_up = 0    # クライアント→サーバのモデルパラメータ転送数
        self.comm_models_down = 0  # サーバ→クライアントのモデルパラメータ転送数
        self.comm_messages_up = 0  # クライアント→サーバの軽量メッセージ数
        self.comm_messages_down = 0  # サーバ→クライアントの軽量メッセージ数
        self.comm_parameter_values_up = 0
        self.comm_parameter_values_down = 0
        self.comm_bytes_up = 0
        self.comm_bytes_down = 0

    def record_parameter_transfer(self, direction, params, count=1):
        """転送したstate dictの値数とバイト数を方向別に加算する。"""
        values, byte_count = parameter_payload_size(params)
        if direction == "up":
            self.comm_parameter_values_up += values * count
            self.comm_bytes_up += byte_count * count
        elif direction == "down":
            self.comm_parameter_values_down += values * count
            self.comm_bytes_down += byte_count * count
        else:
            raise ValueError(f"Unknown communication direction: {direction!r}")

    def record_model_transfer(self, direction, params, count=1):
        """従来のモデル転送回数と、構造非依存の転送量を同時に数える。"""
        if direction == "up":
            self.comm_models_up += count
        elif direction == "down":
            self.comm_models_down += count
        else:
            raise ValueError(f"Unknown communication direction: {direction!r}")
        self.record_parameter_transfer(direction, params, count=count)

    def register_client(self, client):
        self.clients.append(client)

    def record_client_state_summaries(self):
        """モデル割当・ドリフト状態を報告する軽量メッセージを数える。"""
        self.comm_messages_up += sum(
            1 for client in self.clients if client.reports_state_summary
        )

    def request_new_model_id(self):
        new_id = self.next_model_id
        self.next_model_id += 1
        return new_id

    def register_model_params(self, model_id, params):
        self.global_models[model_id] = copy.deepcopy(params)
        self.model_lineage.ensure_model(model_id)

    def record_model_registration(self, model_id, t, client):
        """クライアント由来の新規モデルについて作成元を記録する。"""
        self.model_lineage.register_model(
            model_id,
            round_index=t,
            client_id=getattr(client, "client_id", -1),
        )

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

    def finalize_protocol(self, t):
        """ストリーム終端で、学習を伴わない未完了プロトコル処理を確定する。

        通常のサーバには終端まで持ち越す処理がないため何もしない。終端処理を持つ方式も、
        新しい学習ラウンドや未送信モデルの回収は行わず、既に通信済みの状態だけを確定する。
        """
        return

    def _collect_pending_models(self, t):
        """各クライアントの pending(新規作成)モデルを回収しグローバルIDを発行する。"""
        new_registrations = 0
        for c in self.clients:
            if c.has_pending_model():
                params, stats = c.get_pending_model_info()
                self.record_model_transfer("up", params)
                self.comm_messages_up += 1  # 新規モデル登録通知
                new_global_id = self.request_new_model_id()
                self.register_model_params(new_global_id, params)
                self.register_model_stats(new_global_id, stats)
                self.record_model_registration(new_global_id, t, c)
                c.confirm_model_registration(new_global_id)
                self.comm_messages_down += 1  # 新規モデルIDの割当
                new_registrations += 1

        if new_registrations > 0 and self.verbose:
            print(f"Server [t={t}]: Collected {new_registrations} new models.")

    def _maybe_cluster(self, t, active_ids):
        """クラスタリング/マージのフック。BaseServer では何もしない。"""
        return active_ids

    def update_global_models(self, active_ids):
        """モデルIDごとに、参加クライアントのパラメータをデータ量で加重平均(FedAvg)。

        戻り値: {モデルID: 総データ量}(FedAvg の重みの合計)。マージ時の加重平均など
        呼び出し側で集約重みとして流用できる。
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

                params = c.models[mid].get_params()
                self.record_model_transfer("up", params)
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
        for params in self.global_models.values():
            self.record_model_transfer("down", params, count=len(self.clients))
        if id_mapping:
            self.comm_messages_down += len(self.clients)
        for c in self.clients:
            c.apply_server_mapping(id_mapping or {}, self.global_models, self.global_stats)

    def final_parameter_footprint(self):
        """最終グローバルモデル群が占める値数とバイト数を返す。"""
        values = 0
        byte_count = 0
        for params in self.global_models.values():
            model_values, model_bytes = parameter_payload_size(params)
            values += model_values
            byte_count += model_bytes
        return values, byte_count
