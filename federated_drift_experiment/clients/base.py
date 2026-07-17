"""クライアント共通基底クラス。

BaseClient はモデル保持・損失統計・データストア・新規モデル作成・サーバ連携などの
共通機能を提供する。各手法(FedSDA / FedDrift / Oblivious ...)はこれを継承する。
"""
import copy
import math
import random
import time
from collections import defaultdict

import torch

from .. import config
from ..models import SimpleMLP


class BaseClient:
    reports_state_summary = False

    def __init__(self, client_id, initial_models, initial_stats=None,
                 distance_threshold=None, verbose=True):
        self.client_id = client_id
        self.distance_threshold = (distance_threshold if distance_threshold is not None
                                   else config.DISTANCE_THRESHOLD)
        self.verbose = verbose

        # models: {model_id: SimpleMLP()}
        self.models = copy.deepcopy(initial_models)
        self.current_model_id = 0

        if initial_stats:
            self.model_stats = copy.deepcopy(initial_stats)
        else:
            self.model_stats = {mid: {'n': 0, 'mean': 0.0, 'M2': 0.0} for mid in initial_models}

        self.train_data_store = defaultdict(list)   # 学習用データ(モデルIDごと)
        self.stored_data = defaultdict(list)        # サーバ評価用データ(モデルIDごと)
        self.stored_data_limit = config.STORED_DATA_LIMIT

        # 新規作成モデルの引き渡し。ready=False の間はサーバの回収対象にならない
        self.pending_model_params = None
        self.pending_model_stats = None
        self.pending_model_ready = True

        # per-sample logs
        self.history_model_id = []
        self.history_drift_type = []
        self.history_accuracy = []
        self.history_concept = []
        self.processing_times = defaultdict(list)

        # 計算量は「呼出し回数」と「処理サンプル数」を分けて記録する。
        # 実行時間は環境依存なので、再現性の高いカウンタとは別に保持する。
        self.compute_counters = defaultdict(int)
        self.phase_seconds = defaultdict(float)

        # per-sample index and detection positions
        self.processed_samples = 0                 # number of processed samples for this client
        self.detected_event_positions = []         # detector internal detection positions (debug)
        self.estimated_drift_start_positions = []  # 各検知に対応する検出器推定の変化開始位置
        self.mapping_change_positions = []         # server mapping-induced model changes (debug/plot)
        self.local_switch_positions = []           # ローカルで実際に切替が発生したサンプルインデックス（検出として数えるもの）

        self.batch_size = config.CLIENT_BATCH_SIZE
        self.updates_per_sample = config.UPDATES_PER_SAMPLE
        self._pending_updates = 0   # LOCAL_UPDATE_TAU>1 のとき保留中のローカル更新(サンプル数)

        self.next_temp_id = -100 - self.client_id

    def _record_model_compute(self, phase, examples, calls=1):
        """モデル計算を用途別に記録する。examples はモデルへ入力した標本数。"""
        self.compute_counters[f"{phase}_forward_calls"] += int(calls)
        self.compute_counters[f"{phase}_examples"] += int(examples)

    def telemetry_snapshot(self):
        """ラウンド差分を計算できるよう、累積計測値のコピーを返す。"""
        return {
            "counters": dict(self.compute_counters),
            "phase_seconds": dict(self.phase_seconds),
        }

    # ------------------------------------------------------------
    # 損失統計(Welford法によるオンライン平均・分散)
    # ------------------------------------------------------------
    def _update_model_stats(self, model_id, value):
        stats = self.model_stats.setdefault(model_id, {'n': 0, 'mean': 0.0, 'M2': 0.0})
        stats['n'] += 1
        delta = value - stats['mean']
        stats['mean'] += delta / stats['n']
        delta2 = value - stats['mean']
        stats['M2'] += delta * delta2

    def _get_model_stats(self, model_id):
        stats = self.model_stats.get(model_id)
        if not stats or stats['n'] < 2:
            return 0.0, 0.0
        variance = stats['M2'] / (stats['n'] - 1)
        return stats['mean'], variance

    # ------------------------------------------------------------
    # データ管理
    # ------------------------------------------------------------
    def _store_evaluation_data(self, model_id, data_list):
        if model_id < 0:
            return
        current_stored = self.stored_data[model_id]
        sample_size = min(len(data_list), config.EVAL_STORE_SAMPLE_SIZE)
        if sample_size == 0:
            return
        sampled = random.sample(data_list, sample_size)
        current_stored.extend(sampled)
        if len(current_stored) > self.stored_data_limit:
            self.stored_data[model_id] = current_stored[-self.stored_data_limit:]

    def _absorb_into_store(self, model_id, data_list):
        """データを model_id の学習ストアへ追加し、そのモデルの損失統計を更新する。"""
        model = self.models[model_id]
        for d in data_list:
            self.train_data_store[model_id].append(d)
            with torch.no_grad():
                self._record_model_compute("statistics", len(d[0]))
                l_val = model.get_absolute_error(d[0], d[1])
            self._update_model_stats(model_id, l_val)

    # ------------------------------------------------------------
    # 予測ログ・学習
    # ------------------------------------------------------------
    def _record_prediction(self, x, y, concept_id):
        """現在のモデルで1サンプルを予測し、per-sample ログに記録する。"""
        current_model = self.models[self.current_model_id]
        self._record_model_compute("prediction", len(x))
        pred = current_model.predict(x)
        acc = 1.0 if pred.view(-1)[0].item() == y.view(-1)[0].item() else 0.0

        self.history_accuracy.append(acc)
        self.history_concept.append(concept_id)
        self.history_model_id.append(self.current_model_id)

    def train_step(self):
        """平時の1サンプル分のローカル更新(逐次手法用)。

        LOCAL_UPDATE_TAU(τ)サンプルごとにまとめて τ×UPDATES_PER_SAMPLE 回実行する
        (論文の「t mod τ = 0」)。総更新回数は τ に依らず不変。τ=1 で毎サンプル更新(v1 挙動)。
        """
        self._pending_updates += 1
        if self._pending_updates >= config.LOCAL_UPDATE_TAU:
            self.flush_pending_updates()

    def flush_pending_updates(self):
        """保留中のローカル更新を実行する(ラウンド境界・ドリフト解決前に呼ぶ。τ=1 では実質 no-op)。"""
        if self._pending_updates > 0:
            self.train_all_held_models(count_multiplier=self._pending_updates)
            self._pending_updates = 0

    def train_all_held_models(self, count_multiplier=1):
        start_time = time.perf_counter()
        updates_needed = self.updates_per_sample * count_multiplier
        for m_id, data_list in self.train_data_store.items():
            if m_id not in self.models:
                continue
            if len(data_list) < self.batch_size:
                continue
            model = self.models[m_id]
            for _ in range(updates_needed):
                batch = random.sample(data_list, self.batch_size)
                bx = torch.cat([d[0] for d in batch])
                by = torch.cat([d[1] for d in batch])
                model.update(bx, by)
                self._record_model_compute("training", len(bx))
                self.compute_counters["optimizer_steps"] += 1
        self.phase_seconds["training"] += time.perf_counter() - start_time

    # ------------------------------------------------------------
    # 新規モデルの作成
    # ------------------------------------------------------------
    def _spawn_new_model(self, bx, by, pending_ready):
        """現在のモデルを起点に新規モデルを作成・初期学習し、pending 登録する。

        戻り値: (テンポラリID, 初期学習後の平均損失)
        pending_ready=False の場合、次ラウンドまでサーバに回収されない。
        """
        temp_id = self._alloc_temp_id()
        if self.verbose:
            print(f"  -> Unknown Drift! New Model (Temp ID: {temp_id})")

        m = len(bx)
        new_model = SimpleMLP()
        new_model.set_params(self.models[self.current_model_id].get_params())
        new_model.reset_optimizer(lr=config.NEW_MODEL_LR)

        dataset = torch.utils.data.TensorDataset(bx, by)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(config.CLIENT_BATCH_SIZE, m), shuffle=True)
        training_start = time.perf_counter()
        for _ in range(self.new_model_initial_epochs()):
            for b_x, b_y in loader:
                new_model.update(b_x, b_y)
                self._record_model_compute("training", len(b_x))
                self.compute_counters["optimizer_steps"] += 1
        self.phase_seconds["training"] += time.perf_counter() - training_start

        self.models[temp_id] = new_model

        with torch.no_grad():
            self._record_model_compute("initialization", len(bx))
            preds = new_model(bx)
            final_loss = torch.abs(preds - by)
            init_mean = float(torch.mean(final_loss).item())
            init_var = float(torch.var(final_loss).item())
            if math.isnan(init_var):
                init_var = 0.1

        self.model_stats[temp_id] = {'n': m, 'mean': init_mean, 'M2': init_var * max(1, (m - 1))}
        self.pending_model_params = new_model.get_params()
        self.pending_model_stats = self.model_stats[temp_id]
        self.pending_model_ready = pending_ready
        return temp_id, init_mean

    def new_model_initial_epochs(self):
        """新規モデルの作成時に直ちに実行するローカル学習エポック数。"""
        return config.NEW_MODEL_EPOCHS

    def _alloc_temp_id(self):
        """新しい一意のテンポラリ（負）IDを返す。呼び出すたびに減らしていく。"""
        temp_id = self.next_temp_id
        self.next_temp_id -= 1
        return temp_id

    # ------------------------------------------------------------
    # サーバ連携
    # ------------------------------------------------------------
    def has_pending_model(self):
        return (self.pending_model_params is not None) and self.pending_model_ready

    def get_pending_model_info(self):
        return self.pending_model_params, self.pending_model_stats

    def promote_pending_to_ready(self):
        """ラウンド境界などで呼ぶ: pending がある場合に next-aggregation で回収可能にする"""
        if self.pending_model_params is not None and not self.pending_model_ready:
            self.pending_model_ready = True
            if self.verbose:
                print(f"Client {self.client_id}: pending model promoted to ready for next aggregation.")

    def confirm_model_registration(self, new_global_id):
        """サーバが pending モデルを回収した際、テンポラリIDをグローバルIDへ付け替える。"""
        temp_id = self.current_model_id
        if temp_id >= 0:
            self.pending_model_params = None
            self.pending_model_stats = None
            self.pending_model_ready = True
            return

        if temp_id in self.models:
            self.models[new_global_id] = self.models.pop(temp_id)
        else:
            if self.pending_model_params is not None:
                m = SimpleMLP()
                m.set_params(self.pending_model_params)
                self.models[new_global_id] = m

        if temp_id in self.model_stats:
            self.model_stats[new_global_id] = self.model_stats.pop(temp_id)
        if temp_id in self.train_data_store:
            self.train_data_store[new_global_id] = self.train_data_store.pop(temp_id)
        if temp_id in self.stored_data:
            self.stored_data[new_global_id] = self.stored_data.pop(temp_id)

        self.current_model_id = new_global_id
        self.pending_model_params = None
        self.pending_model_stats = None
        self.pending_model_ready = True

    def get_held_model_ids(self):
        ids = set()
        ids.update([k for k in self.stored_data.keys()])
        ids.update([k for k in self.train_data_store.keys()])
        ids.update(list(self.models.keys()))
        ids.add(self.current_model_id)
        return ids

    def evaluate_model(self, params, target_model_id):
        """サーバからの評価依頼: 指定パラメータのモデルを手元データで評価する。"""
        start_time = time.perf_counter()
        eval_data = []
        if target_model_id in self.stored_data and len(self.stored_data[target_model_id]) > 5:
            eval_data = self.stored_data[target_model_id]
        elif target_model_id == self.current_model_id and len(self.train_data_store[target_model_id]) > 10:
            eval_data = self.train_data_store[target_model_id]

        if len(eval_data) < 5:
            self.phase_seconds["cross_evaluation"] += time.perf_counter() - start_time
            return 0, 0.0, 0.0
        if len(eval_data) > config.EVAL_MAX_SAMPLES:
            eval_data = random.sample(eval_data, config.EVAL_MAX_SAMPLES)

        X = torch.cat([d[0] for d in eval_data])
        y = torch.cat([d[1] for d in eval_data])
        temp_model = SimpleMLP()
        temp_model.set_params(params)
        with torch.no_grad():
            self._record_model_compute("cross_evaluation", len(X))
            preds = temp_model(X)
            errors = torch.abs(preds - y).numpy().flatten()
        self.phase_seconds["cross_evaluation"] += time.perf_counter() - start_time
        return len(errors), float(errors.sum()), float((errors ** 2).sum())

    def apply_cached_merge(self, clusters, cluster_weights, global_stats=None):
        """クライアントが保持済みのモデルから、サーバ指定のマージを適用する。

        通信するのはクラスタ構成とスカラー重みだけでよい。マージ後のパラメータを
        クライアント側で再構成し、次の通常ブロードキャストまでの追加モデル受信を避ける。
        """
        id_mapping = {}
        merged_models = {}

        for cluster in clusters:
            representative = min(cluster)
            weights = cluster_weights[representative]
            total_weight = sum(weights.values())
            if total_weight <= 0:
                weights = {mid: 1.0 for mid in cluster}
                total_weight = float(len(cluster))

            merged_params = None
            for mid in cluster:
                id_mapping[mid] = representative
                params = self.models[mid].get_params()
                weight = weights[mid] / total_weight
                if merged_params is None:
                    merged_params = {name: value * weight for name, value in params.items()}
                else:
                    for name, value in params.items():
                        merged_params[name] = merged_params[name] + value * weight
            merged_models[representative] = merged_params

        self.apply_server_mapping(id_mapping, merged_models, global_stats)

    def apply_server_mapping(self, id_mapping, new_global_models, new_global_stats=None):
        # ID mapping を適用する前に current_model_id が変わるかどうかチェックしてログする
        if id_mapping and (self.current_model_id in id_mapping):
            new_id = id_mapping[self.current_model_id]
            if new_id != self.current_model_id:
                # マッピングによる切替位置を記録（processed_samples は現在まで処理したサンプル数）
                self.mapping_change_positions.append(self.processed_samples)

        # 1. 統計量のマージ（簡易）
        new_stats = {}
        merged_stats_source = defaultdict(list)
        for old_id, stats in self.model_stats.items():
            new_id = id_mapping.get(old_id, old_id)
            merged_stats_source[new_id].append(stats)
        for new_id, stat_list in merged_stats_source.items():
            if len(stat_list) == 1:
                new_stats[new_id] = stat_list[0]
            else:
                best_stat = max(stat_list, key=lambda x: x['n'])
                new_stats[new_id] = best_stat

        if new_global_stats:
            for mid, g_stat in new_global_stats.items():
                if mid not in new_stats or new_stats[mid]['n'] == 0:
                    new_stats[mid] = copy.deepcopy(g_stat)
        self.model_stats = new_stats

        # 2. データストアの再編
        new_stored_data = defaultdict(list)
        for old_id, data_list in self.stored_data.items():
            new_id = id_mapping.get(old_id, old_id)
            new_stored_data[new_id].extend(data_list)
        for mid in new_stored_data:
            if len(new_stored_data[mid]) > self.stored_data_limit:
                new_stored_data[mid] = random.sample(new_stored_data[mid], self.stored_data_limit)
        self.stored_data = new_stored_data

        new_train_data = defaultdict(list)
        for old_id, data_list in self.train_data_store.items():
            new_id = id_mapping.get(old_id, old_id)
            new_train_data[new_id].extend(data_list)
        self.train_data_store = new_train_data

        # 3. グローバル配布モデルで上書き（ただし既に現地にテンポラリがある場合は残す処理を行う）
        temp_models = {mid: m for mid, m in list(self.models.items()) if mid < 0}
        self.models = {}
        for mid, params in new_global_models.items():
            m = SimpleMLP()
            m.set_params(params)
            self.models[mid] = m
        for mid, m in temp_models.items():
            self.models[mid] = m

        # 4. current_model_id の map 適用（先に mapping_change_positions に記録済み）
        if self.current_model_id in id_mapping:
            self.current_model_id = id_mapping[self.current_model_id]
