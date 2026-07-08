"""クライアント実装。

- BaseClient: モデル保持・統計量・データストア・サーバ連携などの共通機能
- AdwinClient: 提案手法 (FedSDA)。ADWIN + FIFOバッファによる逐次ドリフト検出
- PeriodicClient: FedDriftベースライン。固定バッチ単位の損失増分による検出

比較手法を追加する場合は BaseClient を継承し、experiment.py の MODE_SPECS に
エントリを登録する。
"""
import copy
import math
import random
import time
from collections import defaultdict, deque

import torch

from . import config
from .adwin import FullScanADWIN
from .models import SimpleMLP


class BaseClient:
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

        # per-sample index and detection positions
        self.processed_samples = 0                 # number of processed samples for this client
        self.detected_event_positions = []         # detector internal detection positions (debug)
        self.mapping_change_positions = []         # server mapping-induced model changes (debug/plot)
        self.local_switch_positions = []           # ローカルで実際に切替が発生したサンプルインデックス（検出として数えるもの）

        self.batch_size = config.CLIENT_BATCH_SIZE
        self.updates_per_step = config.UPDATES_PER_STEP

        self.next_temp_id = -100 - self.client_id

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
                l_val = model.get_absolute_error(d[0], d[1])
            self._update_model_stats(model_id, l_val)

    # ------------------------------------------------------------
    # 予測ログ・学習
    # ------------------------------------------------------------
    def _record_prediction(self, x, y, concept_id):
        """現在のモデルで1サンプルを予測し、per-sample ログに記録する。"""
        current_model = self.models[self.current_model_id]
        pred = current_model.predict(x)
        acc = 1.0 if pred.view(-1)[0].item() == y.view(-1)[0].item() else 0.0

        self.history_accuracy.append(acc)
        self.history_concept.append(concept_id)
        self.history_model_id.append(self.current_model_id)

    def train_all_held_models(self, count_multiplier=1):
        updates_needed = self.updates_per_step * count_multiplier
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
        for _ in range(config.NEW_MODEL_EPOCHS):
            for b_x, b_y in loader:
                new_model.update(b_x, b_y)

        self.models[temp_id] = new_model

        with torch.no_grad():
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
        eval_data = []
        if target_model_id in self.stored_data and len(self.stored_data[target_model_id]) > 5:
            eval_data = self.stored_data[target_model_id]
        elif target_model_id == self.current_model_id and len(self.train_data_store[target_model_id]) > 10:
            eval_data = self.train_data_store[target_model_id]

        if len(eval_data) < 5:
            return 0, 0.0, 0.0
        if len(eval_data) > config.EVAL_MAX_SAMPLES:
            eval_data = random.sample(eval_data, config.EVAL_MAX_SAMPLES)

        X = torch.cat([d[0] for d in eval_data])
        y = torch.cat([d[1] for d in eval_data])
        temp_model = SimpleMLP()
        temp_model.set_params(params)
        with torch.no_grad():
            preds = temp_model(X)
            errors = torch.abs(preds - y).numpy().flatten()
        return len(errors), float(errors.sum()), float((errors ** 2).sum())

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


class AdwinClient(BaseClient):
    """提案手法 (FedSDA) クライアント: ADWIN + FIFOバッファによる逐次(1サンプル単位)処理。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adwin = FullScanADWIN(delta=config.ADWIN_DELTA)
        self.buffer = deque()                       # FIFOバッファ(検知遅延中のデータ保留)
        self.safe_margin = config.FIFO_BUFFER_SIZE  # バッファ長 N_FIFO

    def process_one_step(self, x_in, y_in, concept_id):
        """1サンプルを処理する: 予測 → ADWIN更新 → (ドリフト解決 | 平時処理) → 学習。"""
        start_time = time.perf_counter()
        x = x_in.unsqueeze(0) if x_in.dim() == 1 else x_in
        y = y_in.unsqueeze(0) if y_in.dim() == 1 else y_in

        # current sample index for this client (before increment)
        idx = self.processed_samples
        self.processed_samples += 1

        self._record_prediction(x, y, concept_id)

        error = self.models[self.current_model_id].get_absolute_error(x, y)
        self.adwin.update(error)
        self.buffer.append((x, y))

        drift_type = 0

        # ADWIN の統計的検知、または保険的な強制チェックのどちらかが発火したら解決処理へ
        if self.adwin.drift_detected or self._forced_drift_check(idx):
            self.detected_event_positions.append(idx)
            drift_type = self._resolve_drift(sample_idx=idx)
        else:
            # 平時: バッファ長 N_FIFO を超えた分だけ古いデータをストアへ確定し、学習する
            while len(self.buffer) > self.safe_margin:
                old_x, old_y = self.buffer.popleft()
                loss_val = self.models[self.current_model_id].get_absolute_error(old_x, old_y)
                self._update_model_stats(self.current_model_id, loss_val)
                self.train_data_store[self.current_model_id].append((old_x, old_y))
            self.train_all_held_models(count_multiplier=1)

        self.history_drift_type.append(drift_type)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        num_global = sum(1 for mid in self.models.keys() if mid >= 0)
        self.processing_times[num_global].append(elapsed_ms)

    def _forced_drift_check(self, idx):
        """ADWIN未検知でも、直近ウィンドウの損失がベースラインから閾値以上悪化して
        いればドリフト解決を強制する保険的チェック。"""
        width = self.adwin.width
        lower_bound = max(0, self.safe_margin - 5)
        upper_bound = max(100, 2 * max(0, (self.safe_margin - 5)))

        if not (lower_bound <= width <= upper_bound and width > 0 and self.current_model_id >= 0):
            return False
        if len(self.buffer) < width:
            return False

        # ADWINウィンドウに対応するバッファ末尾で現行モデルの損失を測る
        tail = list(self.buffer)[-width:]
        bx = torch.cat([d[0] for d in tail])
        by = torch.cat([d[1] for d in tail])
        with torch.no_grad():
            preds = self.models[self.current_model_id](bx)
            window_loss = float(torch.mean(torch.abs(preds - by)).item())
        hist_mean, _ = self._get_model_stats(self.current_model_id)

        if hist_mean > 0.0 and (window_loss >= hist_mean + self.distance_threshold):
            if self.verbose:
                print(f"Client {self.client_id} [sample={idx}]: Forced drift-check triggered "
                      f"(win={width}, loss={window_loss:.3f}, base={hist_mean:.3f}, "
                      f"thr={self.distance_threshold:.3f})")
            return True
        return False

    def _resolve_drift(self, sample_idx):
        """ドリフト解決: バッファを新旧概念に分割し、モデル切替 or 新規作成を行う。"""
        # ADWINの縮小後ウィンドウ幅 = 新概念のデータ数として、バッファを事後分割する
        buffer_list = list(self.buffer)
        n_new_concept = self.adwin.width

        if len(buffer_list) <= n_new_concept:
            drift_data = buffer_list
            old_data = []
        else:
            old_data = buffer_list[:-n_new_concept]
            drift_data = buffer_list[-n_new_concept:]

        # 旧概念のデータは直前まで使っていたモデルへ確定
        if len(old_data) > 0:
            self._store_evaluation_data(self.current_model_id, old_data)
            self._absorb_into_store(self.current_model_id, old_data)

        if len(drift_data) < config.MIN_DRIFT_DATA:
            self.adwin.reset()
            return 0

        if self.verbose:
            print(f"Client {self.client_id} [sample={sample_idx}]: ADWIN Drift Detected.")

        bx = torch.cat([d[0] for d in drift_data])
        by = torch.cat([d[1] for d in drift_data])

        # 各既存モデルの新概念データに対する損失を、ベースラインと比較して適合候補を集める
        valid_candidates = []
        for m_id, model in self.models.items():
            with torch.no_grad():
                preds = model(bx)
                loss = float(torch.mean(torch.abs(preds - by)).item())
            hist_mean, _ = self._get_model_stats(m_id)

            if hist_mean == 0.0:
                if self.verbose:
                    print(f"  Check M{m_id}: No baseline (n=0) -> treat as not-matching. (Loss={loss:.3f})")
                continue

            diff = loss - hist_mean
            if self.verbose:
                print(f"  Check M{m_id}: Diff={diff:.3f} vs Thr={self.distance_threshold:.3f} "
                      f"(Loss={loss:.3f}, Base={hist_mean:.3f})")
            if diff <= self.distance_threshold:
                valid_candidates.append((m_id, loss))

        if valid_candidates:
            # 既知のコンセプト: 適合候補のうち損失最小のモデルへ切替(または現状維持)
            best_model_id, min_loss = min(valid_candidates, key=lambda x: x[1])
            if best_model_id != self.current_model_id:
                if self.verbose:
                    print(f"  -> Switch to Model {best_model_id} (Loss {min_loss:.3f})")
                # 記録は sample_idx で行う（ずれ防止）
                self.local_switch_positions.append(sample_idx)
                self.current_model_id = best_model_id
                drift_type = 1
            else:
                if self.verbose:
                    print(f"  -> Keep current Model {self.current_model_id} (Loss {min_loss:.3f})")
                drift_type = 0

            self._absorb_into_store(self.current_model_id, drift_data)
        else:
            # 未知のコンセプト: 新規モデルを作成(作成ラウンド内ではサーバへ送らない)
            temp_id, _ = self._spawn_new_model(bx, by, pending_ready=False)
            self.local_switch_positions.append(sample_idx)
            self.current_model_id = temp_id
            drift_type = 2
            self.train_data_store[temp_id].extend(drift_data)

        self.adwin.reset()
        self.buffer.clear()
        return drift_type


class PeriodicClient(BaseClient):
    """FedDriftベースライン: 固定バッチ単位で最小損失の増分によりドリフト判定する。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_min_loss = None

    def phase1_detect(self, batch_data, t, concept_id):
        """バッチ全体を予測・ログした後、バッチ単位でドリフト判定とモデル選択を行う。"""
        processed_batch_data = []
        batch_x = []
        batch_y = []

        # batch start index (sample index of first sample in this batch)
        start_idx = self.processed_samples

        for (x_in, y_in) in batch_data:
            x = x_in.unsqueeze(0) if x_in.dim() == 1 else x_in
            y = y_in.unsqueeze(0) if y_in.dim() == 1 else y_in
            processed_batch_data.append((x, y))
            batch_x.append(x)
            batch_y.append(y)

            self._record_prediction(x, y, concept_id)
            self.processed_samples += 1

        bx = torch.cat(batch_x)
        by = torch.cat(batch_y)
        m = len(batch_data)

        # 全モデルの中で最小損失のモデルを求める
        min_loss = float('inf')
        best_model_id = self.current_model_id
        for m_id, model in self.models.items():
            with torch.no_grad():
                preds = model(bx)
                loss = float(torch.mean(torch.abs(preds - by)).item())
            if loss < min_loss:
                min_loss = loss
                best_model_id = m_id

        # ドリフト判定: 最小損失が前バッチから閾値以上増加したか(FedDrift方式)
        is_drift = (self.last_min_loss is not None
                    and min_loss > self.last_min_loss + self.distance_threshold)

        if is_drift:
            if self.verbose:
                print(f"Client {self.client_id} [t={t}]: Drift Detected (Loss {min_loss:.3f})")
            self.detected_event_positions.append(start_idx)

            # FedDriftでは新規モデルを作成ラウンド内でサーバへ送る(ready=True)
            temp_id, init_mean = self._spawn_new_model(bx, by, pending_ready=True)
            # batch の最後のサンプルインデックスで切替記録
            self.local_switch_positions.append(start_idx + m - 1)
            self.current_model_id = temp_id
            drift_type = 2
            self.last_min_loss = init_mean
            self.train_data_store[temp_id].extend(processed_batch_data)
        else:
            drift_type = 0
            if best_model_id != self.current_model_id:
                if self.verbose:
                    print(f"  -> Switch to Model {best_model_id}")
                self.local_switch_positions.append(start_idx + m - 1)
                self.current_model_id = best_model_id
                drift_type = 1

            self._absorb_into_store(self.current_model_id, processed_batch_data)
            self.last_min_loss = min_loss

        self._store_evaluation_data(self.current_model_id, processed_batch_data)
        for _ in range(m):
            self.history_drift_type.append(drift_type)

    def phase2_train(self, k_steps):
        self.train_all_held_models(count_multiplier=k_steps)
