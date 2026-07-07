# 修正版スクリプト（"ローカル切替" を検出カウントに使用するよう修正）
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
import random
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict, deque

# ==========================================
# 0. Utils & Data
# ==========================================
DEFAULT_DISTANCE_THRESHOLD = 0.1

def generate_data(concept_id, n_samples=1):
    x_list = []
    y_list = []

    if concept_id in [0, 2]:
        sigma = 0.6
        if concept_id == 0:
            centers = [(-2, -2), (2, 2)]
        else:
            centers = [(2, 2), (-2, -2)]

        for _ in range(n_samples):
            label = 0.0 if np.random.rand() < 0.5 else 1.0
            center = centers[int(label)]
            x = np.random.randn(2) * sigma + np.array(center)
            x_list.append(x)
            y_list.append(label)

    elif concept_id in [1, 3]:
        for _ in range(n_samples):
            label = 0.0 if np.random.rand() < 0.5 else 1.0
            is_inner = False
            if concept_id == 1:
                if label == 0.0: is_inner = True
            else:
                if label == 1.0: is_inner = True

            if is_inner:
                r = np.random.normal(loc=1.5, scale=0.4)
            else:
                r = np.random.normal(loc=4.5, scale=0.5)

            theta = np.random.uniform(0, 2 * np.pi)
            x = np.array([r * np.cos(theta), r * np.sin(theta)])
            x_list.append(x)
            y_list.append(label)

    if n_samples == 1:
        return torch.FloatTensor(x_list[0]), torch.FloatTensor([y_list[0]])
    else:
        return torch.FloatTensor(np.array(x_list)), torch.FloatTensor(np.array(y_list)).unsqueeze(1)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.05)

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = (self.forward(x) > 0.5).float()
        return out

    def get_absolute_error(self, x, y):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        with torch.no_grad():
            pred = self.forward(x)
            error = torch.abs(pred - y)
            if error.numel() == 1:
                return error.item()
            else:
                return float(torch.mean(error).item())

    def update(self, x, y):
        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def set_optimizer_sgd(self, lr=0.1):
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def get_params(self):
        return copy.deepcopy(self.state_dict())

    def set_params(self, params):
        self.load_state_dict(params)

# ==========================================
# 1. Common Components
# ==========================================
class FullScanADWIN:
    def __init__(self, delta=0.05, max_window_size=1000):
        self.delta = delta
        self.window = deque()
        self.total = 0.0
        self.total_sq = 0.0
        self.width = 0
        self.max_window_size = max_window_size
        self.drift_detected_flag = False

    def update(self, value):
        self.window.append(value)
        self.total += value
        self.total_sq += value ** 2
        self.width += 1
        self.drift_detected_flag = False

        if self.width > self.max_window_size:
            removed = self.window.popleft()
            self.total -= removed
            self.total_sq -= removed ** 2
            self.width -= 1

        self._check_drift()

    def _check_drift(self):
        if self.width < 10:
            return

        window_arr = np.array(self.window)
        cumsum = np.cumsum(window_arr)

        total_sum = self.total
        total_width = self.width

        delta_prime = self.delta / max(1, total_width)
        ln_term = math.log(max(1e-12, 2.0 / delta_prime))

        best_cut_n0 = -1
        max_diff_vs_epsilon = -1.0
        drift_found = False

        mean_W = self.total / self.width
        variance_W = max(0.0, (self.total_sq / self.width) - (mean_W ** 2))

        for n0 in range(1, total_width):
            n1 = total_width - n0
            sum0 = cumsum[n0 - 1]
            sum1 = total_sum - sum0
            mu0 = sum0 / n0
            mu1 = sum1 / n1
            diff = abs(mu0 - mu1)

            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = math.sqrt((2.0 / m) * max(1e-12, variance_W) * ln_term) + (2.0 / (3.0 * m)) * ln_term

            if diff > epsilon:
                metric = diff - epsilon
                if metric > max_diff_vs_epsilon:
                    max_diff_vs_epsilon = metric
                    best_cut_n0 = n0
                    drift_found = True

        if drift_found:
            self.drift_detected_flag = True
            for _ in range(best_cut_n0):
                rm = self.window.popleft()
                self.total -= rm
                self.total_sq -= rm ** 2
                self.width -= 1
            return

    @property
    def drift_detected(self):
        return self.drift_detected_flag

    def reset(self):
        self.window.clear()
        self.total = 0.0
        self.total_sq = 0.0
        self.width = 0
        self.drift_detected_flag = False

class BaseClient:
    def __init__(self, client_id, server, initial_models, initial_stats=None, distance_threshold=0.1, verbose=True):
        self.client_id = client_id
        self.server = server
        self.distance_threshold = distance_threshold
        self.verbose = verbose

        # models: {model_id: SimpleMLP()}
        self.models = copy.deepcopy(initial_models)
        self.current_model_id = 0

        if initial_stats:
            self.model_stats = copy.deepcopy(initial_stats)
        else:
            self.model_stats = {mid: {'n': 0, 'mean': 0.0, 'M2': 0.0} for mid in initial_models}

        self.train_data_store = defaultdict(list)
        self.stored_data = defaultdict(list)
        self.stored_data_limit = 50

        # pending を保持するが、作成直後は next-round まで ready にしない設計
        self.pending_model_params = None
        self.pending_model_stats = None
        self.pending_model_ready = True  # True: server will collect; False: wait one round

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

        self.batch_size = 32
        self.updates_per_step = 1

        self.next_temp_id = -100 - self.client_id

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

    def _store_evaluation_data(self, model_id, data_list):
        if model_id < 0:
            return
        current_stored = self.stored_data[model_id]
        sample_size = min(len(data_list), 20)
        if sample_size == 0:
            return
        sampled = random.sample(data_list, sample_size)
        current_stored.extend(sampled)
        if len(current_stored) > self.stored_data_limit:
            self.stored_data[model_id] = current_stored[-self.stored_data_limit:]

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

    def has_pending_model(self):
        # pending があり、且つ ready フラグが True の場合のみ、サーバが回収対象とみなす
        return (self.pending_model_params is not None) and bool(getattr(self, "pending_model_ready", True))

    def get_pending_model_info(self):
        return self.pending_model_params, self.pending_model_stats

    def promote_pending_to_ready(self):
        """ラウンド境界などで呼ぶ: pending がある場合に next-aggregation で回収可能にする"""
        if self.pending_model_params is not None and not getattr(self, "pending_model_ready", True):
            self.pending_model_ready = True
            if self.verbose:
                print(f"Client {self.client_id}: pending model promoted to ready for next aggregation.")

    def confirm_model_registration(self, new_global_id):
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

        # confirm されれば pending はクリア
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
        eval_data = []
        if target_model_id in self.stored_data and len(self.stored_data[target_model_id]) > 5:
            eval_data = self.stored_data[target_model_id]
        elif target_model_id == self.current_model_id and len(self.train_data_store[target_model_id]) > 10:
            eval_data = self.train_data_store[target_model_id]

        if len(eval_data) < 5:
            return 0, 0.0, 0.0
        if len(eval_data) > 50:
            eval_data = random.sample(eval_data, 50)

        X = torch.cat([d[0] for d in eval_data])
        y = torch.cat([d[1] for d in eval_data])
        temp_model = SimpleMLP()
        temp_model.set_params(params)
        with torch.no_grad():
            preds = temp_model(X)
            errors = torch.abs(preds - y).numpy().flatten()
        return len(errors), float(np.sum(errors)), float(np.sum(errors**2))

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

    def _alloc_temp_id(self):
        """新しい一意のテンポラリ（負）IDを返す。呼び出すたびに減らしていく。"""
        temp_id = self.next_temp_id
        self.next_temp_id -= 1
        return temp_id

# ==========================================
# 2. Specific Clients (ADWIN vs Periodic)
# ==========================================
class AdwinClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adwin = FullScanADWIN(delta=0.05)
        self.buffer = deque()
        self.safe_margin = 30

    # process_one_step: simplified signature, uses self.processed_samples internally
    def process_one_step(self, x_in, y_in, concept_id):
        start_time = time.perf_counter()
        x = x_in.unsqueeze(0) if x_in.dim() == 1 else x_in
        y = y_in.unsqueeze(0) if y_in.dim() == 1 else y_in

        # current sample index for this client (before increment)
        idx = self.processed_samples
        # advance processed_samples (so subsequent call sees next index)
        self.processed_samples += 1

        current_model = self.models[self.current_model_id]
        pred = current_model.predict(x)
        acc = 1.0 if pred.view(-1)[0].item() == y.view(-1)[0].item() else 0.0

        # per-sample logs
        self.history_accuracy.append(acc)
        self.history_concept.append(concept_id)
        self.history_model_id.append(self.current_model_id)

        error = current_model.get_absolute_error(x, y)
        self.adwin.update(error)
        self.buffer.append((x, y))

        drift_type = 0

        if self.adwin.drift_detected:
            # record detector's detection position for potential debug/visualization
            self.detected_event_positions.append(idx)
            # pass sample_idx to _resolve_drift for fine-grained logging
            drift_type = self._resolve_drift(sample_idx=idx)
        else:
            # --- New: forced-check when ADWIN hasn't flagged drift but window size & loss indicate potential drift ---
            width = self.adwin.width
            lower_bound = max(0, self.safe_margin - 5)
            upper_bound = max(100, 2 * max(0, (self.safe_margin - 5)))

            forced_triggered = False
            if lower_bound <= width <= upper_bound and width > 0 and self.current_model_id >= 0:
                # compute current model loss on the ADWIN-windowed buffer (use buffer tail of length width)
                # Ensure buffer length matches or exceeds width
                if len(self.buffer) >= width:
                    tail = list(self.buffer)[-width:]
                    bx = torch.cat([d[0] for d in tail])
                    by = torch.cat([d[1] for d in tail])
                    with torch.no_grad():
                        preds = current_model(bx)
                        window_loss = float(torch.mean(torch.abs(preds - by)).item())
                    hist_mean, _ = self._get_model_stats(self.current_model_id)

                    if hist_mean > 0.0 and (window_loss >= hist_mean + self.distance_threshold):
                        forced_triggered = True
                        if self.verbose:
                            print(f"Client {self.client_id} [sample={idx}]: Forced drift-check triggered (win={width}, loss={window_loss:.3f}, base={hist_mean:.3f}, thr={self.distance_threshold:.3f})")

            if forced_triggered:
                # perform same resolution flow as when ADWIN signals
                self.detected_event_positions.append(idx)
                drift_type = self._resolve_drift(sample_idx=idx)
            else:
                # normal behavior: commit old buffered samples to training stats & data store, then train
                while len(self.buffer) > self.safe_margin:
                    old_x, old_y = self.buffer.popleft()
                    loss_val = current_model.get_absolute_error(old_x, old_y)
                    self._update_model_stats(self.current_model_id, loss_val)
                    self.train_data_store[self.current_model_id].append((old_x, old_y))
                self.train_all_held_models(count_multiplier=1)

        self.history_drift_type.append(drift_type)

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        num_global = sum(1 for mid in self.models.keys() if mid >= 0)
        self.processing_times[num_global].append(elapsed_ms)

    # _resolve_drift records local switches into self.local_switch_positions
    def _resolve_drift(self, sample_idx):
        buffer_list = list(self.buffer)
        n_new_concept = self.adwin.width

        if len(buffer_list) <= n_new_concept:
            drift_data = buffer_list
            old_data = []
        else:
            old_data = buffer_list[:-n_new_concept]
            drift_data = buffer_list[-n_new_concept:]

        if len(old_data) > 0:
            self._store_evaluation_data(self.current_model_id, old_data)
            for d in old_data:
                self.train_data_store[self.current_model_id].append(d)
                with torch.no_grad():
                    l_val = self.models[self.current_model_id].get_absolute_error(d[0], d[1])
                self._update_model_stats(self.current_model_id, l_val)

        if len(drift_data) < 5:
            self.adwin.reset()
            return 0

        if self.verbose:
            print(f"Client {self.client_id} [sample={sample_idx}]: ADWIN Drift Detected.")

        bx = torch.cat([d[0] for d in drift_data])
        by = torch.cat([d[1] for d in drift_data])
        m = len(drift_data)

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
            else:
                diff = loss - hist_mean
                if self.verbose:
                    print(f"  Check M{m_id}: Diff={diff:.3f} vs Thr={self.distance_threshold:.3f} (Loss={loss:.3f}, Base={hist_mean:.3f})")
                if diff <= self.distance_threshold:
                    valid_candidates.append((m_id, loss))

        drift_type = 0

        if valid_candidates:
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

            for d in drift_data:
                self.train_data_store[self.current_model_id].append(d)
                with torch.no_grad():
                    l_val = self.models[self.current_model_id].get_absolute_error(d[0], d[1])
                self._update_model_stats(self.current_model_id, l_val)
        else:
            temp_id = self._alloc_temp_id()
            if self.verbose:
                print(f"  -> Unknown Drift! New Model (Temp ID: {temp_id})")

            new_model = SimpleMLP()
            current_params = self.models[self.current_model_id].get_params()
            new_model.set_params(current_params)
            new_model.set_optimizer_sgd(lr=0.1)

            n_epochs = 30
            dataset = torch.utils.data.TensorDataset(bx, by)
            loader = torch.utils.data.DataLoader(dataset, batch_size=min(32, m), shuffle=True)
            for _ in range(n_epochs):
                for b_x, b_y in loader:
                    new_model.update(b_x, b_y)

            self.models[temp_id] = new_model
            # 記録は sample_idx で行う（ずれ防止）
            self.local_switch_positions.append(sample_idx)
            self.current_model_id = temp_id

            with torch.no_grad():
                preds = new_model(bx)
                final_loss = torch.abs(preds - by)
                init_mean = float(torch.mean(final_loss).item())
                init_var = float(torch.var(final_loss).item())
                if math.isnan(init_var):
                    init_var = 0.1

            self.model_stats[temp_id] = {'n': m, 'mean': init_mean, 'M2': init_var * max(1, (m - 1))}
            # pending に登録するが、作成ラウンド内ではサーバへ送らない（ready=False）
            self.pending_model_params = new_model.get_params()
            self.pending_model_stats = self.model_stats[temp_id]
            self.pending_model_ready = False
            drift_type = 2
            for d in drift_data:
                self.train_data_store[temp_id].append(d)

        self.adwin.reset()
        self.buffer.clear()
        return drift_type

class PeriodicClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_min_loss = None

    def phase1_detect(self, batch_data, t, concept_id):
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

            current_model = self.models[self.current_model_id]
            pred = current_model.predict(x)
            acc = 1.0 if pred.view(-1)[0].item() == y.view(-1)[0].item() else 0.0

            self.history_accuracy.append(acc)
            self.history_concept.append(concept_id)
            self.history_model_id.append(self.current_model_id)

            # mark this sample processed
            self.processed_samples += 1

        bx = torch.cat(batch_x)
        by = torch.cat(batch_y)
        m = len(batch_data)

        min_loss = float('inf')
        best_model_id = self.current_model_id

        for m_id, model in self.models.items():
            with torch.no_grad():
                preds = model(bx)
                loss = float(torch.mean(torch.abs(preds - by)).item())
            if loss < min_loss:
                min_loss = loss
                best_model_id = m_id

        drift_type = 0
        is_drift = False

        if self.last_min_loss is not None:
            if min_loss > self.last_min_loss + self.distance_threshold:
                is_drift = True
                if self.verbose:
                    print(f"Client {self.client_id} [t={t}]: Drift Detected (Loss {min_loss:.3f})")

        if is_drift:
            # record detector's detection position for debugging (not plotted by default)
            self.detected_event_positions.append(start_idx)

            temp_id = self._alloc_temp_id()
            if self.verbose:
                print(f"  -> Unknown Drift! New Model (Temp ID: {temp_id})")
            new_model = SimpleMLP()
            current_params = self.models[self.current_model_id].get_params()
            new_model.set_params(current_params)
            new_model.set_optimizer_sgd(lr=0.1)

            n_epochs = 30
            dataset = torch.utils.data.TensorDataset(bx, by)
            loader = torch.utils.data.DataLoader(dataset, batch_size=min(32, m), shuffle=True)
            for _ in range(n_epochs):
                for b_x, b_y in loader:
                    new_model.update(b_x, b_y)

            self.models[temp_id] = new_model
            # batch の最後のサンプルインデックスで切替記録（start_idx + m - 1）
            switch_idx = start_idx + m - 1
            self.local_switch_positions.append(switch_idx)
            self.current_model_id = temp_id

            with torch.no_grad():
                preds = new_model(bx)
                final_loss = torch.abs(preds - by)
                init_mean = float(torch.mean(final_loss).item())
                init_var = float(torch.var(final_loss).item())
                if math.isnan(init_var):
                    init_var = 0.1

            self.model_stats[temp_id] = {'n': m, 'mean': init_mean, 'M2': init_var * max(1, (m - 1))}
            # FedDriftでは作成ラウンド内でサーバへ送る（ready=True）
            self.pending_model_params = new_model.get_params()
            self.pending_model_stats = self.model_stats[temp_id]
            self.pending_model_ready = True
            drift_type = 2

            self.last_min_loss = init_mean

            for d in processed_batch_data:
                self.train_data_store[temp_id].append(d)
        else:
            if best_model_id != self.current_model_id:
                if self.verbose:
                    print(f"  -> Switch to Model {best_model_id}")
                # batch の最後のサンプルインデックスで切替記録
                switch_idx = start_idx + m - 1
                self.local_switch_positions.append(switch_idx)
                self.current_model_id = best_model_id
                drift_type = 1

            for (x, y) in processed_batch_data:
                with torch.no_grad():
                    l_val = self.models[self.current_model_id].get_absolute_error(x, y)
                self._update_model_stats(self.current_model_id, l_val)
                self.train_data_store[self.current_model_id].append((x, y))

            self.last_min_loss = min_loss

        self._store_evaluation_data(self.current_model_id, processed_batch_data)
        for _ in range(m):
            self.history_drift_type.append(drift_type)

    def phase2_train(self, k_steps):
        self.train_all_held_models(count_multiplier=k_steps)

# ==========================================
# 3. Server (Common)
# ==========================================
class Server:
    def __init__(self, distance_threshold=0.1, verbose=True):
        self.global_models = {}
        self.next_model_id = 1
        self.clients = []
        self.distance_threshold = distance_threshold
        self.verbose = verbose
        self.global_stats = defaultdict(lambda: {'n': 0, 'mean': 0.0, 'M2': 0.0})

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
        new_registrations = 0
        for c in self.clients:
            if c.has_pending_model():
                params, stats = c.get_pending_model_info()
                new_global_id = self.request_new_model_id()
                self.register_model_params(new_global_id, params)
                self.register_model_stats(new_global_id, stats)
                c.confirm_model_registration(new_global_id)
                new_registrations += 1

        if new_registrations > 0 and self.verbose:
            print(f"Server [t={t}]: Collected {new_registrations} new models.")

        active_ids = sorted(list(self.global_models.keys()))
        M = len(active_ids)

        if clustering_enabled and M > 1:
            stats_matrix = self._cross_evaluate(active_ids)
            clusters = self.perform_hierarchical_clustering(active_ids, stats_matrix)

            if len(clusters) < M:
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

                active_ids = sorted(list(self.global_models.keys()))

        self.update_global_models(active_ids)
        self.broadcast_models()

    def update_global_models(self, active_ids):
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
        for c in self.clients:
            c.apply_server_mapping({}, self.global_models, self.global_stats)

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
                if len(target_clients) > 3:
                    target_clients = random.sample(target_clients, 3)

                total_n, total_S, total_SS = 0, 0.0, 0.0
                for c in target_clients:
                    n, S, SS = c.evaluate_model(params_i, target_model_id=id_j)
                    total_n += n; total_S += S; total_SS += SS

                stats_matrix[id_i][id_j] = (total_n, total_S, total_SS)
        return stats_matrix

    def perform_hierarchical_clustering(self, model_ids, stats_matrix):
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

                if stats_ii[0] < 5 or stats_ij[0] < 5 or stats_jj[0] < 5 or stats_ji[0] < 5:
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

# ==========================================
# 4. Main Experiment Logic
# ==========================================
def run_random_drift_experiment(mode='FedDrift', distance_threshold=0.1, random_seed=None, verbose=True, show_plot=True):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    no_federated = False
    if mode == 'NoFed_adwin_serial':
        is_proposed = True
        no_federated = True
    else:
        is_proposed = (mode == 'FedDrift_adwin_serial')

    print(f"=== System Experiment: {mode} (Threshold={distance_threshold}) ===")

    model0 = SimpleMLP()
    stats_0 = {'n': 0, 'mean': 0.0, 'M2': 0.0}

    replay_buf = []
    for _ in range(500):
        x_in, y_in = generate_data(0)
        replay_buf.append((x_in, y_in))

    for _ in range(10):
        random.shuffle(replay_buf)
        batch_size = 32
        for i in range(0, len(replay_buf), batch_size):
            batch = replay_buf[i:i+batch_size]
            bx = torch.stack([d[0] for d in batch])
            by = torch.stack([d[1] for d in batch])
            model0.update(bx, by)

    for x, y in replay_buf:
        loss_val = model0.get_absolute_error(x.unsqueeze(0), y.unsqueeze(0))
        stats_0['n'] += 1
        delta = loss_val - stats_0['mean']
        stats_0['mean'] += delta / stats_0['n']
        delta2 = loss_val - stats_0['mean']
        stats_0['M2'] += delta * delta2

    initial_models = {0: model0}
    initial_stats = {0: stats_0}
    init_params = model0.get_params()

    server = Server(distance_threshold=distance_threshold, verbose=verbose)
    server.register_model_params(0, init_params)
    server.register_model_stats(0, stats_0)

    clients = []

    ClientClass = AdwinClient if is_proposed else PeriodicClient

    for i in range(10):
        c = ClientClass(
            client_id=i,
            server=server,
            initial_models=initial_models,
            initial_stats=initial_stats,
            distance_threshold=distance_threshold,
            verbose=verbose
        )
        # サーバを使うモードのみ server.register_client(c) しておく（NoFedでは登録しない）
        if not no_federated:
            server.register_client(c)
        clients.append(c)

    if verbose:
        print("Clients initialized. All holding Model 0.")

    # R_ROUNDS = 5
    # K_STEPS = 10
    R_ROUNDS = 1
    K_STEPS = 50
    TOTAL_DATA_POINTS = 3000
    DATA_PER_TIME = R_ROUNDS * K_STEPS
    T_STEPS = TOTAL_DATA_POINTS // DATA_PER_TIME

    DRIFT_PROB = 0.0015 * DATA_PER_TIME
    MIN_STABLE_PERIOD = 300 // DATA_PER_TIME
    NUM_CONCEPTS = 4

    client_concept_schedule = []
    for i in range(10):
        schedule = []
        curr = 0
        last_drift = 0
        for data_idx in range(TOTAL_DATA_POINTS):
            if (data_idx - last_drift > 300) and (random.random() < 0.0015):
                candidates = [cid for cid in range(NUM_CONCEPTS) if cid != curr]
                curr = random.choice(candidates)
                last_drift = data_idx
            schedule.append(curr)
        client_concept_schedule.append(schedule)

    true_drift_events = {i: [] for i in range(10)}
    for i in range(10):
        sched = client_concept_schedule[i]
        for idx in range(1, len(sched)):
            if sched[idx] != sched[idx-1]:
                true_drift_events[i].append(idx)

    all_client_data = []
    for i in range(10):
        stream = []
        for idx in range(TOTAL_DATA_POINTS):
            stream.append(generate_data(client_concept_schedule[i][idx]))
        all_client_data.append(stream)

    if verbose:
        print(f"Simulation Start (Total Data={TOTAL_DATA_POINTS}, Mode={mode})...")

    global_data_idx = 0

    # --- measure wall-clock runtime for the whole experiment ---
    exp_start = time.perf_counter()

    for t in range(T_STEPS):
        if verbose and t % 5 == 0:
            print(f"--- Time {t} (Data Index {global_data_idx}) ---")

        start_idx = t * DATA_PER_TIME
        end_idx = (t + 1) * DATA_PER_TIME

        current_time_data = [all_client_data[i][start_idx:end_idx] for i in range(10)]
        current_time_concepts = [client_concept_schedule[i][start_idx:end_idx] for i in range(10)]

        if is_proposed:
            chunk_size = K_STEPS
            for r in range(R_ROUNDS):
                r_offset = r * chunk_size
                for k in range(K_STEPS):
                    k_idx = r_offset + k
                    if k_idx >= len(current_time_data[0]): break
                    for i, c in enumerate(clients):
                        x_in, y_in = current_time_data[i][k_idx]
                        con = current_time_concepts[i][k_idx]
                        c.process_one_step(x_in, y_in, con)

                # Proposed: サーバ同期は no_federated フラグで制御
                if not no_federated:
                    has_new = any(c.has_pending_model() for c in clients)
                    server.run_aggregation_and_merge(t, clustering_enabled=has_new)
                    # aggregation 後に "pending -> ready" を行い、次ラウンドで回収されるようにする
                    for c in clients:
                        c.promote_pending_to_ready()
                else:
                    # NoFed: ローカル処理のみ、サーバは呼ばない
                    if verbose and random.random() < 0.01:
                        # 少しだけ可視化用ログ（毎回だとうるさい）
                        print(f"  [NoFed] t={t}, r={r}: skipped server aggregation (local-only).")
        else:
            # FedDrift baseline behavior (unchanged)
            for i, c in enumerate(clients):
                batch_data = current_time_data[i]
                last_con = current_time_concepts[i][-1]
                c.phase1_detect(batch_data, t, last_con)
            # baseline always performs server cluster/aggregate here (we keep same behavior)
            server.run_aggregation_and_merge(t, clustering_enabled=True)
            for c in clients:
                c.promote_pending_to_ready()

            for r in range(R_ROUNDS):
                for c in clients:
                    c.phase2_train(k_steps=K_STEPS)
                # This aggregation only updates models (no clustering)
                server.run_aggregation_and_merge(t, clustering_enabled=False)
                for c in clients:
                    c.promote_pending_to_ready()

        global_data_idx += DATA_PER_TIME

    # experiment end time
    exp_end = time.perf_counter()
    runtime_seconds = exp_end - exp_start

    if verbose:
        print("Simulation Finished.")
        print(f"  Experiment runtime: {runtime_seconds:.3f} sec")

    # --- Metrics (ローカル切替(local_switch_positions)を検出イベントとみなす) ---
    all_accs = []
    for c in clients:
        all_accs.extend(c.history_accuracy)
    avg_accuracy = sum(all_accs) / len(all_accs) if len(all_accs) > 0 else 0.0

    DELAY_TOLERANCE = 100
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_true_drifts = 0
    delays = []

    total_local_switches = 0
    for i in range(10):
        c = clients[i]
        local_sw = sorted(c.local_switch_positions)
        total_local_switches += len(local_sw)

    # per-client greedy matching
    total_used_switches = 0
    for i in range(10):
        c = clients[i]
        true_drifts = list(true_drift_events[i])  # sample indices where concept changed
        local_sw = sorted(c.local_switch_positions)
        used = set()

        for td_time in true_drifts:
            total_true_drifts += 1
            matched = False
            for j, sw in enumerate(local_sw):
                if j in used:
                    continue
                if td_time <= sw <= td_time + DELAY_TOLERANCE:
                    total_tp += 1
                    delays.append(sw - td_time)
                    used.add(j)
                    matched = True
                    break
            if not matched:
                total_fn += 1

        total_used_switches += len(used)
        # remaining unused local switches count as FP
        total_fp += (len(local_sw) - len(used))

    total_detections = total_local_switches  # 検出は「ローカルで実際に切替を実行した回数」
    fn_rate = total_fn / total_true_drifts if total_true_drifts > 0 else 0.0
    fdr = total_fp / total_detections if total_detections > 0 else 0.0
    recall = total_tp / total_true_drifts if total_true_drifts > 0 else 0.0
    precision = total_tp / total_detections if total_detections > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_delay = sum(delays) / len(delays) if delays else 0.0

    final_model_count = len(server.global_models)

    results = {
        "accuracy": avg_accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "miss_rate": fn_rate,
        "fdr": fdr,
        "avg_delay": avg_delay,
        "total_true": total_true_drifts,
        "total_detect": total_detections,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "final_model_count": final_model_count,
        "runtime_seconds": runtime_seconds,
    }

    if verbose:
        print("\n=== Experiment Metrics ===")
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  Recall (TP Rate): {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Avg Delay: {avg_delay:.1f} steps")
        print(f"  Final Global Models: {final_model_count}")
        print(f"  Total Local Switches (total_detect): {total_detections}")
        print(f"  TP / FP / FN: {total_tp} / {total_fp} / {total_fn}")
        print(f"  Runtime: {runtime_seconds:.3f} sec")

    # Plotting: show model-switch markers (local switches) and concept backgrounds
    if show_plot:
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        window = 50
        avg_acc_history = []
        cmap = plt.get_cmap('tab10')

        for i, c in enumerate(clients):
            if len(c.history_accuracy) >= window:
                sm = np.convolve(c.history_accuracy, np.ones(window)/window, mode='valid')
                plt.plot(range(len(sm)), sm, alpha=0.3, linewidth=1.0, color=cmap(i), label=f'C{i}')
            else:
                plt.plot(range(len(c.history_accuracy)), c.history_accuracy, alpha=0.3, linewidth=1.0, color=cmap(i), label=f'C{i}')

        min_len = min(len(c.history_accuracy) for c in clients)
        for idx in range(min_len):
            accs = [c.history_accuracy[idx] for c in clients]
            avg_acc_history.append(sum(accs)/len(accs))

        if len(avg_acc_history) >= window:
            smooth_acc = np.convolve(avg_acc_history, np.ones(window)/window, mode='valid')
            plt.plot(range(len(smooth_acc)), smooth_acc, color='black', linewidth=2.5, label='Avg')

        plt.title(f"System Accuracy (Avg: {avg_accuracy:.3f})")
        plt.ylim(0, 1.1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']

        for i, c in enumerate(clients):
            ranges = []
            if len(c.history_concept) > 0:
                curr_c = c.history_concept[0]
                start_t = 0
                for t_idx, con in enumerate(c.history_concept):
                    if con != curr_c:
                        ranges.append((start_t, t_idx - start_t, curr_c))
                        start_t = t_idx
                        curr_c = con
                ranges.append((start_t, len(c.history_concept) - start_t, curr_c))

            for (start, width, con) in ranges:
                plt.broken_barh([(start, width)], (i-0.4, 0.8), facecolors=colors[con], alpha=0.5)

            model_ranges = []
            if len(c.history_model_id) > 0:
                curr_m = c.history_model_id[0]
                start_t = 0
                for t_idx, mid in enumerate(c.history_model_id):
                    if mid != curr_m:
                        model_ranges.append((start_t, t_idx - 1, curr_m))
                        start_t = t_idx
                        curr_m = mid
                model_ranges.append((start_t, len(c.history_model_id) - 1, curr_m))

            for start, end, mid in model_ranges:
                if start > 0:
                    plt.vlines(start, i-0.4, i+0.4, colors='black', linestyles='dotted', alpha=0.7)

                mid_str = str(mid) if mid >= 0 else "x"
                center_t = (start + end) / 2
                plt.text(center_t, i, mid_str, fontsize=9, va='center', ha='center', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.5))

            # Draw local-switch markers (red triangles) at recorded local_switch_positions
            for sw in c.local_switch_positions:
                plt.plot(sw, i, marker='^', color='red', markersize=6)

        plt.yticks(range(10), [f"Client {i}" for i in range(10)])
        plt.title(f"Concept (Color) vs Model ID (Text) [{mode}]")

        patches = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(4)]
        marker_handle = Line2D([0], [0], marker='^', color='red', linestyle='None', markersize=8)
        # combine handles and labels so legend shows both concepts and local-switch marker
        handles = patches + [marker_handle]
        labels = [f"Concept {i}" for i in range(4)] + ["Local switch (drift detection)"]
        plt.legend(handles=handles, labels=labels, loc='upper right')

        plt.tight_layout()
        plt.show()

        # 個別詳細プロット
        fig2, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            c = clients[i]
            if len(c.history_accuracy) >= window:
                sm = np.convolve(c.history_accuracy, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(c.history_accuracy)), sm, color='blue', label='Accuracy')
            else:
                ax.plot(range(len(c.history_accuracy)), c.history_accuracy, color='blue', label='Accuracy')

            ranges = []
            if len(c.history_concept) > 0:
                curr_c = c.history_concept[0]
                start_t = 0
                for t_idx, con in enumerate(c.history_concept):
                    if con != curr_c:
                        ranges.append((start_t, t_idx - start_t, curr_c))
                        start_t = t_idx
                        curr_c = con
                ranges.append((start_t, len(c.history_concept) - start_t, curr_c))

            for (start, width, con) in ranges:
                ax.axvspan(start, start+width, facecolor=colors[con], alpha=0.2)

            model_ranges = []
            if len(c.history_model_id) > 0:
                curr_m = c.history_model_id[0]
                start_t = 0
                for t_idx, mid in enumerate(c.history_model_id):
                    if mid != curr_m:
                        model_ranges.append((start_t, t_idx - 1, curr_m))
                        start_t = t_idx
                        curr_m = mid
                model_ranges.append((start_t, len(c.history_model_id) - 1, curr_m))

            for start, end, mid in model_ranges:
                if start > 0:
                    ax.axvline(x=start, color='black', linestyle=':', alpha=0.8)
                mid_str = str(mid) if mid >= 0 else "x"
                center_t = (start + end) / 2
                ax.text(center_t, 0.1, mid_str, fontsize=9, va='center', ha='center',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8))

            # model switch markers (local)
            for sw in c.local_switch_positions:
                ax.plot(sw, 0.05, marker='^', color='red', markersize=8)

            ax.set_title(f"Client {i}")
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True)
            if i >= 8: ax.set_xlabel("Time Step")
            if i % 2 == 0: ax.set_ylabel("Accuracy")

        plt.tight_layout()
        plt.show()

    return results

# if __name__ == "__main__":
#     res1 = run_random_drift_experiment(mode='FedDrift_adwin_serial', random_seed=0, verbose=True, show_plot=True)
#     res2 = run_random_drift_experiment(mode='FedDrift', random_seed=0, verbose=True, show_plot=True)
#     res3 = run_random_drift_experiment(mode='NoFed_adwin_serial', random_seed=0, verbose=True, show_plot=True)
#     print("Proposed:", res1)
#     print("FedDrift:", res2)
#     print("NoFed_adwin_serial:", res3)