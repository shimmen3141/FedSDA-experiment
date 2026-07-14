"""FedSDA(提案手法)クライアント。

ADWIN による統計的ドリフト検出 + FIFOバッファによる逐次(1サンプル単位)処理。
"""
import time
from collections import deque

import torch

from .. import config
from ..adwin import FullScanADWIN
from .base import BaseClient


class AdwinClient(BaseClient):
    """提案手法 (FedSDA) クライアント: ADWIN + FIFOバッファによる逐次(1サンプル単位)処理。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adwin = FullScanADWIN(delta=config.ADWIN_DELTA)
        self.buffer = deque()                       # FIFOバッファ(検知遅延中のデータ保留)
        self.fifo_size = config.FIFO_BUFFER_SIZE    # FIFOバッファ長 N_FIFO

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
            # τ>1 で保留中の更新をドリフト解決前に消化する(τ=1 では no-op)
            self.flush_pending_updates()
            self.detected_event_positions.append(idx)
            drift_type = self._resolve_drift(sample_idx=idx)
        else:
            # 平時: バッファ長 N_FIFO を超えた分だけ古いデータをストアへ確定し、学習する
            while len(self.buffer) > self.fifo_size:
                old_x, old_y = self.buffer.popleft()
                loss_val = self.models[self.current_model_id].get_absolute_error(old_x, old_y)
                self._update_model_stats(self.current_model_id, loss_val)
                self.train_data_store[self.current_model_id].append((old_x, old_y))
            self.train_step()

        self.history_drift_type.append(drift_type)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        num_global = sum(1 for mid in self.models.keys() if mid >= 0)
        self.processing_times[num_global].append(elapsed_ms)

    def _forced_drift_check(self, idx):
        """ADWIN未検知でも、直近ウィンドウの損失がベースラインから閾値以上悪化して
        いればドリフト解決を強制する保険的チェック。"""
        width = self.adwin.width
        lower_bound = max(0, self.fifo_size - 5)
        upper_bound = max(100, 2 * max(0, (self.fifo_size - 5)))

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
