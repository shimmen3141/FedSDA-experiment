"""FedDrift ベースライン クライアント。

「全モデルの最小損失」の増分を監視し、閾値超過でドリフト判定する。検出は
config.FEDDRIFT_DETECT_BATCH 件ごとに行い、時刻粒度(K_STEPS)からは分離している。
"""
import torch

from .. import config
from .base import BaseClient


class PeriodicClient(BaseClient):
    """FedDriftベースライン: 検出バッチ単位で最小損失の増分によりドリフト判定する。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_min_loss = None
        self.detect_buffer = []                             # 検出待ちのデータ
        self.detect_batch_size = config.FEDDRIFT_DETECT_BATCH

    def process_batch(self, batch_data, concept_ids):
        """時刻ブロックのデータを1件ずつ処理(予測ログ + 検出バッファへ蓄積)。

        バッファが検出バッチサイズに達するたびに検出+割り当てを実行する。検出バッチは
        時刻粒度(data_per_time)と独立で、複数時刻にまたがって蓄積されることもある。
        この呼び出し中に検出バッチが1回でも完了したら True を返す(通信タイミングの判定用)。
        """
        fired = False
        for (x_in, y_in), con in zip(batch_data, concept_ids):
            x = x_in.unsqueeze(0) if x_in.dim() == 1 else x_in
            y = y_in.unsqueeze(0) if y_in.dim() == 1 else y_in

            self._record_prediction(x, y, con)
            self.processed_samples += 1
            self.detect_buffer.append((x, y))

            drift_type = 0
            if len(self.detect_buffer) >= self.detect_batch_size:
                drift_type = self._detect_and_assign(self.detect_buffer)
                self.detect_buffer = []
                fired = True
            self.history_drift_type.append(drift_type)
        return fired

    def flush(self):
        """ストリーム終端で残った検出バッファ(部分バッチ)を検出+割り当てする。"""
        if self.detect_buffer:
            self._detect_and_assign(self.detect_buffer)
            self.detect_buffer = []

    def _detect_and_assign(self, batch):
        """検出バッチに対しドリフト判定・モデル選択/新規作成・データ割り当てを行う。"""
        processed_batch_data = list(batch)
        bx = torch.cat([d[0] for d in batch])
        by = torch.cat([d[1] for d in batch])
        m = len(batch)
        first_idx = self.processed_samples - m   # バッチ先頭のサンプルindex
        last_idx = self.processed_samples - 1    # 検出を発火したサンプル(バッチ末尾)

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
                print(f"Client {self.client_id} [sample={last_idx}]: Drift Detected (Loss {min_loss:.3f})")
            self.detected_event_positions.append(first_idx)

            # FedDriftでは新規モデルを作成ラウンド内でサーバへ送る(ready=True)
            temp_id, init_mean = self._spawn_new_model(bx, by, pending_ready=True)
            self.local_switch_positions.append(last_idx)
            self.current_model_id = temp_id
            drift_type = 2
            self.last_min_loss = init_mean
            self.train_data_store[temp_id].extend(processed_batch_data)
        else:
            drift_type = 0
            if best_model_id != self.current_model_id:
                if self.verbose:
                    print(f"  -> Switch to Model {best_model_id}")
                self.local_switch_positions.append(last_idx)
                self.current_model_id = best_model_id
                drift_type = 1

            self._absorb_into_store(self.current_model_id, processed_batch_data)
            self.last_min_loss = min_loss

        self._store_evaluation_data(self.current_model_id, processed_batch_data)
        return drift_type

    def local_train(self, k_steps):
        self.train_all_held_models(count_multiplier=k_steps)
