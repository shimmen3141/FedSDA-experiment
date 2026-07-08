"""FedDrift ベースライン クライアント。

固定バッチ単位で「全モデルの最小損失」の増分を監視し、閾値超過でドリフト判定する。
"""
import torch

from .base import BaseClient


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
