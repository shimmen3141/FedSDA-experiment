"""Oblivious ベースライン クライアント。

単一モデルを FedAvg で学習し、ドリフト検出・モデル切替・新規作成を一切行わない。
"""
import time

from .base import BaseClient


class ObliviousClient(BaseClient):
    """Oblivious ベースライン (FedDrift): 単一モデルを FedAvg で学習し、ドリフト検出・
    モデル切替・新規作成を一切行わない。到着したデータは全て唯一のモデル(ID 0)の
    ストアに蓄積して逐次学習する(全データ利用 = FedDrift の Oblivious/all に相当)。"""

    def process_one_step(self, x_in, y_in, concept_id):
        """1サンプルを処理する: 予測(test-then-train)→ ストア追加 → 学習。"""
        start_time = time.perf_counter()
        x = x_in.unsqueeze(0) if x_in.dim() == 1 else x_in
        y = y_in.unsqueeze(0) if y_in.dim() == 1 else y_in

        self.processed_samples += 1
        self._record_prediction(x, y, concept_id)

        # 常に唯一のモデル(current_model_id=0)へデータを入れて学習(切替・新規作成なし)
        loss_val = self.models[self.current_model_id].get_absolute_error(x, y)
        self._update_model_stats(self.current_model_id, loss_val)
        self.train_data_store[self.current_model_id].append((x, y))
        self.train_all_held_models(count_multiplier=1)

        self.history_drift_type.append(0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        num_global = sum(1 for mid in self.models.keys() if mid >= 0)
        self.processing_times[num_global].append(elapsed_ms)
