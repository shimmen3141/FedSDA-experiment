"""FedSDA(提案手法)クライアント。

ADWIN による統計的ドリフト検出 + FIFOバッファによる逐次(1サンプル単位)処理。
"""
import time
from collections import defaultdict, deque

import torch

from .. import config
from ..adwin import FullScanADWIN
from ..e_detector import BoundedMeanEDetector
from .base import BaseClient


class FedSDAClient(BaseClient):
    """提案手法 (FedSDA) クライアント: ADWIN + FIFOバッファによる逐次(1サンプル単位)処理。"""

    reports_state_summary = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adwin = FullScanADWIN(delta=config.ADWIN_DELTA)
        self.buffer = deque()                       # FIFOバッファ(検知遅延中のデータ保留)
        self.fifo_size = config.FIFO_BUFFER_SIZE    # FIFOバッファ長 N_FIFO
        self.model_upload_delay_rounds = int(config.FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS)
        if self.model_upload_delay_rounds < 1:
            raise ValueError("FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS must be at least 1")
        self._pending_upload_rounds = 0
        # v3 のクロス評価では、ローカル学習中の models ではなく直近の配布時点を使う。
        self.cached_global_model_params = {
            model_id: model.get_params() for model_id, model in self.models.items()
            if model_id >= 0
        }
        self._refresh_cache_on_mapping = True

    def _spawn_new_model(self, bx, by, pending_ready=False):
        """新規モデルを作成し、設定された学習ラウンド数だけアップロードを保留する。"""
        result = super()._spawn_new_model(bx, by, pending_ready=False)
        self._pending_upload_rounds = self.model_upload_delay_rounds
        return result

    def promote_pending_to_ready(self):
        """ラウンド境界で保留期間を進め、満了した新規モデルを送信可能にする。"""
        if self.pending_model_params is None or self.pending_model_ready:
            return
        self._pending_upload_rounds -= 1
        if self._pending_upload_rounds <= 0:
            super().promote_pending_to_ready()

    def evaluate_cached_model(self, model_id, target_model_id):
        """直近のサーバ配布時点のモデルを、指定モデル用の手元データで評価する。"""
        try:
            params = self.cached_global_model_params[model_id]
        except KeyError:
            raise ValueError(f"モデル{model_id}はまだクライアントへ配布されていません") from None
        return self.evaluate_model(params, target_model_id)

    def apply_cached_merge(self, clusters, cluster_weights, global_stats=None):
        """ローカル学習モデルを統合するが、評価用キャッシュは次の配布まで維持する。"""
        self._refresh_cache_on_mapping = False
        try:
            super().apply_cached_merge(clusters, cluster_weights, global_stats)
        finally:
            self._refresh_cache_on_mapping = True

    def apply_server_mapping(self, id_mapping, new_global_models, new_global_stats=None):
        """サーバ配布を適用し、v3の次回クロス評価用キャッシュを更新する。"""
        super().apply_server_mapping(id_mapping, new_global_models, new_global_stats)
        if self._refresh_cache_on_mapping:
            self.cached_global_model_params = {
                model_id: model.get_params() for model_id, model in self.models.items()
                if model_id >= 0
            }

    def process_one_step(self, x_in, y_in, concept_id):
        """1サンプルを処理する: 予測 → ADWIN更新 → (ドリフト解決 | 平時処理) → 学習。"""
        start_time = time.perf_counter()
        training_before = self.phase_seconds["training"]
        x = x_in.unsqueeze(0) if x_in.dim() == 1 else x_in
        y = y_in.unsqueeze(0) if y_in.dim() == 1 else y_in

        # current sample index for this client (before increment)
        idx = self.processed_samples
        self.processed_samples += 1

        self._record_prediction(x, y, concept_id)

        self._record_model_compute("detection", len(x))
        error = self.models[self.current_model_id].get_absolute_error(x, y)
        drift_detected = self._update_drift_detectors(error, y, idx)
        self.buffer.append((x, y))

        drift_type = 0

        # ADWIN の統計的検知、または保険的な強制チェックのどちらかが発火したら解決処理へ
        if drift_detected or self._forced_drift_check(idx):
            # τ>1 で保留中の更新をドリフト解決前に消化する(τ=1 では no-op)
            self.flush_pending_updates()
            self.detected_event_positions.append(idx)
            drift_type = self._resolve_drift(sample_idx=idx)
        else:
            # 平時: バッファ長 N_FIFO を超えた分だけ古いデータをストアへ確定し、学習する
            while len(self.buffer) > self.fifo_size:
                old_x, old_y = self.buffer.popleft()
                self._record_model_compute("statistics", len(old_x))
                loss_val = self.models[self.current_model_id].get_absolute_error(old_x, old_y)
                self._update_model_stats(self.current_model_id, loss_val)
                self.train_data_store[self.current_model_id].append((old_x, old_y))
            self.train_step()

        self.history_drift_type.append(drift_type)

        elapsed = time.perf_counter() - start_time
        training_elapsed = self.phase_seconds["training"] - training_before
        self.phase_seconds["online"] += max(0.0, elapsed - training_elapsed)
        elapsed_ms = elapsed * 1000
        num_global = sum(1 for mid in self.models.keys() if mid >= 0)
        self.processing_times[num_global].append(elapsed_ms)

    def _update_drift_detectors(self, error, y, sample_idx):
        """全体損失ADWINを更新し、ドリフト検知の有無を返す。"""
        scan_width = min(self.adwin.width + 1, self.adwin.max_window_size)
        self.adwin.update(error)
        self.compute_counters["drift_detector_updates"] += 1
        if scan_width >= config.ADWIN_MIN_WIDTH:
            self.compute_counters["drift_detector_hypotheses"] += scan_width - 1
        return self.adwin.drift_detected

    def _new_concept_sample_count(self, buffer_list, sample_idx):
        """検知器が推定した新概念側のサンプル数を返す。"""
        return min(len(buffer_list), self.adwin.width)

    def _reset_drift_detectors(self):
        """ドリフト解決後に検知器を初期状態へ戻す。"""
        self.adwin.reset()

    def _detector_label(self):
        return "ADWIN"

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
            self._record_model_compute("detection", len(bx))
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
        n_new_concept = self._new_concept_sample_count(buffer_list, sample_idx)

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
            self._reset_drift_detectors()
            return 0

        if self.verbose:
            print(f"Client {self.client_id} [sample={sample_idx}]: "
                  f"{self._detector_label()} Drift Detected.")

        bx = torch.cat([d[0] for d in drift_data])
        by = torch.cat([d[1] for d in drift_data])

        # 各既存モデルの新概念データに対する損失を、ベースラインと比較して適合候補を集める
        valid_candidates = []
        for m_id, model in self.models.items():
            with torch.no_grad():
                self._record_model_compute("detection", len(bx))
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

        self._reset_drift_detectors()
        self.buffer.clear()
        return drift_type


class ClassConditionalFedSDAClient(FedSDAClient):
    """全体損失と正解クラス別損失を並列監視するFedSDAクライアント。

    全体ADWINが検知した場合は従来と同じ分割点を使う。全体が未検知で
    クラス別ADWINだけが検知した場合は、そのクラスの新ウィンドウに残った
    最初のサンプル位置を新概念の開始位置としてFIFOバッファを分割する。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_adwins = defaultdict(
            lambda: FullScanADWIN(delta=config.ADWIN_DELTA)
        )
        self.class_adwin_positions = defaultdict(deque)
        self._class_drift_start = None

    def _update_drift_detectors(self, error, y, sample_idx):
        overall_detected = super()._update_drift_detectors(error, y, sample_idx)
        class_id = int(y.view(-1)[0].item())
        detector = self.class_adwins[class_id]
        positions = self.class_adwin_positions[class_id]
        positions.append(sample_idx)
        scan_width = min(detector.width + 1, detector.max_window_size)
        detector.update(error)
        self.compute_counters["drift_detector_updates"] += 1
        if scan_width >= config.ADWIN_MIN_WIDTH:
            self.compute_counters["drift_detector_hypotheses"] += scan_width - 1

        # ADWINが最大窓制限またはドリフト検知で削除した古い標本位置を同期して除く。
        while len(positions) > detector.width:
            positions.popleft()

        class_detected = detector.drift_detected
        if not overall_detected and class_detected and positions:
            self._class_drift_start = positions[0]
        else:
            self._class_drift_start = None
        return overall_detected or class_detected

    def _new_concept_sample_count(self, buffer_list, sample_idx):
        if self.adwin.drift_detected or self._class_drift_start is None:
            return super()._new_concept_sample_count(buffer_list, sample_idx)
        return min(len(buffer_list), sample_idx - self._class_drift_start + 1)

    def _reset_drift_detectors(self):
        super()._reset_drift_detectors()
        for detector in self.class_adwins.values():
            detector.reset()
        self.class_adwin_positions.clear()
        self._class_drift_start = None


class EDetectorFedSDAClient(FedSDAClient):
    """全体損失をbounded mean e-SRで監視するFedSDAクライアント。

    v2.2/v3.2のアブレーション用に、クラス別ADWINと保険的な強制チェックは
    組み合わせない。基準平均は検知区間開始時の現行モデル損失統計から固定する。
    e-detectorの厳密なARL保証には、この値が定常時の条件付き平均上限であることが
    必要であり、標本平均を使う本実装では近似的な仮定になる。
    """

    _MIN_BASELINE = 0.01

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e_detector = BoundedMeanEDetector(
            baseline=self._e_detector_baseline(),
            alpha=config.E_DETECTOR_ALPHA,
            max_candidates=config.ADWIN_MAX_WINDOW,
        )
        self.history_detector_log_e = []

    def _e_detector_baseline(self):
        mean, _ = self._get_model_stats(self.current_model_id)
        return min(1.0 - 1e-6, max(self._MIN_BASELINE, mean))

    def _update_drift_detectors(self, error, y, sample_idx):
        self.e_detector.update(error)
        self.compute_counters["drift_detector_updates"] += 1
        self.compute_counters["drift_detector_hypotheses"] += (
            self.e_detector.active_hypothesis_count
        )
        self.history_detector_log_e.append(self.e_detector.log_e_value)
        return self.e_detector.drift_detected

    def _forced_drift_check(self, idx):
        # 無補正の別経路をOR接続するとe-detectorの誤警報制御を解釈できないため無効化する。
        return False

    def _new_concept_sample_count(self, buffer_list, sample_idx):
        return min(len(buffer_list), self.e_detector.width)

    def _reset_drift_detectors(self):
        self.e_detector.reset(self._e_detector_baseline())

    def _detector_label(self):
        return "e-SR"
