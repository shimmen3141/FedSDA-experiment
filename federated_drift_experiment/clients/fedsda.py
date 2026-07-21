"""FedSDAの共通逐次処理とADWIN・e-SR・HDDM検出器別クライアント。"""
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque

import torch

from .. import config
from ..adwin import FullScanADWIN
from ..e_detector import BoundedMeanEDetector
from ..e_detector_baselines import make_baseline_estimator
from ..hddm import HDDMA, HDDMW
from .base import BaseClient


class FedSDAClient(BaseClient, ABC):
    """検出器に依存しないFedSDAの逐次処理・ドリフト解決基底クラス。"""

    reports_state_summary = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = deque()                       # FIFOバッファ(検知遅延中のデータ保留)
        self.fifo_size = config.FIFO_BUFFER_SIZE    # FIFOバッファ長 N_FIFO
        self.model_upload_delay_rounds = int(config.FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS)
        if self.model_upload_delay_rounds < 1:
            raise ValueError("FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS must be at least 1")
        self._pending_upload_rounds = 0
        # Cachedのクロス評価では、ローカル学習中ではなく直近の配布時点を使う。
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
        """サーバ配布を適用し、Cachedの次回クロス評価用キャッシュを更新する。"""
        super().apply_server_mapping(id_mapping, new_global_models, new_global_stats)
        if self._refresh_cache_on_mapping:
            self.cached_global_model_params = {
                model_id: model.get_params() for model_id, model in self.models.items()
                if model_id >= 0
            }

    def process_one_step(self, x_in, y_in, concept_id):
        """1サンプルを処理する: 予測 → 検出器更新 → (ドリフト解決 | 平時処理) → 学習。"""
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

        # 統計的検知、または検出器固有の補助チェックが発火したら解決処理へ
        if drift_detected or self._forced_drift_check(idx):
            # τ>1 で保留中の更新をドリフト解決前に消化する(τ=1 では no-op)
            self.flush_pending_updates()
            self.detected_event_positions.append(idx)
            self.estimated_drift_start_positions.append(self._estimated_drift_start(idx))
            drift_type = self._resolve_drift(sample_idx=idx)
        else:
            # 平時: バッファ長 N_FIFO を超えた分だけ古いデータをストアへ確定し、学習する
            while len(self.buffer) > self.fifo_size:
                old_x, old_y = self.buffer.popleft()
                self._record_model_compute("statistics", len(old_x))
                loss_val = self.models[self.current_model_id].get_absolute_error(old_x, old_y)
                class_id = int(old_y.view(-1)[0].item())
                self._update_model_stats(
                    self.current_model_id, loss_val, class_id=class_id
                )
                self.train_data_store[self.current_model_id].append((old_x, old_y))
            self.train_step()

        self.history_drift_type.append(drift_type)

        elapsed = time.perf_counter() - start_time
        training_elapsed = self.phase_seconds["training"] - training_before
        self.phase_seconds["online"] += max(0.0, elapsed - training_elapsed)
        elapsed_ms = elapsed * 1000
        num_global = sum(1 for mid in self.models.keys() if mid >= 0)
        self.processing_times[num_global].append(elapsed_ms)

    @abstractmethod
    def _update_drift_detectors(self, error, y, sample_idx):
        """検出器を更新し、ドリフト検知の有無を返す。"""
        raise NotImplementedError

    @abstractmethod
    def _new_concept_sample_count(self, buffer_list, sample_idx):
        """検出器が推定した新概念側のサンプル数を返す。"""
        raise NotImplementedError

    def _estimated_drift_start(self, sample_idx):
        """検知器が推定した新概念側の先頭サンプル位置を返す。"""
        n_new = self._new_concept_sample_count(list(self.buffer), sample_idx)
        return max(0, sample_idx - n_new + 1)

    @abstractmethod
    def _reset_drift_detectors(self):
        """ドリフト解決後に検出器を初期状態へ戻す。"""
        raise NotImplementedError

    @abstractmethod
    def _detector_label(self):
        raise NotImplementedError

    def _forced_drift_check(self, idx):
        """検出器固有の補助チェック。既定では統計的検出だけを使う。"""
        return False

    def _resolve_drift(self, sample_idx):
        """FIFOを新旧概念に分割し、モデル切替または新規作成を行う。"""
        buffer_list = list(self.buffer)
        n_new_concept = self._new_concept_sample_count(buffer_list, sample_idx)

        if len(buffer_list) <= n_new_concept:
            drift_data = buffer_list
            old_data = []
        else:
            old_data = buffer_list[:-n_new_concept]
            drift_data = buffer_list[-n_new_concept:]

        if old_data:
            self._store_evaluation_data(self.current_model_id, old_data)
            self._absorb_into_store(self.current_model_id, old_data)

        if len(drift_data) < config.MIN_DRIFT_DATA:
            self._reset_drift_detectors()
            return 0

        if self.verbose:
            print(f"Client {self.client_id} [sample={sample_idx}]: "
                  f"{self._detector_label()} Drift Detected.")

        bx = torch.cat([data[0] for data in drift_data])
        by = torch.cat([data[1] for data in drift_data])
        valid_candidates = []
        for model_id, model in self.models.items():
            with torch.no_grad():
                self._record_model_compute("detection", len(bx))
                errors = model.per_sample_error(bx, by)
                loss = float(torch.mean(errors).item())
            historical_mean, _ = self._get_model_stats(model_id)

            if historical_mean == 0.0:
                if self.verbose:
                    print(f"  Check M{model_id}: No baseline (n=0) -> "
                          f"treat as not-matching. (Loss={loss:.3f})")
                continue

            difference = loss - historical_mean
            if self.verbose:
                print(f"  Check M{model_id}: Diff={difference:.3f} vs "
                      f"Thr={self.distance_threshold:.3f} "
                      f"(Loss={loss:.3f}, Base={historical_mean:.3f})")
            if difference <= self.distance_threshold:
                valid_candidates.append((model_id, loss))

        if valid_candidates:
            best_model_id, minimum_loss = min(valid_candidates, key=lambda item: item[1])
            if best_model_id != self.current_model_id:
                if self.verbose:
                    print(f"  -> Switch to Model {best_model_id} "
                          f"(Loss {minimum_loss:.3f})")
                self.local_switch_positions.append(sample_idx)
                self.current_model_id = best_model_id
                drift_type = 1
            else:
                if self.verbose:
                    print(f"  -> Keep current Model {self.current_model_id} "
                          f"(Loss {minimum_loss:.3f})")
                drift_type = 0
            self._absorb_into_store(self.current_model_id, drift_data)
        else:
            temporary_id, _ = self._spawn_new_model(bx, by, pending_ready=False)
            self.local_switch_positions.append(sample_idx)
            self.current_model_id = temporary_id
            drift_type = 2
            self.train_data_store[temporary_id].extend(drift_data)

        self._reset_drift_detectors()
        self.buffer.clear()
        return drift_type


class ADWINFedSDAClient(FedSDAClient):
    """全体損失をADWINで監視するFedSDAクライアント。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adwin = FullScanADWIN(delta=config.ADWIN_DELTA)

    def _update_drift_detectors(self, error, y, sample_idx):
        scan_width = min(self.adwin.width + 1, self.adwin.max_window_size)
        self.adwin.update(error)
        self.compute_counters["drift_detector_updates"] += 1
        if scan_width >= config.ADWIN_MIN_WIDTH:
            self.compute_counters["drift_detector_hypotheses"] += scan_width - 1
        return self.adwin.drift_detected

    def _new_concept_sample_count(self, buffer_list, sample_idx):
        return min(len(buffer_list), self.adwin.width)

    def _reset_drift_detectors(self):
        self.adwin.reset()

    def _detector_label(self):
        return "ADWIN"

    def _forced_drift_check(self, idx):
        """ADWIN未検知でも、直近ウィンドウの損失悪化を確認する保険的チェック。"""
        if not config.FEDSDA_ENABLE_FORCED_DRIFT_CHECK:
            return False
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
            errors = self.models[self.current_model_id].per_sample_error(bx, by)
            window_loss = float(torch.mean(errors).item())
        hist_mean, _ = self._get_model_stats(self.current_model_id)

        if hist_mean > 0.0 and (window_loss >= hist_mean + self.distance_threshold):
            if self.verbose:
                print(f"Client {self.client_id} [sample={idx}]: Forced drift-check triggered "
                      f"(win={width}, loss={window_loss:.3f}, base={hist_mean:.3f}, "
                      f"thr={self.distance_threshold:.3f})")
            return True
        return False

class ClassConditionalADWINFedSDAClient(ADWINFedSDAClient):
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


class HDDMFedSDAClient(FedSDAClient):
    """全体損失を一方向HDDM-AまたはHDDM-Wで監視するFedSDAクライアント。"""

    DETECTOR_FACTORIES = {
        "A": lambda: HDDMA(
            drift_confidence=config.HDDM_DRIFT_CONFIDENCE,
            warning_confidence=config.HDDM_WARNING_CONFIDENCE,
        ),
        "W": lambda: HDDMW(
            drift_confidence=config.HDDM_DRIFT_CONFIDENCE,
            warning_confidence=config.HDDM_WARNING_CONFIDENCE,
            lambda_option=config.HDDM_W_LAMBDA,
        ),
    }

    def __init__(self, *args, hddm_variant="A", **kwargs):
        try:
            factory = self.DETECTOR_FACTORIES[hddm_variant]
        except KeyError:
            raise ValueError("hddm_variantは'A'または'W'である必要があります") from None
        super().__init__(*args, **kwargs)
        self.hddm_variant = hddm_variant
        self.hddm = factory()

    def _update_drift_detectors(self, error, y, sample_idx):
        self.hddm.update(error)
        self.compute_counters["drift_detector_updates"] += 1
        self.compute_counters["drift_detector_hypotheses"] += (
            self.hddm.active_hypothesis_count
        )
        return self.hddm.drift_detected

    def _new_concept_sample_count(self, buffer_list, sample_idx):
        return min(len(buffer_list), self.hddm.width)

    def _reset_drift_detectors(self):
        self.hddm.reset()

    def _forced_drift_check(self, idx):
        # 検出器間比較を明確にするため、ADWIN用の補助判定は併用しない。
        return False

    def _detector_label(self):
        return f"HDDM-{self.hddm_variant}"


class ESRFedSDAClient(FedSDAClient):
    """全体損失をbounded mean e-SRで監視するFedSDAクライアント。

    ESRモードのアブレーション用に、クラス別ADWINと保険的な強制チェックは
    組み合わせない。基準平均は検知区間開始時の現行モデル損失統計から固定する。
    e-detectorの厳密なARL保証には、この値が定常時の条件付き平均上限であることが
    必要であり、標本平均を使う本実装では近似的な仮定になる。
    """

    def __init__(self, *args, baseline_strategy="historical_mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_estimator = make_baseline_estimator(
            baseline_strategy, beta=config.E_DETECTOR_BASELINE_BETA
        )
        self.e_detector = BoundedMeanEDetector(
            baseline=self._e_detector_baseline(),
            alpha=config.E_DETECTOR_ALPHA,
            max_candidates=config.ADWIN_MAX_WINDOW,
        )
        self.history_detector_log_e = []

    def _e_detector_baseline(self, class_id=None):
        stats = self.model_stats.get(self.current_model_id, {})
        # ClassESRのmean方式は全体平均を共用し、既存結果を維持する。
        # UCB方式だけはクラス条件付き上限を構成するためクラス別統計を使う。
        if class_id is not None and self.baseline_estimator.name != "historical_mean":
            stats = stats.get("class_stats", {}).get(class_id, {})
        return self.baseline_estimator.estimate(stats)

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


class ClassConditionalESRFedSDAClient(ESRFedSDAClient):
    """全体損失と正解クラス別損失のe-SRを固定重みで混合するクライアント。

    全体系列と各クラス系列へ等しい重みを割り当て、混合e値が閾値を超えたときに
    検知する。クラス別系列は該当クラスのサンプル到着時だけ更新する。クラス別の
    事前統計を保持していないため、開始時の基準平均には全体モデル統計を共用する。
    従ってクラス条件付き平均もこの基準以下という追加仮定が必要になる。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.component_weight = 1.0 / (config.num_classes() + 1)
        self.class_e_detectors = {}
        self.class_e_positions = defaultdict(deque)
        self._class_drift_start = None

    def _new_class_detector(self, class_id):
        return BoundedMeanEDetector(
            baseline=self._e_detector_baseline(class_id=class_id),
            alpha=config.E_DETECTOR_ALPHA,
            max_candidates=config.ADWIN_MAX_WINDOW,
        )

    def _update_component(self, detector, error):
        detector.update(error)
        self.compute_counters["drift_detector_updates"] += 1
        self.compute_counters["drift_detector_hypotheses"] += (
            detector.active_hypothesis_count
        )

    def _update_drift_detectors(self, error, y, sample_idx):
        self._update_component(self.e_detector, error)

        class_id = int(y.view(-1)[0].item())
        if class_id not in range(config.num_classes()):
            raise ValueError(f"クラスIDが範囲外です: {class_id}")
        detector = self.class_e_detectors.get(class_id)
        if detector is None:
            detector = self._new_class_detector(class_id)
            self.class_e_detectors[class_id] = detector
        positions = self.class_e_positions[class_id]
        positions.append(sample_idx)
        self._update_component(detector, error)
        while len(positions) > detector.max_candidates:
            positions.popleft()

        component_logs = {
            "overall": self.e_detector.log_e_value + math.log(self.component_weight),
            class_id: detector.log_e_value + math.log(self.component_weight),
        }
        # 既に観測した他クラスの検出器も、最後に更新したe値で混合する。
        for other_id, other_detector in self.class_e_detectors.items():
            if other_id != class_id:
                component_logs[other_id] = (
                    other_detector.log_e_value + math.log(self.component_weight)
                )

        finite_logs = [value for value in component_logs.values() if math.isfinite(value)]
        if finite_logs:
            maximum = max(finite_logs)
            combined_log_e = maximum + math.log(
                sum(math.exp(value - maximum) for value in finite_logs)
            )
        else:
            combined_log_e = -math.inf
        self.history_detector_log_e.append(combined_log_e)

        if combined_log_e < self.e_detector.log_threshold:
            self._class_drift_start = None
            return False

        best_component = max(component_logs, key=component_logs.get)
        if best_component == "overall":
            self._class_drift_start = None
        else:
            best_detector = self.class_e_detectors[best_component]
            best_positions = self.class_e_positions[best_component]
            offset = best_detector.split_start - best_detector.retained_start_time
            self._class_drift_start = best_positions[max(0, min(offset, len(best_positions) - 1))]
        return True

    def _new_concept_sample_count(self, buffer_list, sample_idx):
        if self._class_drift_start is None:
            return super()._new_concept_sample_count(buffer_list, sample_idx)
        return min(len(buffer_list), sample_idx - self._class_drift_start + 1)

    def _reset_drift_detectors(self):
        super()._reset_drift_detectors()
        self.class_e_detectors.clear()
        self.class_e_positions.clear()
        self._class_drift_start = None

    def _detector_label(self):
        return "overall + class-conditional e-SR mixture"
