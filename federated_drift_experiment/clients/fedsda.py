"""FedSDAの共通逐次処理とADWIN・e-SR・HDDM検出器別クライアント。"""
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque

import torch

from .. import config
from ..adwin import FullScanADWIN
from ..e_detector import BoundedMeanEDetector
from ..hddm import HDDMA, HDDMW
from ..detection_episode import DetectionEpisodeController
from ..models import SimpleMLP
from ..provisional_model import (
    ProvisionalModelDecision,
    has_consistent_validation_advantage,
    temporal_holdout,
    validation_rejection_reason,
)
from .base import BaseClient, USE_CURRENT_MODEL_PARAMS


class FedSDAClient(BaseClient, ABC):
    """検出器に依存しないFedSDAの逐次処理・ドリフト解決基底クラス。"""

    reports_state_summary = True

    def __init__(self, *args, **kwargs):
        if kwargs.get("distance_threshold") is None:
            kwargs["distance_threshold"] = config.FEDSDA_DISTANCE_THRESHOLD
        super().__init__(*args, **kwargs)
        self.buffer = deque()                       # FIFOバッファ(検知遅延中のデータ保留)
        self.fifo_size = config.FIFO_BUFFER_SIZE    # FIFOバッファ長 N_FIFO
        self.detector_candidate_start_positions = []
        self.provisional_model_decisions = []
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
        self.detection_episodes = DetectionEpisodeController(
            enabled=config.FEDSDA_DETECTION_EPISODES_ENABLED,
            length=self.fifo_size,
        )

    def _spawn_new_model(
        self,
        bx,
        by,
        pending_ready=False,
        initialization_params=USE_CURRENT_MODEL_PARAMS,
    ):
        """新規モデルを作成し、設定された学習ラウンド数だけアップロードを保留する。"""
        result = super()._spawn_new_model(
            bx,
            by,
            pending_ready=False,
            initialization_params=initialization_params,
        )
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
            estimated_start = self._estimated_drift_start(idx)
            self.estimated_drift_start_positions.append(estimated_start)
            self.detector_candidate_start_positions.append(
                self._detector_candidate_start(idx)
            )
            operation_allowed, episode_id = self.detection_episodes.observe_detection(idx)
            if operation_allowed:
                drift_type = self._resolve_drift(
                    sample_idx=idx,
                    estimated_start=estimated_start,
                    episode_id=episode_id,
                )
                if drift_type in (1, 2):
                    self.detection_episodes.mark_operation()
            else:
                drift_type = self._resolve_episode_duplicate(
                    sample_idx=idx,
                    estimated_start=estimated_start,
                    episode_id=episode_id,
                )
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
    def _estimated_new_concept_span(self, sample_idx):
        """検出器の候補開始点から現在までの長さを返す。"""
        raise NotImplementedError

    def _estimated_drift_start(self, sample_idx):
        """FIFO内で実際に新概念側として扱う先頭位置を返す。"""
        n_new = min(
            len(self.buffer),
            self._estimated_new_concept_span(sample_idx),
        )
        return max(0, sample_idx - n_new + 1)

    def _detector_candidate_start(self, sample_idx):
        """FIFO長で打ち切らない、検出器本来の候補開始位置を返す。"""
        return max(
            0,
            sample_idx - self._estimated_new_concept_span(sample_idx) + 1,
        )

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

    def _average_model_params(self):
        """クライアントが保持する既存モデルの単純パラメータ平均を返す。"""
        model_params = [model.get_params() for model in self.models.values()]
        if not model_params:
            return None
        averaged = {}
        for name in model_params[0]:
            values = [params[name] for params in model_params]
            if values[0].is_floating_point() or values[0].is_complex():
                averaged[name] = torch.stack(values).mean(dim=0)
            else:
                averaged[name] = values[0].clone()
        return averaged

    def _select_initialization_params(self, evaluated_candidates):
        """設定された方針に従い、新規モデルの初期パラメータを返す。"""
        strategy = config.NEW_MODEL_INITIALIZATION
        if strategy == "current":
            return self.models[self.current_model_id].get_params()
        if strategy == "best_candidate":
            model_id = (
                min(evaluated_candidates, key=lambda item: item[1])[0]
                if evaluated_candidates
                else self.current_model_id
            )
            return self.models[model_id].get_params()
        if strategy == "average":
            return self._average_model_params()
        raise ValueError(
            "NEW_MODEL_INITIALIZATION must be 'current', 'best_candidate', "
            "or 'average'"
        )

    def _spawn_validated_provisional_model(
        self, bx, by, initialization_params, sample_idx
    ):
        """時系列holdoutで既存モデルへの継続的優位を確認してから登録する。"""
        holdout = temporal_holdout(
            bx, by, config.NEW_MODEL_VALIDATION_FRACTION
        )
        if holdout is None:
            self.provisional_model_decisions.append(ProvisionalModelDecision(
                position=sample_idx,
                detector=self._detector_label(),
                accepted=False,
                reason="insufficient_data",
                interval_count=len(bx),
                training_count=0,
                validation_count=0,
                reference_model_id=None,
                candidate_mean_loss=math.nan,
                reference_mean_loss=math.nan,
                candidate_recent_loss=math.nan,
                reference_recent_loss=math.nan,
            ))
            return None

        candidate = SimpleMLP()
        candidate.set_params(initialization_params)
        candidate.reset_optimizer()
        training_start = time.perf_counter()
        self._train_new_model(
            candidate, holdout.training_x, holdout.training_y
        )
        self.phase_seconds["training"] += time.perf_counter() - training_start

        with torch.no_grad():
            self._record_model_compute(
                "initialization", len(holdout.validation_x)
            )
            candidate_losses = candidate.per_sample_error(
                holdout.validation_x, holdout.validation_y
            )

            reference_losses = []
            for model_id, model in self.models.items():
                self._record_model_compute(
                    "detection", len(holdout.validation_x)
                )
                losses = model.per_sample_error(
                    holdout.validation_x, holdout.validation_y
                )
                reference_losses.append((model_id, losses))

        if not reference_losses:
            return None
        reference_model_id, best_reference = min(
            reference_losses,
            key=lambda item: float(item[1].mean().item()),
        )
        recent_start = len(candidate_losses) // 2
        reason = validation_rejection_reason(
            candidate_losses,
            best_reference,
            min_delta=config.NEW_MODEL_EARLY_STOPPING_MIN_DELTA,
        )
        accepted = has_consistent_validation_advantage(
            candidate_losses,
            best_reference,
            min_delta=config.NEW_MODEL_EARLY_STOPPING_MIN_DELTA,
        )
        self.provisional_model_decisions.append(ProvisionalModelDecision(
            position=sample_idx,
            detector=self._detector_label(),
            accepted=accepted,
            reason=reason,
            interval_count=len(bx),
            training_count=len(holdout.training_x),
            validation_count=len(holdout.validation_x),
            reference_model_id=reference_model_id,
            candidate_mean_loss=float(candidate_losses.mean().item()),
            reference_mean_loss=float(best_reference.mean().item()),
            candidate_recent_loss=float(
                candidate_losses[recent_start:].mean().item()
            ),
            reference_recent_loss=float(
                best_reference[recent_start:].mean().item()
            ),
        ))
        if not accepted:
            return None

        temp_id = self._alloc_temp_id()
        if self.verbose:
            print(f"  -> Validated New Model (Temp ID: {temp_id})")
        self._register_trained_new_model(
            temp_id,
            candidate,
            bx,
            by,
            pending_ready=False,
        )
        self._pending_upload_rounds = self.model_upload_delay_rounds
        return temp_id

    def _resolve_episode_duplicate(self, sample_idx, estimated_start, episode_id):
        """同一エピソードの追加検出を記録し、モデルを再操作せず学習へ反映する。"""
        old_model_id = self.current_model_id
        buffered_data = list(self.buffer)
        if buffered_data:
            self._absorb_into_store(self.current_model_id, buffered_data)
        self._record_adaptation_event(
            position=sample_idx,
            detector=self._detector_label(),
            action="episode_suppressed",
            old_model_id=old_model_id,
            new_model_id=self.current_model_id,
            estimated_change_point=estimated_start,
            episode_id=episode_id,
        )
        self._reset_drift_detectors()
        self.buffer.clear()
        return 0

    def _resolve_drift(self, sample_idx, estimated_start=None, episode_id=None):
        """FIFOを新旧概念に分割し、モデル切替または新規作成を行う。"""
        old_model_id = self.current_model_id
        buffer_list = list(self.buffer)
        estimated_span = self._estimated_new_concept_span(sample_idx)
        n_new_concept = min(
            len(buffer_list), estimated_span
        )

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
            self._record_adaptation_event(
                position=sample_idx,
                detector=self._detector_label(),
                action="insufficient_data",
                old_model_id=old_model_id,
                new_model_id=self.current_model_id,
                estimated_change_point=estimated_start,
                episode_id=episode_id,
            )
            self._reset_drift_detectors()
            return 0

        buffer_drift_data = drift_data

        if self.verbose:
            print(f"Client {self.client_id} [sample={sample_idx}]: "
                  f"{self._detector_label()} Drift Detected.")

        # 既存モデルの適合判定には、検出器が保持していたFIFOだけを使う。
        bx = torch.cat([data[0] for data in buffer_drift_data])
        by = torch.cat([data[1] for data in buffer_drift_data])
        evaluated_candidates = []
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
            evaluated_candidates.append((model_id, loss))
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
                action = "reuse"
                drift_data = buffer_drift_data
            else:
                if self.verbose:
                    print(f"  -> Keep current Model {self.current_model_id} "
                          f"(Loss {minimum_loss:.3f})")
                drift_type = 0
                action = "maintain"
                drift_data = buffer_drift_data
            self._absorb_into_store(self.current_model_id, drift_data)
        else:
            initial_bx = bx
            initial_by = by
            # 再利用閾値には届かなかった既存モデルのうち、ドリフト後データへ
            # 最も適合するモデルを新規モデルの初期値として利用する。
            initialization_params = self._select_initialization_params(
                evaluated_candidates
            )
            if config.NEW_MODEL_CREATION_POLICY == "immediate":
                temporary_id, _ = self._spawn_new_model(
                    initial_bx,
                    initial_by,
                    pending_ready=False,
                    initialization_params=initialization_params,
                )
            elif config.NEW_MODEL_CREATION_POLICY == "validated":
                temporary_id = self._spawn_validated_provisional_model(
                    initial_bx,
                    initial_by,
                    initialization_params,
                    sample_idx,
                )
            else:
                raise ValueError(
                    "NEW_MODEL_CREATION_POLICY must be 'immediate' or 'validated'"
                )

            drift_data = buffer_drift_data
            if temporary_id is None:
                self._absorb_into_store(self.current_model_id, drift_data)
                drift_type = 0
                action = "create_rejected"
            else:
                self.local_switch_positions.append(sample_idx)
                self.current_model_id = temporary_id
                drift_type = 2
                action = "create"
                self.train_data_store[temporary_id].extend(drift_data)

        self._record_adaptation_event(
            position=sample_idx,
            detector=self._detector_label(),
            action=action,
            old_model_id=old_model_id,
            new_model_id=self.current_model_id,
            estimated_change_point=estimated_start,
            episode_id=episode_id,
        )

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

    def _estimated_new_concept_span(self, sample_idx):
        return self.adwin.width

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

    def _estimated_new_concept_span(self, sample_idx):
        if self.adwin.drift_detected or self._class_drift_start is None:
            return super()._estimated_new_concept_span(sample_idx)
        return sample_idx - self._class_drift_start + 1

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

    def _estimated_new_concept_span(self, sample_idx):
        return self.hddm.width

    def _reset_drift_detectors(self):
        self.hddm.reset()

    def _forced_drift_check(self, idx):
        # 検出器間比較を明確にするため、ADWIN用の補助判定は併用しない。
        return False

    def _detector_label(self):
        return f"HDDM-{self.hddm_variant}"


class ClassConditionalHDDMAFedSDAClient(HDDMFedSDAClient):
    """全体損失と正解クラス別損失をHDDM-Aで並列監視するクライアント。"""

    def __init__(self, *args, **kwargs):
        kwargs["hddm_variant"] = "A"
        super().__init__(*args, **kwargs)
        # 各系列は同じHDDM設定を使う。多重検定補正が必要な場合は、
        # 実験側で系列数を考慮したconfidenceを明示的に設定する。
        self.component_drift_confidence = config.HDDM_DRIFT_CONFIDENCE
        self.component_warning_confidence = config.HDDM_WARNING_CONFIDENCE
        self.hddm = self._new_component_detector()
        self.class_hddms = defaultdict(self._new_component_detector)
        self.class_hddm_positions = defaultdict(
            lambda: deque(maxlen=self.fifo_size)
        )
        self._class_drift_start = None

    def _new_component_detector(self):
        return HDDMA(
            drift_confidence=self.component_drift_confidence,
            warning_confidence=self.component_warning_confidence,
        )

    def _update_drift_detectors(self, error, y, sample_idx):
        overall_detected = super()._update_drift_detectors(error, y, sample_idx)
        class_id = int(y.view(-1)[0].item())
        detector = self.class_hddms[class_id]
        positions = self.class_hddm_positions[class_id]
        positions.append(sample_idx)
        detector.update(error)
        self.compute_counters["drift_detector_updates"] += 1
        self.compute_counters["drift_detector_hypotheses"] += (
            detector.active_hypothesis_count
        )

        if overall_detected:
            self._class_drift_start = None
            return True
        if not detector.drift_detected:
            self._class_drift_start = None
            return False

        retained_width = min(detector.width, len(positions))
        self._class_drift_start = positions[-retained_width]
        return True

    def _estimated_new_concept_span(self, sample_idx):
        if self.hddm.drift_detected or self._class_drift_start is None:
            return super()._estimated_new_concept_span(sample_idx)
        return sample_idx - self._class_drift_start + 1

    def _reset_drift_detectors(self):
        super()._reset_drift_detectors()
        for detector in self.class_hddms.values():
            detector.reset()
        self.class_hddms.clear()
        self.class_hddm_positions.clear()
        self._class_drift_start = None

    def _detector_label(self):
        return "overall + class-conditional HDDM-A"


class ESRFedSDAClient(FedSDAClient):
    """全体損失をbounded mean e-SRで監視するFedSDAクライアント。

    ESRモードのアブレーション用に、クラス別ADWINと保険的な強制チェックは
    組み合わせない。基準平均は検知区間開始時の現行モデル損失統計から固定する。
    e-detectorの厳密なARL保証には、この値が定常時の条件付き平均上限であることが
    必要であり、標本平均を使う本実装では近似的な仮定になる。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e_detector = BoundedMeanEDetector(
            baseline=self._e_detector_baseline(),
            alpha=config.E_DETECTOR_ALPHA,
            max_candidates=config.ADWIN_MAX_WINDOW,
        )
        self.history_detector_log_e = []

    def _e_detector_baseline(self):
        stats = self.model_stats.get(self.current_model_id, {})
        if not stats or stats.get("n", 0) < 1:
            return 0.01
        return min(1.0 - 1e-6, max(0.01, float(stats["mean"])))

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

    def _estimated_new_concept_span(self, sample_idx):
        # e-SRの最大wealth候補の年齢であり、変化点推定の保証は持たない。
        return self.e_detector.width

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
            baseline=self._e_detector_baseline(),
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

    def _estimated_new_concept_span(self, sample_idx):
        if self._class_drift_start is None:
            return super()._estimated_new_concept_span(sample_idx)
        return sample_idx - self._class_drift_start + 1

    def _reset_drift_detectors(self):
        super()._reset_drift_detectors()
        self.class_e_detectors.clear()
        self.class_e_positions.clear()
        self._class_drift_start = None

    def _detector_label(self):
        return "overall + class-conditional e-SR mixture"
