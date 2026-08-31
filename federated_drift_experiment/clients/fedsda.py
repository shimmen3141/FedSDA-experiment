"""FedSDAの共通逐次処理とADWIN・e-SR・HDDM検出器別クライアント。"""
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque

import torch

from .. import config
from ..diagnostics import RoutingLeaveOneOutDiagnostics
from ..drift_detectors import BoundedMeanEDetector, FullScanADWIN, HDDMA, HDDMW
from ..detection_episode import DetectionEpisodeController
from ..expert_routing import (
    AdaHedgeRouter,
    PeriodicForwardProbeActiveSet,
    SwitchingExpertRouter,
)
from ..provisional_model import (
    ForwardValidationSession,
    ProvisionalModelDecision,
    disjoint_validation_rejection_reason,
    forward_creation_policy,
    has_consistent_validation_advantage,
    has_disjoint_validation_advantage,
    select_forward_fitting_reference,
    temporal_holdout,
    validation_rejection_reason,
)
from .base import BaseClient, USE_CURRENT_MODEL_PARAMS


_CONTEXTUAL_ROUTING_MODES = frozenset({
    "predicted_class", "meta_predicted_class", "meta_switching",
})


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
        self.reuse_selection_counts = defaultdict(int)
        self._forward_validation = None
        self.forward_validation_samples = int(
            config.NEW_MODEL_FORWARD_VALIDATION_SAMPLES
        )
        if self.forward_validation_samples < 2:
            raise ValueError(
                "NEW_MODEL_FORWARD_VALIDATION_SAMPLES must be at least 2"
            )
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

    def _on_local_model_change(self, old_model_id, new_model_id):
        """クライアント内で確定したモデル切替後の拡張フック。"""

    def _set_local_current_model(self, model_id):
        """ローカルな再利用・新規作成による現行モデル変更を一元化する。"""
        old_model_id = self.current_model_id
        self.current_model_id = model_id
        if model_id != old_model_id:
            self._on_local_model_change(old_model_id, model_id)

    def _snapshot_reference_models(self):
        """前向き検証中の比較対象を警報時点のパラメータで固定する。"""
        snapshots = {}
        for model_id, model in self.models.items():
            snapshot = self._new_model()
            snapshot.set_params(model.get_params())
            snapshots[model_id] = snapshot
        return snapshots

    def _train_reference_shadows(self, reference_models, bx, by):
        """既存モデルのshadowへ候補と同じ検知区間学習を適用する。"""
        training_start = time.perf_counter()
        for model in reference_models.values():
            model.reset_optimizer()
            self._train_new_model(model, bx, by)
        self.phase_seconds["training"] += time.perf_counter() - training_start

    def _update_tournament_shadows(self, session, x, y):
        """評価済みのforwardサンプルで全shadowを同じ回数だけ更新する。"""
        models = [session.candidate, *session.reference_models.values()]
        training_start = time.perf_counter()
        for model in models:
            model.update(x, y)
            self._record_model_compute("training", len(x))
            self.compute_counters["optimizer_steps"] += 1
        self.phase_seconds["training"] += time.perf_counter() - training_start

    def _begin_forward_validation(
        self,
        bx,
        by,
        drift_data,
        initialization_params,
        sample_idx,
        estimated_start,
        episode_id,
    ):
        """推定区間で候補を学習し、警報後データによる採否判定を開始する。"""
        candidate = self._new_model()
        candidate.set_params(initialization_params)
        candidate.reset_optimizer()
        training_start = time.perf_counter()
        training_examples_before = self.compute_counters["training_examples"]
        optimizer_steps_before = self.compute_counters["optimizer_steps"]
        self._train_new_model(candidate, bx, by)
        self.phase_seconds["training"] += time.perf_counter() - training_start
        candidate_training_examples = (
            self.compute_counters["training_examples"] - training_examples_before
        )
        candidate_optimizer_steps = (
            self.compute_counters["optimizer_steps"] - optimizer_steps_before
        )
        reference_models = self._snapshot_reference_models()
        policy = forward_creation_policy(config.NEW_MODEL_CREATION_POLICY)
        if policy is not None and policy.train_reference_shadows:
            self._train_reference_shadows(reference_models, bx, by)
        reference_historical_means = {}
        for model_id in reference_models:
            stats = self.model_stats.get(model_id)
            if stats is not None and stats.get("n", 0) >= 2:
                reference_historical_means[model_id] = float(stats["mean"])
        self._forward_validation = ForwardValidationSession(
            proposal_position=sample_idx,
            estimated_change_point=estimated_start,
            episode_id=episode_id,
            old_model_id=self.current_model_id,
            detector=self._detector_label(),
            candidate=candidate,
            training_x=bx,
            training_y=by,
            held_data=list(drift_data),
            reference_models=reference_models,
            target_count=self.forward_validation_samples,
            candidate_training_examples=candidate_training_examples,
            candidate_optimizer_steps=candidate_optimizer_steps,
            reference_historical_means=reference_historical_means,
        )

    def _observe_forward_validation(self, x, y, sample_idx):
        """最新サンプルをshadow candidateと警報時点の既存モデルで評価する。"""
        session = self._forward_validation
        if session is None:
            return 0
        with torch.no_grad():
            self._record_model_compute("detection", len(x))
            candidate_loss = float(
                session.candidate.per_sample_error(x, y).mean().item()
            )
            reference_losses = {}
            for model_id, model in session.reference_models.items():
                self._record_model_compute("detection", len(x))
                reference_losses[model_id] = float(
                    model.per_sample_error(x, y).mean().item()
                )
        session.append_losses(candidate_loss, reference_losses)
        policy = forward_creation_policy(config.NEW_MODEL_CREATION_POLICY)
        if policy is not None and policy.train_reference_shadows:
            self._update_tournament_shadows(session, x, y)
        if not session.ready:
            return 0
        return self._finalize_forward_validation(sample_idx)

    def _finalize_forward_validation(self, sample_idx):
        """規定数の警報後サンプルから候補を正式採用または棄却する。"""
        session = self._forward_validation
        if session is None or not session.ready:
            return 0
        candidate_losses = torch.tensor(session.candidate_losses)
        policy = forward_creation_policy(config.NEW_MODEL_CREATION_POLICY)
        if policy is None:
            raise ValueError(
                "forward validation requires a forward creation policy"
            )
        reference_model_id = min(
            session.reference_losses,
            key=lambda model_id: sum(session.reference_losses[model_id]),
        )
        requalified_model_id = None
        if policy.requalify_references:
            available_losses = {
                model_id: losses
                for model_id, losses in session.reference_losses.items()
                if model_id in self.models
            }
            requalified_model_id = select_forward_fitting_reference(
                available_losses,
                session.reference_historical_means,
                self.distance_threshold,
                preferred_model_id=(
                    self.current_model_id if policy.prefer_current_reference else None
                ),
            )
            if requalified_model_id is not None:
                reference_model_id = requalified_model_id
        reference_losses = torch.tensor(
            session.reference_losses[reference_model_id]
        )
        tournament_reference_won = False
        if policy.train_reference_shadows:
            accepted = (
                float(candidate_losses.mean().item())
                < float(reference_losses.mean().item())
                - config.NEW_MODEL_EARLY_STOPPING_MIN_DELTA
            )
            tournament_reference_won = not accepted
            reason = "candidate_won" if accepted else "reference_won"
        elif requalified_model_id is not None:
            accepted = False
            if policy.prefer_current_reference:
                reason = (
                    "current_reference_refit"
                    if requalified_model_id == self.current_model_id
                    else "alternative_reference_refit"
                )
            else:
                reason = "reference_refit"
        elif policy.require_disjoint_persistence:
            accepted = has_disjoint_validation_advantage(
                candidate_losses,
                reference_losses,
                min_delta=config.NEW_MODEL_EARLY_STOPPING_MIN_DELTA,
            )
            reason = disjoint_validation_rejection_reason(
                candidate_losses,
                reference_losses,
                min_delta=config.NEW_MODEL_EARLY_STOPPING_MIN_DELTA,
            )
        else:
            accepted = has_consistent_validation_advantage(
                candidate_losses,
                reference_losses,
                min_delta=config.NEW_MODEL_EARLY_STOPPING_MIN_DELTA,
            )
            reason = validation_rejection_reason(
                candidate_losses,
                reference_losses,
                min_delta=config.NEW_MODEL_EARLY_STOPPING_MIN_DELTA,
            )
        recent_start = len(candidate_losses) // 2
        self.provisional_model_decisions.append(ProvisionalModelDecision(
            position=session.proposal_position,
            detector=session.detector,
            accepted=accepted,
            reason=reason,
            interval_count=len(session.training_x),
            training_count=len(session.training_x),
            validation_count=session.validation_count,
            reference_model_id=reference_model_id,
            candidate_mean_loss=float(candidate_losses.mean().item()),
            reference_mean_loss=float(reference_losses.mean().item()),
            candidate_recent_loss=float(
                candidate_losses[recent_start:].mean().item()
            ),
            reference_recent_loss=float(
                reference_losses[recent_start:].mean().item()
            ),
            resolution_position=sample_idx,
            validation_source="forward",
            reference_historical_mean=session.reference_historical_means.get(
                reference_model_id, math.nan
            ),
        ))

        old_model_id = self.current_model_id
        if accepted:
            temp_id = self._alloc_temp_id()
            self._register_trained_new_model(
                temp_id,
                session.candidate,
                session.training_x,
                session.training_y,
                pending_ready=False,
            )
            self.model_training_examples[temp_id] += (
                session.candidate_training_examples
            )
            self.model_optimizer_steps[temp_id] += (
                session.candidate_optimizer_steps
            )
            self._pending_upload_rounds = self.model_upload_delay_rounds
            self.train_data_store[temp_id].extend(session.held_data)
            self._set_local_current_model(temp_id)
            self.local_switch_positions.append(sample_idx)
            self.detection_episodes.mark_operation()
            action = "create"
            drift_type = 2
        elif tournament_reference_won:
            winning_shadow = session.reference_models[reference_model_id]
            self.models[reference_model_id].set_params(winning_shadow.get_params())
            self.models[reference_model_id].reset_optimizer()
            self._set_local_current_model(reference_model_id)
            self._absorb_into_store(self.current_model_id, session.held_data)
            if self.current_model_id != old_model_id:
                self.local_switch_positions.append(sample_idx)
                self.detection_episodes.mark_operation()
                action = "reuse"
                drift_type = 1
            else:
                action = "maintain"
                drift_type = 0
        elif requalified_model_id is not None:
            self._set_local_current_model(requalified_model_id)
            self._absorb_into_store(self.current_model_id, session.held_data)
            if self.current_model_id != old_model_id:
                self.local_switch_positions.append(sample_idx)
                self.detection_episodes.mark_operation()
                action = "reuse"
                drift_type = 1
            else:
                action = "maintain"
                drift_type = 0
        else:
            self._absorb_into_store(self.current_model_id, session.held_data)
            action = "create_rejected"
            drift_type = 0
        self._record_adaptation_event(
            position=sample_idx,
            detector=session.detector,
            action=action,
            old_model_id=old_model_id,
            new_model_id=self.current_model_id,
            estimated_change_point=session.estimated_change_point,
            episode_id=session.episode_id,
        )
        self._forward_validation = None
        return drift_type

    def finalize_incomplete_forward_validation(self):
        """実験終端で未完了の前向き検証を棄却し、保留データを回収する。"""
        session = self._forward_validation
        if session is None:
            return
        resolution_position = max(
            session.proposal_position, self.processed_samples - 1
        )
        self.provisional_model_decisions.append(ProvisionalModelDecision(
            position=session.proposal_position,
            detector=session.detector,
            accepted=False,
            reason="insufficient_forward_data",
            interval_count=len(session.training_x),
            training_count=len(session.training_x),
            validation_count=session.validation_count,
            reference_model_id=None,
            candidate_mean_loss=math.nan,
            reference_mean_loss=math.nan,
            candidate_recent_loss=math.nan,
            reference_recent_loss=math.nan,
            resolution_position=resolution_position,
            validation_source="forward",
        ))
        self._absorb_into_store(self.current_model_id, session.held_data)
        self._record_adaptation_event(
            position=resolution_position,
            detector=session.detector,
            action="create_rejected",
            old_model_id=self.current_model_id,
            new_model_id=self.current_model_id,
            estimated_change_point=session.estimated_change_point,
            episode_id=session.episode_id,
        )
        self._forward_validation = None

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

    def evaluate_cached_model_diagnostics(
        self, model_id, target_model_id, include_class_correctness=False
    ):
        """配布済みキャッシュを使ってモデル対の正誤相補性も評価する。"""
        try:
            params = self.cached_global_model_params[model_id]
        except KeyError:
            raise ValueError(
                f"モデル{model_id}はまだクライアントへ配布されていません"
            ) from None
        return self.evaluate_model_diagnostics(
            params,
            target_model_id,
            include_class_correctness=include_class_correctness,
        )

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

        drift_type = self._observe_forward_validation(x, y, idx)

        self._record_model_compute("detection", len(x))
        error = self.models[self.current_model_id].get_absolute_error(x, y)
        drift_detected = self._update_drift_detectors(error, y, idx)
        # 真の概念IDはoracle診断にのみ使い、検出・学習ロジックには渡さない。
        self.buffer.append((x, y, concept_id))

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
                old_data = self.buffer.popleft()
                old_x, old_y = old_data[:2]
                self._record_model_compute("statistics", len(old_x))
                loss_val = self.models[self.current_model_id].get_absolute_error(old_x, old_y)
                class_id = int(old_y.view(-1)[0].item())
                self._update_model_stats(
                    self.current_model_id, loss_val, class_id=class_id
                )
                self.train_data_store[self.current_model_id].append(old_data)
                self._record_model_concept(
                    self.current_model_id, old_data[2]
                )
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

    def _select_reuse_candidate(self, valid_candidates):
        """適合済み既存モデルのうち、区間平均損失が最小のものを選ぶ。"""
        return min(valid_candidates, key=lambda item: item[1])

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

        candidate = self._new_model()
        candidate.set_params(initialization_params)
        candidate.reset_optimizer()
        training_start = time.perf_counter()
        training_examples_before = self.compute_counters["training_examples"]
        optimizer_steps_before = self.compute_counters["optimizer_steps"]
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
        self._attribute_model_training(
            temp_id, training_examples_before, optimizer_steps_before
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
        if self._forward_validation is not None:
            self._absorb_into_store(self.current_model_id, buffer_list)
            self._record_adaptation_event(
                position=sample_idx,
                detector=self._detector_label(),
                action="forward_validation_pending",
                old_model_id=old_model_id,
                new_model_id=self.current_model_id,
                estimated_change_point=estimated_start,
                episode_id=episode_id,
            )
            self._reset_drift_detectors()
            self.buffer.clear()
            return 0
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
            best_model_id, minimum_loss = self._select_reuse_candidate(
                valid_candidates
            )
            if best_model_id != self.current_model_id:
                self.reuse_selection_counts["alternative_fit"] += 1
                if self.verbose:
                    print(f"  -> Switch to Model {best_model_id} "
                          f"(Loss {minimum_loss:.3f})")
                self.local_switch_positions.append(sample_idx)
                self._set_local_current_model(best_model_id)
                drift_type = 1
                action = "reuse"
                drift_data = buffer_drift_data
            else:
                self.reuse_selection_counts["current_fit"] += 1
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
            elif forward_creation_policy(config.NEW_MODEL_CREATION_POLICY) is not None:
                self._begin_forward_validation(
                    initial_bx,
                    initial_by,
                    drift_data,
                    initialization_params,
                    sample_idx,
                    estimated_start,
                    episode_id,
                )
                temporary_id = None
            else:
                raise ValueError(
                    "NEW_MODEL_CREATION_POLICY must be one of "
                    f"{config.NEW_MODEL_CREATION_POLICIES}"
                )

            drift_data = buffer_drift_data
            if forward_creation_policy(config.NEW_MODEL_CREATION_POLICY) is not None:
                drift_type = 0
                action = "create_pending"
            elif temporary_id is None:
                self._absorb_into_store(self.current_model_id, drift_data)
                drift_type = 0
                action = "create_rejected"
            else:
                self.local_switch_positions.append(sample_idx)
                self._set_local_current_model(temporary_id)
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

    def __init__(self, *args, overall_component_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        class_count = config.num_classes()
        if overall_component_weight is None:
            # 従来方式では、全体系列と各クラス系列へ均等に重みを配分する。
            overall_component_weight = 1.0 / (class_count + 1)
        if not 0.0 < overall_component_weight < 1.0:
            raise ValueError("overall_component_weight must be between 0 and 1")
        self.overall_component_weight = float(overall_component_weight)
        self.class_component_weight = (
            (1.0 - self.overall_component_weight) / class_count
        )
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
            "overall": (
                self.e_detector.log_e_value
                + math.log(self.overall_component_weight)
            ),
            class_id: (
                detector.log_e_value
                + math.log(self.class_component_weight)
            ),
        }
        # 既に観測した他クラスの検出器も、最後に更新したe値で混合する。
        for other_id, other_detector in self.class_e_detectors.items():
            if other_id != class_id:
                component_logs[other_id] = (
                    other_detector.log_e_value
                    + math.log(self.class_component_weight)
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


class _AdaHedgeRoutingFedSDAClientMixin:
    """検出器から独立して保持モデルの予測をAdaHedgeで統合するmixin。

    検出・モデル操作・学習・通信は後続の検出器クラスに委ね、prequential
    予測だけをソフト化する。これにより検出器とroutingを独立に組み合わせる。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_router = AdaHedgeRouter()
        self.switching_expert_router = SwitchingExpertRouter(
            config.FIFO_BUFFER_SIZE
        )
        # 現行Meta mixtureとモデル追従型mixtureを固定閾値なしで選ぶ上位ルータ。
        self.meta_switching_router = SwitchingExpertRouter(
            config.FIFO_BUFFER_SIZE
        )
        self.context_expert_routers = defaultdict(AdaHedgeRouter)
        self.shadow_meta_routers = defaultdict(AdaHedgeRouter)
        # 真の概念IDを診断時だけ与え、概念状態を完全に識別できた場合の
        # 因果的なルーティング上限を測る。実予測や学習割当には使用しない。
        self.oracle_concept_expert_routers = defaultdict(AdaHedgeRouter)
        self.history_routing_effective_experts = []
        self.history_routing_max_weight = []
        self.history_routing_gate_open = []
        self.history_routing_oracle_correct = []
        self.history_routing_leader_correct = []
        self.history_routing_oracle_concept_correct = []
        self.history_routing_meta_correct = []
        self.history_routing_meta_global_correct = []
        self.history_routing_meta_context_mixture_correct = []
        self.history_routing_meta_context_leader_correct = []
        self.history_routing_meta_context_leader_weight = []
        self.history_routing_switching_correct = []
        self.history_routing_switching_leader_id = []
        self.history_routing_switching_effective_experts = []
        self.history_routing_meta_switching_correct = []
        self.history_routing_meta_switching_selected_switching = []
        self.routing_diagnostics = {
            "sample_count": 0,
            "oracle_correct_count": 0,
            "mixture_correct_count": 0,
            "leader_correct_count": 0,
            "confidence_leader_correct_count": 0,
            "missed_oracle_count": 0,
            "confidence_leader_missed_oracle_count": 0,
        }
        self.routing_oracle_concept_diagnostics = {
            "sample_count": 0,
            "correct_count": 0,
        }
        # 実予測を変えず、global mixtureと文脈別leaderの選択可能性を診断する。
        self.routing_meta_diagnostics = {
            "sample_count": 0,
            "correct_count": 0,
            "actual_correct_count": 0,
            "global_correct_count": 0,
            "context_mixture_correct_count": 0,
            "context_leader_correct_count": 0,
            "context_leader_weight_sum": 0.0,
            "context_leader_preferred_count": 0,
        }
        self.routing_switching_diagnostics = {
            "sample_count": 0,
            "correct_count": 0,
            "actual_correct_count": 0,
            "global_correct_count": 0,
            "effective_experts_sum": 0.0,
        }
        self.routing_meta_switching_diagnostics = {
            "sample_count": 0,
            "correct_count": 0,
            "actual_correct_count": 0,
            "meta_correct_count": 0,
            "switching_correct_count": 0,
            "selected_switching_count": 0,
        }
        # 入力文脈を使うルータへ進む前に、既存ルータの未回収余地が
        # 正解クラスへ偏っているかを追加forwardなしで記録する。
        self.routing_class_diagnostics = defaultdict(
            lambda: defaultdict(int)
        )
        self.routing_leave_one_out_diagnostics = (
            RoutingLeaveOneOutDiagnostics()
        )
        if config.ROUTING_ACTIVE_SET_POLICY == "periodic_forward_probe":
            if config.ROUTING_ARCHIVE_SHADOW_DIAGNOSTICS:
                raise ValueError(
                    "routing active集合とarchive shadow診断は同時に有効化できません"
                )
            self.routing_active_set = PeriodicForwardProbeActiveSet(
                config.NEW_MODEL_FORWARD_VALIDATION_SAMPLES
            )
        elif config.ROUTING_ACTIVE_SET_POLICY == "all":
            self.routing_active_set = None
        else:
            raise ValueError(
                "未知のrouting active-set方針です: "
                f"{config.ROUTING_ACTIVE_SET_POLICY!r}"
            )

    def _prediction_probabilities(self, proposal_probabilities):
        """AdaHedgeの提案重みを実際の予測重みへ変換する。"""
        self.history_routing_gate_open.append(True)
        return proposal_probabilities

    def _routing_scores(self, x, model_ids):
        """各独立モデルを評価し、SoftRouting用の出力を返す。"""
        scores = {model_id: self.models[model_id].forward(x) for model_id in model_ids}
        self._record_model_compute(
            "prediction", len(x) * len(model_ids), calls=len(model_ids)
        )
        return scores

    @staticmethod
    def _restrict_routing_probabilities(probabilities, active_model_ids):
        """全repository上の重みをactive集合へ制限して再正規化する。"""
        active_model_ids = tuple(active_model_ids)
        total = sum(probabilities[model_id] for model_id in active_model_ids)
        if total > 0.0:
            return {
                model_id: probabilities[model_id] / total
                for model_id in active_model_ids
            }
        uniform = 1.0 / len(active_model_ids)
        return {model_id: uniform for model_id in active_model_ids}

    @staticmethod
    def _weighted_routing_scores(prediction_scores, probabilities):
        """モデル別出力を指定された確率で混合する。"""
        return sum(
            prediction_scores[model_id] * probabilities[model_id]
            for model_id in probabilities
        )

    def _routing_leader(self, probabilities):
        """最大重みモデルを、同率時は現行モデル優先で選ぶ。"""
        maximum = max(probabilities.values())
        leaders = [
            model_id
            for model_id, weight in probabilities.items()
            if weight == maximum
        ]
        leader = (
            self.current_model_id
            if self.current_model_id in leaders
            else min(leaders)
        )
        return leader, maximum

    @staticmethod
    def _routing_prediction(scores, num_classes):
        """確率出力をクラス予測へ変換する。"""
        if num_classes == 2:
            return (scores > 0.5).float()
        return torch.argmax(scores, dim=1, keepdim=True).float()

    @staticmethod
    def _routing_score_loss(scores, y, num_classes):
        """AdaHedge更新用の[0, 1]有界損失を返す。"""
        if num_classes == 2:
            return float(
                torch.abs(scores.view(-1) - y.view(-1).float()).mean().item()
            )
        labels = y.view(-1).long()
        return float(
            (1.0 - scores.gather(1, labels.unsqueeze(1)).squeeze(1))
            .mean().item()
        )

    @classmethod
    def _routing_correct(cls, scores, y, num_classes):
        prediction = cls._routing_prediction(scores, num_classes)
        return bool(
            prediction.view(-1)[0].item() == y.view(-1)[0].item()
        )

    def _record_prediction(self, x, y, concept_id):
        repository_model_ids = tuple(sorted(self.models))
        sample_index = max(0, self.processed_samples - 1)
        if self.routing_active_set is None:
            model_ids = repository_model_ids
            update_routing_evidence = True
        else:
            (
                model_ids,
                update_routing_evidence,
            ) = self.routing_active_set.select(
                repository_model_ids, self.current_model_id, sample_index,
            )
        proposal_repository_probabilities = self.expert_router.probabilities(
            repository_model_ids
        )
        proposal_probabilities = self._restrict_routing_probabilities(
            proposal_repository_probabilities, model_ids
        )
        model_losses = {}
        model_correctness = {}
        model_confidences = {}
        prediction_scores = {}

        with torch.no_grad():
            scores_by_model = self._routing_scores(x, model_ids)
            for model_id in model_ids:
                model = self.models[model_id]
                scores = scores_by_model[model_id]
                if model.num_classes > 2:
                    scores = torch.softmax(scores, dim=1)
                prediction_scores[model_id] = scores
                if model.num_classes > 2:
                    labels = y.view(-1).long()
                    losses = 1.0 - scores.gather(
                        1, labels.unsqueeze(1)
                    ).squeeze(1)
                    model_prediction = torch.argmax(scores, dim=1, keepdim=True)
                    model_confidences[model_id] = float(
                        scores.max(dim=1).values.mean().item()
                    )
                else:
                    losses = torch.abs(scores.view(-1) - y.view(-1).float())
                    model_prediction = (scores > 0.5).float()
                    model_confidences[model_id] = float(
                        torch.abs(scores.view(-1) - 0.5).mean().item()
                    )
                # 予測に用いた出力から損失も計算し、同一モデルの二重forwardを避ける。
                model_losses[model_id] = float(losses.mean().item())
                model_correctness[model_id] = bool(
                    model_prediction.view(-1)[0].item()
                    == y.view(-1)[0].item()
                )

        num_classes = self.models[model_ids[0]].num_classes
        global_scores = self._weighted_routing_scores(
            prediction_scores, proposal_probabilities
        )
        switching_repository_probabilities = (
            self.switching_expert_router.probabilities(repository_model_ids)
        )
        switching_probabilities = self._restrict_routing_probabilities(
            switching_repository_probabilities, model_ids
        )
        switching_scores = self._weighted_routing_scores(
            prediction_scores, switching_probabilities
        )
        oracle_concept_router = self.oracle_concept_expert_routers[
            int(concept_id)
        ]
        oracle_concept_repository_probabilities = (
            oracle_concept_router.probabilities(repository_model_ids)
        )
        oracle_concept_probabilities = self._restrict_routing_probabilities(
            oracle_concept_repository_probabilities, model_ids
        )
        oracle_concept_scores = self._weighted_routing_scores(
            prediction_scores, oracle_concept_probabilities
        )
        context_router = None
        context_probabilities = None
        context_scores = None
        context_leader_model_id = None
        meta_router = None
        meta_probabilities = None
        meta_scores = None
        meta_model_probabilities = None
        meta_switching_probabilities = None
        meta_switching_selected = None
        meta_switching_scores = None

        if config.SOFT_ROUTING_CONTEXT in _CONTEXTUAL_ROUTING_MODES:
            context_id = int(
                self._routing_prediction(global_scores, num_classes)
                .view(-1)[0].item()
            )
            context_router = self.context_expert_routers[context_id]
            context_repository_proposal = context_router.probabilities(
                repository_model_ids
            )
            context_proposal = self._restrict_routing_probabilities(
                context_repository_proposal, model_ids
            )
            context_probabilities = self._prediction_probabilities(
                context_proposal
            )
            context_scores = self._weighted_routing_scores(
                prediction_scores, context_probabilities
            )
            context_leader_model_id, _ = self._routing_leader(
                context_probabilities
            )

            meta_router = self.shadow_meta_routers[context_id]
            meta_expert_ids = ("global_mixture", "context_leader")
            meta_probabilities = meta_router.probabilities(meta_expert_ids)
            meta_scores = (
                global_scores * meta_probabilities["global_mixture"]
                + prediction_scores[context_leader_model_id]
                * meta_probabilities["context_leader"]
            )
            meta_model_probabilities = {
                model_id: (
                    meta_probabilities["global_mixture"]
                    * proposal_probabilities[model_id]
                    + meta_probabilities["context_leader"]
                    * int(model_id == context_leader_model_id)
                )
                for model_id in model_ids
            }
            meta_switching_probabilities = (
                self.meta_switching_router.probabilities(
                    ("meta", "switching")
                )
            )
            meta_switching_selected = self.meta_switching_router.leader(
                meta_switching_probabilities, preferred_id="meta",
            )
            if config.SOFT_ROUTING_TOP_COMBINATION == "leader":
                meta_switching_scores = (
                    meta_scores
                    if meta_switching_selected == "meta"
                    else switching_scores
                )
            elif config.SOFT_ROUTING_TOP_COMBINATION == "mixture":
                meta_switching_scores = (
                    meta_scores * meta_switching_probabilities["meta"]
                    + switching_scores
                    * meta_switching_probabilities["switching"]
                )
            else:
                raise ValueError(
                    "未知のMeta-switching上位統合方式です: "
                    f"{config.SOFT_ROUTING_TOP_COMBINATION!r}"
                )
        elif config.SOFT_ROUTING_CONTEXT != "global":
            raise ValueError(
                f"未知のSoftRouting文脈です: {config.SOFT_ROUTING_CONTEXT!r}"
            )

        if config.SOFT_ROUTING_CONTEXT == "global":
            probabilities = self._prediction_probabilities(
                proposal_probabilities
            )
            weighted_scores = self._weighted_routing_scores(
                prediction_scores, probabilities
            )
        elif config.SOFT_ROUTING_CONTEXT == "predicted_class":
            probabilities = context_probabilities
            weighted_scores = context_scores
        elif config.SOFT_ROUTING_CONTEXT == "meta_predicted_class":
            probabilities = meta_model_probabilities
            weighted_scores = meta_scores
        else:
            if config.SOFT_ROUTING_TOP_COMBINATION == "leader":
                probabilities = (
                    meta_model_probabilities
                    if meta_switching_selected == "meta"
                    else switching_probabilities
                )
            else:
                probabilities = {
                    model_id: (
                        meta_switching_probabilities["meta"]
                        * meta_model_probabilities[model_id]
                        + meta_switching_probabilities["switching"]
                        * switching_probabilities[model_id]
                    )
                    for model_id in model_ids
                }
            weighted_scores = meta_switching_scores

        accuracy = float(
            self._routing_correct(weighted_scores, y, num_classes)
        )
        routing_contributions = self.routing_leave_one_out_diagnostics.observe(
            prediction_scores=prediction_scores,
            effective_probabilities=probabilities,
            fallback_probabilities=proposal_probabilities,
            target=y,
            num_classes=num_classes,
            current_model_id=self.current_model_id,
            sample_index=sample_index,
            aggregation_interval=config.AGGREGATION_INTERVAL,
            archive_shadow_enabled=(
                config.ROUTING_ARCHIVE_SHADOW_DIAGNOSTICS
            ),
            archive_shadow_policy=config.ROUTING_ARCHIVE_SHADOW_POLICY,
            forward_probe_samples=(
                config.NEW_MODEL_FORWARD_VALIDATION_SAMPLES
            ),
            repository_model_ids=repository_model_ids,
        )
        if self.routing_active_set is not None:
            self.routing_active_set.observe(
                routing_contributions,
                sample_index,
            )
        switching_correct = self._routing_correct(
            switching_scores, y, num_classes
        )
        oracle_concept_correct = self._routing_correct(
            oracle_concept_scores, y, num_classes
        )
        self.routing_oracle_concept_diagnostics["sample_count"] += 1
        self.routing_oracle_concept_diagnostics["correct_count"] += int(
            oracle_concept_correct
        )
        self.history_routing_oracle_concept_correct.append(
            int(oracle_concept_correct)
        )
        switching_global_correct = self._routing_correct(
            global_scores, y, num_classes
        )
        switching_leader_id, _ = self._routing_leader(
            switching_probabilities
        )
        switching_effective_experts = self.expert_router.effective_expert_count(
            switching_probabilities
        )
        self.routing_switching_diagnostics["sample_count"] += 1
        self.routing_switching_diagnostics["correct_count"] += int(
            switching_correct
        )
        self.routing_switching_diagnostics["actual_correct_count"] += int(
            accuracy
        )
        self.routing_switching_diagnostics["global_correct_count"] += int(
            switching_global_correct
        )
        self.routing_switching_diagnostics[
            "effective_experts_sum"
        ] += switching_effective_experts
        self.history_routing_switching_correct.append(int(switching_correct))
        self.history_routing_switching_leader_id.append(switching_leader_id)
        self.history_routing_switching_effective_experts.append(
            switching_effective_experts
        )
        routed_model_id, maximum = self._routing_leader(probabilities)
        # 混合自体が全単体モデルより良い場合もあるため、oracle候補には実混合も含める。
        oracle_correct = bool(accuracy) or any(model_correctness.values())
        leader_correct = model_correctness[routed_model_id]

        # 適用区間でactive expertが全て誤った場合は、除外集合に正解expertが
        # いる可能性を次標本から確認する。混合だけが誤った場合はrouting重みの
        # 問題なので発火させず、新しい閾値を増やさない。
        if (
            self.routing_active_set is not None
            and not update_routing_evidence
            and not accuracy
            and not any(model_correctness.values())
        ):
            self.routing_active_set.restart_after_active_failure(
                repository_model_ids,
                sample_index + 1,
            )

        meta_correct = None
        global_correct = None
        context_correct = None
        if meta_router is not None:
            meta_correct = self._routing_correct(meta_scores, y, num_classes)
            global_correct = self._routing_correct(global_scores, y, num_classes)
            context_correct = self._routing_correct(
                context_scores, y, num_classes
            )
            if config.SOFT_ROUTING_META_LOSS == "bounded_score":
                meta_losses = {
                    "global_mixture": self._routing_score_loss(
                        global_scores, y, num_classes
                    ),
                    "context_leader": model_losses[context_leader_model_id],
                }
            elif config.SOFT_ROUTING_META_LOSS == "zero_one":
                # Metaの目的を最終accuracyへ揃え、較正差による選択の逆転を避ける。
                meta_losses = {
                    "global_mixture": float(not global_correct),
                    "context_leader": float(
                        not model_correctness[context_leader_model_id]
                    ),
                }
            else:
                raise ValueError(
                    "未知のMeta-router更新損失です: "
                    f"{config.SOFT_ROUTING_META_LOSS!r}"
                )
            context_leader_weight = meta_probabilities["context_leader"]
            self.routing_meta_diagnostics["sample_count"] += 1
            self.routing_meta_diagnostics["correct_count"] += int(meta_correct)
            self.routing_meta_diagnostics["actual_correct_count"] += int(accuracy)
            self.routing_meta_diagnostics["global_correct_count"] += int(
                global_correct
            )
            self.routing_meta_diagnostics[
                "context_mixture_correct_count"
            ] += int(context_correct)
            self.routing_meta_diagnostics[
                "context_leader_correct_count"
            ] += int(model_correctness[context_leader_model_id])
            self.routing_meta_diagnostics[
                "context_leader_weight_sum"
            ] += context_leader_weight
            self.routing_meta_diagnostics[
                "context_leader_preferred_count"
            ] += int(context_leader_weight > 0.5)
            self.history_routing_meta_correct.append(int(meta_correct))
            self.history_routing_meta_global_correct.append(int(global_correct))
            self.history_routing_meta_context_mixture_correct.append(
                int(context_correct)
            )
            self.history_routing_meta_context_leader_correct.append(
                int(model_correctness[context_leader_model_id])
            )
            self.history_routing_meta_context_leader_weight.append(
                context_leader_weight
            )
            if update_routing_evidence:
                meta_router.update(meta_losses, meta_probabilities)

            meta_switching_correct = self._routing_correct(
                meta_switching_scores, y, num_classes
            )
            top_diagnostics = self.routing_meta_switching_diagnostics
            top_diagnostics["sample_count"] += 1
            top_diagnostics["correct_count"] += int(meta_switching_correct)
            top_diagnostics["actual_correct_count"] += int(accuracy)
            top_diagnostics["meta_correct_count"] += int(meta_correct)
            top_diagnostics["switching_correct_count"] += int(
                switching_correct
            )
            top_diagnostics["selected_switching_count"] += int(
                meta_switching_selected == "switching"
            )
            self.history_routing_meta_switching_correct.append(
                int(meta_switching_correct)
            )
            self.history_routing_meta_switching_selected_switching.append(
                int(meta_switching_selected == "switching")
            )
            if update_routing_evidence:
                self.meta_switching_router.update(
                    {
                        "meta": float(not meta_correct),
                        "switching": float(not switching_correct),
                    },
                    meta_switching_probabilities,
                )
        max_confidence = max(model_confidences.values())
        confidence_leaders = [
            model_id for model_id, confidence in model_confidences.items()
            if confidence == max_confidence
        ]
        confidence_model_id = max(
            confidence_leaders,
            key=lambda model_id: (
                probabilities[model_id],
                model_id == self.current_model_id,
                -model_id,
            ),
        )
        confidence_leader_correct = model_correctness[confidence_model_id]
        self.routing_diagnostics["sample_count"] += 1
        self.routing_diagnostics["oracle_correct_count"] += int(oracle_correct)
        self.routing_diagnostics["mixture_correct_count"] += int(accuracy)
        self.routing_diagnostics["leader_correct_count"] += int(leader_correct)
        self.routing_diagnostics["confidence_leader_correct_count"] += int(
            confidence_leader_correct
        )
        self.routing_diagnostics["missed_oracle_count"] += int(
            oracle_correct and not accuracy
        )
        self.routing_diagnostics["confidence_leader_missed_oracle_count"] += int(
            oracle_correct and not confidence_leader_correct
        )
        class_id = int(y.view(-1)[0].item())
        class_diagnostics = self.routing_class_diagnostics[class_id]
        class_diagnostics["sample_count"] += 1
        class_diagnostics["oracle_correct_count"] += int(oracle_correct)
        class_diagnostics["mixture_correct_count"] += int(accuracy)
        class_diagnostics["leader_correct_count"] += int(leader_correct)
        class_diagnostics["confidence_leader_correct_count"] += int(
            confidence_leader_correct
        )
        class_diagnostics["missed_oracle_count"] += int(
            oracle_correct and not accuracy
        )
        class_diagnostics["confidence_leader_missed_oracle_count"] += int(
            oracle_correct and not confidence_leader_correct
        )
        if meta_correct is not None:
            class_diagnostics["meta_sample_count"] += 1
            class_diagnostics["meta_correct_count"] += int(meta_correct)
            class_diagnostics["meta_global_correct_count"] += int(
                global_correct
            )
            class_diagnostics["meta_context_mixture_correct_count"] += int(
                context_correct
            )
            class_diagnostics["meta_context_leader_correct_count"] += int(
                model_correctness[context_leader_model_id]
            )
        self.history_routing_oracle_correct.append(int(oracle_correct))
        self.history_routing_leader_correct.append(int(leader_correct))
        self.history_accuracy.append(accuracy)
        self.history_concept.append(concept_id)
        self.history_model_id.append(routed_model_id)
        self.history_routing_effective_experts.append(
            self.expert_router.effective_expert_count(probabilities)
        )
        self.history_routing_max_weight.append(maximum)

        # prequential順序を守り、予測後に正解ラベルで重みを更新する。
        # 保護方式でもAdaHedge自体は提案分布で更新し、反実仮想の学習を続ける。
        # active-set方式では、全expertの真の損失を観測できるprobe中だけ更新する。
        # 適用区間で未観測expertへ推定損失を代入すると、次のprobeでも累積証拠が
        # 歪んだまま残るため、全ルータの証拠を同じfull-information標本に揃える。
        if update_routing_evidence:
            self.expert_router.update(model_losses, proposal_probabilities)
            oracle_concept_router.update(
                model_losses, oracle_concept_repository_probabilities
            )
            self.switching_expert_router.update(
                model_losses, switching_probabilities
            )
            if context_router is not None:
                context_router.update(model_losses, context_proposal)


class _RestartingSoftRoutingFedSDAClientMixin(
    _AdaHedgeRoutingFedSDAClientMixin
):
    """確定した概念切替ごとにAdaHedgeを再始動するmixin。"""

    def _on_local_model_change(self, old_model_id, new_model_id):
        self.expert_router.restart_for_concept()
        for router in self.context_expert_routers.values():
            router.restart_for_concept()
        for router in self.shadow_meta_routers.values():
            router.restart_for_concept()
        if self.routing_active_set is not None:
            self.routing_active_set.restart_for_concept(
                self.models,
                self.processed_samples,
            )


class RestartingSoftRoutingClassConditionalESRFedSDAClient(
    _RestartingSoftRoutingFedSDAClientMixin,
    ClassConditionalESRFedSDAClient,
):
    """ClassESRとRestarting SoftRoutingを組み合わせるクライアント。"""


class RestartingSoftRoutingESRFedSDAClient(
    _RestartingSoftRoutingFedSDAClientMixin,
    ESRFedSDAClient,
):
    """全体損失e-SRとRestarting SoftRoutingを組み合わせるクライアント。"""


class RestartingSoftRoutingClassConditionalADWINFedSDAClient(
    _RestartingSoftRoutingFedSDAClientMixin,
    ClassConditionalADWINFedSDAClient,
):
    """ClassADWINとRestarting SoftRoutingを組み合わせるクライアント。"""


class RestartingSoftRoutingADWINFedSDAClient(
    _RestartingSoftRoutingFedSDAClientMixin,
    ADWINFedSDAClient,
):
    """全体損失ADWINとRestarting SoftRoutingを組み合わせるクライアント。"""


class ProtectedSoftRoutingClassConditionalESRFedSDAClient(
    RestartingSoftRoutingClassConditionalESRFedSDAClient
):
    """現行モデルより累積損失が小さい場合だけ混合予測を採用する。"""

    def _prediction_probabilities(self, proposal_probabilities):
        cumulative_losses = self.expert_router.cumulative_losses
        incumbent_loss = cumulative_losses[self.current_model_id]
        proposal_loss = sum(
            proposal_probabilities[model_id] * cumulative_losses[model_id]
            for model_id in proposal_probabilities
        )
        gate_open = proposal_loss < incumbent_loss
        self.history_routing_gate_open.append(gate_open)
        if gate_open:
            return proposal_probabilities
        return {
            model_id: 1.0 if model_id == self.current_model_id else 0.0
            for model_id in proposal_probabilities
        }
