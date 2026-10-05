"""全体と正解クラス系列のe-SRを混合し、候補をglobal位置へ対応させる。"""

import math
from collections import deque
from dataclasses import dataclass

from .bounded_loss_e_sr_detection import (
    BoundedLossESRDetector,
    BoundedLossESRState,
    _validate_bounded_number,
)
from .loss_change_detection_settings import LossChangeDetectionSettings


@dataclass(frozen=True, kw_only=True)
class LossMonitoringObservation:
    """警報とFIFO切詰め前の候補global位置、その回の計算件数。"""

    sample_index: int
    log_e_value: float
    drift_detected: bool
    detector_candidate_start_sample_index: int
    estimated_change_span_sample_count: int
    component_update_count: int
    evaluated_candidate_bet_count: int


@dataclass(frozen=True, kw_only=True)
class OverallAndTrueClassLossMonitoringState:
    """保持capitalとglobal位置を独立tupleとして観測する。"""

    overall_esr_state: BoundedLossESRState
    class_esr_states_by_class_id: tuple[tuple[int, BoundedLossESRState], ...]
    class_sample_indices_by_class_id: tuple[tuple[int, tuple[int, ...]], ...]
    last_sample_index: int | None
    last_observation: LossMonitoringObservation | None


class OverallAndTrueClassLossMonitor:
    """モデルを持たず、観測後の現行モデル損失だけを監視する。"""

    def __init__(
        self,
        *,
        loss_change_detection_settings: LossChangeDetectionSettings,
        class_count: int,
        initial_baseline_loss_mean: float,
        maximum_retained_candidate_count: int,
        betting_fractions: tuple[float, ...],
    ) -> None:
        if type(loss_change_detection_settings) is not LossChangeDetectionSettings:
            raise TypeError(
                "loss_change_detection_settingsはLossChangeDetectionSettingsにしてください。"
            )
        loss_change_detection_settings.__post_init__()
        if type(class_count) is not int:
            raise TypeError("class_countはbool以外のbuiltin intにしてください。")
        if class_count < 2:
            raise ValueError("class_countは2以上にしてください。")
        self._overall_detector = BoundedLossESRDetector(
            baseline_loss_mean=initial_baseline_loss_mean,
            false_alarm_control_alpha=float(
                loss_change_detection_settings.e_sr_false_alarm_control_alpha
            ),
            maximum_retained_candidate_count=maximum_retained_candidate_count,
            betting_fractions=betting_fractions,
        )
        self._loss_change_detection_settings = loss_change_detection_settings
        self._class_count = class_count
        self._maximum_retained_candidate_count = maximum_retained_candidate_count
        self._betting_fractions = betting_fractions
        self._overall_component_weight = 1.0 / (class_count + 1)
        self._class_component_weight = (1.0 - self._overall_component_weight) / class_count
        self._log_alarm_threshold = math.log(
            1.0 / float(loss_change_detection_settings.e_sr_false_alarm_control_alpha)
        )
        self._class_detectors_by_class_id: dict[int, BoundedLossESRDetector] = {}
        self._class_sample_indices_by_class_id: dict[int, deque[int]] = {}
        self._last_sample_index: int | None = None
        self._last_observation: LossMonitoringObservation | None = None

    @property
    def last_observation(self) -> LossMonitoringObservation | None:
        """未観測ならNone、それ以外はimmutable結果を返す。"""
        return self._last_observation

    def _validate_monitoring_observation(
        self,
        *,
        observed_loss: float,
        observed_class_id: int,
        sample_index: int,
        current_model_baseline_loss_mean: float,
    ) -> None:
        _validate_bounded_number(specified_value=observed_loss, parameter_name="observed_loss")
        _validate_bounded_number(
            specified_value=current_model_baseline_loss_mean,
            parameter_name="current_model_baseline_loss_mean",
        )
        if type(observed_class_id) is not int:
            raise TypeError("observed_class_idはbool以外のbuiltin intにしてください。")
        if not 0 <= observed_class_id < self._class_count:
            raise ValueError("observed_class_idは0～K-1にしてください。")
        if type(sample_index) is not int:
            raise TypeError("sample_indexはbool以外のbuiltin intにしてください。")
        if sample_index < 0 or (
            self._last_sample_index is not None and sample_index != self._last_sample_index + 1
        ):
            raise ValueError("sample_indexは非負、同区間内は直前位置+1にしてください。")

    def observe_loss_after_label_observation(
        self,
        *,
        observed_loss: float,
        observed_class_id: int,
        sample_index: int,
        current_model_baseline_loss_mean: float,
    ) -> LossMonitoringObservation:
        """全入力検証後、全体と正解classだけを更新して固定混合する。"""
        self._validate_monitoring_observation(
            observed_loss=observed_loss,
            observed_class_id=observed_class_id,
            sample_index=sample_index,
            current_model_baseline_loss_mean=current_model_baseline_loss_mean,
        )
        overall_observation = self._overall_detector.observe_loss(observed_loss=observed_loss)
        class_detector = self._class_detectors_by_class_id.get(observed_class_id)
        if class_detector is None:
            class_detector = BoundedLossESRDetector(
                baseline_loss_mean=current_model_baseline_loss_mean,
                false_alarm_control_alpha=float(
                    self._loss_change_detection_settings.e_sr_false_alarm_control_alpha
                ),
                maximum_retained_candidate_count=self._maximum_retained_candidate_count,
                betting_fractions=self._betting_fractions,
            )
            self._class_detectors_by_class_id[observed_class_id] = class_detector
            self._class_sample_indices_by_class_id[observed_class_id] = deque()
        class_sample_indices = self._class_sample_indices_by_class_id[observed_class_id]
        class_sample_indices.append(sample_index)
        class_observation = class_detector.observe_loss(observed_loss=observed_loss)
        while len(class_sample_indices) > self._maximum_retained_candidate_count:
            class_sample_indices.popleft()
        component_logs = {
            "overall": overall_observation.log_e_value + math.log(self._overall_component_weight),
            observed_class_id: class_observation.log_e_value
            + math.log(self._class_component_weight),
        }
        for other_class_id, other_class_detector in self._class_detectors_by_class_id.items():
            if other_class_id != observed_class_id:
                component_logs[other_class_id] = (
                    other_class_detector.last_observation.log_e_value
                    + math.log(self._class_component_weight)
                )
        finite_component_logs = [
            log_e_value for log_e_value in component_logs.values() if math.isfinite(log_e_value)
        ]
        if finite_component_logs:
            maximum_log_value = max(finite_component_logs)
            combined_log_e_value = maximum_log_value + math.log(
                sum(
                    math.exp(log_e_value - maximum_log_value)
                    for log_e_value in finite_component_logs
                )
            )
        else:
            combined_log_e_value = -math.inf
        drift_detected = combined_log_e_value >= self._log_alarm_threshold
        detector_candidate_start_sample_index = (
            sample_index - overall_observation.estimated_change_span_sample_count + 1
        )
        if drift_detected:
            best_component = max(component_logs, key=component_logs.get)
            if best_component != "overall":
                class_observation = self._class_detectors_by_class_id[
                    best_component
                ].last_observation
                class_sample_indices = self._class_sample_indices_by_class_id[best_component]
                candidate_position_offset = (
                    class_observation.candidate_start_observation_number
                    - class_observation.oldest_retained_candidate_observation_number
                )
                detector_candidate_start_sample_index = class_sample_indices[
                    max(0, min(candidate_position_offset, len(class_sample_indices) - 1))
                ]
        self._last_sample_index = sample_index
        self._last_observation = LossMonitoringObservation(
            sample_index=sample_index,
            log_e_value=combined_log_e_value,
            drift_detected=drift_detected,
            detector_candidate_start_sample_index=detector_candidate_start_sample_index,
            estimated_change_span_sample_count=sample_index
            - detector_candidate_start_sample_index
            + 1,
            component_update_count=2,
            evaluated_candidate_bet_count=overall_observation.evaluated_candidate_bet_count
            + self._class_detectors_by_class_id[
                observed_class_id
            ].last_observation.evaluated_candidate_bet_count,
        )
        return self._last_observation

    def reset(self, *, baseline_loss_mean: float) -> None:
        """検証済みbaselineでoverallを再開し、class/global位置を破棄する。"""
        self._overall_detector.reset(baseline_loss_mean=baseline_loss_mean)
        self._class_detectors_by_class_id.clear()
        self._class_sample_indices_by_class_id.clear()
        self._last_sample_index = None
        self._last_observation = None

    def get_state_snapshot(self) -> OverallAndTrueClassLossMonitoringState:
        """class初出順を保存し、配列/dequeを独立したimmutable値へコピーする。"""
        return OverallAndTrueClassLossMonitoringState(
            overall_esr_state=self._overall_detector.get_state_snapshot(),
            class_esr_states_by_class_id=tuple(
                (class_id, class_detector.get_state_snapshot())
                for class_id, class_detector in self._class_detectors_by_class_id.items()
            ),
            class_sample_indices_by_class_id=tuple(
                (class_id, tuple(class_sample_indices))
                for class_id, class_sample_indices in self._class_sample_indices_by_class_id.items()
            ),
            last_sample_index=self._last_sample_index,
            last_observation=self._last_observation,
        )
