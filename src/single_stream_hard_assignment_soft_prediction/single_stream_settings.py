"""手法の設定と既定値。riverとtorchをimportしない。

監視・Fixed-Share・MDLの値のうち、この手法で働くものは製品のFedHASPの既定と同じにする（設計の「設定の一覧」）。
"""

from dataclasses import dataclass
from math import isfinite

from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    LOWER_BASELINE_ON_LOSS_DECREASE_POLICY_NAME,
    WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME,
    LossChangeDetectionSettings,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import (
    PredictionCombinationSettings,
)

# 製品のFedHASPの既定と同じ値（runtime/fedsda_final_configuration_run_settings.py）。一致はtestで確かめる。
_PENDING_ASSIGNMENT_BUFFER_CAPACITY_SAMPLES = 30
_MINIMUM_CHANGE_INTERVAL_SAMPLE_COUNT = 5
_FIXED_SHARE_WEIGHT_REDISTRIBUTION_TIME_SCALE_SAMPLES = 120
_E_SR_FALSE_ALARM_CONTROL_ALPHA = 0.001
_LOSS_MONITOR_BETTING_FRACTIONS = (0.05, 0.1, 0.2, 0.4, 0.8)
_LOSS_MONITOR_MAXIMUM_RETAINED_CANDIDATE_COUNT = 1000
_ADWIN_CONFIDENCE_DELTA = 0.05
_ADWIN_MINIMUM_TEST_WINDOW_SAMPLE_COUNT = 10
NEW_MODEL_HYPOTHESIS_LEARNERS = "reinitialized_and_warm_started_learners"
CHANGE_INTERVAL_SEARCH = "every_sample_with_reverse_prequential_code"
# この手法だけの値（設計の「製品と違う点」5・6）。
_WARM_UP_SAMPLE_COUNT = 200
_PROBABILITY_SMOOTHING_WEIGHT = 0.01


def _validate_positive_int(*, value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name}はbool以外のbuiltin intが必要です。")
    if value < 1:
        raise ValueError(f"{name}は1以上が必要です。")


@dataclass(frozen=True)
class SingleStreamHardAssignmentSoftPredictionSettings:
    """手法の設定。値はデータによらず固定し、runの記録に残す。"""

    pending_assignment_buffer_capacity_samples: int
    minimum_change_interval_sample_count: int
    warm_up_sample_count: int
    probability_smoothing_weight: float
    loss_change_detection_settings: LossChangeDetectionSettings
    loss_monitor_betting_fractions: tuple[float, ...]
    loss_monitor_maximum_retained_candidate_count: int
    prediction_combination_settings: PredictionCombinationSettings
    new_model_hypothesis_learners: str
    change_interval_search: str

    def __post_init__(self) -> None:
        for name in (
            "pending_assignment_buffer_capacity_samples",
            "minimum_change_interval_sample_count",
            "warm_up_sample_count",
            "loss_monitor_maximum_retained_candidate_count",
        ):
            _validate_positive_int(value=getattr(self, name), name=name)
        weight = self.probability_smoothing_weight
        if type(weight) is not float:
            raise TypeError("probability_smoothing_weightはbuiltin floatが必要です。")
        if not isfinite(weight) or not 0.0 < weight < 1.0:
            raise ValueError("probability_smoothing_weightは0より大きく1より小さい値が必要です。")
        if type(self.loss_change_detection_settings) is not LossChangeDetectionSettings:
            raise TypeError(
                "loss_change_detection_settingsはLossChangeDetectionSettingsが必要です。"
            )
        if type(self.prediction_combination_settings) is not PredictionCombinationSettings:
            raise TypeError(
                "prediction_combination_settingsはPredictionCombinationSettingsが必要です。"
            )
        fractions = self.loss_monitor_betting_fractions
        if (
            type(fractions) is not tuple
            or not fractions
            or any(type(f) is not float or not isfinite(f) or not 0.0 < f < 1.0 for f in fractions)
        ):
            raise ValueError(
                "loss_monitor_betting_fractionsは(0, 1)のfloatの空でないtupleが必要です。"
            )
        if self.new_model_hypothesis_learners != NEW_MODEL_HYPOTHESIS_LEARNERS:
            raise ValueError("新しいモデルの学習者は、resetとkeepの2つだけを扱います。")
        if self.change_interval_search != CHANGE_INTERVAL_SEARCH:
            raise ValueError("変化区間の探索は、1件刻みの逆順の1本の流れだけを扱います。")

    @property
    def fixed_share_weight_redistribution_time_scale_samples(self) -> int:
        """Fixed-ShareのT（予測の重みの分け直しと、切り替わりの符号長に使う）。"""
        return self.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples

    @property
    def switching_probability(self) -> float:
        """MDLの切り替わりの確率1/T（製品と同じ）。"""
        return 1.0 / self.fixed_share_weight_redistribution_time_scale_samples


def build_default_single_stream_settings() -> SingleStreamHardAssignmentSoftPredictionSettings:
    """既定の設定（監視・Fixed-Share・MDLは製品のFedHASPの既定と同じ値）。"""
    return SingleStreamHardAssignmentSoftPredictionSettings(
        pending_assignment_buffer_capacity_samples=_PENDING_ASSIGNMENT_BUFFER_CAPACITY_SAMPLES,
        minimum_change_interval_sample_count=_MINIMUM_CHANGE_INTERVAL_SAMPLE_COUNT,
        warm_up_sample_count=_WARM_UP_SAMPLE_COUNT,
        probability_smoothing_weight=_PROBABILITY_SMOOTHING_WEIGHT,
        loss_change_detection_settings=LossChangeDetectionSettings(
            loss_monitoring_baseline_update_policy=LOWER_BASELINE_ON_LOSS_DECREASE_POLICY_NAME,
            short_change_interval_alarm_policy=WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME,
            drift_detector_name="e_sr",
            adwin_confidence_delta=_ADWIN_CONFIDENCE_DELTA,
            adwin_minimum_test_window_sample_count=_ADWIN_MINIMUM_TEST_WINDOW_SAMPLE_COUNT,
            loss_monitoring_scope="overall_and_true_class_losses",
            e_sr_false_alarm_control_alpha=_E_SR_FALSE_ALARM_CONTROL_ALPHA,
        ),
        loss_monitor_betting_fractions=_LOSS_MONITOR_BETTING_FRACTIONS,
        loss_monitor_maximum_retained_candidate_count=_LOSS_MONITOR_MAXIMUM_RETAINED_CANDIDATE_COUNT,
        prediction_combination_settings=PredictionCombinationSettings(
            prediction_combination_strategy="fixed_share_weighted_prediction",
            prediction_mixture_activation_policy="always",
            prediction_weight_recalibration_after_aggregation_policy=(
                "recompute_buffer_losses_and_replay_weight_updates"
            ),
            prediction_state_reset_on_training_assignment_change_policy=(
                "restart_adahedge_preserve_fixed_share_prediction_state"
            ),
            fixed_share_weight_redistribution_time_scale_samples=(
                _FIXED_SHARE_WEIGHT_REDISTRIBUTION_TIME_SCALE_SAMPLES
            ),
            # 同期の無い単一ストリームでは、局所の版だけで予測する（設計の「製品と違う点」1）。
            prediction_model_parameter_source="current_local_parameters",
        ),
        new_model_hypothesis_learners=NEW_MODEL_HYPOTHESIS_LEARNERS,
        change_interval_search=CHANGE_INTERVAL_SEARCH,
    )
