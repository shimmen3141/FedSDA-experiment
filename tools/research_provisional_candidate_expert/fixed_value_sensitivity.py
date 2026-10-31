"""基準版16の固定値を研究runだけで置換し、消費側で照合する。"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from unittest.mock import patch


@dataclass(frozen=True)
class FixedValues:
    time_scale_samples: int = 120
    esr_alpha: float = 0.001
    fifo_capacity_samples: int = 30


FIXED_VALUE_SENSITIVITY_VARIANTS = {
    "product-default-time-scale-60": FixedValues(time_scale_samples=60),
    "product-default-time-scale-240": FixedValues(time_scale_samples=240),
    "product-default-esr-alpha-0.01": FixedValues(esr_alpha=0.01),
    "product-default-esr-alpha-0.0001": FixedValues(esr_alpha=0.0001),
    "product-default-fifo-capacity-15": FixedValues(fifo_capacity_samples=15),
    "product-default-fifo-capacity-60": FixedValues(fifo_capacity_samples=60),
}


def replace_fixed_values(settings, values):
    """定数ではなく、ビルダーの戻り値を検証付きのreplaceで作り直す。"""
    participant = settings.run_participant_settings
    client = participant.run_client_settings
    return replace(
        settings,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                client,
                prediction_combination_settings=replace(
                    client.prediction_combination_settings,
                    fixed_share_weight_redistribution_time_scale_samples=values.time_scale_samples,
                ),
                loss_change_detection_settings=replace(
                    client.loss_change_detection_settings,
                    e_sr_false_alarm_control_alpha=values.esr_alpha,
                ),
                training_data_assignment_settings=replace(
                    client.training_data_assignment_settings,
                    pending_assignment_buffer_capacity_samples=values.fifo_capacity_samples,
                ),
            ),
        ),
    )


@contextmanager
def fixed_value_settings(values):
    """研究CLIが呼ぶ最終設定ビルダーの参照だけを、一時的に包む。"""
    from federated_learning_experiments.sweep import (
        experiment_configurations as configurations,
    )

    original = configurations.build_final_configuration_fedsda_run_settings

    def build(**kwargs):
        return replace_fixed_values(original(**kwargs), values)

    with patch.object(configurations, "build_final_configuration_fedsda_run_settings", build):
        yield


@contextmanager
def observe_fixed_value_consumers(*, values, diagnostics):
    """実際の更新・選択・検出・FIFO操作を包み、渡された値を検査する。

    初期統計(500標本/5分割)、e-SR候補保持数(1000)、最小区間(5)は容量に非依存。
    ADWIN補助窓は容量から派生するが、このe-SR構成では呼ばれない。
    """
    from federated_learning_experiments.methods.fedsda.loss_change_detection.bounded_loss_e_sr_detection import (
        BoundedLossESRDetector,
    )
    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )
    from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_and_local_prediction_mixture import (
        SynchronizedAndLocalPredictionMixture,
    )
    from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
        PendingTrainingAssignmentBuffer,
    )
    from federated_learning_experiments.runtime import (
        fedsda_run_client as client_module,
    )
    from federated_learning_experiments.runtime import (
        observed_sample_processing as processing,
    )

    capacity = values.fifo_capacity_samples
    probability = 1.0 / values.time_scale_samples
    diagnostics.update(maximum_pending_sample_count=0, mdl_selection_count=0)

    def verify(name, actual, expected):
        if actual != expected:
            raise RuntimeError(f"固定値の消費側が不一致: {name}: {actual!r} != {expected!r}")
        diagnostics[name] = actual
        count = name + "_observation_count"
        diagnostics[count] = diagnostics.get(count, 0) + 1

    update = FixedShareWeightController.update_weights_after_loss_observation
    mixture_update = SynchronizedAndLocalPredictionMixture.update_weights_after_loss_observation
    detector_init = BoundedLossESRDetector.__init__
    append = PendingTrainingAssignmentBuffer.append_observed_sample_index
    process = client_module.process_observed_sample
    select = processing.select_alarm_interval_assignment

    def update_weights(self, **kwargs):
        verify(
            "fixed_share_rate",
            1.0
            / self._prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples,
            probability,
        )
        return update(self, **kwargs)

    def update_mixture_weights(self, *args, **kwargs):
        # 同期と局所の予測の混合も、分け直しの率に同じ1/Tを使う。
        verify("synchronized_local_share_rate", self._share_rate, probability)
        return mixture_update(self, *args, **kwargs)

    def initialize_detector(self, **kwargs):
        verify("detector_alpha", kwargs["false_alarm_control_alpha"], values.esr_alpha)
        return detector_init(self, **kwargs)

    def append_sample(self, **kwargs):
        verify("fifo_capacity_samples", self.capacity_sample_count, capacity)
        result = append(self, **kwargs)
        length = len(self.get_state_snapshot().pending_sample_indices)
        if length > capacity + 1:
            raise RuntimeError("帰属保留の長さが置換後の容量＋1を超えています")
        diagnostics["maximum_pending_sample_count"] = max(
            diagnostics["maximum_pending_sample_count"], length
        )
        return result

    def process_sample(**kwargs):
        # 警報が無いrunでも、MDLへ渡す値を標本処理の入口で確認する。
        verify("mdl_switching_probability", kwargs["switching_probability"], probability)
        verify(
            "fifo_capacity_samples",
            kwargs["pending_training_assignment_buffer"].capacity_sample_count,
            capacity,
        )
        return process(**kwargs)

    def select_assignment(**kwargs):
        verify(
            "mdl_selection_switching_probability",
            kwargs["switching_probability"],
            probability,
        )
        verify(
            "change_interval_grid_sample_count",
            kwargs["change_interval_grid_sample_count"],
            5,
        )
        length = len(kwargs["pending_sample_observations"])
        if length > capacity + 1:
            raise RuntimeError("探索範囲が置換後の容量＋1を超えています")
        diagnostics["maximum_search_sample_count"] = max(
            diagnostics.get("maximum_search_sample_count", 0), length
        )
        diagnostics["mdl_selection_count"] += 1
        return select(**kwargs)

    with ExitStack() as stack:
        for target, name, replacement in (
            (
                FixedShareWeightController,
                "update_weights_after_loss_observation",
                update_weights,
            ),
            (
                SynchronizedAndLocalPredictionMixture,
                "update_weights_after_loss_observation",
                update_mixture_weights,
            ),
            (BoundedLossESRDetector, "__init__", initialize_detector),
            (
                PendingTrainingAssignmentBuffer,
                "append_observed_sample_index",
                append_sample,
            ),
            (client_module, "process_observed_sample", process_sample),
            (processing, "select_alarm_interval_assignment", select_assignment),
        ):
            stack.enter_context(patch.object(target, name, replacement))
        yield
