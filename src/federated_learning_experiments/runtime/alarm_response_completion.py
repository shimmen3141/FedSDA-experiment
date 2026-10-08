"""警報応答の後に、損失監視の再開と保留位置の消費を行い、記録用の情報を返す。"""

from dataclasses import dataclass

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
    select_loss_monitoring_baseline_mean_loss,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.runtime.alarm_buffer_response import AlarmBufferResponse


def _validate_optional_nonnegative_index(
    *, specified_value: int | None, parameter_name: str
) -> None:
    if specified_value is None:
        return
    if type(specified_value) is not int:
        raise TypeError(f"{parameter_name} must be builtin int or None")
    if specified_value < 0:
        raise ValueError(f"{parameter_name} must be nonnegative")


@dataclass(frozen=True, kw_only=True)
class AlarmResponseCompletion:
    """完了した警報応答と、呼出側の記録・通知・session保持に必要な情報。"""

    alarm_buffer_response: AlarmBufferResponse
    alarm_sample_index: int
    previous_training_model_id: int
    current_training_model_id: int
    estimated_change_point_sample_index: int | None
    detection_episode_id: int | None
    loss_monitoring_baseline_mean_loss: float
    drained_pending_sample_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.alarm_buffer_response) is not AlarmBufferResponse:
            raise TypeError("alarm_buffer_response must be exact AlarmBufferResponse")
        if type(self.alarm_sample_index) is not int:
            raise TypeError("alarm_sample_index must be builtin int")
        if self.alarm_sample_index < 0:
            raise ValueError("alarm_sample_index must be nonnegative")
        for model_id in (self.previous_training_model_id, self.current_training_model_id):
            if type(model_id) is not int:
                raise TypeError("training model IDs must be builtin int")
        _validate_optional_nonnegative_index(
            specified_value=self.estimated_change_point_sample_index,
            parameter_name="estimated_change_point_sample_index",
        )
        _validate_optional_nonnegative_index(
            specified_value=self.detection_episode_id, parameter_name="detection_episode_id"
        )
        if type(self.loss_monitoring_baseline_mean_loss) is not float:
            raise TypeError("loss_monitoring_baseline_mean_loss must be builtin float")
        if not 0.0 < self.loss_monitoring_baseline_mean_loss < 1.0:
            raise ValueError("loss_monitoring_baseline_mean_loss must be inside (0, 1)")
        if type(self.drained_pending_sample_indices) is not tuple:
            raise TypeError("drained_pending_sample_indices must be exact tuple")
        for sample_index in self.drained_pending_sample_indices:
            if type(sample_index) is not int:
                raise TypeError("drained sample indices must be builtin int")
            if sample_index < 0:
                raise ValueError("drained sample indices must be nonnegative")
        training_model_changed = self.previous_training_model_id != self.current_training_model_id
        if training_model_changed != (
            self.alarm_buffer_response.response_outcome == "alarm_interval_held_model_reused"
        ):
            raise ValueError("training model IDs must differ exactly for a reused held model")
        if (
            not self.alarm_buffer_response.pending_assignment_buffer_should_be_cleared
            and self.drained_pending_sample_indices
        ):
            raise ValueError("a too short change interval must keep pending sample indices")

    @property
    def training_model_switch_sample_index(self) -> int | None:
        if self.previous_training_model_id == self.current_training_model_id:
            return None
        return self.alarm_sample_index

    @property
    def detection_episode_operation_required(self) -> bool:
        return self.previous_training_model_id != self.current_training_model_id


def complete_alarm_buffer_response(
    *,
    alarm_buffer_response: AlarmBufferResponse,
    alarm_sample_index: int,
    estimated_change_point_sample_index: int | None,
    detection_episode_id: int | None,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    loss_change_monitor: OverallAndTrueClassLossMonitor,
    pending_training_assignment_buffer: PendingTrainingAssignmentBuffer,
) -> AlarmResponseCompletion:
    if type(alarm_buffer_response) is not AlarmBufferResponse:
        raise TypeError("alarm_buffer_response must be exact AlarmBufferResponse")
    if type(current_training_model_assignment) is not CurrentTrainingModelAssignment:
        raise TypeError(
            "current_training_model_assignment must be exact CurrentTrainingModelAssignment"
        )
    if type(loss_statistics_store) is not ModelAndClassLossStatisticsStore:
        raise TypeError("loss_statistics_store must be exact ModelAndClassLossStatisticsStore")
    if type(loss_change_monitor) is not OverallAndTrueClassLossMonitor:
        raise TypeError("loss_change_monitor must be exact OverallAndTrueClassLossMonitor")
    if type(pending_training_assignment_buffer) is not PendingTrainingAssignmentBuffer:
        raise TypeError(
            "pending_training_assignment_buffer must be exact PendingTrainingAssignmentBuffer"
        )
    if type(alarm_sample_index) is not int:
        raise TypeError("alarm_sample_index must be builtin int")
    if alarm_sample_index < 0:
        raise ValueError("alarm_sample_index must be nonnegative")
    _validate_optional_nonnegative_index(
        specified_value=estimated_change_point_sample_index,
        parameter_name="estimated_change_point_sample_index",
    )
    _validate_optional_nonnegative_index(
        specified_value=detection_episode_id, parameter_name="detection_episode_id"
    )
    pending_assignment_state = pending_training_assignment_buffer.get_state_snapshot()
    if pending_assignment_state.last_observed_sample_index not in (None, alarm_sample_index):
        raise ValueError("alarm_sample_index must be the last observed sample index")
    prepared_alarm_training_intervals = alarm_buffer_response.prepared_alarm_training_intervals
    if prepared_alarm_training_intervals is not None and (
        tuple(
            indexed_observation.sample_index
            for indexed_observation in (
                *prepared_alarm_training_intervals.earlier_observations,
                *prepared_alarm_training_intervals.change_interval_observations,
            )
        )
        != pending_assignment_state.pending_sample_indices
    ):
        raise ValueError("prepared intervals must cover the pending sample indices exactly")
    current_training_model_id = current_training_model_assignment.current_training_model_id
    previous_training_model_id = current_training_model_id
    change_interval_resolution = alarm_buffer_response.change_interval_resolution
    if (
        change_interval_resolution is not None
        and change_interval_resolution.training_model_assignment_change is not None
    ):
        training_model_assignment_change = (
            change_interval_resolution.training_model_assignment_change
        )
        if training_model_assignment_change.current_model_id != current_training_model_id:
            raise ValueError("response must describe the current training model assignment")
        previous_training_model_id = training_model_assignment_change.previous_model_id
    # 応答で吸収した後の現行モデルの履歴統計から、監視の基準平均を選ぶ。
    current_model_loss_statistics = loss_statistics_store.get_model_loss_statistics(
        model_id=current_training_model_id
    )
    loss_monitoring_baseline_mean_loss = select_loss_monitoring_baseline_mean_loss(
        loss_moments=(
            None
            if current_model_loss_statistics is None
            else current_model_loss_statistics.overall_loss_moments
        )
    )
    # ここまで読取りだけ。更新は監視の再開、保留位置の消費の順（旧のreset→clear）。
    loss_change_monitor.reset(baseline_loss_mean=loss_monitoring_baseline_mean_loss)
    drained_pending_sample_indices = (
        pending_training_assignment_buffer.drain_pending_sample_indices()
        if alarm_buffer_response.pending_assignment_buffer_should_be_cleared
        else ()
    )
    return AlarmResponseCompletion(
        alarm_buffer_response=alarm_buffer_response,
        alarm_sample_index=alarm_sample_index,
        previous_training_model_id=previous_training_model_id,
        current_training_model_id=current_training_model_id,
        estimated_change_point_sample_index=estimated_change_point_sample_index,
        detection_episode_id=detection_episode_id,
        loss_monitoring_baseline_mean_loss=loss_monitoring_baseline_mean_loss,
        drained_pending_sample_indices=drained_pending_sample_indices,
    )
