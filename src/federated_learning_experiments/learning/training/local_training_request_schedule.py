"""学習要求の保留件数と反復予算を管理し、成功確認後だけ消化する。"""

from .local_training_schedule_settings import LocalTrainingScheduleSettings


class LocalTrainingRequestSchedule:
    """同期する単一所有者のcounter。実学習は外側で行う。"""

    def __init__(self, *, local_training_schedule_settings: LocalTrainingScheduleSettings) -> None:
        if type(local_training_schedule_settings) is not LocalTrainingScheduleSettings:
            raise ValueError(
                "local_training_schedule_settingsはexact LocalTrainingScheduleSettingsが必要です。"
            )
        local_training_schedule_settings.__post_init__()
        self._local_training_schedule_settings = local_training_schedule_settings
        self._pending_training_request_count = 0

    @property
    def pending_training_request_count(self) -> int:
        """未完了要求件数を返し、外側へ可変参照を公開しない。"""
        return self._pending_training_request_count

    def record_training_request(self) -> int:
        """一要求を追加し、間隔到達時の試行回数を返す。"""
        self._pending_training_request_count += 1
        if (
            self._pending_training_request_count
            < self._local_training_schedule_settings.training_requests_per_update_interval
        ):
            return 0
        return self.calculate_pending_joint_update_iteration_count()

    def calculate_pending_joint_update_iteration_count(self) -> int:
        """明示flush用の全保留予算を、件数を変更せず返す。"""
        return (
            self._pending_training_request_count
            * self._local_training_schedule_settings.joint_update_iterations_per_training_request
        )

    def acknowledge_completed_training_requests(
        self, *, completed_training_request_count: int
    ) -> None:
        """現在の全保留が正常終了したという外側の確認を受ける。"""
        if (
            type(completed_training_request_count) is not int
            or completed_training_request_count < 1
            or completed_training_request_count != self._pending_training_request_count
        ):
            raise ValueError(
                "completed_training_request_countは現在の全保留件数と等しい正のbuiltin intが必要です。"
            )
        self._pending_training_request_count = 0
