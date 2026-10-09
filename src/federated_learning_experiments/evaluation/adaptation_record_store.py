"""適応の不変記録と、入力順の履歴・切替位置・再利用件数を保持する。"""

from dataclasses import dataclass, replace
from typing import Literal, get_args

AdaptationOutcome = Literal[
    "alarm_during_candidate_validation",
    "alarm_change_interval_too_short",
    "alarm_interval_held_model_reused",
    "alarm_interval_current_model_maintained",
    "alarm_interval_candidate_validation_started",
    "post_alarm_validation_candidate_adopted",
    "post_alarm_validation_held_model_reused",
    "post_alarm_validation_current_model_maintained",
    "post_alarm_validation_candidate_rejected",
    "post_alarm_validation_incomplete_candidate_rejected",
    "server_consolidation_training_model_remapped",
]

# 学習帰属IDが変わる結果。この結果のときだけ変更前後のIDが異なり、切替位置を記録する。
TRAINING_MODEL_SWITCH_ADAPTATION_OUTCOMES: tuple[AdaptationOutcome, ...] = (
    "alarm_interval_held_model_reused",
    "post_alarm_validation_candidate_adopted",
    "post_alarm_validation_held_model_reused",
)
# サーバの統合で、学習帰属IDが付け替わった結果。IDは変わるが、警報による切替ではないので、
# 切替位置ではなく、付け替えの位置として記録する。
SERVER_REMAP_ADAPTATION_OUTCOME: AdaptationOutcome = "server_consolidation_training_model_remapped"


@dataclass(frozen=True, kw_only=True)
class AdaptationRecord:
    """一回の完了した適応を、判断や学習状態から独立して保持する。"""

    adaptation_sample_index: int
    detector_name: str
    adaptation_outcome: AdaptationOutcome
    previous_training_model_id: int
    current_training_model_id: int
    estimated_change_point_sample_index: int | None
    detection_episode_id: int | None

    def __post_init__(self) -> None:
        if type(self.adaptation_sample_index) is not int:
            raise TypeError("adaptation_sample_index must be builtin int")
        if self.adaptation_sample_index < 0:
            raise ValueError("adaptation_sample_index must be nonnegative")
        if type(self.detector_name) is not str:
            raise TypeError("detector_name must be builtin str")
        if not self.detector_name.strip():
            raise ValueError("detector_name must contain a non-whitespace character")
        if type(self.adaptation_outcome) is not str:
            raise TypeError("adaptation_outcome must be builtin str")
        if self.adaptation_outcome not in get_args(AdaptationOutcome):
            raise ValueError("adaptation_outcome must describe a supported adaptation")
        for model_id in (self.previous_training_model_id, self.current_training_model_id):
            if type(model_id) is not int:
                raise TypeError("training model IDs must be builtin int")
        for parameter_name, specified_value in (
            ("estimated_change_point_sample_index", self.estimated_change_point_sample_index),
            ("detection_episode_id", self.detection_episode_id),
        ):
            if specified_value is None:
                continue
            if type(specified_value) is not int:
                raise TypeError(f"{parameter_name} must be builtin int or None")
            if specified_value < 0:
                raise ValueError(f"{parameter_name} must be nonnegative")
        if (self.previous_training_model_id != self.current_training_model_id) != (
            self.adaptation_outcome in TRAINING_MODEL_SWITCH_ADAPTATION_OUTCOMES
            or self.adaptation_outcome == SERVER_REMAP_ADAPTATION_OUTCOME
        ):
            raise ValueError(
                "training model IDs must differ exactly for a switching or server remap outcome"
            )


@dataclass(frozen=True, kw_only=True)
class AdaptationRecordSnapshot:
    """取得後の追加によって変化しない、記録ownerの状態。"""

    adaptation_records: tuple[AdaptationRecord, ...]
    training_model_switch_sample_indices: tuple[int, ...]
    server_remapped_sample_indices: tuple[int, ...]
    alternative_model_reuse_count: int
    current_model_fit_count: int


class AdaptationRecordStore:
    """適応記録・切替位置・二種類の再利用件数を一箇所で所有する。"""

    def __init__(self) -> None:
        self._adaptation_records: list[AdaptationRecord] = []
        self._training_model_switch_sample_indices: list[int] = []
        self._server_remapped_sample_indices: list[int] = []
        self._alternative_model_reuse_count = 0
        self._current_model_fit_count = 0

    def append_adaptation_record(self, *, adaptation_record: AdaptationRecord) -> None:
        if type(adaptation_record) is not AdaptationRecord:
            raise TypeError("adaptation_record must be exact AdaptationRecord")
        # 入力が手動で破壊されていても全fieldを再検査し、保存用copyを更新前に確定する。
        validated_adaptation_record = replace(adaptation_record)
        self._adaptation_records.append(validated_adaptation_record)
        if (
            validated_adaptation_record.adaptation_outcome
            in TRAINING_MODEL_SWITCH_ADAPTATION_OUTCOMES
        ):
            self._training_model_switch_sample_indices.append(
                validated_adaptation_record.adaptation_sample_index
            )
        if validated_adaptation_record.adaptation_outcome == SERVER_REMAP_ADAPTATION_OUTCOME:
            self._server_remapped_sample_indices.append(
                validated_adaptation_record.adaptation_sample_index
            )
        # 二種類の件数は、警報時の区間評価による再利用・現行適合だけを数える（旧の計数箇所と同じ）。
        if validated_adaptation_record.adaptation_outcome == "alarm_interval_held_model_reused":
            self._alternative_model_reuse_count += 1
        elif (
            validated_adaptation_record.adaptation_outcome
            == "alarm_interval_current_model_maintained"
        ):
            self._current_model_fit_count += 1

    def get_state_snapshot(self) -> AdaptationRecordSnapshot:
        return AdaptationRecordSnapshot(
            adaptation_records=tuple(self._adaptation_records),
            training_model_switch_sample_indices=tuple(self._training_model_switch_sample_indices),
            server_remapped_sample_indices=tuple(self._server_remapped_sample_indices),
            alternative_model_reuse_count=self._alternative_model_reuse_count,
            current_model_fit_count=self._current_model_fit_count,
        )
