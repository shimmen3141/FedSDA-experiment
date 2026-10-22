"""警報の後の、最小記述長の原理による選択ごとの記録（選んだ仮説と、その後に警報にしたか）を持つ。"""

from dataclasses import dataclass
from math import isfinite

from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    MinimumDescriptionLengthAssignmentSelection,
)


@dataclass(frozen=True, kw_only=True)
class AlarmIntervalAssignmentSelectionRecord:
    """選択1回の記録。行き先の保有モデルのIDが無い（現在のモデルのまま、新しいモデル）ときは−1。"""

    alarm_sample_index: int
    selected_hypothesis: str
    change_interval_sample_count: int
    destination_model_id: int
    description_length: float
    # 選択の後に、短い警報の扱いで警報にしなかったなら、False。
    alarm_raised: bool
    # 「新しいモデル」を選んだときの、その変化区間の長さで最短だった学習者。ほかの仮説では空の文字列。
    new_model_learner_name: str


class AlarmIntervalAssignmentSelectionRecordStore:
    """記録だけを所有し、選択や応答の判断をしない。"""

    def __init__(self) -> None:
        self._selection_records: list[AlarmIntervalAssignmentSelectionRecord] = []

    def append_selection_record(
        self,
        *,
        alarm_sample_index: int,
        selection: MinimumDescriptionLengthAssignmentSelection,
        alarm_raised: bool,
    ) -> None:
        """選択1回を足す。警報の標本の番号は、前の記録より後であること。"""
        if type(alarm_sample_index) is not int or alarm_sample_index < 0:
            raise ValueError("alarm_sample_index must be a nonnegative builtin int")
        if type(selection) is not MinimumDescriptionLengthAssignmentSelection:
            raise TypeError("selection must be exact MinimumDescriptionLengthAssignmentSelection")
        if type(alarm_raised) is not bool:
            raise TypeError("alarm_raised must be builtin bool")
        if (
            self._selection_records
            and alarm_sample_index <= self._selection_records[-1].alarm_sample_index
        ):
            raise ValueError("alarm_sample_index must follow the last recorded selection")
        if not isfinite(selection.description_length):
            raise ValueError("description_length must be finite")
        self._selection_records.append(
            AlarmIntervalAssignmentSelectionRecord(
                alarm_sample_index=alarm_sample_index,
                selected_hypothesis=selection.selected_hypothesis,
                change_interval_sample_count=selection.change_interval_sample_count,
                destination_model_id=-1
                if selection.destination_model_id is None
                else selection.destination_model_id,
                description_length=selection.description_length,
                alarm_raised=alarm_raised,
                new_model_learner_name=selection.new_model_learner_name,
            )
        )

    def snapshot_selection_records(self) -> tuple[AlarmIntervalAssignmentSelectionRecord, ...]:
        return tuple(self._selection_records)
