"""モデルへ帰属する学習量と真概念別割当件数を独立に保持する。"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ModelTrainingAndAssignmentCountsSnapshot:
    """辞書は呼出し側が変更できる独立copy。ownerへ変更を伝播しない。"""

    trained_sample_counts_by_model_id: dict[int, int]
    parameter_update_step_counts_by_model_id: dict[int, int]
    assigned_sample_counts_by_model_and_concept_id: dict[int, dict[int, int]]


def _validate_integer(
    *, parameter_value: int, parameter_name: str, minimum_value: int | None = None
) -> None:
    if type(parameter_value) is not int:
        raise TypeError(f"{parameter_name}はbool・派生型以外のbuiltin intが必要です。")
    if minimum_value is not None and parameter_value < minimum_value:
        raise ValueError(f"{parameter_name}は{minimum_value}以上が必要です。")


class ModelTrainingAndAssignmentCountsStore:
    """完了済み学習の計数と診断用割当計数を所有し、実行の判断をしない。"""

    def __init__(self) -> None:
        self._trained_sample_counts_by_model_id: dict[int, int] = {}
        self._parameter_update_step_counts_by_model_id: dict[int, int] = {}
        self._assigned_sample_counts_by_model_and_concept_id: dict[int, dict[int, int]] = {}

    def record_completed_model_training(
        self, *, model_id: int, trained_sample_count: int, parameter_update_step_count: int
    ) -> None:
        """両増分を検証してから加算する。stepは個別parameter更新回数。"""
        _validate_integer(parameter_value=model_id, parameter_name="model_id")
        _validate_integer(
            parameter_value=trained_sample_count,
            parameter_name="trained_sample_count",
            minimum_value=0,
        )
        _validate_integer(
            parameter_value=parameter_update_step_count,
            parameter_name="parameter_update_step_count",
            minimum_value=0,
        )
        self._trained_sample_counts_by_model_id[model_id] = (
            self._trained_sample_counts_by_model_id.get(model_id, 0) + trained_sample_count
        )
        self._parameter_update_step_counts_by_model_id[model_id] = (
            self._parameter_update_step_counts_by_model_id.get(model_id, 0)
            + parameter_update_step_count
        )

    def record_assigned_sample_concept(
        self, *, model_id: int, observed_concept_id: int | None
    ) -> None:
        """真概念が既知の帰属標本だけを1件記録する。予測判断には使わない。"""
        _validate_integer(parameter_value=model_id, parameter_name="model_id")
        if observed_concept_id is None:
            return
        _validate_integer(parameter_value=observed_concept_id, parameter_name="observed_concept_id")
        assigned_sample_counts = self._assigned_sample_counts_by_model_and_concept_id.setdefault(
            model_id, {}
        )
        assigned_sample_counts[observed_concept_id] = (
            assigned_sample_counts.get(observed_concept_id, 0) + 1
        )

    def get_model_assigned_sample_concept_counts(self, *, model_id: int) -> dict[int, int]:
        """欠落項目を生成せず、概念別件数を独立copyで返す。"""
        _validate_integer(parameter_value=model_id, parameter_name="model_id")
        return dict(self._assigned_sample_counts_by_model_and_concept_id.get(model_id, {}))

    def snapshot_model_training_and_assignment_counts(
        self,
    ) -> ModelTrainingAndAssignmentCountsSnapshot:
        """3辞書と各概念内辞書を構造分離する。"""
        return ModelTrainingAndAssignmentCountsSnapshot(
            trained_sample_counts_by_model_id=dict(self._trained_sample_counts_by_model_id),
            parameter_update_step_counts_by_model_id=dict(
                self._parameter_update_step_counts_by_model_id
            ),
            assigned_sample_counts_by_model_and_concept_id={
                model_id: dict(assigned_sample_counts)
                for model_id, assigned_sample_counts in self._assigned_sample_counts_by_model_and_concept_id.items()
            },
        )

    def transfer_model_training_and_assignment_counts(
        self, *, original_model_id: int, receiving_model_id: int
    ) -> None:
        """異なる元IDの計数を除き、先へ加算する。先上書きではない。"""
        _validate_integer(parameter_value=original_model_id, parameter_name="original_model_id")
        _validate_integer(parameter_value=receiving_model_id, parameter_name="receiving_model_id")
        if original_model_id == receiving_model_id:
            raise ValueError("receiving_model_idはoriginal_model_idと異なるIDが必要です。")
        if original_model_id in self._trained_sample_counts_by_model_id:
            trained_sample_count = self._trained_sample_counts_by_model_id.pop(original_model_id)
            self._trained_sample_counts_by_model_id[receiving_model_id] = (
                self._trained_sample_counts_by_model_id.get(receiving_model_id, 0)
                + trained_sample_count
            )
        if original_model_id in self._parameter_update_step_counts_by_model_id:
            parameter_update_step_count = self._parameter_update_step_counts_by_model_id.pop(
                original_model_id
            )
            self._parameter_update_step_counts_by_model_id[receiving_model_id] = (
                self._parameter_update_step_counts_by_model_id.get(receiving_model_id, 0)
                + parameter_update_step_count
            )
        if original_model_id in self._assigned_sample_counts_by_model_and_concept_id:
            assigned_sample_counts = self._assigned_sample_counts_by_model_and_concept_id.pop(
                original_model_id
            )
            receiving_assigned_sample_counts = (
                self._assigned_sample_counts_by_model_and_concept_id.setdefault(
                    receiving_model_id, {}
                )
            )
            for observed_concept_id, assigned_sample_count in assigned_sample_counts.items():
                receiving_assigned_sample_counts[observed_concept_id] = (
                    receiving_assigned_sample_counts.get(observed_concept_id, 0)
                    + assigned_sample_count
                )

    def remap_model_training_and_assignment_counts(
        self, *, model_id_mapping: dict[int, int]
    ) -> None:
        """元順の一回ID対応で加算し、完成した3辞書を交換する。"""
        if type(model_id_mapping) is not dict:
            raise TypeError("model_id_mappingはexact dictが必要です。")
        for original_model_id, mapped_model_id in model_id_mapping.items():
            _validate_integer(
                parameter_value=original_model_id, parameter_name="model_id_mappingのキー"
            )
            _validate_integer(
                parameter_value=mapped_model_id, parameter_name="model_id_mappingの値"
            )
        remapped_trained_sample_counts: dict[int, int] = {}
        remapped_parameter_update_step_counts: dict[int, int] = {}
        remapped_assigned_sample_counts: dict[int, dict[int, int]] = {}
        for (
            original_model_id,
            trained_sample_count,
        ) in self._trained_sample_counts_by_model_id.items():
            mapped_model_id = model_id_mapping.get(original_model_id, original_model_id)
            remapped_trained_sample_counts[mapped_model_id] = (
                remapped_trained_sample_counts.get(mapped_model_id, 0) + trained_sample_count
            )
        for (
            original_model_id,
            parameter_update_step_count,
        ) in self._parameter_update_step_counts_by_model_id.items():
            mapped_model_id = model_id_mapping.get(original_model_id, original_model_id)
            remapped_parameter_update_step_counts[mapped_model_id] = (
                remapped_parameter_update_step_counts.get(mapped_model_id, 0)
                + parameter_update_step_count
            )
        for (
            original_model_id,
            assigned_sample_counts,
        ) in self._assigned_sample_counts_by_model_and_concept_id.items():
            mapped_model_id = model_id_mapping.get(original_model_id, original_model_id)
            receiving_assigned_sample_counts = remapped_assigned_sample_counts.setdefault(
                mapped_model_id, {}
            )
            for observed_concept_id, assigned_sample_count in assigned_sample_counts.items():
                receiving_assigned_sample_counts[observed_concept_id] = (
                    receiving_assigned_sample_counts.get(observed_concept_id, 0)
                    + assigned_sample_count
                )
        self._trained_sample_counts_by_model_id = remapped_trained_sample_counts
        self._parameter_update_step_counts_by_model_id = remapped_parameter_update_step_counts
        self._assigned_sample_counts_by_model_and_concept_id = remapped_assigned_sample_counts
