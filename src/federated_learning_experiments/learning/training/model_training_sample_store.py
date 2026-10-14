"""割り当て済み学習標本の構造をモデル別に所有する。"""

from __future__ import annotations

from .model_training_sample_records import ModelTrainingSampleCollection, ObservedTrainingSample


class ModelTrainingSampleStore:
    """モデル初出順と標本追加順を保持し、payload参照を借用する。"""

    def __init__(self) -> None:
        self._training_samples_by_model_id: dict[int, list[ObservedTrainingSample]] = {}
        # モデルごとの最新の区切り（列の位置）。警報の後に同じモデルで続けると決まったときに付く。
        self._latest_alarm_boundary_position_by_model_id: dict[int, int] = {}

    def count_model_training_samples(self, *, model_id: int) -> int:
        """そのモデルの列の標本の数を返す（列が無ければ0）。"""
        if type(model_id) is not int:
            raise TypeError("model_idはbool以外のbuiltin intが必要です。")
        return len(self._training_samples_by_model_id.get(model_id, ()))

    def mark_latest_alarm_boundary(self, *, model_id: int, position: int) -> None:
        """指定の位置を、そのモデルの最新の区切りとして記録する（列があり、位置が0以上かつ列の長さ以下）。"""
        if type(model_id) is not int:
            raise TypeError("model_idはbool以外のbuiltin intが必要です。")
        if type(position) is not int:
            raise TypeError("positionはbool以外のbuiltin intが必要です。")
        if model_id not in self._training_samples_by_model_id:
            raise ValueError("区切りは、標本の列があるモデルにだけ付けられます。")
        if not 0 <= position <= len(self._training_samples_by_model_id[model_id]):
            raise ValueError("positionは0以上かつ標本の列の長さ以下が必要です。")
        self._latest_alarm_boundary_position_by_model_id[model_id] = position

    def append_model_training_samples(
        self, *, model_id: int, training_samples: tuple[ObservedTrainingSample, ...]
    ) -> None:
        """全入力を検証してから追加する。空列もモデルを登録する。"""
        if type(model_id) is not int:
            raise TypeError("model_idはbool以外のbuiltin intが必要です。")
        if type(training_samples) is not tuple:
            raise TypeError("training_samplesはexact tupleが必要です。")
        for training_sample in training_samples:
            if type(training_sample) is not ObservedTrainingSample:
                raise TypeError(
                    "training_samplesの各要素はexact ObservedTrainingSampleが必要です。"
                )
        self._training_samples_by_model_id.setdefault(model_id, []).extend(training_samples)

    def reassign_model_training_samples_id(
        self, *, original_model_id: int, reassigned_model_id: int
    ) -> None:
        """元の標本列で先を上書きし、payloadと取得済みsnapshotには触れない。"""
        for parameter_name, model_id in (
            ("original_model_id", original_model_id),
            ("reassigned_model_id", reassigned_model_id),
        ):
            if type(model_id) is not int:
                raise TypeError(f"{parameter_name}はbool・派生型以外のbuiltin intが必要です。")
        if original_model_id in self._training_samples_by_model_id:
            training_samples = self._training_samples_by_model_id.pop(original_model_id)
            self._training_samples_by_model_id[reassigned_model_id] = training_samples
            # 区切りも、列と同じく移す（元に区切りが無ければ、先の区切りも消す）。
            boundary_position = self._latest_alarm_boundary_position_by_model_id.pop(
                original_model_id, None
            )
            if boundary_position is None:
                self._latest_alarm_boundary_position_by_model_id.pop(reassigned_model_id, None)
            else:
                self._latest_alarm_boundary_position_by_model_id[reassigned_model_id] = (
                    boundary_position
                )

    def snapshot_ordered_model_training_samples(self) -> tuple[ModelTrainingSampleCollection, ...]:
        """構造を分離したsnapshotを返す。標本recordとTensorは共有する。"""
        return tuple(
            ModelTrainingSampleCollection(
                model_id=model_id,
                training_samples=tuple(training_samples),
                latest_alarm_boundary_position=(
                    self._latest_alarm_boundary_position_by_model_id.get(model_id)
                ),
            )
            for model_id, training_samples in self._training_samples_by_model_id.items()
        )

    def remap_model_training_sample_collections(self, *, model_id_mapping: dict[int, int]) -> None:
        """一回対応で列を元順に連結し、完成した構造へ交換する。"""
        if type(model_id_mapping) is not dict:
            raise TypeError("model_id_mappingはexact dictが必要です。")
        for original_model_id, mapped_model_id in model_id_mapping.items():
            if type(original_model_id) is not int or type(mapped_model_id) is not int:
                raise TypeError("model_id_mappingの全キーと値はbool以外のbuiltin intが必要です。")
        remapped_training_samples_by_model_id: dict[int, list[ObservedTrainingSample]] = {}
        original_model_ids_by_mapped_model_id: dict[int, list[int]] = {}
        for original_model_id, training_samples in self._training_samples_by_model_id.items():
            mapped_model_id = model_id_mapping.get(original_model_id, original_model_id)
            remapped_training_samples_by_model_id.setdefault(mapped_model_id, []).extend(
                training_samples
            )
            original_model_ids_by_mapped_model_id.setdefault(mapped_model_id, []).append(
                original_model_id
            )
        # 区切り: 1つの列がIDだけ変わったなら移し、複数の列がつながったなら消す（どこが新しい概念か決められない）。
        remapped_boundary_positions: dict[int, int] = {}
        for mapped_model_id, original_model_ids in original_model_ids_by_mapped_model_id.items():
            if len(original_model_ids) != 1:
                continue
            boundary_position = self._latest_alarm_boundary_position_by_model_id.get(
                original_model_ids[0]
            )
            if boundary_position is not None:
                remapped_boundary_positions[mapped_model_id] = boundary_position
        self._training_samples_by_model_id = remapped_training_samples_by_model_id
        self._latest_alarm_boundary_position_by_model_id = remapped_boundary_positions
