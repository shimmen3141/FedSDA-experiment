"""割り当て済み学習標本の構造をモデル別に所有する。"""

from __future__ import annotations

from .model_training_sample_records import ModelTrainingSampleCollection, ObservedTrainingSample


class ModelTrainingSampleStore:
    """モデル初出順と標本追加順を保持し、payload参照を借用する。"""

    def __init__(self) -> None:
        self._training_samples_by_model_id: dict[int, list[ObservedTrainingSample]] = {}

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

    def snapshot_ordered_model_training_samples(self) -> tuple[ModelTrainingSampleCollection, ...]:
        """構造を分離したsnapshotを返す。標本recordとTensorは共有する。"""
        return tuple(
            ModelTrainingSampleCollection(
                model_id=model_id, training_samples=tuple(training_samples)
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
        for original_model_id, training_samples in self._training_samples_by_model_id.items():
            mapped_model_id = model_id_mapping.get(original_model_id, original_model_id)
            remapped_training_samples_by_model_id.setdefault(mapped_model_id, []).extend(
                training_samples
            )
        self._training_samples_by_model_id = remapped_training_samples_by_model_id
