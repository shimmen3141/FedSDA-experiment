"""評価標本の列を所有し、明示乱数で抽出追加とID再編を行う。"""

from random import Random

from .model_evaluation_sample_records import (
    ModelEvaluationSampleCollection,
    ObservedEvaluationSample,
)


def _validate_model_id(*, model_id: int, parameter_name: str) -> None:
    if type(model_id) is not int:
        raise TypeError(f"{parameter_name}はbool・派生型以外のbuiltin intが必要です。")


class ModelEvaluationSampleStore:
    """モデル初出順と抽出順を保持し、recordとpayloadは借用する。"""

    def __init__(
        self, *, maximum_stored_sample_count_per_model: int, added_batch_sample_count: int
    ) -> None:
        if (
            type(maximum_stored_sample_count_per_model) is not int
            or maximum_stored_sample_count_per_model < 1
        ):
            raise ValueError("maximum_stored_sample_count_per_modelは正のbuiltin intが必要です。")
        if type(added_batch_sample_count) is not int or added_batch_sample_count < 0:
            raise ValueError("added_batch_sample_countは非負のbuiltin intが必要です。")
        self._maximum_stored_sample_count_per_model = maximum_stored_sample_count_per_model
        self._added_batch_sample_count = added_batch_sample_count
        self._evaluation_samples_by_model_id: dict[int, list[ObservedEvaluationSample]] = {}

    def sample_and_append_model_evaluation_samples(
        self,
        *,
        model_id: int,
        evaluation_samples: tuple[ObservedEvaluationSample, ...],
        python_random_generator: Random,
    ) -> None:
        """抽出を追加し、超過時に末尾を残す。負IDは事前検査後に無視する。"""
        _validate_model_id(model_id=model_id, parameter_name="model_id")
        if type(evaluation_samples) is not tuple:
            raise TypeError("evaluation_samplesはexact tupleが必要です。")
        for evaluation_sample in evaluation_samples:
            if type(evaluation_sample) is not ObservedEvaluationSample:
                raise TypeError(
                    "evaluation_samplesの各要素はexact ObservedEvaluationSampleが必要です。"
                )
        if type(python_random_generator) is not Random:
            raise TypeError("python_random_generatorはexact random.Randomが必要です。")
        if model_id < 0:
            return
        self._evaluation_samples_by_model_id.setdefault(model_id, [])
        sample_count_to_append = min(len(evaluation_samples), self._added_batch_sample_count)
        if sample_count_to_append == 0:
            return
        sampled_evaluation_samples = python_random_generator.sample(
            evaluation_samples, sample_count_to_append
        )
        self._evaluation_samples_by_model_id[model_id].extend(sampled_evaluation_samples)
        if (
            len(self._evaluation_samples_by_model_id[model_id])
            > self._maximum_stored_sample_count_per_model
        ):
            self._evaluation_samples_by_model_id[model_id] = self._evaluation_samples_by_model_id[
                model_id
            ][-self._maximum_stored_sample_count_per_model :]

    def reassign_model_evaluation_samples_id(
        self, *, original_model_id: int, reassigned_model_id: int
    ) -> None:
        """元列で先を上書きし、容量抽出やpayload変更を行わない。"""
        _validate_model_id(model_id=original_model_id, parameter_name="original_model_id")
        _validate_model_id(model_id=reassigned_model_id, parameter_name="reassigned_model_id")
        if original_model_id in self._evaluation_samples_by_model_id:
            evaluation_samples = self._evaluation_samples_by_model_id.pop(original_model_id)
            self._evaluation_samples_by_model_id[reassigned_model_id] = evaluation_samples

    def remap_model_evaluation_sample_collections(
        self, *, model_id_mapping: dict[int, int], python_random_generator: Random
    ) -> None:
        """一回ID対応で元順に連結し、先順に超過列だけ再抽出して交換する。"""
        if type(model_id_mapping) is not dict:
            raise TypeError("model_id_mappingはexact dictが必要です。")
        for original_model_id, mapped_model_id in model_id_mapping.items():
            _validate_model_id(model_id=original_model_id, parameter_name="model_id_mappingのキー")
            _validate_model_id(model_id=mapped_model_id, parameter_name="model_id_mappingの値")
        if type(python_random_generator) is not Random:
            raise TypeError("python_random_generatorはexact random.Randomが必要です。")
        remapped_evaluation_samples_by_model_id: dict[int, list[ObservedEvaluationSample]] = {}
        for original_model_id, evaluation_samples in self._evaluation_samples_by_model_id.items():
            mapped_model_id = model_id_mapping.get(original_model_id, original_model_id)
            remapped_evaluation_samples_by_model_id.setdefault(mapped_model_id, []).extend(
                evaluation_samples
            )
        for mapped_model_id, evaluation_samples in remapped_evaluation_samples_by_model_id.items():
            if len(evaluation_samples) > self._maximum_stored_sample_count_per_model:
                remapped_evaluation_samples_by_model_id[mapped_model_id] = (
                    python_random_generator.sample(
                        evaluation_samples, self._maximum_stored_sample_count_per_model
                    )
                )
        self._evaluation_samples_by_model_id = remapped_evaluation_samples_by_model_id

    def snapshot_ordered_model_evaluation_samples(
        self,
    ) -> tuple[ModelEvaluationSampleCollection, ...]:
        """列構造を分離し、record/Tensor参照は共有する。"""
        return tuple(
            ModelEvaluationSampleCollection(
                model_id=model_id, evaluation_samples=tuple(evaluation_samples)
            )
            for model_id, evaluation_samples in self._evaluation_samples_by_model_id.items()
        )
