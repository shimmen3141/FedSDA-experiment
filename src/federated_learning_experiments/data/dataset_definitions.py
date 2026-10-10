"""datasetごとの、入力の特徴数・概念数・クラス数。"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class DatasetDefinition:
    """datasetを決める、手法に依らない値。隠れ層の幅や学習率は、手法の設定が持つ。"""

    dataset_name: str
    input_feature_count: int
    concept_count: int
    class_count: int


_DATASET_DEFINITIONS_BY_NAME = {
    dataset_definition.dataset_name: dataset_definition
    for dataset_definition in (
        DatasetDefinition(
            dataset_name="sine2", input_feature_count=2, concept_count=2, class_count=2
        ),
        DatasetDefinition(
            dataset_name="sea2", input_feature_count=3, concept_count=2, class_count=2
        ),
        DatasetDefinition(
            dataset_name="sea4", input_feature_count=3, concept_count=4, class_count=2
        ),
        DatasetDefinition(
            dataset_name="circle2", input_feature_count=2, concept_count=2, class_count=2
        ),
        DatasetDefinition(
            dataset_name="mnist2", input_feature_count=784, concept_count=2, class_count=10
        ),
        DatasetDefinition(
            dataset_name="mnist4", input_feature_count=784, concept_count=4, class_count=10
        ),
    )
}


# 定義のあるdataset名（定義した順）。
DEFINED_DATASET_NAMES: tuple[str, ...] = tuple(_DATASET_DEFINITIONS_BY_NAME)


def get_dataset_definition(*, dataset_name: str) -> DatasetDefinition:
    """dataset名から、定義を返す。定義のない名前は拒否する。"""
    if type(dataset_name) is not str:
        raise TypeError("dataset_name must be builtin str")
    if dataset_name not in _DATASET_DEFINITIONS_BY_NAME:
        raise ValueError(
            f"dataset_name must be one of {sorted(_DATASET_DEFINITIONS_BY_NAME)}: {dataset_name!r}"
        )
    return _DATASET_DEFINITIONS_BY_NAME[dataset_name]
