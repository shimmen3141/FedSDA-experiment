"""通常の観測値と評価用の真の概念を分けた不変記録。"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ObservedSample:
    """真の概念を含まない、2特徴と二値クラスの観測標本。"""

    feature_values: tuple[float, float]
    class_label: int

    def __post_init__(self) -> None:
        if type(self.feature_values) is not tuple:
            raise TypeError("feature_valuesにはfloatのtupleを指定してください。")
        if len(self.feature_values) != 2:
            raise ValueError("feature_valuesには長さ2のtupleを指定してください。")
        for feature_value in self.feature_values:
            if type(feature_value) is not float:
                raise TypeError("feature_valuesの各要素にはfloatを指定してください。")
        if type(self.class_label) is not int:
            raise TypeError("class_labelには整数の0/1を指定してください。boolは受理しません。")
        if self.class_label not in (0, 1):
            raise ValueError("class_labelには0または1を指定してください。")


@dataclass(frozen=True, kw_only=True)
class ClientObservedStream:
    """clientの観測標本をtupleの位置順に保持する。"""

    client_id: int
    observed_samples: tuple[ObservedSample, ...]

    def __post_init__(self) -> None:
        if type(self.client_id) is not int:
            raise TypeError("client_idには非負の整数を指定してください。boolは受理しません。")
        if self.client_id < 0:
            raise ValueError("client_idには0以上の整数を指定してください。")
        if type(self.observed_samples) is not tuple:
            raise TypeError("observed_samplesにはObservedSampleのtupleを指定してください。")
        for observed_sample in self.observed_samples:
            if type(observed_sample) is not ObservedSample:
                raise TypeError("observed_samplesの各要素にはObservedSampleを指定してください。")


@dataclass(frozen=True, kw_only=True)
class ClientConceptTrace:
    """学習入力から分離した、clientの評価用真値を位置順に保持する。"""

    client_id: int
    concept_ids_by_sample_index: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.client_id) is not int:
            raise TypeError("client_idには非負の整数を指定してください。boolは受理しません。")
        if self.client_id < 0:
            raise ValueError("client_idには0以上の整数を指定してください。")
        if type(self.concept_ids_by_sample_index) is not tuple:
            raise TypeError("concept_ids_by_sample_indexには整数のtupleを指定してください。")
        for concept_id in self.concept_ids_by_sample_index:
            if type(concept_id) is not int:
                raise TypeError(
                    "concept_ids_by_sample_indexの各要素には整数の0/1を指定してください。"
                    "boolは受理しません。"
                )
            if concept_id not in (0, 1):
                raise ValueError(
                    "concept_ids_by_sample_indexの各要素には0または1を指定してください。"
                )
