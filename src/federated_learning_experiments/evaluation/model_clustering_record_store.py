"""モデルのクラスタリングの診断の記録（モデル対ごとの観測と、モデルごとの観測）を、足された順に持つ。"""

from dataclasses import dataclass, replace
from math import isnan


def _validate_nonnegative_integer(*, value: int, value_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{value_name} must be builtin int")
    if value < 0:
        raise ValueError(f"{value_name} must be nonnegative")


def _validate_number(*, value: float, value_name: str) -> None:
    """NaNでないbuiltin float（無限大は許す）。"""
    if type(value) is not float:
        raise TypeError(f"{value_name} must be builtin float")
    if isnan(value):
        raise ValueError(f"{value_name} must not be NaN")


def _validate_flag(*, value: bool, value_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{value_name} must be builtin bool")


@dataclass(frozen=True, kw_only=True)
class ModelPairClusteringObservation:
    """1回のクラスタリングの、モデル対1つの観測。損失の距離を持つ対だけが記録される。

    `decision_score`は、判定に使った値（判定の値を持たない対では、距離）。
    `true_concepts_match`と`concept_specific_parameter_distance`は診断値で、判定には使われていない。
    """

    round_index: int
    lower_model_id: int
    higher_model_id: int
    loss_increase_distance: float
    decision_score: float
    assigned_to_same_cluster: bool
    # 両方のモデルに、割当の最多の真の概念が一意にあるときだけ、その一致。なければNone。
    true_concepts_match: bool | None
    concept_specific_parameter_distance: float

    def __post_init__(self) -> None:
        for identifier_name in ("round_index", "lower_model_id", "higher_model_id"):
            _validate_nonnegative_integer(
                value=getattr(self, identifier_name), value_name=identifier_name
            )
        if not self.lower_model_id < self.higher_model_id:
            raise ValueError("lower_model_id must be less than higher_model_id")
        for number_name in (
            "loss_increase_distance",
            "decision_score",
            "concept_specific_parameter_distance",
        ):
            _validate_number(value=getattr(self, number_name), value_name=number_name)
        _validate_flag(value=self.assigned_to_same_cluster, value_name="assigned_to_same_cluster")
        if self.true_concepts_match is not None:
            _validate_flag(value=self.true_concepts_match, value_name="true_concepts_match")


@dataclass(frozen=True, kw_only=True)
class ModelClusteringObservation:
    """1回のクラスタリングの、モデル1つの観測。距離は、損失の距離（互いの平均損失の増加）。"""

    round_index: int
    model_id: int
    # 損失の距離が最も小さい相手（同じ距離なら小さいID）。距離を持つ対がなければNone。
    nearest_model_id: int | None
    nearest_loss_increase_distance: float | None
    # 自分のクラスタの最小のID。
    representative_model_id: int
    cluster_model_count: int
    # 自分のクラスタの中の対の、損失の距離の最大。距離を持つ対がなければNone。
    maximum_within_cluster_distance: float | None
    evaluated_within_cluster_pair_count: int
    possible_within_cluster_pair_count: int
    merged_with_other_models: bool
    absorbed_into_representative: bool

    def __post_init__(self) -> None:
        for identifier_name in (
            "round_index",
            "model_id",
            "representative_model_id",
            "evaluated_within_cluster_pair_count",
            "possible_within_cluster_pair_count",
        ):
            _validate_nonnegative_integer(
                value=getattr(self, identifier_name), value_name=identifier_name
            )
        _validate_nonnegative_integer(
            value=self.cluster_model_count, value_name="cluster_model_count"
        )
        if self.cluster_model_count < 1:
            raise ValueError("cluster_model_count must be positive")
        if (self.nearest_model_id is None) != (self.nearest_loss_increase_distance is None):
            raise ValueError(
                "nearest_model_id and nearest_loss_increase_distance must be given together"
            )
        if self.nearest_model_id is not None:
            _validate_nonnegative_integer(
                value=self.nearest_model_id, value_name="nearest_model_id"
            )
            if self.nearest_model_id == self.model_id:
                raise ValueError("nearest_model_id must differ from model_id")
        if self.nearest_loss_increase_distance is not None:
            _validate_number(
                value=self.nearest_loss_increase_distance,
                value_name="nearest_loss_increase_distance",
            )
        if self.maximum_within_cluster_distance is not None:
            _validate_number(
                value=self.maximum_within_cluster_distance,
                value_name="maximum_within_cluster_distance",
            )
        _validate_flag(value=self.merged_with_other_models, value_name="merged_with_other_models")
        _validate_flag(
            value=self.absorbed_into_representative, value_name="absorbed_into_representative"
        )
        if self.representative_model_id > self.model_id:
            raise ValueError("representative_model_id must not exceed model_id")
        if self.merged_with_other_models != (self.cluster_model_count > 1):
            raise ValueError("merged_with_other_models must agree with cluster_model_count")
        if self.absorbed_into_representative != (self.model_id != self.representative_model_id):
            raise ValueError("absorbed_into_representative must agree with representative_model_id")
        if self.possible_within_cluster_pair_count != (
            self.cluster_model_count * (self.cluster_model_count - 1) // 2
        ):
            raise ValueError(
                "possible_within_cluster_pair_count must agree with cluster_model_count"
            )
        if self.evaluated_within_cluster_pair_count > self.possible_within_cluster_pair_count:
            raise ValueError(
                "evaluated_within_cluster_pair_count must not exceed the possible pair count"
            )
        if (self.maximum_within_cluster_distance is None) != (
            self.evaluated_within_cluster_pair_count == 0
        ):
            raise ValueError(
                "maximum_within_cluster_distance must be given exactly when a pair was evaluated"
            )


class ModelClusteringRecordStore:
    """クラスタリングの観測を、足された順に所有する。集計や判定はしない。"""

    def __init__(self) -> None:
        self._pair_clustering_observations: list[ModelPairClusteringObservation] = []
        self._model_clustering_observations: list[ModelClusteringObservation] = []

    def append_clustering_observations(
        self,
        *,
        pair_observations: tuple[ModelPairClusteringObservation, ...],
        model_observations: tuple[ModelClusteringObservation, ...],
    ) -> None:
        """1回のクラスタリングの観測を、全部の検査の後で足す。不正なら、何も変えない。"""
        validated_observations = []
        for observations, observation_type, observations_name in (
            (pair_observations, ModelPairClusteringObservation, "pair_observations"),
            (model_observations, ModelClusteringObservation, "model_observations"),
        ):
            if type(observations) is not tuple:
                raise TypeError(f"{observations_name} must be builtin tuple")
            for observation in observations:
                if type(observation) is not observation_type:
                    raise TypeError(
                        f"{observations_name} must hold exact {observation_type.__name__}"
                    )
            # 入力が手動で書き換えられていても全fieldを再検査し、保存用の写しを更新前に確定する。
            validated_observations.append([replace(observation) for observation in observations])
        self._pair_clustering_observations.extend(validated_observations[0])
        self._model_clustering_observations.extend(validated_observations[1])

    def snapshot_pair_clustering_observations(self) -> tuple[ModelPairClusteringObservation, ...]:
        return tuple(self._pair_clustering_observations)

    def snapshot_model_clustering_observations(self) -> tuple[ModelClusteringObservation, ...]:
        return tuple(self._model_clustering_observations)
