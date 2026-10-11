"""FedDriftのサーバ進行の設定。"""

from dataclasses import dataclass
from math import isfinite

from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)


@dataclass(frozen=True, kw_only=True)
class FeddriftServerSettings:
    """サーバが受け取る固定設定。"""

    maximum_same_cluster_loss_increase: float
    cluster_linkage: str
    minimum_cross_evaluation_sample_count: int
    maximum_cross_evaluation_client_count: int
    maximum_client_evaluation_sample_count: int
    new_model_isolation_timestep_count: int
    local_training_round_count: int
    initial_model_aggregation_weight: int
    rebuilt_model_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings

    def __post_init__(self) -> None:
        if type(self.maximum_same_cluster_loss_increase) is not float:
            raise TypeError("距離閾値はbuiltin floatが必要です。")
        if not isfinite(self.maximum_same_cluster_loss_increase):
            raise ValueError("距離閾値は有限値が必要です。")
        if type(self.cluster_linkage) is not str:
            raise TypeError("linkageはbuiltin strが必要です。")
        if self.cluster_linkage not in ("connected", "average", "complete"):
            raise ValueError("linkageが未対応です。")
        for name in (
            "minimum_cross_evaluation_sample_count",
            "maximum_cross_evaluation_client_count",
            "maximum_client_evaluation_sample_count",
            "new_model_isolation_timestep_count",
            "local_training_round_count",
            "initial_model_aggregation_weight",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name}はbuiltin intが必要です。")
            minimum = 0 if name == "initial_model_aggregation_weight" else 1
            if value < minimum:
                raise ValueError(f"{name}は{minimum}以上が必要です。")
        if type(self.rebuilt_model_optimizer_settings) not in (
            AdamParameterOptimizerSettings,
            SgdParameterOptimizerSettings,
        ):
            raise TypeError("再構築optimizer設定の型が不正です。")
        self.rebuilt_model_optimizer_settings.__post_init__()
