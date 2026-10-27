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
    model_comparison_criterion: str
    training_model_eligibility: str
    consolidation_weight_timing: str

    def __post_init__(self) -> None:
        for name, choices in (
            ("model_comparison_criterion", ("bounded_loss", "classification_error_rate")),
            (
                "training_model_eligibility",
                ("held_models_with_training_samples", "models_assigned_in_current_timestep"),
            ),
            (
                "consolidation_weight_timing",
                (
                    "weights_before_current_timestep",
                    "weights_including_current_timestep_assignments",
                ),
            ),
        ):
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"{name}はbuiltin strが必要です。")
            if value not in choices:
                raise ValueError(f"{name}が未対応です。")
        if (
            self.model_comparison_criterion != "bounded_loss"
            or self.training_model_eligibility != "held_models_with_training_samples"
            or self.consolidation_weight_timing != "weights_before_current_timestep"
        ) and self.new_model_isolation_timestep_count != 1:
            raise ValueError("新しいFedDriftの扱いでは隔離期間は1だけです。")
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
