"""FedDrift clientの組立と進行に必要な条件。"""

from dataclasses import dataclass, field
from math import isfinite

from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.feddrift.adaptive_fedavg_learning_rate import (
    AdaptiveFedavgLearningRateSettings,
)


@dataclass(frozen=True, kw_only=True)
class FeddriftRunClientSettings:
    """一clientの検出・学習・評価保持条件。"""

    loss_increase_threshold: float
    detection_batch_sample_count: int
    local_training_batch_sample_count: int
    updates_per_sample: int
    maximum_stored_evaluation_sample_count_per_model: int
    added_evaluation_batch_sample_count: int
    new_model_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings
    model_comparison_criterion: str
    new_model_parameter_source: str
    detection_reference_after_drift: str
    model_assignment_policy: str = field(
        default="detection_batch_assignment", metadata={"omit_default_from_saved_settings": True}
    )
    training_sample_window: str = field(
        default="all_timesteps", metadata={"omit_default_from_saved_settings": True}
    )
    cluster_model_initialization: str = field(
        default="all_pretrained", metadata={"omit_default_from_saved_settings": True}
    )

    adaptive_learning_rate_settings: AdaptiveFedavgLearningRateSettings | None = field(
        default=None, metadata={"omit_default_from_saved_settings": True}
    )

    def __post_init__(self) -> None:
        if self.adaptive_learning_rate_settings is not None:
            if type(self.adaptive_learning_rate_settings) is not AdaptiveFedavgLearningRateSettings:
                raise TypeError("exact AdaptiveFedavgLearningRateSettingsが必要です。")
            self.adaptive_learning_rate_settings.__post_init__()
            if self.model_assignment_policy != "single_global_model_assignment":
                raise ValueError("学習率の調整は単一モデルの構成専用です。")
        for name, choices in (
            (
                "model_assignment_policy",
                (
                    "detection_batch_assignment",
                    "true_concept_assignment",
                    "minimum_loss_cluster_assignment",
                    "single_global_model_assignment",
                ),
            ),
            ("training_sample_window", ("all_timesteps", "current_timestep")),
            (
                "cluster_model_initialization",
                ("all_pretrained", "all_random", "pretrained_cluster0"),
            ),
            ("model_comparison_criterion", ("bounded_loss", "classification_error_rate")),
            (
                "new_model_parameter_source",
                ("current_training_model", "minimum_criterion_held_model"),
            ),
            (
                "detection_reference_after_drift",
                ("new_model_batch_value", "minimum_held_model_batch_value"),
            ),
        ):
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"{name}はbuiltin strが必要です。")
            if value not in choices:
                raise ValueError(f"{name}が未対応です。")
        if self.model_assignment_policy == "minimum_loss_cluster_assignment":
            if (
                self.training_sample_window != "current_timestep"
                or self.cluster_model_initialization == "all_pretrained"
            ):
                raise ValueError("IFCA+Windowには1時刻の窓とクラスタごとの初期化が必要です。")
        elif self.model_assignment_policy == "single_global_model_assignment":
            if (
                self.training_sample_window != "current_timestep"
                or self.cluster_model_initialization != "all_pretrained"
            ):
                raise ValueError("単一モデルには1時刻の窓と事前学習の初期化が必要です。")
        elif (
            self.training_sample_window != "all_timesteps"
            or self.cluster_model_initialization != "all_pretrained"
        ):
            raise ValueError("窓とクラスタごとの初期化はIFCA+Window専用です。")
        if type(self.loss_increase_threshold) is not float or (
            not isfinite(self.loss_increase_threshold) or self.loss_increase_threshold < 0
        ):
            raise ValueError("損失増分閾値は非負有限のbuiltin floatが必要です。")
        for name, minimum in (
            ("detection_batch_sample_count", 1),
            ("local_training_batch_sample_count", 1),
            ("updates_per_sample", 0),
            ("maximum_stored_evaluation_sample_count_per_model", 1),
            ("added_evaluation_batch_sample_count", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name}は{minimum}以上のbuiltin intが必要です。")
        if type(self.new_model_optimizer_settings) not in (
            AdamParameterOptimizerSettings,
            SgdParameterOptimizerSettings,
        ):
            raise TypeError("新規optimizerにはexact Adam/SGD設定が必要です。")
        self.new_model_optimizer_settings.__post_init__()
