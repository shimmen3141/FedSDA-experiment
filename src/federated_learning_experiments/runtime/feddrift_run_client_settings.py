"""FedDrift clientの組立と進行に必要な条件。"""

from dataclasses import dataclass
from math import isfinite

from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
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

    def __post_init__(self) -> None:
        for name, choices in (
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
