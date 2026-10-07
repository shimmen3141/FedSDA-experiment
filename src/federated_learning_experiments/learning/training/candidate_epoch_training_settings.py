"""候補の区間学習方式とエポック・検証停止条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class CandidateEpochTrainingSettings:
    """学習率をoptimizer設定へ分離した必須の固定条件。"""

    candidate_training_strategy: str = field(
        metadata={
            "allowed_parameter_values": (
                "fixed_epoch_training",
                "validation_loss_early_stopping",
                "skip_training",
            )
        }
    )
    maximum_epoch_count: int = field(
        metadata={"minimum_allowed_value": 0, "minimum_value_is_inclusive": True}
    )
    maximum_batch_sample_count: int = field(
        metadata={"minimum_allowed_value": 1, "minimum_value_is_inclusive": True}
    )
    validation_sample_fraction: float = field(
        metadata={
            "minimum_allowed_value": 0,
            "minimum_value_is_inclusive": False,
            "maximum_allowed_value": 1,
            "maximum_value_is_inclusive": False,
        }
    )
    consecutive_non_improving_epoch_limit: int = field(
        metadata={"minimum_allowed_value": 1, "minimum_value_is_inclusive": True}
    )
    minimum_validation_loss_decrease: float = field(
        metadata={"minimum_allowed_value": 0, "minimum_value_is_inclusive": True}
    )

    def __post_init__(self) -> None:
        for configuration_parameter_name in (
            "maximum_epoch_count",
            "maximum_batch_sample_count",
            "consecutive_non_improving_epoch_limit",
        ):
            if type(getattr(self, configuration_parameter_name)) is not int:
                raise ValueError(f"{configuration_parameter_name}はbuiltin intが必要です。")
        for configuration_parameter_name in (
            "validation_sample_fraction",
            "minimum_validation_loss_decrease",
        ):
            if type(getattr(self, configuration_parameter_name)) not in (int, float):
                raise ValueError(f"{configuration_parameter_name}はbuiltin int/floatが必要です。")
        validate_settings_field_values(self)
