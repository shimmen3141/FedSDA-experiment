"""ランダム概念系列の方式と変更試行条件を保持する。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class RandomConceptScheduleSettings:
    """実行状態を持たず、必須の固定条件を構築時に検証する。"""

    concept_schedule_strategy: str = field(
        metadata={"allowed_parameter_values": ("random_changes_after_minimum_index_gap",)},
    )
    minimum_sample_index_gap_before_change_trial: int = field(
        metadata={
            "parameter_unit": "sample/client",
            "minimum_allowed_value": 0,
            "minimum_value_is_inclusive": True,
        },
    )
    per_eligible_sample_concept_change_probability: float = field(
        metadata={
            "parameter_unit": "dimensionless",
            "minimum_allowed_value": 0,
            "minimum_value_is_inclusive": True,
            "maximum_allowed_value": 1,
            "maximum_value_is_inclusive": True,
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
