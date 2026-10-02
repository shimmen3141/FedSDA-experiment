"""帰属確定を保留するFIFOの容量に関する固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class TrainingDataAssignmentSettings:
    """バッファ内容や割当先を持たず、構築時に設定容量を検証する。"""

    pending_assignment_buffer_capacity_samples: int = field(
        metadata={
            "parameter_unit": "sample/client",
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
