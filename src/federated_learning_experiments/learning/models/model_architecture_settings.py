"""モデル構造の正式名と要求rankの固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class ModelArchitectureSettings:
    """要求rankを丸めず保持し、構築時に宣言された型と値域を検証する。"""

    model_architecture_name: str = field(
        metadata={"allowed_parameter_values": ("shared_backbone_residual_adapter",)},
    )
    residual_adapter_requested_rank: int = field(
        metadata={
            "parameter_unit": "rank",
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
