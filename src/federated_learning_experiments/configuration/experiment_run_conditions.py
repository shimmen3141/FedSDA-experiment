"""dataset・seed・実験規模・集約間隔の固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class ExperimentRunConditions:
    """実行状態を持たず、構築時に宣言された型と値域を検証する。"""

    dataset_name: str = field(
        metadata={"allowed_parameter_values": ("sine2", "sea2", "mnist2")},
    )
    random_seed: int = field(
        metadata={
            "parameter_unit": "dimensionless",
            "minimum_allowed_value": 0,
            "minimum_value_is_inclusive": True,
        },
    )
    client_count: int = field(
        metadata={
            "parameter_unit": "client",
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
        },
    )
    per_client_sample_count: int = field(
        metadata={
            "parameter_unit": "sample/client",
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
        },
    )
    server_aggregation_interval_per_client_samples: int = field(
        metadata={
            "parameter_unit": "sample/client",
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
