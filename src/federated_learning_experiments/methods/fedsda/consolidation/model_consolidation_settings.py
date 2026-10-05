"""モデルクラスタリングと統合の固定方針。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class ModelConsolidationSettings:
    """開始条件・対比較・linkage・統合方針を保持し、正式値を検証する。"""

    model_clustering_trigger_policy: str = field(
        metadata={
            "allowed_parameter_values": ("on_new_model_registration",),
        },
    )
    model_pair_comparison_strategy: str = field(
        metadata={
            "allowed_parameter_values": ("classwise_unique_correctness_lower_confidence_bound",),
        },
    )
    model_clustering_linkage: str = field(
        metadata={
            "allowed_parameter_values": ("average_linkage",),
        },
    )
    model_consolidation_policy: str = field(
        metadata={
            "allowed_parameter_values": ("weighted_parameter_average_and_merge_ids",),
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
