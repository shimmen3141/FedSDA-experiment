"""候補モデルの学習・採否方針と将来検証件数に関する固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class CandidateModelTrainingAndAcceptanceSettings:
    """候補方針と標本件数を保持し、構築時に正式値と値域を検証する。"""

    candidate_model_acceptance_policy: str = field(
        metadata={
            "allowed_parameter_values": (
                "current_model_first_reuse_then_two_segment_candidate_validation",
            ),
        },
    )
    candidate_post_alarm_validation_sample_count: int = field(
        metadata={
            "parameter_unit": "sample/client",
            "minimum_allowed_value": 2,
            "minimum_value_is_inclusive": True,
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
