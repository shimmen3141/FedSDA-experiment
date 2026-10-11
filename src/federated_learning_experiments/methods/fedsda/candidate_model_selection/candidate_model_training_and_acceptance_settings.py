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
                # 警報時に、保有モデルの再利用を先に試し、合うモデルが無ければ、候補を学習して、
                # 警報の後の標本で、2つの区間に分けて検証してから採否を決める（提案構成）。
                "current_model_first_reuse_then_two_segment_candidate_validation",
                # 合うモデルが無ければ、学習した候補を、検証せずに、その場で採用する（候補検証を外した構成）。
                # このとき、検証の標本件数は、使わない。
                "current_model_first_reuse_then_candidate_adoption_without_validation",
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
