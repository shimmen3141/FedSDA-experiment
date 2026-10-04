"""候補の初期parameter snapshotを選ぶ固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import validate_settings_field_values


@dataclass(frozen=True, kw_only=True)
class CandidateParameterInitializationSettings:
    """初期化元だけを宣言し、学習と採否の状態は持たない。"""

    candidate_parameter_initialization_source: str = field(metadata={
        "allowed_parameter_values": (
            "assigned_training_model",
            "lowest_evaluated_mean_loss_model",
            "equal_mean_of_available_models",
        ),
    })

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
