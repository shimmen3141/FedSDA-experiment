"""学習要求の実行間隔と一要求あたりの共同更新試行予算。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class LocalTrainingScheduleSettings:
    """実更新方式や学習実体を含めない不変設定。"""

    training_requests_per_update_interval: int = field(
        metadata={
            "parameter_unit": "training_requests",
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
        }
    )
    joint_update_iterations_per_training_request: int = field(
        metadata={
            "parameter_unit": "joint_update_iterations_per_training_request",
            "minimum_allowed_value": 0,
            "minimum_value_is_inclusive": True,
        }
    )

    def __post_init__(self) -> None:
        for parameter_name, parameter_value in (
            ("training_requests_per_update_interval", self.training_requests_per_update_interval),
            (
                "joint_update_iterations_per_training_request",
                self.joint_update_iterations_per_training_request,
            ),
        ):
            if type(parameter_value) is not int:
                raise ValueError(f"{parameter_name}はbool以外のbuiltin intが必要です。")
        validate_settings_field_values(self)
