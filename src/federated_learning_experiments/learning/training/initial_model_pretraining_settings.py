"""runの最初に1回だけ行う、初期モデルの事前学習の条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class InitialModelPretrainingSettings:
    """学習の実体や標本を持たない不変設定。epoch数0は、学習せずに統計だけを求める。"""

    pretraining_sample_count: int = field(
        metadata={
            "parameter_unit": "sample",
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
        }
    )
    pretraining_epoch_count: int = field(
        metadata={
            "parameter_unit": "epoch",
            "minimum_allowed_value": 0,
            "minimum_value_is_inclusive": True,
        }
    )
    pretraining_batch_sample_count: int = field(
        metadata={
            "parameter_unit": "sample",
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
        }
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
        # 共通の検査はintの派生型を受け入れる。件数として使うので、builtin intだけを受け入れる。
        for configuration_parameter_name in (
            "pretraining_sample_count",
            "pretraining_epoch_count",
            "pretraining_batch_sample_count",
        ):
            if type(getattr(self, configuration_parameter_name)) is not int:
                raise RunSettingsValidationError(
                    configuration_parameter_name=configuration_parameter_name,
                    specified_parameter_value=getattr(self, configuration_parameter_name),
                    validation_failure_reason="bool・派生型以外のbuiltin intを指定してください。",
                )
