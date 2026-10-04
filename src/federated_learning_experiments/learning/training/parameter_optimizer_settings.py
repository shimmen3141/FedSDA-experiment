"""標準Adam/SGDへ渡す方式別の固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import validate_settings_field_values


@dataclass(frozen=True, kw_only=True)
class AdamParameterOptimizerSettings:
    """Adam専用条件を必須fieldとして宣言する。"""

    learning_rate: float = field(metadata={
        "parameter_unit": "更新係数", "minimum_allowed_value": 0,
        "minimum_value_is_inclusive": True,
    })
    weight_decay: float = field(metadata={
        "parameter_unit": "L2係数", "minimum_allowed_value": 0,
        "minimum_value_is_inclusive": True,
    })
    adam_variant: str = field(metadata={"allowed_parameter_values": ("standard", "amsgrad")})

    def __post_init__(self) -> None:
        for configuration_parameter_name, specified_parameter_value in (
            ("learning_rate", self.learning_rate), ("weight_decay", self.weight_decay),
        ):
            if type(specified_parameter_value) not in (int, float):
                raise ValueError(f"{configuration_parameter_name}はbool以外のbuiltin int/floatが必要です。")
        validate_settings_field_values(self)


@dataclass(frozen=True, kw_only=True)
class SgdParameterOptimizerSettings:
    """momentum/weight decayなしのSGD学習率だけを宣言する。"""

    learning_rate: float = field(metadata={
        "parameter_unit": "更新係数", "minimum_allowed_value": 0,
        "minimum_value_is_inclusive": True,
    })

    def __post_init__(self) -> None:
        if type(self.learning_rate) not in (int, float):
            raise ValueError("learning_rateはbool以外のbuiltin int/floatが必要です。")
        validate_settings_field_values(self)
