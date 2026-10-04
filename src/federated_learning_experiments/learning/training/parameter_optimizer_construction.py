"""明示Parameter列を検査して標準optimizerを新規生成する。"""

from torch import float32, strided
from torch.nn import Parameter
from torch.optim import Optimizer, Adam, SGD

from .parameter_optimizer_settings import AdamParameterOptimizerSettings, SgdParameterOptimizerSettings


def _validate_optimizer_settings(
    *, optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
) -> AdamParameterOptimizerSettings | SgdParameterOptimizerSettings:
    try:
        if type(optimizer_settings) is AdamParameterOptimizerSettings:
            return AdamParameterOptimizerSettings(
                learning_rate=optimizer_settings.learning_rate,
                weight_decay=optimizer_settings.weight_decay,
                adam_variant=optimizer_settings.adam_variant)
        if type(optimizer_settings) is SgdParameterOptimizerSettings:
            return SgdParameterOptimizerSettings(learning_rate=optimizer_settings.learning_rate)
        raise ValueError("optimizer_settingsはexact AdamParameterOptimizerSettings/SgdParameterOptimizerSettingsが必要です。")
    except (TypeError, ValueError) as validation_error:
        raise ValueError(f"optimizer_settings: {validation_error}") from validation_error


def _validate_optimizer_parameters(*, parameters: tuple[Parameter, ...]) -> None:
    if type(parameters) is not tuple:
        raise ValueError("parametersはexact tupleが必要です。")
    if not parameters:
        raise ValueError("parametersは空でないtupleが必要です。")
    parameter_ids = set()
    for parameter_index, parameter in enumerate(parameters):
        if type(parameter) is not Parameter:
            raise ValueError(f"parameters[{parameter_index}]はexact torch.nn.Parameterが必要です。")
        if (parameter.device.type != "cpu" or parameter.dtype != float32
                or parameter.layout != strided or parameter.is_nested):
            raise ValueError(f"parameters[{parameter_index}]はCPU float32 strided/notnestedが必要です。")
        if id(parameter) in parameter_ids:
            raise ValueError(f"parameters[{parameter_index}]は同一Parameterが重複しています。")
        parameter_ids.add(id(parameter))


def create_parameter_optimizer(
    *, parameters: tuple[Parameter, ...],
    optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
) -> Optimizer:
    """入力参照・順序を保ち、初期stateが空のAdam/SGDだけを返す。"""
    validated_optimizer_settings = _validate_optimizer_settings(optimizer_settings=optimizer_settings)
    _validate_optimizer_parameters(parameters=parameters)
    if type(validated_optimizer_settings) is AdamParameterOptimizerSettings:
        return Adam(parameters, lr=validated_optimizer_settings.learning_rate,
            weight_decay=validated_optimizer_settings.weight_decay,
            amsgrad=validated_optimizer_settings.adam_variant == "amsgrad")
    return SGD(parameters, lr=validated_optimizer_settings.learning_rate)
