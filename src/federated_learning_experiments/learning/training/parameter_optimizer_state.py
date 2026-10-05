"""固定parameter列の現在optimizerと明示リセットを管理する。"""

from __future__ import annotations

from torch.nn import Parameter
from torch.optim import Optimizer

from .parameter_optimizer_construction import create_parameter_optimizer
from .parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)


class ParameterOptimizerState:
    """parameterと固定設定を借用し、現在のoptimizerだけを所有する。"""

    def __init__(
        self,
        *,
        parameters: tuple[Parameter, ...],
        optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    ) -> None:
        self._parameter_optimizer = create_parameter_optimizer(
            parameters=parameters, optimizer_settings=optimizer_settings
        )
        self._parameters = parameters
        self._optimizer_settings = optimizer_settings

    @property
    def parameter_optimizer(self) -> Optimizer:
        """現在参照を借用で返す。外側の学習更新をそのまま保持する。"""
        return self._parameter_optimizer

    def reset_parameter_optimizer(self) -> None:
        """生成成功後だけ交換する。旧借用記録やNN値・gradは変更しない。"""
        self._parameter_optimizer = create_parameter_optimizer(
            parameters=self._parameters, optimizer_settings=self._optimizer_settings
        )
