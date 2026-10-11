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
    """parameterと固定設定を借用し、現在のoptimizerだけを所有する。

    parameterが空（共有部を持たない構造の、共有部）のときは、optimizerを持たない。
    """

    def __init__(
        self,
        *,
        parameters: tuple[Parameter, ...],
        optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    ) -> None:
        self._parameter_optimizer = self._create_parameter_optimizer_if_any(
            parameters=parameters, optimizer_settings=optimizer_settings
        )
        self._parameters = parameters
        self._optimizer_settings = optimizer_settings

    @staticmethod
    def _create_parameter_optimizer_if_any(
        *,
        parameters: tuple[Parameter, ...],
        optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    ) -> Optimizer | None:
        # 空のtupleだけを、optimizerなしとして受け入れる（ほかの不正は、生成の検査が拒否する）。
        # 設定は、optimizerを作らなくても、確かめる。
        if type(parameters) is tuple and not parameters:
            if type(optimizer_settings) not in (
                AdamParameterOptimizerSettings,
                SgdParameterOptimizerSettings,
            ):
                raise ValueError("optimizer_settingsはexact Adam/SGD設定型が必要です。")
            try:
                optimizer_settings.__post_init__()
            except (TypeError, ValueError) as validation_error:
                raise ValueError(f"optimizer_settings: {validation_error}") from validation_error
            return None
        return create_parameter_optimizer(
            parameters=parameters, optimizer_settings=optimizer_settings
        )

    @property
    def parameter_optimizer(self) -> Optimizer:
        """現在参照を借用で返す。外側の学習更新をそのまま保持する。parameterが空なら、拒否する。"""
        if self._parameter_optimizer is None:
            raise LookupError("parameterが空のoptimizerの状態は、optimizerを持ちません。")
        return self._parameter_optimizer

    @property
    def parameter_optimizer_if_any(self) -> Optimizer | None:
        """現在のoptimizer。parameterが空なら、None。"""
        return self._parameter_optimizer

    def reset_parameter_optimizer(self) -> None:
        """生成成功後だけ交換する。旧借用記録やNN値・gradは変更しない。"""
        self._parameter_optimizer = self._create_parameter_optimizer_if_any(
            parameters=self._parameters, optimizer_settings=self._optimizer_settings
        )
