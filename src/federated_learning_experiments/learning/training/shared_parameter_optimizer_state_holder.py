"""client 1つの、現在の共有部のoptimizerの状態への参照を保持する。"""

from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)


def _validate_shared_parameter_optimizer_state(
    *, shared_parameter_optimizer_state: ParameterOptimizerState
) -> None:
    if type(shared_parameter_optimizer_state) is not ParameterOptimizerState:
        raise TypeError("shared_parameter_optimizer_state must be exact ParameterOptimizerState")


class SharedParameterOptimizerStateHolder:
    """共有部のoptimizerの状態を1つ持つ。配布で保有モデルを作り直すと、つなぎ先の共有部のものへ置き換わる。

    optimizerの状態そのものは作らず、学習もしない。
    """

    def __init__(self, *, shared_parameter_optimizer_state: ParameterOptimizerState) -> None:
        _validate_shared_parameter_optimizer_state(
            shared_parameter_optimizer_state=shared_parameter_optimizer_state
        )
        self._shared_parameter_optimizer_state = shared_parameter_optimizer_state

    @property
    def held_shared_parameter_optimizer_state(self) -> ParameterOptimizerState:
        """現在の共有部のoptimizerの状態。"""
        return self._shared_parameter_optimizer_state

    def replace_shared_parameter_optimizer_state(
        self, *, shared_parameter_optimizer_state: ParameterOptimizerState
    ) -> None:
        _validate_shared_parameter_optimizer_state(
            shared_parameter_optimizer_state=shared_parameter_optimizer_state
        )
        self._shared_parameter_optimizer_state = shared_parameter_optimizer_state
