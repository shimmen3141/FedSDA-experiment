"""実験の構成の名前（手法と、その選択肢の組）から、完全なrun設定を作る。

構成は、実行条件（dataset、seed、規模、集約間隔）とは別のもの。構成を足すときは、ここへ足す。
"""

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
    build_final_configuration_fedsda_run_settings,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings

# 提案構成（docs/overview/proposed-method.mdの「最終提案構成」）。
PROPOSED_FEDSDA_CONFIGURATION_NAME = "fedsda"
# コマンドで指定できる構成の名前。
EXPERIMENT_CONFIGURATION_NAMES: tuple[str, ...] = (PROPOSED_FEDSDA_CONFIGURATION_NAME,)


def build_experiment_run_settings(
    *, configuration_name: str, experiment_run_conditions: ExperimentRunConditions
) -> FedsdaRunSettings:
    """構成の名前と実行条件から、1 runの完全なrun設定を作る。"""
    if type(configuration_name) is not str:
        raise TypeError("configuration_name must be builtin str")
    if type(experiment_run_conditions) is not ExperimentRunConditions:
        raise TypeError("experiment_run_conditions must be exact ExperimentRunConditions")
    if configuration_name == PROPOSED_FEDSDA_CONFIGURATION_NAME:
        return build_final_configuration_fedsda_run_settings(
            experiment_run_conditions=experiment_run_conditions
        )
    raise ValueError(
        f"unknown experiment configuration: {configuration_name!r} "
        f"(available: {', '.join(EXPERIMENT_CONFIGURATION_NAMES)})"
    )
