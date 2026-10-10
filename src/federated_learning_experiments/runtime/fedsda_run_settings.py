"""最終構成のFedSDAの1 runに要る、全部の設定を束ねて検証する。"""

from dataclasses import dataclass

from federated_learning_experiments.configuration.run_settings import (
    ValidatedExperimentRunSettingsSubset,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
    validate_stream_protocol_execution_settings,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_consolidation_settings import (
    ModelConsolidationSettings,
)
from federated_learning_experiments.runtime.fedsda_run_participant_factory import (
    FedsdaRunParticipantSettings,
)

# 手法の正式名（機能の組合せの検証へ渡す）。
_METHOD_NAME = "fedsda"

# fieldの名前から、受け入れるexact型への対応（宣言順）。
_REQUIRED_SETTINGS_TYPE_BY_FIELD_NAME: dict[str, type] = {
    "execution_settings": StreamProtocolExecutionSettings,
    "run_participant_settings": FedsdaRunParticipantSettings,
    "model_consolidation_settings": ModelConsolidationSettings,
    "run_metric_settings": RunMetricSettings,
}


@dataclass(frozen=True, kw_only=True)
class FedsdaRunSettings:
    """1 runの条件の全部。生成時に、各部分と、機能の組合せを確かめる。

    `execution_settings`は、実行の枠（実行条件、概念の変更、実行の方式）。`run_participant_settings`は、
    参加者の準備（clientの設定の束、事前学習、モデルの構造と隠れ層の幅、クラスタリングの判定の基準、
    クロス評価のclientの上限）。`model_consolidation_settings`は、統合の方式の宣言（サーバは、最終構成の
    方式だけを実装しているので、条件の保存と、組合せの検証に使う）。`run_metric_settings`は、指標の計算。
    """

    execution_settings: StreamProtocolExecutionSettings
    run_participant_settings: FedsdaRunParticipantSettings
    model_consolidation_settings: ModelConsolidationSettings
    run_metric_settings: RunMetricSettings

    def __post_init__(self) -> None:
        for settings_field_name, required_type in _REQUIRED_SETTINGS_TYPE_BY_FIELD_NAME.items():
            component_settings = getattr(self, settings_field_name)
            if type(component_settings) is not required_type:
                raise RunSettingsValidationError(
                    configuration_parameter_name=settings_field_name,
                    specified_parameter_value=component_settings,
                    validation_failure_reason=(
                        f"{required_type.__name__}型の値を指定してください。派生型は受理しません。"
                    ),
                )
        # frozenを回避して組み立てた値も拒否できるよう、各部分の検査をもう一度行う。
        validate_stream_protocol_execution_settings(execution_settings=self.execution_settings)
        self.run_participant_settings.__post_init__()
        self.model_consolidation_settings.__post_init__()
        self.run_metric_settings.__post_init__()
        # 機能の組合せ（最終構成の条件）は、既存の部分型の検証で確かめる。
        run_client_settings = self.run_participant_settings.run_client_settings
        ValidatedExperimentRunSettingsSubset(
            method_name=_METHOD_NAME,
            experiment_run_conditions=self.execution_settings.experiment_run_conditions,
            model_architecture_settings=self.run_participant_settings.model_architecture_settings,
            loss_change_detection_settings=run_client_settings.loss_change_detection_settings,
            prediction_combination_settings=run_client_settings.prediction_combination_settings,
            local_training_settings=run_client_settings.local_training_settings,
            training_data_assignment_settings=run_client_settings.training_data_assignment_settings,
            candidate_model_training_and_acceptance_settings=(
                run_client_settings.candidate_model_training_and_acceptance_settings
            ),
            model_consolidation_settings=self.model_consolidation_settings,
        )
