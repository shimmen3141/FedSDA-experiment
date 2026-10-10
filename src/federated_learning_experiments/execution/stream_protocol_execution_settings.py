"""観測標本の供給と区間進行だけを扱う単一runの固定条件。"""

from dataclasses import dataclass

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.data.dataset_definitions import DEFINED_DATASET_NAMES


@dataclass(frozen=True, kw_only=True)
class StreamProtocolExecutionSettings:
    """部分設定ではなく、今回の実行窓口に必要な条件をすべて保持する。"""

    experiment_run_conditions: ExperimentRunConditions
    concept_schedule_settings: RandomConceptScheduleSettings
    execution_strategy: str

    def __post_init__(self) -> None:
        validate_stream_protocol_execution_settings(execution_settings=self)


def validate_stream_protocol_execution_settings(*, execution_settings: object) -> None:
    """構築時と実行直前で同じ検証を行い、元の値を報告する。"""
    if type(execution_settings) is not StreamProtocolExecutionSettings:
        raise RunSettingsValidationError(
            configuration_parameter_name="execution_settings",
            specified_parameter_value=execution_settings,
            validation_failure_reason="StreamProtocolExecutionSettingsを指定してください。",
        )
    if type(execution_settings.experiment_run_conditions) is not ExperimentRunConditions:
        raise RunSettingsValidationError(
            configuration_parameter_name="experiment_run_conditions",
            specified_parameter_value=execution_settings.experiment_run_conditions,
            validation_failure_reason="ExperimentRunConditionsを指定してください。",
        )
    if type(execution_settings.concept_schedule_settings) is not RandomConceptScheduleSettings:
        raise RunSettingsValidationError(
            configuration_parameter_name="concept_schedule_settings",
            specified_parameter_value=execution_settings.concept_schedule_settings,
            validation_failure_reason="RandomConceptScheduleSettingsを指定してください。",
        )
    validate_settings_field_values(execution_settings.experiment_run_conditions)
    validate_settings_field_values(execution_settings.concept_schedule_settings)
    if (
        type(execution_settings.execution_strategy) is not str
        or execution_settings.execution_strategy
        != "sample_index_then_client_order_with_interval_synchronization"
    ):
        raise RunSettingsValidationError(
            configuration_parameter_name="execution_strategy",
            specified_parameter_value=execution_settings.execution_strategy,
            validation_failure_reason="sample_index_then_client_order_with_interval_synchronizationを指定してください。",
        )
    if execution_settings.experiment_run_conditions.dataset_name not in DEFINED_DATASET_NAMES:
        raise RunSettingsValidationError(
            configuration_parameter_name="dataset_name",
            specified_parameter_value=execution_settings.experiment_run_conditions.dataset_name,
            validation_failure_reason=(
                "この実行窓口では"
                + "・".join(DEFINED_DATASET_NAMES)
                + "のいずれかを指定してください。"
            ),
        )
    if execution_settings.experiment_run_conditions.random_seed > 2**32 - 1:
        raise RunSettingsValidationError(
            configuration_parameter_name="random_seed",
            specified_parameter_value=execution_settings.experiment_run_conditions.random_seed,
            validation_failure_reason="0以上2**32-1以下の整数を指定してください。",
        )
