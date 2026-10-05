"""初回の機能設定と機能間条件を、入力を変更せずに検証する。"""

from collections.abc import Mapping
from typing import cast

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import (
    RunSettingsValidationError,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_consolidation_settings import (
    ModelConsolidationSettings,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    LossChangeDetectionSettings,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import (
    PredictionCombinationSettings,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    TrainingDataAssignmentSettings,
)

initial_component_settings_types = (
    ("method_name", str),
    ("experiment_run_conditions", ExperimentRunConditions),
    ("model_architecture_settings", ModelArchitectureSettings),
    ("loss_change_detection_settings", LossChangeDetectionSettings),
    ("prediction_combination_settings", PredictionCombinationSettings),
    ("local_training_settings", LocalTrainingSettings),
    ("training_data_assignment_settings", TrainingDataAssignmentSettings),
    (
        "candidate_model_training_and_acceptance_settings",
        CandidateModelTrainingAndAcceptanceSettings,
    ),
    ("model_consolidation_settings", ModelConsolidationSettings),
)


def validate_experiment_run_settings(
    unvalidated_run_settings: Mapping[str, object],
) -> None:
    """正式キー・機能型・最終FedSDAの組合せ条件を宣言順に検査する。"""
    # Pythonの型注釈は実行時に強制されないため、契約外入力も明示拒否する。
    if not isinstance(unvalidated_run_settings, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise RunSettingsValidationError(
            configuration_parameter_name="unvalidated_run_settings",
            specified_parameter_value=unvalidated_run_settings,
            validation_failure_reason="正式な設定キーと機能設定型のmappingを指定してください。",
        )

    # 未定義キーには宣言順がないため、表記の順を用いて入力順に依存させない。
    unknown_settings_names = sorted(
        (
            component_settings_name
            for component_settings_name in unvalidated_run_settings
            if component_settings_name
            not in (
                component_settings_name
                for component_settings_name, _component_settings_type in initial_component_settings_types
            )
        ),
        key=repr,
    )
    if unknown_settings_names:
        raise RunSettingsValidationError(
            configuration_parameter_name=unknown_settings_names[0],
            specified_parameter_value=unvalidated_run_settings[unknown_settings_names[0]],
            validation_failure_reason=(
                "未定義の設定項目です。正式な設定キーだけを指定してください。"
                f"許容する設定キー: {tuple(component_settings_name for component_settings_name, _component_settings_type in initial_component_settings_types)!r}。"
            ),
        )

    for component_settings_name, component_settings_type in initial_component_settings_types:
        if component_settings_name not in unvalidated_run_settings:
            raise RunSettingsValidationError(
                configuration_parameter_name=component_settings_name,
                specified_parameter_value=None,
                validation_failure_reason=(
                    "必須の設定項目がありません。"
                    f"{component_settings_type.__name__}型の値を明示してください。"
                ),
            )
        component_settings = unvalidated_run_settings[component_settings_name]
        if not isinstance(component_settings, component_settings_type):
            raise RunSettingsValidationError(
                configuration_parameter_name=component_settings_name,
                specified_parameter_value=component_settings,
                validation_failure_reason=(
                    f"{component_settings_type.__name__}型の値を指定してください。"
                ),
            )
        if component_settings_name == "method_name" and component_settings != "fedsda":
            raise RunSettingsValidationError(
                configuration_parameter_name=component_settings_name,
                specified_parameter_value=component_settings,
                validation_failure_reason="初回の対応手法は正式名'fedsda'だけです。",
            )

    # 上の全項目検査で型を確定済み。castは値や受理条件を変更しない。
    component_settings = cast(
        PredictionCombinationSettings, unvalidated_run_settings["prediction_combination_settings"]
    )
    if component_settings.fixed_share_weight_redistribution_time_scale_samples != (
        cast(
            TrainingDataAssignmentSettings,
            unvalidated_run_settings["training_data_assignment_settings"],
        ).pending_assignment_buffer_capacity_samples
    ):
        raise RunSettingsValidationError(
            configuration_parameter_name="fixed_share_weight_redistribution_time_scale_samples",
            specified_parameter_value=(
                component_settings.fixed_share_weight_redistribution_time_scale_samples
            ),
            validation_failure_reason=(
                "最終FedSDA構成ではFixed-Shareの時間尺度を帰属保留FIFO容量と同値にしてください。"
                "pending_assignment_buffer_capacity_samples="
                f"{cast(TrainingDataAssignmentSettings, unvalidated_run_settings['training_data_assignment_settings']).pending_assignment_buffer_capacity_samples!r}。"
            ),
        )
