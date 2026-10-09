"""FedSDAのclient 1つの組立てが受け取る設定と値の束。

機能別の設定型に置き場所がまだない値を含む、組立てのための形である。保存表現・preset・値の
機能別の設定型への配置は、完全なrun設定を決める段階で定める。
"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.local_training_schedule_settings import (
    LocalTrainingScheduleSettings,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
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

_NONNEGATIVE = {"minimum_allowed_value": 0, "minimum_value_is_inclusive": True}
_POSITIVE_COUNT = {"minimum_allowed_value": 1, "minimum_value_is_inclusive": True}
_COUNT_FIELD_NAMES = (
    "new_model_upload_delay_round_count",
    "minimum_change_interval_sample_count",
    "local_training_batch_sample_count",
    "maximum_stored_evaluation_sample_count_per_model",
    "added_evaluation_batch_sample_count",
    "loss_monitor_maximum_retained_candidate_count",
)
_MEAN_LOSS_FIELD_NAMES = (
    "maximum_tolerated_mean_loss_increase",
    "minimum_candidate_mean_loss_improvement",
)


@dataclass(frozen=True, kw_only=True)
class FedsdaRunClientScalarSettings:
    """機能別の設定型に置き場所がまだない、数値と文字列の値。範囲は、値を使う部品の検査と同じ。

    `maximum_tolerated_mean_loss_increase`は、履歴の平均損失からの増加をどこまで「このモデルに合う」と
    みなすかの許容量で、候補検証の参照モデルの評価と、警報区間での保有モデルの再利用評価の両方に使う。
    """

    maximum_tolerated_mean_loss_increase: float = field(
        metadata={"parameter_unit": "mean bounded loss"} | _NONNEGATIVE
    )
    minimum_candidate_mean_loss_improvement: float = field(
        metadata={"parameter_unit": "mean bounded loss"} | _NONNEGATIVE
    )
    new_model_upload_delay_round_count: int = field(
        metadata={"parameter_unit": "round"} | _POSITIVE_COUNT
    )
    minimum_change_interval_sample_count: int = field(
        metadata={"parameter_unit": "sample"} | _POSITIVE_COUNT
    )
    local_training_batch_sample_count: int = field(
        metadata={"parameter_unit": "sample"} | _POSITIVE_COUNT
    )
    maximum_stored_evaluation_sample_count_per_model: int = field(
        metadata={"parameter_unit": "sample/model"} | _POSITIVE_COUNT
    )
    added_evaluation_batch_sample_count: int = field(
        metadata={"parameter_unit": "sample"} | _NONNEGATIVE
    )
    loss_monitor_maximum_retained_candidate_count: int = field(
        metadata={"parameter_unit": "candidate"} | _POSITIVE_COUNT
    )
    detector_name: str = field(metadata={})

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
        # 共通の検査は、intの派生型と、floatの派生型を受け入れる。値を使う部品はbuiltinの型だけを受け入れる。
        for configuration_parameter_name in _COUNT_FIELD_NAMES:
            if type(getattr(self, configuration_parameter_name)) is not int:
                raise RunSettingsValidationError(
                    configuration_parameter_name=configuration_parameter_name,
                    specified_parameter_value=getattr(self, configuration_parameter_name),
                    validation_failure_reason="bool・派生型以外のbuiltin intを指定してください。",
                )
        for configuration_parameter_name in _MEAN_LOSS_FIELD_NAMES:
            if type(getattr(self, configuration_parameter_name)) not in (int, float):
                raise RunSettingsValidationError(
                    configuration_parameter_name=configuration_parameter_name,
                    specified_parameter_value=getattr(self, configuration_parameter_name),
                    validation_failure_reason="bool・派生型以外のbuiltin int/floatを指定してください。",
                )
        if type(self.detector_name) is not str or not self.detector_name.strip():
            raise RunSettingsValidationError(
                configuration_parameter_name="detector_name",
                specified_parameter_value=self.detector_name,
                validation_failure_reason="空白以外の文字を含むbuiltin strを指定してください。",
            )


# 束のfield名から、受け入れるexact型への対応（宣言順）。
_REQUIRED_SETTINGS_TYPES_BY_FIELD_NAME: dict[str, tuple[type, ...]] = {
    "loss_change_detection_settings": (LossChangeDetectionSettings,),
    "prediction_combination_settings": (PredictionCombinationSettings,),
    "local_training_settings": (LocalTrainingSettings,),
    "local_training_schedule_settings": (LocalTrainingScheduleSettings,),
    "training_data_assignment_settings": (TrainingDataAssignmentSettings,),
    "candidate_model_training_and_acceptance_settings": (
        CandidateModelTrainingAndAcceptanceSettings,
    ),
    "candidate_parameter_initialization_settings": (CandidateParameterInitializationSettings,),
    "candidate_epoch_training_settings": (CandidateEpochTrainingSettings,),
    "parameter_optimizer_settings": (
        AdamParameterOptimizerSettings,
        SgdParameterOptimizerSettings,
    ),
    "scalar_settings": (FedsdaRunClientScalarSettings,),
}


@dataclass(frozen=True, kw_only=True)
class FedsdaRunClientSettings:
    """client 1つの組立てに要る、機能別の設定・置き場所のない値・検出器の賭け率。生成時に全部を確かめる。"""

    loss_change_detection_settings: LossChangeDetectionSettings
    prediction_combination_settings: PredictionCombinationSettings
    local_training_settings: LocalTrainingSettings
    local_training_schedule_settings: LocalTrainingScheduleSettings
    training_data_assignment_settings: TrainingDataAssignmentSettings
    candidate_model_training_and_acceptance_settings: CandidateModelTrainingAndAcceptanceSettings
    candidate_parameter_initialization_settings: CandidateParameterInitializationSettings
    candidate_epoch_training_settings: CandidateEpochTrainingSettings
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings
    scalar_settings: FedsdaRunClientScalarSettings
    loss_monitor_betting_fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        for settings_field_name, required_types in _REQUIRED_SETTINGS_TYPES_BY_FIELD_NAME.items():
            component_settings = getattr(self, settings_field_name)
            if type(component_settings) not in required_types:
                raise RunSettingsValidationError(
                    configuration_parameter_name=settings_field_name,
                    specified_parameter_value=component_settings,
                    validation_failure_reason=(
                        "・".join(required_type.__name__ for required_type in required_types)
                        + "型の値を指定してください。派生型は受理しません。"
                    ),
                )
            # frozenを回避して組み立てた値も拒否できるよう、各設定の検査をもう一度行う。
            component_settings.__post_init__()
        betting_fractions = self.loss_monitor_betting_fractions
        if (
            type(betting_fractions) is not tuple
            or not betting_fractions
            or any(
                type(betting_fraction) is not float or not 0.0 < betting_fraction < 1.0
                for betting_fraction in betting_fractions
            )
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="loss_monitor_betting_fractions",
                specified_parameter_value=betting_fractions,
                validation_failure_reason=(
                    "0より大きく1未満のbuiltin floatの、空でないtupleを指定してください。"
                ),
            )
        fixed_share_time_scale = self.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        pending_capacity = (
            self.training_data_assignment_settings.pending_assignment_buffer_capacity_samples
        )
        if fixed_share_time_scale != pending_capacity:
            raise RunSettingsValidationError(
                configuration_parameter_name="fixed_share_weight_redistribution_time_scale_samples",
                specified_parameter_value=fixed_share_time_scale,
                validation_failure_reason=(
                    "最終FedSDA構成ではFixed-Shareの時間尺度を帰属保留FIFO容量と同値にしてください。"
                    f"pending_assignment_buffer_capacity_samples={pending_capacity!r}。"
                ),
            )
        minimum_improvement = self.scalar_settings.minimum_candidate_mean_loss_improvement
        minimum_decrease = self.candidate_epoch_training_settings.minimum_validation_loss_decrease
        if minimum_improvement != minimum_decrease:
            raise RunSettingsValidationError(
                configuration_parameter_name="minimum_candidate_mean_loss_improvement",
                specified_parameter_value=minimum_improvement,
                validation_failure_reason=(
                    "最終FedSDA構成では候補の平均損失の最小改善量を、候補の学習の早期終了の最小改善量と同値にしてください。"
                    f"minimum_validation_loss_decrease={minimum_decrease!r}。"
                ),
            )
        local_batch_sample_count = self.scalar_settings.local_training_batch_sample_count
        candidate_batch_sample_count = (
            self.candidate_epoch_training_settings.maximum_batch_sample_count
        )
        if local_batch_sample_count != candidate_batch_sample_count:
            raise RunSettingsValidationError(
                configuration_parameter_name="local_training_batch_sample_count",
                specified_parameter_value=local_batch_sample_count,
                validation_failure_reason=(
                    "最終FedSDA構成ではローカル学習のbatchの件数を、候補の学習のbatchの件数の上限と同値にしてください。"
                    f"maximum_batch_sample_count={candidate_batch_sample_count!r}。"
                ),
            )
