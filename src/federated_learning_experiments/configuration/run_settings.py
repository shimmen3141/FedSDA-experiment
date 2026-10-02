"""初回の固定条件を束ね、値と機能間の組合せを検証した部分型を表す。"""

from dataclasses import dataclass, fields

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.configuration.run_settings_validation import (
    validate_experiment_run_settings,
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


@dataclass(frozen=True, kw_only=True)
class ValidatedExperimentRunSettingsSubset:
    """検証済みの初回条件。実行・保存用の完全なrun設定ではない。"""

    method_name: str
    experiment_run_conditions: ExperimentRunConditions
    model_architecture_settings: ModelArchitectureSettings
    loss_change_detection_settings: LossChangeDetectionSettings
    prediction_combination_settings: PredictionCombinationSettings
    local_training_settings: LocalTrainingSettings
    training_data_assignment_settings: TrainingDataAssignmentSettings
    candidate_model_training_and_acceptance_settings: CandidateModelTrainingAndAcceptanceSettings
    model_consolidation_settings: ModelConsolidationSettings

    def __post_init__(self) -> None:
        """直接構築を含めて、正式な全フィールドの組合せを検証する。"""
        validate_experiment_run_settings({
            settings_field.name: getattr(self, settings_field.name)
            for settings_field in fields(self)
        })
