"""単一runの各層で許可する依存とpackage境界を確認する。"""

import ast
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest


# BEGIN held_candidate_validation_diagnostic_notification dependency contract
@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import AdaHedgeDiagnosticEvidenceCollection",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import AdaHedgeDiagnosticEvidenceCollection as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import notify_diagnostics_of_training_assignment_change",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import notify_diagnostics_of_training_assignment_change as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.runtime.training_assignment_diagnostic_notification",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence import AdaHedgeDiagnosticEvidence",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.alarm_occurrence_handling import handle_alarm_occurrence",
            False,
        ),
    ],
)
def test_held_candidate_validation_diagnostic_notification_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


# END held_candidate_validation_diagnostic_notification dependency contract


# BEGIN alarm_occurrence_handling dependency contract
@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        ("runtime/alarm_occurrence_handling.py", "from dataclasses import dataclass", True),
        (
            "runtime/alarm_occurrence_handling.py",
            "from dataclasses import dataclass as AcceptedDependency",
            True,
        ),
        ("runtime/alarm_occurrence_handling.py", "import dataclasses", False),
        ("runtime/alarm_occurrence_handling.py", "from dataclasses import _private", False),
        ("runtime/alarm_occurrence_handling.py", "from dataclasses import *", False),
        ("runtime/alarm_occurrence_handling.py", "from random import Random", True),
        (
            "runtime/alarm_occurrence_handling.py",
            "from random import Random as AcceptedDependency",
            True,
        ),
        ("runtime/alarm_occurrence_handling.py", "import random", False),
        ("runtime/alarm_occurrence_handling.py", "from random import _private", False),
        ("runtime/alarm_occurrence_handling.py", "from random import *", False),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import AdaHedgeDiagnosticEvidenceCollection",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import AdaHedgeDiagnosticEvidenceCollection as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecord",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecord as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.evaluation.adaptation_record_store",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecordStore",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecordStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.evaluation.adaptation_record_store",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.evaluation.model_evaluation_sample_store",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.learning.models.residual_adapter_classifier",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.learning.training.candidate_epoch_training_settings",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.learning.training.current_training_model_assignment",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.learning.training.held_model_training_state_registry",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.learning.training.indexed_observed_training_sample",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.learning.training.model_training_and_assignment_counts",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.learning.training.model_training_sample_store",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import SgdParameterOptimizerSettings",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import SgdParameterOptimizerSettings as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import CandidateParameterInitializationSettings",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import CandidateParameterInitializationSettings as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import OverallAndTrueClassLossMonitor",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import OverallAndTrueClassLossMonitor as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_adaptation_recording import record_completed_alarm_response",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_adaptation_recording import record_completed_alarm_response as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.runtime.alarm_adaptation_recording",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_adaptation_recording import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_adaptation_recording import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_buffer_response import respond_to_alarm_with_buffered_samples",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_buffer_response import respond_to_alarm_with_buffered_samples as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.runtime.alarm_buffer_response",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_buffer_response import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_buffer_response import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_response_completion import AlarmResponseCompletion",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_response_completion import AlarmResponseCompletion as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.runtime.alarm_response_completion",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_response_completion import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_response_completion import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_response_completion import complete_alarm_buffer_response",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_response_completion import complete_alarm_buffer_response as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.runtime.alarm_response_completion",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_response_completion import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_response_completion import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.candidate_validation_session_holder import CandidateValidationSessionHolder",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.candidate_validation_session_holder import CandidateValidationSessionHolder as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.runtime.candidate_validation_session_holder",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.candidate_validation_session_holder import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.candidate_validation_session_holder import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.held_candidate_validation_progress import apply_alarm_response_to_validation_session_holder",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.held_candidate_validation_progress import apply_alarm_response_to_validation_session_holder as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.runtime.held_candidate_validation_progress",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.held_candidate_validation_progress import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.held_candidate_validation_progress import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import notify_diagnostics_of_training_assignment_change",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import notify_diagnostics_of_training_assignment_change as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "import federated_learning_experiments.runtime.training_assignment_diagnostic_notification",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import _private",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import *",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.held_candidate_validation_progress import advance_held_candidate_validation",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.held_candidate_validation_progress import finalize_held_incomplete_candidate_validation",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import advance_post_alarm_candidate_validation",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_buffer_response import AlarmBufferResponse",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import resolve_alarm_change_interval",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence import AdaHedgeDiagnosticEvidence",
            False,
        ),
        ("runtime/alarm_occurrence_handling.py", "from random import random", False),
        ("runtime/alarm_occurrence_handling.py", "import torch", False),
        ("runtime/alarm_occurrence_handling.py", "import os", False),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_drift_experiment import config",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_drift_experiment.detection_episode import DetectionEpisodeController",
            False,
        ),
        (
            "runtime/alarm_occurrence_handling.py",
            "from federated_learning_experiments.runtime import alarm_buffer_response",
            False,
        ),
    ],
)
def test_alarm_occurrence_handling_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


# END alarm_occurrence_handling dependency contract


# BEGIN held_candidate_validation_progress dependency contract
@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        (
            "runtime/candidate_validation_session_holder.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession",
            True,
        ),
        (
            "runtime/candidate_validation_session_holder.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession as AcceptedDependency",
            True,
        ),
        (
            "runtime/candidate_validation_session_holder.py",
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start",
            False,
        ),
        (
            "runtime/candidate_validation_session_holder.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import *",
            False,
        ),
        (
            "runtime/candidate_validation_session_holder.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import start_post_alarm_candidate_validation_session",
            False,
        ),
        (
            "runtime/candidate_validation_session_holder.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecordStore",
            False,
        ),
        (
            "runtime/candidate_validation_session_holder.py",
            "from dataclasses import dataclass",
            False,
        ),
        ("runtime/candidate_validation_session_holder.py", "import torch", False),
        (
            "runtime/candidate_validation_session_holder.py",
            "from federated_drift_experiment import config",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from dataclasses import dataclass",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from dataclasses import dataclass as AcceptedDependency",
            True,
        ),
        ("runtime/held_candidate_validation_progress.py", "import dataclasses", False),
        (
            "runtime/held_candidate_validation_progress.py",
            "from dataclasses import _private",
            False,
        ),
        ("runtime/held_candidate_validation_progress.py", "from dataclasses import *", False),
        ("runtime/held_candidate_validation_progress.py", "from torch import Tensor", True),
        (
            "runtime/held_candidate_validation_progress.py",
            "from torch import Tensor as AcceptedDependency",
            True,
        ),
        ("runtime/held_candidate_validation_progress.py", "import torch", False),
        ("runtime/held_candidate_validation_progress.py", "from torch import _private", False),
        ("runtime/held_candidate_validation_progress.py", "from torch import *", False),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecord",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecord as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.evaluation.adaptation_record_store",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecordStore",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecordStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.evaluation.adaptation_record_store",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.learning.training.current_training_model_assignment",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.learning.training.held_model_training_state_registry",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.learning.training.model_training_and_assignment_counts",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.learning.training.model_training_sample_store",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import TemporaryModelIdAllocator",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import TemporaryModelIdAllocator as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.learning.training.temporary_model_id_allocation",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.alarm_buffer_response import AlarmBufferResponse",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.alarm_buffer_response import AlarmBufferResponse as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.runtime.alarm_buffer_response",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.alarm_buffer_response import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.alarm_buffer_response import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_adaptation_recording import record_completed_candidate_validation",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_adaptation_recording import record_completed_candidate_validation as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.runtime.candidate_validation_adaptation_recording",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_adaptation_recording import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_adaptation_recording import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_adaptation_recording import record_incomplete_candidate_validation_finalization",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_adaptation_recording import record_incomplete_candidate_validation_finalization as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.runtime.candidate_validation_adaptation_recording",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_adaptation_recording import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_adaptation_recording import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_session_holder import CandidateValidationSessionHolder",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_session_holder import CandidateValidationSessionHolder as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.runtime.candidate_validation_session_holder",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_session_holder import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.candidate_validation_session_holder import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import IncompletePostAlarmCandidateValidationFinalization",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import IncompletePostAlarmCandidateValidationFinalization as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import finalize_incomplete_post_alarm_candidate_validation",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import finalize_incomplete_post_alarm_candidate_validation as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import PostAlarmCandidateValidationProgress",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import PostAlarmCandidateValidationProgress as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_progress",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import advance_post_alarm_candidate_validation",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import advance_post_alarm_candidate_validation as AcceptedDependency",
            True,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_progress",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import _private",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import *",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.alarm_buffer_response import respond_to_alarm_with_buffered_samples",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.alarm_response_completion import complete_alarm_buffer_response",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.alarm_adaptation_recording import record_completed_alarm_response",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import apply_post_alarm_candidate_validation_resolution",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession",
            False,
        ),
        ("runtime/held_candidate_validation_progress.py", "from torch import no_grad", False),
        ("runtime/held_candidate_validation_progress.py", "import os", False),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_drift_experiment import config",
            False,
        ),
        (
            "runtime/held_candidate_validation_progress.py",
            "from federated_learning_experiments.runtime import candidate_validation_session_holder",
            False,
        ),
    ],
)
def test_held_candidate_validation_progress_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


# END held_candidate_validation_progress dependency contract


# BEGIN candidate_validation_adaptation_recording dependency contract
@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationOutcome",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationOutcome as AcceptedDependency",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.evaluation.adaptation_record_store",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.evaluation.adaptation_record_store.AdaptationOutcome",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import _private",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import *",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecord",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecord as AcceptedDependency",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.evaluation.adaptation_record_store",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecord",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import _private",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import *",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecordStore",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecordStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.evaluation.adaptation_record_store",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecordStore",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import _private",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import *",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange as AcceptedDependency",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.learning.training.current_training_model_assignment",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import _private",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import *",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import IncompletePostAlarmCandidateValidationDecisionRecord",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import IncompletePostAlarmCandidateValidationDecisionRecord as AcceptedDependency",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record.IncompletePostAlarmCandidateValidationDecisionRecord",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import _private",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import *",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import PostAlarmCandidateValidationDecisionRecord",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import PostAlarmCandidateValidationDecisionRecord as AcceptedDependency",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record.PostAlarmCandidateValidationDecisionRecord",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import _private",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import *",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import IncompletePostAlarmCandidateValidationFinalization",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import IncompletePostAlarmCandidateValidationFinalization as AcceptedDependency",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization.IncompletePostAlarmCandidateValidationFinalization",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import _private",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import *",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import PostAlarmCandidateValidationCompletion",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import PostAlarmCandidateValidationCompletion as AcceptedDependency",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_progress",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_progress.PostAlarmCandidateValidationCompletion",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import _private",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import *",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import PostAlarmCandidateValidationResolution",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import PostAlarmCandidateValidationResolution as AcceptedDependency",
            True,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution.PostAlarmCandidateValidationResolution",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import _private",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import *",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import advance_post_alarm_candidate_validation",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import apply_post_alarm_candidate_validation_resolution",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import finalize_incomplete_post_alarm_candidate_validation",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecordSnapshot",
            False,
        ),
        ("runtime/candidate_validation_adaptation_recording.py", "from typing import cast", False),
        ("runtime/candidate_validation_adaptation_recording.py", "import torch", False),
        ("runtime/candidate_validation_adaptation_recording.py", "import os", False),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_drift_experiment import config",
            False,
        ),
        (
            "runtime/candidate_validation_adaptation_recording.py",
            "from federated_learning_experiments.runtime import alarm_adaptation_recording",
            False,
        ),
        ("evaluation/adaptation_record_store.py", "from typing import get_args", True),
        ("evaluation/adaptation_record_store.py", "from typing import Literal, get_args", True),
        ("evaluation/adaptation_record_store.py", "from typing import cast", False),
        ("evaluation/adaptation_record_store.py", "import typing", False),
        (
            "evaluation/adaptation_record_store.py",
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import PostAlarmCandidateValidationCompletion",
            False,
        ),
    ],
)
def test_candidate_validation_adaptation_recording_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


# END candidate_validation_adaptation_recording dependency contract


# BEGIN alarm_response_completion dependency contract
@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("from dataclasses import dataclass", True),
        ("from dataclasses import dataclass as AcceptedDependency", True),
        ("import dataclasses", False),
        ("import dataclasses as AcceptedDependency", False),
        ("import dataclasses.dataclass", False),
        ("from dataclasses import _private", False),
        ("from dataclasses.child import dataclass", False),
        ("from dataclasses import *", False),
        ("from dataclasses import field", False),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            False,
        ),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.child import ModelAndClassLossStatisticsStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import field",
            False,
        ),
        (
            "from ..learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment.child import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import field",
            False,
        ),
        (
            "from ..learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment, TrainingModelAssignmentChange",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment.child import TrainingModelAssignmentChange",
            False,
        ),
        (
            "from ..learning.training.current_training_model_assignment import TrainingModelAssignmentChange",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import _validate_model_id",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import OverallAndTrueClassLossMonitor",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import OverallAndTrueClassLossMonitor as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring.OverallAndTrueClassLossMonitor",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import _private",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring.child import OverallAndTrueClassLossMonitor",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import *",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import field",
            False,
        ),
        (
            "from ..methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import OverallAndTrueClassLossMonitor",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_loss_monitoring_baseline_mean_loss",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_loss_monitoring_baseline_mean_loss as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.select_loss_monitoring_baseline_mean_loss",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import _private",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.child import select_loss_monitoring_baseline_mean_loss",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import *",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import field",
            False,
        ),
        (
            "from ..methods.fedsda.loss_statistics.loss_baseline_selection import select_loss_monitoring_baseline_mean_loss",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import _private",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.child import PendingTrainingAssignmentBuffer",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import *",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import field",
            False,
        ),
        (
            "from ..methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_buffer_response import AlarmBufferResponse",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_buffer_response import AlarmBufferResponse as AcceptedDependency",
            True,
        ),
        ("import federated_learning_experiments.runtime.alarm_buffer_response", False),
        (
            "import federated_learning_experiments.runtime.alarm_buffer_response as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.alarm_buffer_response.AlarmBufferResponse",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_buffer_response import _private",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_buffer_response.child import AlarmBufferResponse",
            False,
        ),
        ("from federated_learning_experiments.runtime.alarm_buffer_response import *", False),
        ("from federated_learning_experiments.runtime.alarm_buffer_response import field", False),
        ("from ..runtime.alarm_buffer_response import AlarmBufferResponse", True),
        (
            "from federated_learning_experiments.runtime.alarm_buffer_response import respond_to_alarm_with_buffered_samples",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_buffer_response import ALARM_BUFFER_RESPONSE_OUTCOMES",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_alarm_interval_reuse_baseline_mean_loss",
            False,
        ),
        ("import os", False),
        ("import numpy", False),
        ("import torch", False),
        ("from federated_drift_experiment import config", False),
        ("from federated_learning_experiments import runtime", False),
        ("from federated_learning_experiments.runtime import alarm_response_completion", False),
        ("from federated_learning_experiments.cli import main", False),
        ("from ..configuration.run_settings import RunSettings", False),
    ],
)
def test_alarm_response_completion_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="runtime/alarm_response_completion.py", source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


# END alarm_response_completion dependency contract


# BEGIN alarm_buffer_response dependency contract
@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("from dataclasses import dataclass", True),
        ("from dataclasses import dataclass as AcceptedDependency", True),
        ("import dataclasses", False),
        ("import dataclasses as AcceptedDependency", False),
        ("import dataclasses.dataclass", False),
        ("from dataclasses import _private", False),
        ("from dataclasses.child import dataclass", False),
        ("from dataclasses import *", False),
        ("from dataclasses import field", False),
        ("from random import Random", True),
        ("from random import Random as AcceptedDependency", True),
        ("import random", False),
        ("import random as AcceptedDependency", False),
        ("import random.Random", False),
        ("from random import _private", False),
        ("from random.child import Random", False),
        ("from random import *", False),
        ("from random import field", False),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore as AcceptedDependency",
            True,
        ),
        ("import federated_learning_experiments.evaluation.model_evaluation_sample_store", False),
        (
            "import federated_learning_experiments.evaluation.model_evaluation_sample_store as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import _private",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store.child import ModelEvaluationSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import *",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import field",
            False,
        ),
        ("from ..evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore", True),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            False,
        ),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.child import ModelAndClassLossStatisticsStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import field",
            False,
        ),
        (
            "from ..learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier.child import ResidualAdapterClassifier",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import field",
            False,
        ),
        (
            "from ..learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training_settings",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training_settings as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings.child import CandidateEpochTrainingSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import field",
            False,
        ),
        (
            "from ..learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment.child import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import field",
            False,
        ),
        (
            "from ..learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry.child import HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import field",
            False,
        ),
        (
            "from ..learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.indexed_observed_training_sample",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.indexed_observed_training_sample as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample.child import IndexedObservedTrainingSample",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import field",
            False,
        ),
        (
            "from ..learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_and_assignment_counts",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_and_assignment_counts as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts.child import ModelTrainingAndAssignmentCountsStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import field",
            False,
        ),
        (
            "from ..learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_store",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_store as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store.child import ModelTrainingSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import field",
            False,
        ),
        (
            "from ..learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings.child import AdamParameterOptimizerSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import field",
            False,
        ),
        (
            "from ..learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import SgdParameterOptimizerSettings",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import SgdParameterOptimizerSettings as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import _private",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings.child import SgdParameterOptimizerSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import field",
            False,
        ),
        (
            "from ..learning.training.parameter_optimizer_settings import SgdParameterOptimizerSettings",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import _private",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.child import CandidateModelTrainingAndAcceptanceSettings",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import *",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import field",
            False,
        ),
        (
            "from ..methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import CandidateParameterInitializationSettings",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import CandidateParameterInitializationSettings as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import _private",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.child import CandidateParameterInitializationSettings",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import *",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import field",
            False,
        ),
        (
            "from ..methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import CandidateParameterInitializationSettings",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import _private",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.child import PendingTrainingAssignmentBuffer",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import *",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import field",
            False,
        ),
        (
            "from ..methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import PreparedAlarmTrainingIntervals",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import PreparedAlarmTrainingIntervals as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals.PreparedAlarmTrainingIntervals",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import _private",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals.child import PreparedAlarmTrainingIntervals",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import *",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import field",
            False,
        ),
        (
            "from ..methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import PreparedAlarmTrainingIntervals",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES as AcceptedDependency",
            True,
        ),
        ("import federated_learning_experiments.runtime.alarm_change_interval_resolution", False),
        (
            "import federated_learning_experiments.runtime.alarm_change_interval_resolution as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.alarm_change_interval_resolution.ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import _private",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution.child import ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import *",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import field",
            False,
        ),
        (
            "from ..runtime.alarm_change_interval_resolution import ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import AlarmChangeIntervalResolution",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import AlarmChangeIntervalResolution as AcceptedDependency",
            True,
        ),
        ("import federated_learning_experiments.runtime.alarm_change_interval_resolution", False),
        (
            "import federated_learning_experiments.runtime.alarm_change_interval_resolution as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.alarm_change_interval_resolution.AlarmChangeIntervalResolution",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import _private",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution.child import AlarmChangeIntervalResolution",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import *",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import field",
            False,
        ),
        (
            "from ..runtime.alarm_change_interval_resolution import AlarmChangeIntervalResolution",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import resolve_alarm_change_interval",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import resolve_alarm_change_interval as AcceptedDependency",
            True,
        ),
        ("import federated_learning_experiments.runtime.alarm_change_interval_resolution", False),
        (
            "import federated_learning_experiments.runtime.alarm_change_interval_resolution as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.alarm_change_interval_resolution.resolve_alarm_change_interval",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import _private",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution.child import resolve_alarm_change_interval",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import *",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_change_interval_resolution import field",
            False,
        ),
        (
            "from ..runtime.alarm_change_interval_resolution import resolve_alarm_change_interval",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_training_interval_preparation import prepare_alarm_training_intervals",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_training_interval_preparation import prepare_alarm_training_intervals as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.runtime.alarm_training_interval_preparation",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.alarm_training_interval_preparation as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.alarm_training_interval_preparation.prepare_alarm_training_intervals",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_training_interval_preparation import _private",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_training_interval_preparation.child import prepare_alarm_training_intervals",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_training_interval_preparation import *",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.alarm_training_interval_preparation import field",
            False,
        ),
        (
            "from ..runtime.alarm_training_interval_preparation import prepare_alarm_training_intervals",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import absorb_assigned_training_samples_into_held_model",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import absorb_assigned_training_samples_into_held_model as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.runtime.assigned_training_sample_absorption",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.assigned_training_sample_absorption as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import _private",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption.child import absorb_assigned_training_samples_into_held_model",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import *",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import field",
            False,
        ),
        (
            "from ..runtime.assigned_training_sample_absorption import absorb_assigned_training_samples_into_held_model",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import _private",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.child import PostAlarmCandidateValidationSession",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import *",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import field",
            False,
        ),
        (
            "from ..runtime.post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession",
            True,
        ),
        ("import os", False),
        ("import numpy", False),
        ("from federated_drift_experiment import config", False),
        ("from federated_learning_experiments import runtime", False),
        ("from federated_learning_experiments.runtime import alarm_buffer_response", False),
        ("from federated_learning_experiments.cli import main", False),
        ("from ..configuration.run_settings import RunSettings", False),
    ],
)
def test_alarm_buffer_response_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="runtime/alarm_buffer_response.py", source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


# END alarm_buffer_response dependency contract


# BEGIN alarm_training_interval_preparation dependency contract
@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        (
            "learning/training/indexed_observed_training_sample.py",
            "from dataclasses import dataclass",
            True,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from dataclasses import dataclass as AcceptedDependency",
            True,
        ),
        ("learning/training/indexed_observed_training_sample.py", "import dataclasses", False),
        (
            "learning/training/indexed_observed_training_sample.py",
            "import dataclasses as AcceptedDependency",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "import dataclasses.dataclass",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from dataclasses import _private",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from dataclasses.child import dataclass",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from dataclasses import *",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from dataclasses import field",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample as AcceptedDependency",
            True,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "import federated_learning_experiments.learning.training.model_training_sample_records",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "import federated_learning_experiments.learning.training.model_training_sample_records as AcceptedDependency",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "import federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import _private",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records.child import ObservedTrainingSample",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import *",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import field",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from .model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        ("learning/training/indexed_observed_training_sample.py", "import os", False),
        ("learning/training/indexed_observed_training_sample.py", "import numpy", False),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from federated_drift_experiment import config",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from federated_learning_experiments import runtime",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from federated_learning_experiments.runtime import alarm_training_interval_preparation",
            False,
        ),
        (
            "learning/training/indexed_observed_training_sample.py",
            "from federated_learning_experiments.cli import main",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from dataclasses import dataclass",
            True,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from dataclasses import dataclass as AcceptedDependency",
            True,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "import dataclasses",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "import dataclasses as AcceptedDependency",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "import dataclasses.dataclass",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from dataclasses import _private",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from dataclasses.child import dataclass",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from dataclasses import *",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from dataclasses import field",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample",
            True,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample as AcceptedDependency",
            True,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "import federated_learning_experiments.learning.training.indexed_observed_training_sample",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "import federated_learning_experiments.learning.training.indexed_observed_training_sample as AcceptedDependency",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "import federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import _private",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample.child import IndexedObservedTrainingSample",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import *",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import field",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from ....learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample",
            True,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "import os",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "import numpy",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from federated_drift_experiment import config",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from federated_learning_experiments import runtime",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from federated_learning_experiments.runtime import alarm_training_interval_preparation",
            False,
        ),
        (
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "from federated_learning_experiments.cli import main",
            False,
        ),
        ("runtime/alarm_training_interval_preparation.py", "from random import Random", True),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from random import Random as AcceptedDependency",
            True,
        ),
        ("runtime/alarm_training_interval_preparation.py", "import random", False),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import random as AcceptedDependency",
            False,
        ),
        ("runtime/alarm_training_interval_preparation.py", "import random.Random", False),
        ("runtime/alarm_training_interval_preparation.py", "from random import _private", False),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from random.child import Random",
            False,
        ),
        ("runtime/alarm_training_interval_preparation.py", "from random import *", False),
        ("runtime/alarm_training_interval_preparation.py", "from random import field", False),
        ("runtime/alarm_training_interval_preparation.py", "from torch import Tensor", True),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from torch import Tensor as AcceptedDependency",
            True,
        ),
        ("runtime/alarm_training_interval_preparation.py", "import torch", False),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import torch as AcceptedDependency",
            False,
        ),
        ("runtime/alarm_training_interval_preparation.py", "import torch.Tensor", False),
        ("runtime/alarm_training_interval_preparation.py", "from torch import _private", False),
        ("runtime/alarm_training_interval_preparation.py", "from torch.child import Tensor", False),
        ("runtime/alarm_training_interval_preparation.py", "from torch import *", False),
        ("runtime/alarm_training_interval_preparation.py", "from torch import field", False),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.indexed_observed_training_sample",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.indexed_observed_training_sample as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample.child import IndexedObservedTrainingSample",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.indexed_observed_training_sample import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..learning.training.indexed_observed_training_sample import IndexedObservedTrainingSample",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import PreparedAlarmTrainingIntervals",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import PreparedAlarmTrainingIntervals as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals.PreparedAlarmTrainingIntervals",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals.child import PreparedAlarmTrainingIntervals",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import PreparedAlarmTrainingIntervals",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.current_training_model_assignment",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.current_training_model_assignment as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment.child import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.current_training_model_assignment import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.child import PendingTrainingAssignmentBuffer",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.evaluation.model_evaluation_sample_store",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.evaluation.model_evaluation_sample_store as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store.child import ModelEvaluationSampleStore",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_records import ObservedEvaluationSample",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_records import ObservedEvaluationSample as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.evaluation.model_evaluation_sample_records",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.evaluation.model_evaluation_sample_records as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.evaluation.model_evaluation_sample_records.ObservedEvaluationSample",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_records import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_records.child import ObservedEvaluationSample",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_records import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.evaluation.model_evaluation_sample_records import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..evaluation.model_evaluation_sample_records import ObservedEvaluationSample",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.model_training_sample_records",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.model_training_sample_records as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records.child import ObservedTrainingSample",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.held_model_training_state_registry",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.held_model_training_state_registry as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry.child import HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.model_training_sample_store",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.model_training_sample_store as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store.child import ModelTrainingSampleStore",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_sample_store import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.model_training_and_assignment_counts",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.model_training_and_assignment_counts as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts.child import ModelTrainingAndAssignmentCountsStore",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.child import ModelAndClassLossStatisticsStore",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.child import evaluate_classifier_per_sample_bounded_losses",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from ..learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import absorb_assigned_training_samples_into_held_model",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import absorb_assigned_training_samples_into_held_model as AcceptedDependency",
            True,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.runtime.assigned_training_sample_absorption",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.runtime.assigned_training_sample_absorption as AcceptedDependency",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "import federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import _private",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption.child import absorb_assigned_training_samples_into_held_model",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import *",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import field",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from .assigned_training_sample_absorption import absorb_assigned_training_samples_into_held_model",
            True,
        ),
        ("runtime/alarm_training_interval_preparation.py", "import os", False),
        ("runtime/alarm_training_interval_preparation.py", "import numpy", False),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_drift_experiment import config",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments import runtime",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.runtime import alarm_training_interval_preparation",
            False,
        ),
        (
            "runtime/alarm_training_interval_preparation.py",
            "from federated_learning_experiments.cli import main",
            False,
        ),
    ],
)
def test_alarm_training_interval_preparation_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


# END alarm_training_interval_preparation dependency contract


# BEGIN alarm_change_interval_resolution dependency contract
@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        *[
            (source_text, expected_acceptance)
            for imported_module_name in (
                "dataclasses.dataclass",
                "torch.Tensor",
                "torch.cat",
                "torch.float32",
                "torch.strided",
                "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
                "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
                "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
                "federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
                "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
                "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
                "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
                "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
                "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
                "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
                "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
                "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment.AlarmIntervalModelReuseAssessment",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization.select_candidate_initial_parameter_snapshot",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings",
                "federated_learning_experiments.runtime.alarm_interval_model_reuse_assessment.evaluate_held_models_for_alarm_interval_reuse",
                "federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
                "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
                "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.start_post_alarm_candidate_validation_session",
            )
            for source_text, expected_acceptance in (
                (
                    f"from {imported_module_name.rsplit('.', 1)[0]} import {imported_module_name.rsplit('.', 1)[1]}",
                    True,
                ),
                (
                    f"from {imported_module_name.rsplit('.', 1)[0]} import {imported_module_name.rsplit('.', 1)[1]} as AcceptedDependency",
                    True,
                ),
                (f"import {imported_module_name.rsplit('.', 1)[0]}", False),
                (f"import {imported_module_name.rsplit('.', 1)[0]} as AcceptedDependency", False),
                (f"import {imported_module_name}", False),
                (f"import {imported_module_name} as AcceptedDependency", False),
                (f"from {imported_module_name.rsplit('.', 1)[0]} import _private", False),
                (
                    f"from {imported_module_name.rsplit('.', 1)[0]}.child import {imported_module_name.rsplit('.', 1)[1]}",
                    False,
                ),
                (f"from {imported_module_name.rsplit('.', 1)[0]} import field", False),
            )
        ],
        *[
            (source_text, expected_acceptance)
            for imported_module_name in (
                "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
                "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
                "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
                "federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
                "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
                "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
                "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
                "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
                "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
                "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
                "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
                "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment.AlarmIntervalModelReuseAssessment",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization.select_candidate_initial_parameter_snapshot",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings",
                "federated_learning_experiments.runtime.alarm_interval_model_reuse_assessment.evaluate_held_models_for_alarm_interval_reuse",
                "federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
                "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
                "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.start_post_alarm_candidate_validation_session",
            )
            for source_text, expected_acceptance in (
                (
                    f"from {imported_module_name.rsplit('.', 1)[0].replace('federated_learning_experiments.runtime.', '.').replace('federated_learning_experiments.', '..')} import {imported_module_name.rsplit('.', 1)[1]}",
                    True,
                ),
                (
                    f"from {imported_module_name.rsplit('.', 2)[0]} import {imported_module_name.rsplit('.', 2)[1]}",
                    False,
                ),
            )
        ],
        (
            "from math import isfinite",
            False,
        ),
        (
            "from random import Random",
            False,
        ),
        (
            "from torch import no_grad",
            False,
        ),
        (
            "from torch import manual_seed",
            False,
        ),
        (
            "from torch import stack",
            False,
        ),
        (
            "from numpy import concatenate",
            False,
        ),
        (
            "from copy import deepcopy",
            False,
        ),
        (
            "from __future__ import annotations",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_drift_experiment import config",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import _validate_absorption_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import _validate_post_alarm_candidate_validation_start_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import apply_post_alarm_candidate_validation_resolution",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import advance_post_alarm_candidate_validation",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import finalize_incomplete_post_alarm_candidate_validation",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_local_adoption import adopt_candidate_as_current_training_model",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import assess_alarm_interval_model_reuse",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_alarm_interval_reuse_baseline_mean_loss",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer",
            False,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingState",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "from federated_learning_experiments.runtime import alarm_interval_model_reuse_assessment",
            False,
        ),
    ],
)
def test_alarm_change_interval_resolution_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="runtime/alarm_change_interval_resolution.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


# END alarm_change_interval_resolution dependency contract
@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        *[
            (source_text, expected_acceptance)
            for imported_module_name in (
                "math.isfinite",
                "torch.Tensor",
                "torch.mean",
                "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
                "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
                "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment.AlarmIntervalModelReuseAssessment",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment.assess_alarm_interval_model_reuse",
                "federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.select_alarm_interval_reuse_baseline_mean_loss",
            )
            for source_text, expected_acceptance in (
                (
                    f"from {imported_module_name.rsplit('.', 1)[0]} import {imported_module_name.rsplit('.', 1)[1]}",
                    True,
                ),
                (
                    f"from {imported_module_name.rsplit('.', 1)[0]} import {imported_module_name.rsplit('.', 1)[1]} as AcceptedDependency",
                    True,
                ),
                (f"import {imported_module_name.rsplit('.', 1)[0]}", False),
                (f"import {imported_module_name.rsplit('.', 1)[0]} as AcceptedDependency", False),
                (f"import {imported_module_name}", False),
                (f"import {imported_module_name} as AcceptedDependency", False),
                (f"from {imported_module_name.rsplit('.', 1)[0]} import _private", False),
                (
                    f"from {imported_module_name.rsplit('.', 1)[0]}.child import {imported_module_name.rsplit('.', 1)[1]}",
                    False,
                ),
                (f"from {imported_module_name.rsplit('.', 1)[0]} import field", False),
            )
        ],
        *[
            (source_text, expected_acceptance)
            for imported_module_name in (
                "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
                "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
                "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment.AlarmIntervalModelReuseAssessment",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment.assess_alarm_interval_model_reuse",
                "federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.select_alarm_interval_reuse_baseline_mean_loss",
            )
            for source_text, expected_acceptance in (
                (
                    f"from {imported_module_name.rsplit('.', 1)[0].replace('federated_learning_experiments.', '..')} import {imported_module_name.rsplit('.', 1)[1]}",
                    True,
                ),
                (
                    f"from {imported_module_name.rsplit('.', 2)[0]} import {imported_module_name.rsplit('.', 2)[1]}",
                    False,
                ),
            )
        ],
        ("from dataclasses import dataclass", False),
        ("from math import isnan", False),
        ("from random import Random", False),
        ("from torch import no_grad", False),
        ("from torch import manual_seed", False),
        ("from numpy import mean", False),
        ("from __future__ import annotations", False),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        ("from federated_drift_experiment import config", False),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import _validate_classifier_bounded_loss_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingState",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_post_alarm_reference_historical_mean_loss",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_loss_monitoring_baseline_mean_loss",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import _validate_alarm_interval_reuse_assessment_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        (
            "from .post_alarm_candidate_validation_session_start import start_post_alarm_candidate_validation_session",
            False,
        ),
        (
            "from .assigned_training_sample_absorption import absorb_assigned_training_samples_into_held_model",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import HeldModelTrainingStateRegistry",
            False,
        ),
    ],
)
def test_alarm_interval_reuse_evaluation_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="runtime/alarm_interval_model_reuse_assessment.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("from dataclasses import dataclass", True),
        ("from dataclasses import dataclass as AcceptedDependency", True),
        ("from math import isfinite", True),
        ("from math import isfinite as AcceptedDependency", True),
        ("import dataclasses", False),
        ("import dataclasses as AcceptedDependency", False),
        ("import dataclasses.dataclass", False),
        ("import math", False),
        ("import math as AcceptedDependency", False),
        ("import math.isfinite", False),
        ("from dataclasses import field", False),
        ("from dataclasses import _private", False),
        ("from dataclasses.child import dataclass", False),
        ("from math import isnan", False),
        ("from math import inf", False),
        ("from math.child import isfinite", False),
        ("from random import Random", False),
        ("from torch import Tensor", False),
        ("from torch import mean", False),
        ("import torch", False),
        ("import numpy", False),
        ("from __future__ import annotations", False),
        ("from federated_drift_experiment import config", False),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from ....runtime.alarm_interval_model_reuse_assessment import evaluate_held_models_for_alarm_interval_reuse",
            False,
        ),
        (
            "from .post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            False,
        ),
        (
            "from .candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        (
            "from ..loss_statistics.loss_baseline_selection import select_alarm_interval_reuse_baseline_mean_loss",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection import dataclass",
            False,
        ),
    ],
)
def test_alarm_interval_reuse_assessment_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="methods/fedsda/candidate_model_selection/alarm_interval_model_reuse_assessment.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        *[
            (source_text, expected_acceptance)
            for imported_module_name in (
                "dataclasses.dataclass",
                "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
                "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
                "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
                "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
                "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record.IncompletePostAlarmCandidateValidationDecisionRecord",
                "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
                "federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
            )
            for source_text, expected_acceptance in (
                (
                    f"from {imported_module_name.rsplit('.', 1)[0]} import {imported_module_name.rsplit('.', 1)[1]}",
                    True,
                ),
                (
                    f"from {imported_module_name.rsplit('.', 1)[0]} import {imported_module_name.rsplit('.', 1)[1]} as AcceptedDependency",
                    True,
                ),
                (f"import {imported_module_name.rsplit('.', 1)[0]}", False),
                (f"import {imported_module_name.rsplit('.', 1)[0]} as AcceptedDependency", False),
                (f"import {imported_module_name}", False),
                (f"import {imported_module_name} as AcceptedDependency", False),
                (f"from {imported_module_name.rsplit('.', 1)[0]} import _private", False),
                (
                    f"from {imported_module_name.rsplit('.', 1)[0]}.child import {imported_module_name.rsplit('.', 1)[1]}",
                    False,
                ),
                (f"from {imported_module_name.rsplit('.', 1)[0]} import field", False),
            )
        ],
        *[
            (source_text, expected_acceptance)
            for imported_module_name in (
                "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
                "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
                "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
                "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
                "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record.IncompletePostAlarmCandidateValidationDecisionRecord",
                "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
                "federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
            )
            for source_text, expected_acceptance in (
                (
                    f"from {imported_module_name.rsplit('.', 1)[0].replace('federated_learning_experiments.runtime.', '.').replace('federated_learning_experiments.', '..')} import {imported_module_name.rsplit('.', 1)[1]}",
                    True,
                ),
                (
                    f"from {imported_module_name.rsplit('.', 2)[0]} import {imported_module_name.rsplit('.', 2)[1]}",
                    False,
                ),
            )
        ],
        ("from math import isfinite", False),
        ("from random import Random", False),
        ("from torch import Tensor", False),
        ("from __future__ import annotations", False),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        ("from federated_drift_experiment import config", False),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            False,
        ),
        (
            "from .candidate_classifier_construction import create_independent_candidate_training_state",
            False,
        ),
        (
            "from .post_alarm_candidate_validation_progress import progress_post_alarm_candidate_validation",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_change_detection.detection_episode import DetectionEpisodeController",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import HeldModelTrainingStateRegistry",
            False,
        ),
    ],
)
def test_incomplete_validation_finalization_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="runtime/incomplete_post_alarm_candidate_validation_finalization.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("from dataclasses import dataclass", True),
        ("from dataclasses import dataclass as AcceptedDependency", True),
        ("import dataclasses", False),
        ("import dataclasses as AcceptedDependency", False),
        ("import dataclasses.dataclass", False),
        ("import dataclasses.dataclass as AcceptedDependency", False),
        ("from dataclasses import field", False),
        ("from dataclasses import _private", False),
        ("from dataclasses.child import dataclass", False),
        ("from math import isfinite", False),
        ("from random import Random", False),
        ("from torch import Tensor", False),
        ("from __future__ import annotations", False),
        ("from federated_drift_experiment import config", False),
        (
            "from ....runtime.incomplete_post_alarm_candidate_validation_finalization import IncompletePostAlarmCandidateValidationFinalization",
            False,
        ),
        (
            "from .post_alarm_candidate_loss_evaluation import PostAlarmCandidateLossEvaluation",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection import dataclass",
            False,
        ),
    ],
)
def test_incomplete_validation_decision_record_dependency_contract(
    source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="methods/fedsda/candidate_model_selection/incomplete_post_alarm_candidate_validation_decision_record.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import torch", False),
        ("import dataclasses", False),
        ("import math", False),
        ("from random import Random", False),
        ("import numpy", False),
        ("from dataclasses import field", False),
        ("from typing import Any", False),
        ("from copy import deepcopy", False),
        ("from torch import nn", False),
        ("from torch import clone", False),
        ("from torch import mean", False),
        ("from torch import no_grad", False),
        ("from torch.optim import Adam", False),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        ("from federated_learning_experiments.runtime import run", False),
        ("import federated_drift_experiment", False),
        ("import pathlib", False),
        ("from torch import *", False),
        ("from dataclasses import *", False),
        ("from dataclasses import dataclass", True),
        ("from torch import Tensor", True),
        ("from torch import float32, isfinite", True),
        ("from torch import strided", True),
        ("from torch import Tensor as ParameterValues", True),
    ],
)
def test_pending_model_upload_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="methods/fedsda/model_registration/pending_model_upload.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import torch", False),
        ("import torch.nn", False),
        ("import math", False),
        ("from random import Random", False),
        ("import numpy", False),
        ("from torch import nn", False),
        ("from torch import rand", False),
        ("from torch import mean", False),
        ("from torch import clamp", False),
        ("from torch.optim import Adam", False),
        ("from .residual_adapter_classifier import _validate_classifier_inputs", False),
        ("from . import residual_adapter_classifier", False),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier",
            False,
        ),
        (
            "from ..training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from ..loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        ("from federated_learning_experiments.runtime import run", False),
        ("from federated_learning_experiments.configuration import config", False),
        ("import federated_drift_experiment", False),
        ("import pathlib", False),
        ("from torch import *", False),
        ("from torch import Tensor", True),
        ("from torch import float32, isfinite", True),
        ("from torch import no_grad, strided", True),
        ("from .residual_adapter_classifier import ResidualAdapterClassifier", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier as Classifier",
            True,
        ),
    ],
)
def test_classifier_parameter_snapshot_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/models/classifier_parameter_snapshot.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import torch", False),
        ("import torch.nn", False),
        ("import math", False),
        ("from random import Random", False),
        ("import numpy", False),
        ("from torch import nn", False),
        ("from torch import rand", False),
        ("from torch import clamp", False),
        ("from torch import mean", False),
        ("from torch.nn import BCELoss", False),
        ("from torch.optim import Adam", False),
        (
            "from .class_probability_calculations import compute_model_mean_bounded_losses_after_label_observation",
            False,
        ),
        ("from ..models.residual_adapter_classifier import _validate_classifier_inputs", False),
        ("from ..models import residual_adapter_classifier", False),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier",
            False,
        ),
        (
            "from ..loss_statistics.batch_loss_statistics_initialization import initialize_model_and_class_loss_statistics_from_batch",
            False,
        ),
        (
            "from ..training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            False,
        ),
        ("from federated_learning_experiments.runtime import run", False),
        ("from federated_learning_experiments.configuration import config", False),
        ("import federated_drift_experiment", False),
        ("import pathlib", False),
        ("from torch import *", False),
        ("from torch import Tensor", True),
        ("from torch import abs, float32, isfinite", True),
        ("from torch import no_grad, softmax, strided, trunc", True),
        ("from ..models.residual_adapter_classifier import ResidualAdapterClassifier", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier as Classifier",
            True,
        ),
    ],
)
def test_classifier_bounded_loss_evaluation_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/prediction/classifier_bounded_loss_evaluation.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("from random import Random", False),
        ("import numpy", False),
        ("from dataclasses import asdict", False),
        ("from torch import rand", False),
        ("from torch.nn import Linear", False),
        ("from torch.optim import AdamW", False),
        ("from .parameter_optimizer_state import _private", False),
        ("from .parameter_optimizer_construction import create_parameter_optimizer", False),
        ("from .joint_model_parameter_update import perform_joint_model_parameter_update", False),
        ("from .model_training_sample_store import ModelTrainingSampleStore", False),
        ("from .held_model_training_binding import _private", False),
        ("from federated_learning_experiments.runtime import run", False),
        ("from federated_learning_experiments.configuration import config", False),
        ("import federated_drift_experiment", False),
        ("from . import parameter_optimizer_state", False),
        ("from dataclasses import dataclass", True),
        ("from torch import float32, strided", True),
        ("from torch.nn import Parameter", True),
        ("from torch.optim import Adam, SGD", True),
        ("from .parameter_optimizer_state import ParameterOptimizerState", True),
        ("from .held_model_training_binding import HeldModelTrainingBinding", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
    ],
)
def test_held_model_training_state_registry_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/held_model_training_state_registry.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text",
    [
        "import torch",
        "import torch.nn",
        "import torch.optim as optim",
        "import dataclasses",
        "import federated_learning_experiments.learning.training.parameter_optimizer_state as state",
        "from torch import nn",
        "from torch import optim",
    ],
)
def test_held_model_training_state_registry_rejects_module_imports(source_text):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/held_model_training_state_registry.py",
        source_text=source_text,
    )
    assert dependency_boundary_violations


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("import random", False),
        ("import numpy", False),
        ("from dataclasses import asdict", False),
        ("from torch import Tensor", False),
        ("from torch import rand", False),
        ("from torch.nn import Linear", False),
        ("from torch.optim import AdamW", False),
        ("from .joint_model_parameter_update import perform_joint_model_parameter_update", False),
        ("from .held_model_training_binding import HeldModelTrainingBinding", False),
        ("from .parameter_optimizer_construction import create_parameter_optimizer", False),
        ("from .parameter_optimizer_state import _private", False),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import _validate_classifier_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import _private",
            False,
        ),
        ("from federated_learning_experiments.runtime import run", False),
        ("import federated_drift_experiment", False),
        ("from . import parameter_optimizer_state", False),
        ("from dataclasses import dataclass", False),
        ("from torch import float32, strided", True),
        ("from torch.nn import Parameter", True),
        ("from torch.optim import Adam, SGD", True),
        ("from .parameter_optimizer_state import ParameterOptimizerState", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
            True,
        ),
    ],
)
def test_adopted_candidate_shared_feature_integration_dependency_contract(
    source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/adopted_candidate_shared_feature_integration.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("import random", False),
        ("import numpy", False),
        ("from dataclasses import asdict", False),
        ("from torch import Tensor", False),
        ("from torch import rand", False),
        ("from torch.nn import Linear", False),
        ("from torch.optim import AdamW", False),
        ("from .joint_model_parameter_update import perform_joint_model_parameter_update", False),
        ("from .held_model_training_binding import HeldModelTrainingBinding", False),
        ("from .parameter_optimizer_construction import create_parameter_optimizer", False),
        ("from .parameter_optimizer_state import _private", False),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import _validate_classifier_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import _private",
            False,
        ),
        ("from federated_learning_experiments.runtime import run", False),
        ("import federated_drift_experiment", False),
        ("from . import parameter_optimizer_state", False),
        ("from dataclasses import dataclass", True),
        ("from torch import float32, strided", True),
        ("from torch.nn import Parameter", True),
        ("from torch.optim import Adam, SGD", True),
        ("from .parameter_optimizer_state import ParameterOptimizerState", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
            True,
        ),
    ],
)
def test_held_model_shared_feature_reconnection_dependency_contract(
    source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/held_model_shared_feature_reconnection.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


def is_configuration_foundation_module(*, source_module_path):
    """設定基盤の実装と機能別の設定宣言を選ぶ。"""
    return source_module_path.startswith(("core/", "configuration/")) or (
        source_module_path.startswith(("learning/", "methods/"))
        and source_module_path.endswith("_settings.py")
    )


def resolve_imported_module_names(*, import_statement, importing_package_name):
    """fromの別名も展開し、package経由の禁止モジュールを見逃さない。"""
    if isinstance(import_statement, ast.Import):
        return tuple(imported_module_alias.name for imported_module_alias in import_statement.names)
    if not isinstance(import_statement, ast.ImportFrom):
        return ()
    relative_import_prefix = "." * import_statement.level
    imported_base_module_name = resolve_name(
        relative_import_prefix + (import_statement.module or ""),
        importing_package_name,
    )
    # package名自体は依存先の層を特定しないため、選択された属性まで確認する。
    if imported_base_module_name in (
        "federated_learning_experiments",
        "federated_learning_experiments.configuration",
        "federated_learning_experiments.core",
        "federated_learning_experiments.data",
        "federated_learning_experiments.data.concept_schedules",
        "federated_learning_experiments.data.sine",
        "federated_learning_experiments.execution",
        "federated_learning_experiments.learning",
        "federated_learning_experiments.learning.models",
        "federated_learning_experiments.runtime",
    ):
        return tuple(
            imported_base_module_name + "." + imported_module_alias.name
            for imported_module_alias in import_statement.names
        )
    return (imported_base_module_name,) + tuple(
        imported_base_module_name + "." + imported_module_alias.name
        for imported_module_alias in import_statement.names
    )


def dependency_is_allowed(*, source_module_path, imported_module_name):
    """各層の依存方向と数値ライブラリを参照できる場所を判定する。"""
    if source_module_path == "evaluation/adahedge_diagnostic_evidence_collection.py":
        return imported_module_name in (
            "federated_learning_experiments.evaluation.adahedge_diagnostic_evidence.AdaHedgeDiagnosticEvidence",
        )
    if source_module_path == "runtime/training_assignment_diagnostic_notification.py":
        return imported_module_name in (
            "federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection.AdaHedgeDiagnosticEvidenceCollection",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
        )
    if source_module_path == "evaluation/adahedge_diagnostic_evidence.py":
        return imported_module_name in (
            "math",
            "collections.abc.Iterable",
            "collections.abc.Mapping",
        )
    if source_module_path == "evaluation/adaptation_record_store.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "dataclasses.replace",
            "typing.Literal",
            "typing.get_args",
        )
    if source_module_path == "runtime/released_pending_sample_assignment.py":
        return imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
            "federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
        )
    if source_module_path == "runtime/held_model_training_request_handling.py":
        return imported_module_name in (
            "random.Random",
            "torch.optim.Optimizer",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.training.held_model_joint_training_iterations.perform_held_model_joint_training_iterations",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.local_training_request_schedule.LocalTrainingRequestSchedule",
            "federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
        )
    if (
        source_module_path
        == "methods/fedsda/training_data_assignment/pending_sample_observation_store.py"
    ):
        return imported_module_name in (
            "federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
        )
    if source_module_path == "learning/training/shared_parameter_optimizer_state_holder.py":
        return imported_module_name in (
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
        )
    if source_module_path == "runtime/server_model_registration_and_aggregation.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.Tensor",
            "federated_learning_experiments.evaluation.communication_volume_record_store.CommunicationVolumeRecordStore",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.BoundedLossMoments",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
            "federated_learning_experiments.learning.loss_statistics.server_loss_mean_aggregation.aggregate_participating_client_loss_means",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            "federated_learning_experiments.methods.fedsda.model_registration.global_model_repository.GlobalModelRepository",
            "federated_learning_experiments.runtime.fedsda_run_client.FedsdaRunClient",
            "federated_learning_experiments.runtime.held_model_registration_confirmation.confirm_held_model_registration",
        )
    if (
        source_module_path
        == "methods/fedsda/candidate_model_selection/candidate_validation_decision_record_store.py"
    ):
        return imported_module_name in (
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record.IncompletePostAlarmCandidateValidationDecisionRecord",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record.PostAlarmCandidateValidationDecisionRecord",
        )
    if source_module_path == "methods/fedsda/consolidation/model_clustering_calculations.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "math.isfinite",
            "math.sqrt",
            "statistics.NormalDist",
        )
    if source_module_path == "methods/fedsda/model_registration/global_model_repository.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.Tensor",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
        )
    if source_module_path == "learning/models/model_computation_measurement.py":
        return imported_module_name in (
            "collections.abc.Iterator",
            "contextlib.contextmanager",
            "dataclasses.dataclass",
            "dataclasses.fields",
            "weakref.ReferenceType",
            "weakref.ref",
            "torch.Tensor",
            "torch.is_grad_enabled",
            "torch.nn.Linear",
            "torch.nn.Module",
            "torch.nn.modules.module.register_module_forward_hook",
            "torch.nn.modules.module.register_module_forward_pre_hook",
            "torch.optim.Optimizer",
            "torch.optim.optimizer.register_optimizer_step_post_hook",
            "federated_learning_experiments.learning.models.nonlinear_residual_adapter.NonlinearResidualAdapter",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
        )
    if source_module_path == "evaluation/loss_monitoring_computation_count_store.py":
        return imported_module_name in ("dataclasses.dataclass",)
    if source_module_path == "evaluation/held_model_count_record_store.py":
        return imported_module_name in ()
    if source_module_path == "evaluation/computation_cost_summary.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "dataclasses.fields",
            "federated_learning_experiments.learning.models.model_computation_measurement.ModelComputationCounts",
        )
    if source_module_path == "evaluation/run_metric_calculations.py":
        return imported_module_name in (
            "bisect.bisect_right",
            "dataclasses.dataclass",
        )
    if source_module_path == "evaluation/model_clustering_record_store.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "dataclasses.replace",
            "math.isnan",
        )
    if source_module_path == "evaluation/cross_evaluation_record_store.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "dataclasses.replace",
            "math.isfinite",
        )
    if source_module_path == "evaluation/communication_volume_record_store.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.Tensor",
        )
    if source_module_path == "runtime/initial_model_pretraining.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "random.Random",
            "numpy.random.RandomState",
            "torch.float32",
            "torch.is_grad_enabled",
            "torch.tensor",
            "federated_learning_experiments.data.observed_streams.ObservedSample",
            "federated_learning_experiments.data.sine.sine_sample_generation.SineSampleGenerator",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.model_architecture_settings.ModelArchitectureSettings",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            "federated_learning_experiments.learning.training.initial_model_pretraining_settings.InitialModelPretrainingSettings",
            "federated_learning_experiments.learning.training.joint_model_parameter_update.perform_joint_model_parameter_update",
            "federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.learning.training.participating_model_training_batch.ParticipatingModelTrainingBatch",
        )
    if source_module_path == "learning/training/initial_model_pretraining_settings.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "dataclasses.field",
            "federated_learning_experiments.core.configuration_errors.RunSettingsValidationError",
            "federated_learning_experiments.core.settings_field_validation.validate_settings_field_values",
        )
    if source_module_path == "runtime/fedsda_measured_run_execution.py":
        return imported_module_name in (
            "federated_learning_experiments.execution.run_participant_contracts.RunServerOperations",
            "dataclasses.dataclass",
            "federated_learning_experiments.configuration.experiment_run_conditions.ExperimentRunConditions",
            "federated_learning_experiments.data.sine.sine_sample_generation.SineSampleGenerator",
            "federated_learning_experiments.execution.run_execution_records.StreamProtocolRunResult",
            "federated_learning_experiments.execution.run_participant_contracts.RunParticipants",
            "federated_learning_experiments.execution.run_random_sources.RunRandomSources",
            "federated_learning_experiments.execution.stream_protocol_execution_settings.StreamProtocolExecutionSettings",
            "federated_learning_experiments.learning.models.model_computation_measurement.ModelComputationCounts",
            "federated_learning_experiments.learning.models.model_computation_measurement.ModelComputationMeter",
            "federated_learning_experiments.learning.models.model_computation_measurement.measure_model_computation",
            "federated_learning_experiments.learning.models.model_computation_measurement.subtract_model_computation_counts",
            "federated_learning_experiments.runtime.fedsda_run_participant_factory.FedsdaRunParticipantFactory",
            "federated_learning_experiments.runtime.fedsda_run_participant_factory.FedsdaRunParticipantSettings",
            "federated_learning_experiments.runtime.single_run_execution.execute_stream_protocol_run",
        )
    if source_module_path == "runtime/fedsda_run_metric_derivation.py":
        return imported_module_name in (
            "typing.cast",
            "federated_learning_experiments.evaluation.computation_cost_summary.ComputationCostSummary",
            "federated_learning_experiments.evaluation.computation_cost_summary.ServerComputationCounts",
            "federated_learning_experiments.evaluation.computation_cost_summary.summarize_computation_cost",
            "federated_learning_experiments.evaluation.loss_monitoring_computation_count_store.LossMonitoringComputationCounts",
            "federated_learning_experiments.learning.models.model_computation_measurement.ModelComputationCounts",
            "dataclasses.dataclass",
            "torch.Tensor",
            "federated_learning_experiments.evaluation.communication_volume_record_store.CommunicationVolumeSnapshot",
            "federated_learning_experiments.evaluation.run_metric_calculations.DetectionMetrics",
            "federated_learning_experiments.evaluation.run_metric_calculations.RunMetricSettings",
            "federated_learning_experiments.evaluation.run_metric_calculations.calculate_detection_metrics",
            "federated_learning_experiments.evaluation.run_metric_calculations.calculate_prediction_accuracy",
            "federated_learning_experiments.evaluation.run_metric_calculations.calculate_stable_period_prediction_accuracy",
            "federated_learning_experiments.evaluation.run_metric_calculations.extract_concept_change_sample_indices",
            "federated_learning_experiments.execution.run_execution_records.StreamProtocolRunResult",
            "federated_learning_experiments.execution.run_participant_contracts.RunParticipants",
            "federated_learning_experiments.methods.fedsda.model_registration.global_model_repository.GlobalModelRepository",
            "federated_learning_experiments.runtime.fedsda_run_client.FedsdaRunClient",
            "federated_learning_experiments.runtime.fedsda_run_server.FedsdaRunServer",
            "federated_learning_experiments.runtime.server_model_registration_and_aggregation.split_shared_and_concept_specific_parameters",
        )
    if source_module_path == "runtime/fedsda_run_participant_factory.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "typing.cast",
            "federated_learning_experiments.configuration.experiment_run_conditions.ExperimentRunConditions",
            "federated_learning_experiments.core.configuration_errors.RunSettingsValidationError",
            "federated_learning_experiments.data.sine.sine_sample_generation.SineSampleGenerator",
            "federated_learning_experiments.execution.run_participant_contracts.RunClientOperations",
            "federated_learning_experiments.execution.run_participant_contracts.RunParticipants",
            "federated_learning_experiments.execution.run_random_sources.RunRandomSources",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.models.model_architecture_settings.ModelArchitectureSettings",
            "federated_learning_experiments.learning.training.initial_model_pretraining_settings.InitialModelPretrainingSettings",
            "federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations.ModelClusteringCriteria",
            "federated_learning_experiments.runtime.fedsda_run_client.assemble_fedsda_run_client",
            "federated_learning_experiments.runtime.fedsda_run_client_settings.FedsdaRunClientSettings",
            "federated_learning_experiments.runtime.fedsda_run_server.assemble_fedsda_run_server",
            "federated_learning_experiments.runtime.initial_model_pretraining.pretrain_initial_model",
        )
    if source_module_path == "runtime/fedsda_run_server.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "random.Random",
            "torch.Tensor",
            "federated_learning_experiments.evaluation.communication_volume_record_store.CommunicationVolumeRecordStore",
            "federated_learning_experiments.evaluation.cross_evaluation_record_store.CrossEvaluationRecordStore",
            "federated_learning_experiments.evaluation.model_clustering_record_store.ModelClusteringRecordStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
            "federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations.ModelClusteringCriteria",
            "federated_learning_experiments.methods.fedsda.model_registration.global_model_repository.GlobalModelRepository",
            "federated_learning_experiments.runtime.fedsda_run_client.FedsdaRunClient",
            "federated_learning_experiments.runtime.server_round_synchronization.ServerRoundSynchronization",
            "federated_learning_experiments.runtime.server_round_synchronization.synchronize_models_in_server_round",
        )
    if source_module_path == "runtime/server_round_synchronization.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "random.Random",
            "federated_learning_experiments.evaluation.communication_volume_record_store.CommunicationVolumeRecordStore",
            "federated_learning_experiments.evaluation.cross_evaluation_record_store.CrossEvaluationRecordStore",
            "federated_learning_experiments.evaluation.model_clustering_record_store.ModelClusteringRecordStore",
            "federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations.ModelClusteringCriteria",
            "federated_learning_experiments.methods.fedsda.model_registration.global_model_repository.GlobalModelRepository",
            "federated_learning_experiments.runtime.fedsda_run_client.FedsdaRunClient",
            "federated_learning_experiments.runtime.global_model_distribution.distribute_global_models_to_clients",
            "federated_learning_experiments.runtime.global_model_distribution_application.GlobalModelDistributionApplication",
            "federated_learning_experiments.runtime.model_clustering_and_consolidation.ModelConsolidation",
            "federated_learning_experiments.runtime.model_clustering_and_consolidation.cluster_and_consolidate_global_models",
            "federated_learning_experiments.runtime.model_cross_evaluation.ModelCrossEvaluation",
            "federated_learning_experiments.runtime.model_cross_evaluation.cross_evaluate_global_models",
            "federated_learning_experiments.runtime.post_aggregation_prediction_recalibration.PostAggregationPredictionRecalibration",
            "federated_learning_experiments.runtime.server_model_registration_and_aggregation.ClientModelAggregation",
            "federated_learning_experiments.runtime.server_model_registration_and_aggregation.RegisteredClientModel",
            "federated_learning_experiments.runtime.server_model_registration_and_aggregation.aggregate_client_models_into_global_models",
            "federated_learning_experiments.runtime.server_model_registration_and_aggregation.register_ready_client_models",
        )
    if source_module_path == "runtime/model_clustering_and_consolidation.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "math.inf",
            "math.sqrt",
            "torch.Tensor",
            "federated_learning_experiments.evaluation.model_clustering_record_store.ModelClusteringObservation",
            "federated_learning_experiments.evaluation.model_clustering_record_store.ModelClusteringRecordStore",
            "federated_learning_experiments.evaluation.model_clustering_record_store.ModelPairClusteringObservation",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.BoundedLossMoments",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
            "federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations.ModelClusteringCriteria",
            "federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations.cluster_model_ids_by_average_linkage",
            "federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations.compute_classwise_unique_correctness_decision_score",
            "federated_learning_experiments.methods.fedsda.model_registration.global_model_repository.GlobalModelRepository",
            "federated_learning_experiments.runtime.fedsda_run_client.FedsdaRunClient",
            "federated_learning_experiments.runtime.model_cross_evaluation.CrossEvaluationLossSums",
            "federated_learning_experiments.runtime.model_cross_evaluation.ModelCrossEvaluation",
            "federated_learning_experiments.runtime.server_model_registration_and_aggregation.split_shared_and_concept_specific_parameters",
        )
    if source_module_path == "runtime/model_cross_evaluation.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "random.Random",
            "federated_learning_experiments.evaluation.communication_volume_record_store.CommunicationVolumeRecordStore",
            "federated_learning_experiments.evaluation.cross_evaluation_record_store.ClientCrossEvaluationRecord",
            "federated_learning_experiments.evaluation.cross_evaluation_record_store.CrossEvaluationRecordStore",
            "federated_learning_experiments.methods.fedsda.model_registration.global_model_repository.GlobalModelRepository",
            "federated_learning_experiments.runtime.client_model_cross_evaluation.ClientModelCrossEvaluation",
            "federated_learning_experiments.runtime.client_model_cross_evaluation.ModelPairCorrectnessCounts",
            "federated_learning_experiments.runtime.fedsda_run_client.FedsdaRunClient",
            "federated_learning_experiments.runtime.server_model_registration_and_aggregation.split_shared_and_concept_specific_parameters",
        )
    if source_module_path == "runtime/client_model_cross_evaluation.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "random.Random",
            "torch.Tensor",
            "torch.cat",
            "torch.float32",
            "torch.no_grad",
            "torch.strided",
            "torch.unique",
            "federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.prediction.class_probability_calculations.predict_class_labels_from_prediction_scores",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses_and_outputs",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
        )
    if source_module_path == "runtime/post_aggregation_prediction_recalibration.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.cat",
            "federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection.AdaHedgeDiagnosticEvidenceCollection",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifiers_per_sample_bounded_losses_from_shared_features",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights.FixedSharePredictionWeightController",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store.PendingSampleObservationStore",
        )
    if source_module_path == "runtime/global_model_distribution.py":
        return imported_module_name in (
            "federated_learning_experiments.evaluation.communication_volume_record_store.CommunicationVolumeRecordStore",
            "federated_learning_experiments.methods.fedsda.model_registration.global_model_repository.GlobalModelRepository",
            "federated_learning_experiments.runtime.fedsda_run_client.FedsdaRunClient",
            "federated_learning_experiments.runtime.global_model_distribution_application.GlobalModelDistributionApplication",
            "federated_learning_experiments.runtime.server_model_registration_and_aggregation.split_shared_and_concept_specific_parameters",
        )
    if source_module_path == "runtime/global_model_distribution_application.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "random.Random",
            "torch.Tensor",
            "torch.float32",
            "torch.strided",
            "federated_learning_experiments.evaluation.adaptation_record_store.SERVER_REMAP_ADAPTATION_OUTCOME",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecord",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecordStore",
            "federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection.select_loss_statistics_after_model_id_mapping",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            "federated_learning_experiments.learning.training.held_model_shared_feature_reconnection.HeldModelOptimizerBinding",
            "federated_learning_experiments.learning.training.held_model_shared_feature_reconnection.reconnect_held_models_to_shared_feature_extractor",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingState",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.learning.training.shared_parameter_optimizer_state_holder.SharedParameterOptimizerStateHolder",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
        )
    if source_module_path == "runtime/fedsda_run_client.py":
        return imported_module_name in (
            "federated_learning_experiments.evaluation.held_model_count_record_store.HeldModelCountRecordStore",
            "federated_learning_experiments.evaluation.loss_monitoring_computation_count_store.LossMonitoringComputationCountStore",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_validation_decision_record_store.CandidateValidationDecisionRecordStore",
            "copy.deepcopy",
            "dataclasses.dataclass",
            "random.Random",
            "torch.Tensor",
            "torch.float32",
            "torch.tensor",
            "federated_learning_experiments.data.observed_streams.ObservedSample",
            "federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection.AdaHedgeDiagnosticEvidenceCollection",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecordStore",
            "federated_learning_experiments.evaluation.loss_change_alarm_record_store.LossChangeAlarmRecordStore",
            "federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            "federated_learning_experiments.evaluation.sample_prediction_record_store.SamplePredictionRecordStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
            "federated_learning_experiments.learning.training.local_training_request_schedule.LocalTrainingRequestSchedule",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.learning.training.shared_parameter_optimizer_state_holder.SharedParameterOptimizerStateHolder",
            "federated_learning_experiments.learning.training.temporary_model_id_allocation.TemporaryModelIdAllocator",
            "federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring.OverallAndTrueClassLossMonitor",
            "federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.select_loss_monitoring_baseline_mean_loss",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
            "federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights.FixedSharePredictionWeightController",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store.PendingSampleObservationStore",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
            "federated_learning_experiments.runtime.candidate_validation_session_holder.CandidateValidationSessionHolder",
            "federated_learning_experiments.runtime.client_model_cross_evaluation.ClientModelCrossEvaluation",
            "federated_learning_experiments.runtime.client_model_cross_evaluation.evaluate_candidate_model_on_target_model_samples",
            "federated_learning_experiments.runtime.fedsda_run_client_settings.FedsdaRunClientSettings",
            "federated_learning_experiments.runtime.global_model_distribution_application.GlobalModelDistributionApplication",
            "federated_learning_experiments.runtime.global_model_distribution_application.apply_global_model_distribution",
            "federated_learning_experiments.runtime.held_candidate_validation_progress.HeldIncompleteCandidateValidationFinalization",
            "federated_learning_experiments.runtime.held_candidate_validation_progress.finalize_held_incomplete_candidate_validation",
            "federated_learning_experiments.runtime.held_model_training_request_handling.train_held_models_for_pending_training_requests",
            "federated_learning_experiments.runtime.observed_sample_processing.ObservedSampleProcessing",
            "federated_learning_experiments.runtime.observed_sample_processing.process_observed_sample",
            "federated_learning_experiments.runtime.post_aggregation_prediction_recalibration.PostAggregationPredictionRecalibration",
            "federated_learning_experiments.runtime.post_aggregation_prediction_recalibration.recalibrate_prediction_state_after_aggregation",
        )
    if source_module_path == "runtime/fedsda_run_client_settings.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "dataclasses.field",
            "federated_learning_experiments.core.configuration_errors.RunSettingsValidationError",
            "federated_learning_experiments.core.settings_field_validation.validate_settings_field_values",
            "federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
            "federated_learning_experiments.learning.training.local_training_schedule_settings.LocalTrainingScheduleSettings",
            "federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings",
            "federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings.LossChangeDetectionSettings",
            "federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings.PredictionCombinationSettings",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings.TrainingDataAssignmentSettings",
        )
    if source_module_path == "runtime/observed_sample_prediction.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.Tensor",
            "torch.float32",
            "torch.isfinite",
            "torch.no_grad",
            "torch.strided",
            "federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection.AdaHedgeDiagnosticEvidenceCollection",
            "federated_learning_experiments.evaluation.sample_prediction_record_store.SamplePredictionRecord",
            "federated_learning_experiments.evaluation.sample_prediction_record_store.SamplePredictionRecordStore",
            "federated_learning_experiments.learning.prediction.class_probability_calculations.combine_model_prediction_probabilities",
            "federated_learning_experiments.learning.prediction.class_probability_calculations.compute_model_mean_bounded_losses_after_label_observation",
            "federated_learning_experiments.learning.prediction.class_probability_calculations.convert_model_outputs_to_prediction_probabilities",
            "federated_learning_experiments.learning.prediction.class_probability_calculations.normalize_model_prediction_weights",
            "federated_learning_experiments.learning.prediction.class_probability_calculations.predict_class_labels_from_prediction_scores",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights.FixedSharePredictionWeightController",
        )
    if source_module_path == "evaluation/sample_prediction_record_store.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "math.isfinite",
        )
    if source_module_path == "evaluation/loss_change_alarm_record_store.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "math.inf",
            "math.isnan",
        )
    if source_module_path == "runtime/observed_sample_processing.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "random.Random",
            "torch.Tensor",
            "torch.optim.Optimizer",
            "federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection.AdaHedgeDiagnosticEvidenceCollection",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecordStore",
            "federated_learning_experiments.evaluation.loss_change_alarm_record_store.LossChangeAlarmRecordStore",
            "federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            "federated_learning_experiments.evaluation.sample_prediction_record_store.SamplePredictionRecordStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            "federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
            "federated_learning_experiments.learning.training.local_training_request_schedule.LocalTrainingRequestSchedule",
            "federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.temporary_model_id_allocation.TemporaryModelIdAllocator",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings",
            "federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring.LossMonitoringObservation",
            "federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring.OverallAndTrueClassLossMonitor",
            "federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.select_loss_monitoring_baseline_mean_loss",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store.PendingSampleObservationStore",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
            "federated_learning_experiments.runtime.alarm_occurrence_handling.AlarmOccurrenceHandling",
            "federated_learning_experiments.runtime.alarm_occurrence_handling.handle_alarm_occurrence",
            "federated_learning_experiments.runtime.candidate_validation_session_holder.CandidateValidationSessionHolder",
            "federated_learning_experiments.runtime.held_candidate_validation_progress.HeldCandidateValidationAdvance",
            "federated_learning_experiments.runtime.held_candidate_validation_progress.advance_held_candidate_validation",
            "federated_learning_experiments.runtime.held_model_training_request_handling.record_training_request_and_train_held_models_when_due",
            "federated_learning_experiments.runtime.held_model_training_request_handling.train_held_models_for_pending_training_requests",
            "federated_learning_experiments.runtime.observed_sample_prediction.ObservedSamplePrediction",
            "federated_learning_experiments.runtime.observed_sample_prediction.predict_observed_sample_and_update_prediction_weights",
            "federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights.FixedSharePredictionWeightController",
            "federated_learning_experiments.runtime.released_pending_sample_assignment.assign_released_pending_samples_to_current_training_model",
        )
    if source_module_path == "runtime/alarm_occurrence_handling.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "random.Random",
            "federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection.AdaHedgeDiagnosticEvidenceCollection",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecord",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecordStore",
            "federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings",
            "federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring.OverallAndTrueClassLossMonitor",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
            "federated_learning_experiments.runtime.alarm_adaptation_recording.record_completed_alarm_response",
            "federated_learning_experiments.runtime.alarm_buffer_response.respond_to_alarm_with_buffered_samples",
            "federated_learning_experiments.runtime.alarm_response_completion.AlarmResponseCompletion",
            "federated_learning_experiments.runtime.alarm_response_completion.complete_alarm_buffer_response",
            "federated_learning_experiments.runtime.candidate_validation_session_holder.CandidateValidationSessionHolder",
            "federated_learning_experiments.runtime.held_candidate_validation_progress.apply_alarm_response_to_validation_session_holder",
            "federated_learning_experiments.runtime.training_assignment_diagnostic_notification.notify_diagnostics_of_training_assignment_change",
        )
    if source_module_path == "runtime/candidate_validation_session_holder.py":
        return imported_module_name in (
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
        )
    if source_module_path == "runtime/held_candidate_validation_progress.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.Tensor",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecord",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecordStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.temporary_model_id_allocation.TemporaryModelIdAllocator",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
            "federated_learning_experiments.runtime.alarm_buffer_response.AlarmBufferResponse",
            "federated_learning_experiments.runtime.candidate_validation_adaptation_recording.record_completed_candidate_validation",
            "federated_learning_experiments.runtime.candidate_validation_adaptation_recording.record_incomplete_candidate_validation_finalization",
            "federated_learning_experiments.runtime.candidate_validation_session_holder.CandidateValidationSessionHolder",
            "federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization.IncompletePostAlarmCandidateValidationFinalization",
            "federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization.finalize_incomplete_post_alarm_candidate_validation",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_progress.PostAlarmCandidateValidationProgress",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_progress.advance_post_alarm_candidate_validation",
            "federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection.AdaHedgeDiagnosticEvidenceCollection",
            "federated_learning_experiments.runtime.training_assignment_diagnostic_notification.notify_diagnostics_of_training_assignment_change",
        )
    if source_module_path == "runtime/candidate_validation_adaptation_recording.py":
        return imported_module_name in (
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationOutcome",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecord",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecordStore",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record.IncompletePostAlarmCandidateValidationDecisionRecord",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record.PostAlarmCandidateValidationDecisionRecord",
            "federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization.IncompletePostAlarmCandidateValidationFinalization",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_progress.PostAlarmCandidateValidationCompletion",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution.PostAlarmCandidateValidationResolution",
        )
    if source_module_path == "runtime/alarm_adaptation_recording.py":
        return imported_module_name in (
            "typing.cast",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationOutcome",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecord",
            "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecordStore",
            "federated_learning_experiments.runtime.alarm_response_completion.AlarmResponseCompletion",
        )
    if source_module_path == "learning/training/indexed_observed_training_sample.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
        )
    if (
        source_module_path
        == "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py"
    ):
        return imported_module_name in (
            "dataclasses.dataclass",
            "federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
        )
    if source_module_path == "runtime/alarm_training_interval_preparation.py":
        return imported_module_name in (
            "random.Random",
            "torch.Tensor",
            "federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals.PreparedAlarmTrainingIntervals",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
            "federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            "federated_learning_experiments.evaluation.model_evaluation_sample_records.ObservedEvaluationSample",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            "federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
        )
    if source_module_path == "runtime/alarm_response_completion.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            "federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring.OverallAndTrueClassLossMonitor",
            "federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.select_loss_monitoring_baseline_mean_loss",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
            "federated_learning_experiments.runtime.alarm_buffer_response.AlarmBufferResponse",
        )
    if source_module_path == "runtime/alarm_buffer_response.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "random.Random",
            "federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer",
            "federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals.PreparedAlarmTrainingIntervals",
            "federated_learning_experiments.runtime.alarm_change_interval_resolution.ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES",
            "federated_learning_experiments.runtime.alarm_change_interval_resolution.AlarmChangeIntervalResolution",
            "federated_learning_experiments.runtime.alarm_change_interval_resolution.resolve_alarm_change_interval",
            "federated_learning_experiments.runtime.alarm_training_interval_preparation.prepare_alarm_training_intervals",
            "federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
        )
    if source_module_path == "runtime/alarm_change_interval_resolution.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.Tensor",
            "torch.cat",
            "torch.float32",
            "torch.strided",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment.AlarmIntervalModelReuseAssessment",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization.select_candidate_initial_parameter_snapshot",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings",
            "federated_learning_experiments.runtime.alarm_interval_model_reuse_assessment.evaluate_held_models_for_alarm_interval_reuse",
            "federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.start_post_alarm_candidate_validation_session",
        )
    if source_module_path == "runtime/alarm_interval_model_reuse_assessment.py":
        return imported_module_name in (
            "math.isfinite",
            "torch.Tensor",
            "torch.mean",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment.AlarmIntervalModelReuseAssessment",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment.assess_alarm_interval_model_reuse",
            "federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.select_alarm_interval_reuse_baseline_mean_loss",
        )
    if source_module_path == (
        "methods/fedsda/candidate_model_selection/alarm_interval_model_reuse_assessment.py"
    ):
        return imported_module_name in (
            "dataclasses.dataclass",
            "math.isfinite",
        )
    if source_module_path == "runtime/incomplete_post_alarm_candidate_validation_finalization.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record.IncompletePostAlarmCandidateValidationDecisionRecord",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
            "federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
        )
    if source_module_path == (
        "methods/fedsda/candidate_model_selection/incomplete_post_alarm_candidate_validation_decision_record.py"
    ):
        return imported_module_name == "dataclasses.dataclass"
    if source_module_path == "runtime/post_alarm_candidate_validation_progress.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "math.isfinite",
            "torch.Tensor",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.temporary_model_id_allocation.TemporaryModelIdAllocator",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation.evaluate_candidate_using_post_alarm_losses",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record.PostAlarmCandidateValidationDecisionRecord",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution.PostAlarmCandidateValidationResolution",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution.apply_post_alarm_candidate_validation_resolution",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation.observe_post_alarm_candidate_validation_sample",
            "federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
        )
    if (
        source_module_path
        == "methods/fedsda/candidate_model_selection/post_alarm_candidate_validation_decision_record.py"
    ):
        return imported_module_name in (
            "dataclasses.dataclass",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation.PostAlarmCandidateLossEvaluation",
        )
    if source_module_path == "runtime/post_alarm_candidate_validation_session_start.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.Tensor",
            "torch.float32",
            "torch.is_grad_enabled",
            "torch.isfinite",
            "torch.strided",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.candidate_epoch_training.CandidateEpochTrainingResult",
            "federated_learning_experiments.learning.training.candidate_epoch_training.train_candidate_classifier_epochs",
            "federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection.PostAlarmCandidateLossCollection",
            "federated_learning_experiments.runtime.candidate_classifier_construction.IndependentCandidateTrainingState",
            "federated_learning_experiments.runtime.candidate_classifier_construction.create_independent_candidate_training_state",
            "federated_learning_experiments.runtime.post_alarm_reference_model_fixation.FixedPostAlarmReferenceModels",
            "federated_learning_experiments.runtime.post_alarm_reference_model_fixation.fix_reference_models_at_alarm",
        )
    if source_module_path == "learning/training/candidate_epoch_training_settings.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "dataclasses.field",
            "federated_learning_experiments.core.settings_field_validation.validate_settings_field_values",
        )
    if source_module_path == "learning/training/candidate_epoch_training.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "math.inf",
            "torch.Tensor",
            "torch.float32",
            "torch.is_grad_enabled",
            "torch.isfinite",
            "torch.randperm",
            "torch.strided",
            "torch.optim.SGD",
            "torch.optim.Adam",
            "torch.utils.data.DataLoader",
            "torch.utils.data.TensorDataset",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            "federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
            "federated_learning_experiments.learning.training.joint_model_parameter_update.perform_joint_model_parameter_update",
            "federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.learning.training.participating_model_training_batch.ParticipatingModelTrainingBatch",
        )
    if source_module_path == "runtime/candidate_classifier_construction.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.Tensor",
            "torch.float32",
            "torch.isfinite",
            "torch.strided",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
        )
    if source_module_path == "learning/training/temporary_model_id_allocation.py":
        return imported_module_name == "__future__.annotations"
    if source_module_path == "runtime/post_alarm_reference_model_fixation.py":
        return imported_module_name in (
            "__future__.annotations",
            "dataclasses.dataclass",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.select_post_alarm_reference_historical_mean_loss",
        )
    if source_module_path == "runtime/post_alarm_candidate_validation_sample_observation.py":
        return imported_module_name in (
            "__future__.annotations",
            "torch.Tensor",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection.PostAlarmCandidateLossCollection",
        )
    if source_module_path == "runtime/post_alarm_candidate_validation_resolution.py":
        return imported_module_name in (
            "__future__.annotations",
            "dataclasses.dataclass",
            "torch.Tensor",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation.PostAlarmCandidateLossEvaluation",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            "federated_learning_experiments.learning.training.temporary_model_id_allocation.TemporaryModelIdAllocator",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
            "federated_learning_experiments.runtime.adopted_candidate_local_adoption.adopt_candidate_as_current_training_model",
            "federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model",
        )
    if source_module_path == "runtime/assigned_training_sample_absorption.py":
        return imported_module_name in (
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.validate_classifier_bounded_loss_inputs",
            "__future__.annotations",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
        )
    if source_module_path == "runtime/adopted_candidate_local_adoption.py":
        return imported_module_name in (
            "__future__.annotations",
            "torch.Tensor",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.learning.training.temporary_model_id_allocation.TemporaryModelIdAllocator",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
            "federated_learning_experiments.runtime.adopted_candidate_initial_local_registration.register_adopted_candidate_as_temporary_held_model",
        )
    if source_module_path == "runtime/adopted_candidate_initial_local_registration.py":
        return imported_module_name in (
            "__future__.annotations",
            "torch.Tensor",
            "federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization.initialize_model_and_class_loss_statistics_from_batch",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses",
            "federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration.integrate_adopted_candidate_shared_features",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingState",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
        )
    if source_module_path == "runtime/held_model_registration_confirmation.py":
        return imported_module_name in (
            "__future__.annotations",
            "federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            "federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange",
            "federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            "federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            "federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            "federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
        )
    if source_module_path in (
        "learning/training/model_training_and_assignment_counts.py",
        "learning/training/current_training_model_assignment.py",
    ):
        return imported_module_name in ("__future__.annotations", "dataclasses.dataclass")
    if source_module_path == "evaluation/model_evaluation_sample_records.py":
        return imported_module_name in (
            "__future__.annotations",
            "dataclasses.dataclass",
            "torch.Tensor",
        )
    if source_module_path == "evaluation/model_evaluation_sample_store.py":
        return imported_module_name in (
            "__future__.annotations",
            "random.Random",
            "federated_learning_experiments.evaluation.model_evaluation_sample_records.ObservedEvaluationSample",
            "federated_learning_experiments.evaluation.model_evaluation_sample_records.ModelEvaluationSampleCollection",
        )
    if source_module_path == "methods/fedsda/model_registration/pending_model_upload.py":
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.Tensor",
            "torch.float32",
            "torch.isfinite",
            "torch.strided",
        )
    if source_module_path == "learning/models/classifier_parameter_snapshot.py":
        return imported_module_name in (
            "torch.Tensor",
            "torch.float32",
            "torch.isfinite",
            "torch.no_grad",
            "torch.strided",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
        )
    if source_module_path == "learning/prediction/classifier_bounded_loss_evaluation.py":
        return imported_module_name in (
            "torch.Tensor",
            "torch.abs",
            "torch.float32",
            "torch.isfinite",
            "torch.no_grad",
            "torch.softmax",
            "torch.strided",
            "torch.trunc",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
        )
    if source_module_path == "learning/training/held_model_training_state_registry.py":
        # 一覧構造と公開状態参照だけを許可し、上位処理の呼出しを防ぐ。
        return imported_module_name in (
            "dataclasses.dataclass",
            "torch.float32",
            "torch.strided",
            "torch.nn.Parameter",
            "torch.optim.Adam",
            "torch.optim.SGD",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
            "federated_learning_experiments.learning.training.held_model_training_binding.HeldModelTrainingBinding",
        )
    if source_module_path == "learning/training/adopted_candidate_shared_feature_integration.py":
        # 外側接続はモデルとownerの公開型/属性だけを使い、学習計算へ依存しない。
        return imported_module_name in (
            "torch",
            "torch.float32",
            "torch.strided",
            "torch.nn",
            "torch.nn.Parameter",
            "torch.optim",
            "torch.optim.Adam",
            "torch.optim.SGD",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.training.parameter_optimizer_state",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
        )
    if source_module_path == "learning/training/held_model_shared_feature_reconnection.py":
        # 外側接続はモデルとownerの公開型/属性だけを使い、学習計算へ依存しない。
        return imported_module_name in (
            "dataclasses",
            "dataclasses.dataclass",
            "torch",
            "torch.float32",
            "torch.strided",
            "torch.nn",
            "torch.nn.Parameter",
            "torch.optim",
            "torch.optim.Adam",
            "torch.optim.SGD",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.training.parameter_optimizer_state",
            "federated_learning_experiments.learning.training.parameter_optimizer_state.ParameterOptimizerState",
        )
    if source_module_path == "learning/training/parameter_optimizer_state.py":
        # 状態所有者は公開型・固定設定・既存生成関数だけへ依存する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "torch.nn",
            "torch.nn.Parameter",
            "torch.optim",
            "torch.optim.Optimizer",
            "federated_learning_experiments.learning.training.parameter_optimizer_construction",
            "federated_learning_experiments.learning.training.parameter_optimizer_construction.create_parameter_optimizer",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
        )
    if source_module_path == "learning/training/model_training_sample_store.py":
        # 構造保持は同階層の公開recordのみを参照し、演算や抽出を呼ばない。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "federated_learning_experiments.learning.training.model_training_sample_records",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_records.ModelTrainingSampleCollection",
        )
    if source_module_path == "learning/training/local_training_schedule_settings.py":
        # 機能別の設定宣言と公開値検査だけを許可する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "dataclasses",
            "dataclasses.dataclass",
            "dataclasses.field",
            "federated_learning_experiments.core.settings_field_validation",
            "federated_learning_experiments.core.settings_field_validation.validate_settings_field_values",
        )
    if source_module_path == "learning/training/local_training_request_schedule.py":
        # 要求counterは自分の設定だけを参照し、学習実体や上位をimportしない。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "federated_learning_experiments.learning.training.local_training_schedule_settings",
            "federated_learning_experiments.learning.training.local_training_schedule_settings.LocalTrainingScheduleSettings",
        )
    if source_module_path == "learning/training/held_model_training_binding.py":
        # 借用参照と公開部品の接続だけに限定し、演算・上位層・private依存を拒否する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "dataclasses",
            "dataclasses.dataclass",
            "torch.optim",
            "torch.optim.Optimizer",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
        )
    if source_module_path == "learning/training/held_model_joint_training_iterations.py":
        # 借用参照と公開部品の接続だけに限定し、演算・上位層・private依存を拒否する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "random",
            "random.Random",
            "torch.optim",
            "torch.optim.Optimizer",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.training.held_model_training_binding",
            "federated_learning_experiments.learning.training.held_model_training_binding.HeldModelTrainingBinding",
            "federated_learning_experiments.learning.training.model_training_sample_records",
            "federated_learning_experiments.learning.training.model_training_sample_records.ModelTrainingSampleCollection",
            "federated_learning_experiments.learning.training.participating_model_training_batch",
            "federated_learning_experiments.learning.training.participating_model_training_batch.ParticipatingModelTrainingBatch",
            "federated_learning_experiments.learning.training.local_training_settings",
            "federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings",
            "federated_learning_experiments.learning.training.held_model_training_batch_sampling",
            "federated_learning_experiments.learning.training.held_model_training_batch_sampling.sample_training_batches_for_held_models",
            "federated_learning_experiments.learning.training.joint_model_parameter_update",
            "federated_learning_experiments.learning.training.joint_model_parameter_update.perform_joint_model_parameter_update",
        )
    if source_module_path == "learning/training/model_training_sample_records.py":
        # 標本記録はdataclass宣言と借用Tensor型だけを参照する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "dataclasses",
            "dataclasses.dataclass",
            "torch",
            "torch.Tensor",
        )
    if source_module_path == "learning/training/held_model_training_batch_sampling.py":
        # 抽出は借用Randomと明示Tensor演算・同feature公開記録だけに限定する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "random",
            "random.Random",
            "torch",
            "torch.Tensor",
            "torch.cat",
            "torch.isfinite",
            "torch.float32",
            "torch.strided",
            "federated_learning_experiments.learning.training.model_training_sample_records",
            "federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            "federated_learning_experiments.learning.training.model_training_sample_records.ModelTrainingSampleCollection",
            "federated_learning_experiments.learning.training.model_training_sample_records.SampledModelTrainingBatch",
        )
    if source_module_path == "learning/training/participating_model_training_batch.py":
        # 借用記録は宣言に必要な公開型だけを参照する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "dataclasses",
            "dataclasses.dataclass",
            "torch",
            "torch.Tensor",
            "torch.optim",
            "torch.optim.Optimizer",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
        )
    if source_module_path == "learning/training/joint_model_parameter_update.py":
        # 共同更新は明示演算と兄弟記録/設定・モデル公開型へ限定する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "torch",
            "torch.Tensor",
            "torch.cat",
            "torch.isfinite",
            "torch.no_grad",
            "torch.is_grad_enabled",
            "torch.float32",
            "torch.strided",
            "torch.nn",
            "torch.nn.Parameter",
            "torch.nn.BCELoss",
            "torch.nn.CrossEntropyLoss",
            "torch.optim",
            "torch.optim.Optimizer",
            "torch.optim.Adam",
            "torch.optim.SGD",
            "federated_learning_experiments.learning.training.local_training_settings",
            "federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings",
            "federated_learning_experiments.learning.training.participating_model_training_batch",
            "federated_learning_experiments.learning.training.participating_model_training_batch.ParticipatingModelTrainingBatch",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
        )
    if source_module_path == "learning/training/parameter_optimizer_settings.py":
        # optimizer設定は一般stdlib/core許可より前に公開field検査へ限定する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "dataclasses",
            "dataclasses.dataclass",
            "dataclasses.field",
            "federated_learning_experiments.core.settings_field_validation",
            "federated_learning_experiments.core.settings_field_validation.validate_settings_field_values",
        )
    if source_module_path == "learning/training/parameter_optimizer_construction.py":
        # 生成部は指定Parameter/optimizerと専用公開設定型だけを参照する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "torch",
            "torch.float32",
            "torch.strided",
            "torch.nn",
            "torch.nn.Parameter",
            "torch.optim",
            "torch.optim.Optimizer",
            "torch.optim.Adam",
            "torch.optim.SGD",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
        )
    if source_module_path in (
        "learning/models/shared_feature_extractor.py",
        "learning/models/nonlinear_residual_adapter.py",
        "learning/models/residual_adapter_classifier.py",
    ):
        # モデル構造は一般stdlib許可より先にexact module/public symbolで制限する。
        return imported_module_name in (
            "__future__",
            "__future__.annotations",
            "torch",
            "torch.Tensor",
            "torch.float32",
            "torch.device",
            "torch.strided",
            "torch.nn",
            "torch.nn.Module",
            "torch.nn.Sequential",
            "torch.nn.Linear",
            "torch.nn.ReLU",
            "torch.nn.Sigmoid",
            "torch.nn.Identity",
            "torch.nn.init",
            "torch.nn.init.zeros_",
        ) or (
            source_module_path == "learning/models/residual_adapter_classifier.py"
            and imported_module_name
            in (
                "federated_learning_experiments.learning.models.model_architecture_settings",
                "federated_learning_experiments.learning.models.model_architecture_settings.ModelArchitectureSettings",
                "federated_learning_experiments.learning.models.shared_feature_extractor",
                "federated_learning_experiments.learning.models.shared_feature_extractor.SharedFeatureExtractor",
                "federated_learning_experiments.learning.models.nonlinear_residual_adapter",
                "federated_learning_experiments.learning.models.nonlinear_residual_adapter.NonlinearResidualAdapter",
            )
        )
    if imported_module_name.split(".")[0] in sys.stdlib_module_names:
        return True
    if (
        source_module_path
        == "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py"
    ):
        # 初期snapshot作成だけにtorchと同機能の公開設定型を許可する。
        return imported_module_name in (
            "torch",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings",
        )
    if (
        source_module_path
        == "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py"
    ):
        # 設定宣言は既存の公開field検査関数だけを参照する。
        return imported_module_name in (
            "federated_learning_experiments.core.settings_field_validation",
            "federated_learning_experiments.core.settings_field_validation.validate_settings_field_values",
        )
    if source_module_path == "learning/loss_statistics/server_loss_mean_aggregation.py":
        # サーバ用途の平均集約は公開集計値型だけに依存する。
        return imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.BoundedLossMoments",
        )
    if (
        source_module_path
        == "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py"
    ):
        # ID対応後の選択は公開統計型だけを参照する。
        return imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
        )
    if source_module_path == "learning/loss_statistics/batch_loss_statistics_initialization.py":
        # batch初期化だけにtorchと既存の集計値型を許可する。
        return imported_module_name in ("torch", "torch.Tensor") or imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.BoundedLossMoments",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatistics",
        )
    if source_module_path == "learning/loss_statistics/model_and_class_loss_statistics.py":
        # 統計所有は公開集計型と一件追加だけに依存する。
        return imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.BoundedLossMoments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.accumulate_bounded_loss_observation",
        )
    if source_module_path == "methods/fedsda/loss_statistics/loss_baseline_selection.py":
        # 基準値方針は公開集計型だけを参照し、他の数値処理へ依存しない。
        return imported_module_name in (
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.BoundedLossMoments",
        )
    if (
        source_module_path
        == "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py"
    ):
        # 収集状態は同機能設定だけを参照し、数値評価へ依存しない。
        return imported_module_name in (
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
        )
    if (
        source_module_path
        == "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py"
    ):
        # 保留位置FIFOは同機能の容量条件だけを参照する。
        return (
            imported_module_name
            == "federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings"
            or imported_module_name.startswith(
                "federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings."
            )
        )
    if source_module_path == "methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py":
        # 単系列の数値検出器だけにNumPyを許可する。
        return imported_module_name == "numpy" or imported_module_name.startswith("numpy.")
    if (
        source_module_path
        == "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py"
    ):
        # 候補loss評価だけにtorchと同機能の固定条件を許可する。
        return (
            imported_module_name == "torch"
            or imported_module_name.startswith("torch.")
            or imported_module_name
            == "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings"
            or imported_module_name.startswith(
                "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings."
            )
        )
    if source_module_path in (
        "learning/models/torch_random_state_scope.py",
        "learning/prediction/class_probability_calculations.py",
    ):
        # torchを必要とする責務のexact moduleだけを例外にする。
        return imported_module_name == "torch" or imported_module_name.startswith("torch.")
    if imported_module_name == "numpy" or imported_module_name.startswith("numpy."):
        return (
            source_module_path.startswith("data/")
            and not source_module_path.endswith("_settings.py")
        ) or source_module_path == "execution/run_random_sources.py"
    if not imported_module_name.startswith("federated_learning_experiments."):
        return False
    source_module_layer = source_module_path.split("/")[0]
    configuration_foundation_module = is_configuration_foundation_module(
        source_module_path=source_module_path,
    )
    if configuration_foundation_module:
        if source_module_layer == "configuration" and source_module_path != (
            "configuration/experiment_run_conditions.py"
        ):
            allowed_internal_module_prefixes = (
                "federated_learning_experiments.core.",
                "federated_learning_experiments.learning.",
                "federated_learning_experiments.methods.",
                "federated_learning_experiments.configuration.experiment_run_conditions.",
            )
            if source_module_path == "configuration/run_settings.py":
                allowed_internal_module_prefixes += (
                    "federated_learning_experiments.configuration.run_settings_validation.",
                )
        else:
            allowed_internal_module_prefixes = ("federated_learning_experiments.core.",)
    elif source_module_path.endswith("_settings.py"):
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.core.",
            "federated_learning_experiments.configuration.experiment_run_conditions.",
        )
        if source_module_layer == "execution":
            allowed_internal_module_prefixes += (
                "federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings.",
            )
    elif (
        source_module_path
        == "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py"
    ):
        # 数値状態部品は同機能の固定条件だけを参照する。
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings.",
        )
    elif (
        source_module_path
        == "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py"
    ):
        # 混合監視は同機能の検出器と固定条件だけを参照する。
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.methods.fedsda.loss_change_detection.bounded_loss_e_sr_detection.",
            "federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings.",
        )
    elif source_module_layer == "data":
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.data.",
            "federated_learning_experiments.configuration.experiment_run_conditions.",
        )
    elif source_module_layer == "execution":
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.execution.",
            "federated_learning_experiments.data.",
        )
        if source_module_path == "execution/run_participant_contracts.py":
            allowed_internal_module_prefixes += (
                "federated_learning_experiments.configuration.experiment_run_conditions.",
            )
    elif source_module_layer == "runtime":
        allowed_internal_module_prefixes = (
            "federated_learning_experiments.core.",
            "federated_learning_experiments.configuration.",
            "federated_learning_experiments.data.",
            "federated_learning_experiments.execution.",
            "federated_learning_experiments.learning.models.torch_random_state_scope.",
        )
    else:
        allowed_internal_module_prefixes = ()
    return any(
        imported_module_name == imported_base_module_name.rstrip(".")
        or imported_module_name.startswith(imported_base_module_name)
        for imported_base_module_name in allowed_internal_module_prefixes
    )


def collect_dependency_boundary_violations(*, source_module_path, source_text):
    """実際のソースと注入用ソースに同じAST検査を適用する。"""
    importing_package_name = "federated_learning_experiments"
    if "/" in source_module_path:
        importing_package_name += "." + source_module_path.rsplit("/", 1)[0].replace("/", ".")
    parsed_source_module = ast.parse(source_text)
    dependency_boundary_violations = ()
    for import_statement in ast.walk(parsed_source_module):
        imported_module_names = resolve_imported_module_names(
            import_statement=import_statement,
            importing_package_name=importing_package_name,
        )
        if source_module_path in (
            "evaluation/adahedge_diagnostic_evidence_collection.py",
            "runtime/training_assignment_diagnostic_notification.py",
            "evaluation/adahedge_diagnostic_evidence.py",
            "evaluation/adaptation_record_store.py",
            "runtime/alarm_adaptation_recording.py",
            "runtime/candidate_validation_adaptation_recording.py",
            "runtime/candidate_validation_session_holder.py",
            "runtime/held_candidate_validation_progress.py",
            "runtime/alarm_occurrence_handling.py",
            "methods/fedsda/training_data_assignment/pending_sample_observation_store.py",
            "evaluation/loss_change_alarm_record_store.py",
            "runtime/observed_sample_processing.py",
            "evaluation/sample_prediction_record_store.py",
            "runtime/observed_sample_prediction.py",
            "runtime/fedsda_run_client_settings.py",
            "runtime/fedsda_run_client.py",
            "runtime/global_model_distribution_application.py",
            "runtime/global_model_distribution.py",
            "runtime/post_aggregation_prediction_recalibration.py",
            "runtime/client_model_cross_evaluation.py",
            "runtime/model_cross_evaluation.py",
            "runtime/model_clustering_and_consolidation.py",
            "runtime/server_round_synchronization.py",
            "runtime/fedsda_run_server.py",
            "runtime/fedsda_run_participant_factory.py",
            "runtime/fedsda_run_metric_derivation.py",
            "runtime/fedsda_measured_run_execution.py",
            "learning/training/initial_model_pretraining_settings.py",
            "runtime/initial_model_pretraining.py",
            "evaluation/communication_volume_record_store.py",
            "evaluation/cross_evaluation_record_store.py",
            "evaluation/model_clustering_record_store.py",
            "evaluation/run_metric_calculations.py",
            "evaluation/computation_cost_summary.py",
            "evaluation/held_model_count_record_store.py",
            "evaluation/loss_monitoring_computation_count_store.py",
            "learning/models/model_computation_measurement.py",
            "methods/fedsda/model_registration/global_model_repository.py",
            "methods/fedsda/consolidation/model_clustering_calculations.py",
            "methods/fedsda/candidate_model_selection/candidate_validation_decision_record_store.py",
            "runtime/server_model_registration_and_aggregation.py",
            "learning/training/shared_parameter_optimizer_state_holder.py",
            "runtime/held_model_training_request_handling.py",
            "runtime/released_pending_sample_assignment.py",
            "learning/training/candidate_epoch_training_settings.py",
            "learning/training/candidate_epoch_training.py",
            "runtime/candidate_classifier_construction.py",
            "learning/training/held_model_training_state_registry.py",
            "learning/prediction/classifier_bounded_loss_evaluation.py",
            "learning/models/classifier_parameter_snapshot.py",
            "methods/fedsda/model_registration/pending_model_upload.py",
            "evaluation/model_evaluation_sample_records.py",
            "evaluation/model_evaluation_sample_store.py",
            "learning/training/model_training_and_assignment_counts.py",
            "learning/training/current_training_model_assignment.py",
            "runtime/held_model_registration_confirmation.py",
            "runtime/adopted_candidate_initial_local_registration.py",
            "learning/training/temporary_model_id_allocation.py",
            "runtime/adopted_candidate_local_adoption.py",
            "runtime/assigned_training_sample_absorption.py",
            "runtime/post_alarm_candidate_validation_resolution.py",
            "runtime/post_alarm_candidate_validation_sample_observation.py",
            "runtime/post_alarm_reference_model_fixation.py",
            "runtime/post_alarm_candidate_validation_session_start.py",
            "runtime/post_alarm_candidate_validation_progress.py",
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_validation_decision_record.py",
            "runtime/incomplete_post_alarm_candidate_validation_finalization.py",
            "methods/fedsda/candidate_model_selection/incomplete_post_alarm_candidate_validation_decision_record.py",
            "runtime/alarm_interval_model_reuse_assessment.py",
            "methods/fedsda/candidate_model_selection/alarm_interval_model_reuse_assessment.py",
            "learning/training/indexed_observed_training_sample.py",
            "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
            "runtime/alarm_training_interval_preparation.py",
            "runtime/alarm_buffer_response.py",
            "runtime/alarm_response_completion.py",
            "runtime/alarm_change_interval_resolution.py",
        ) and isinstance(import_statement, ast.ImportFrom):
            # 通常resolverのpackage別返却差に依存せず、束縛symbolを直接解決する。
            imported_module_names = tuple(
                resolve_name(
                    "." * import_statement.level + (import_statement.module or ""),
                    importing_package_name,
                )
                + "."
                + imported_module_alias.name
                for imported_module_alias in import_statement.names
            )
        for imported_module_name in imported_module_names:
            if (
                source_module_path
                in (
                    "evaluation/adahedge_diagnostic_evidence_collection.py",
                    "runtime/training_assignment_diagnostic_notification.py",
                    "evaluation/adahedge_diagnostic_evidence.py",
                    "evaluation/adaptation_record_store.py",
                    "runtime/alarm_adaptation_recording.py",
                    "runtime/candidate_validation_adaptation_recording.py",
                    "runtime/candidate_validation_session_holder.py",
                    "runtime/held_candidate_validation_progress.py",
                    "runtime/alarm_occurrence_handling.py",
                    "methods/fedsda/training_data_assignment/pending_sample_observation_store.py",
                    "evaluation/loss_change_alarm_record_store.py",
                    "runtime/observed_sample_processing.py",
                    "evaluation/sample_prediction_record_store.py",
                    "runtime/observed_sample_prediction.py",
                    "runtime/fedsda_run_client_settings.py",
                    "runtime/fedsda_run_client.py",
                    "runtime/global_model_distribution_application.py",
                    "runtime/global_model_distribution.py",
                    "runtime/post_aggregation_prediction_recalibration.py",
                    "runtime/client_model_cross_evaluation.py",
                    "runtime/model_cross_evaluation.py",
                    "runtime/model_clustering_and_consolidation.py",
                    "runtime/server_round_synchronization.py",
                    "runtime/fedsda_run_server.py",
                    "runtime/fedsda_run_participant_factory.py",
                    "runtime/fedsda_run_metric_derivation.py",
                    "runtime/fedsda_measured_run_execution.py",
                    "learning/training/initial_model_pretraining_settings.py",
                    "runtime/initial_model_pretraining.py",
                    "evaluation/communication_volume_record_store.py",
                    "evaluation/cross_evaluation_record_store.py",
                    "evaluation/model_clustering_record_store.py",
                    "evaluation/run_metric_calculations.py",
                    "evaluation/computation_cost_summary.py",
                    "evaluation/held_model_count_record_store.py",
                    "evaluation/loss_monitoring_computation_count_store.py",
                    "learning/models/model_computation_measurement.py",
                    "methods/fedsda/model_registration/global_model_repository.py",
                    "methods/fedsda/consolidation/model_clustering_calculations.py",
                    "methods/fedsda/candidate_model_selection/candidate_validation_decision_record_store.py",
                    "runtime/server_model_registration_and_aggregation.py",
                    "learning/training/shared_parameter_optimizer_state_holder.py",
                    "runtime/held_model_training_request_handling.py",
                    "runtime/released_pending_sample_assignment.py",
                    "learning/training/candidate_epoch_training_settings.py",
                    "learning/training/candidate_epoch_training.py",
                    "runtime/candidate_classifier_construction.py",
                    "learning/training/held_model_training_state_registry.py",
                    "learning/prediction/classifier_bounded_loss_evaluation.py",
                    "learning/models/classifier_parameter_snapshot.py",
                    "methods/fedsda/model_registration/pending_model_upload.py",
                    "evaluation/model_evaluation_sample_records.py",
                    "evaluation/model_evaluation_sample_store.py",
                    "learning/training/model_training_and_assignment_counts.py",
                    "learning/training/current_training_model_assignment.py",
                    "runtime/held_model_registration_confirmation.py",
                    "runtime/adopted_candidate_initial_local_registration.py",
                    "learning/training/temporary_model_id_allocation.py",
                    "runtime/adopted_candidate_local_adoption.py",
                    "runtime/assigned_training_sample_absorption.py",
                    "runtime/post_alarm_candidate_validation_resolution.py",
                    "runtime/post_alarm_candidate_validation_sample_observation.py",
                    "runtime/post_alarm_reference_model_fixation.py",
                    "runtime/post_alarm_candidate_validation_session_start.py",
                    "runtime/post_alarm_candidate_validation_progress.py",
                    "methods/fedsda/candidate_model_selection/post_alarm_candidate_validation_decision_record.py",
                    "runtime/incomplete_post_alarm_candidate_validation_finalization.py",
                    "methods/fedsda/candidate_model_selection/incomplete_post_alarm_candidate_validation_decision_record.py",
                    "runtime/alarm_interval_model_reuse_assessment.py",
                    "methods/fedsda/candidate_model_selection/alarm_interval_model_reuse_assessment.py",
                    "learning/training/indexed_observed_training_sample.py",
                    "methods/fedsda/training_data_assignment/prepared_alarm_training_intervals.py",
                    "runtime/alarm_training_interval_preparation.py",
                    "runtime/alarm_buffer_response.py",
                    "runtime/alarm_response_completion.py",
                    "runtime/alarm_change_interval_resolution.py",
                )
                and isinstance(import_statement, ast.Import)
                and not (
                    source_module_path == "evaluation/adahedge_diagnostic_evidence.py"
                    and imported_module_name == "math"
                )
            ) or not dependency_is_allowed(
                source_module_path=source_module_path,
                imported_module_name=imported_module_name,
            ):
                dependency_boundary_violations += (
                    (
                        imported_module_name,
                        "この層では許可されない依存先です",
                    ),
                )
    return dependency_boundary_violations


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import torch", False),
        ("from torch import Tensor", False),
        ("import numpy", False),
        ("import math", False),
        ("import random", False),
        ("from random import Random", False),
        ("from dataclasses import dataclass", False),
        ("import federated_drift_experiment", False),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import _validate_model_id",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUpload",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import *",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment, TrainingModelAssignmentChange",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState",
            True,
        ),
        (
            "from ..methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState as UploadState",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_held_model_registration_confirmation_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/held_model_registration_confirmation.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import torch",
            False,
        ),
        (
            "from torch import float32",
            False,
        ),
        (
            "from torch import no_grad",
            False,
        ),
        (
            "from torch.nn import Parameter",
            False,
        ),
        (
            "from torch.optim import Adam",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "from random import Random",
            False,
        ),
        (
            "from dataclasses import dataclass",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.held_model_registration_confirmation import confirm_held_model_registration",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_post_alarm_candidate_losses",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration import _validate_adopted_candidate_integration_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import _validate_classifier_bounded_loss_inputs",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUpload",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import *",
            False,
        ),
        (
            "from torch import Tensor",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import initialize_model_and_class_loss_statistics_from_batch",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
            True,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration import integrate_adopted_candidate_shared_features",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingState",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import ParameterOptimizerState",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState",
            True,
        ),
        (
            "from ..methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState as UploadState",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_adopted_candidate_initial_local_registration_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/adopted_candidate_initial_local_registration.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "import random as random_module",
            False,
        ),
        (
            "import dataclasses",
            False,
        ),
        (
            "from dataclasses import dataclass",
            False,
        ),
        (
            "from dataclasses import dataclass as Record",
            False,
        ),
        (
            "from random import Random",
            False,
        ),
        (
            "import torch",
            False,
        ),
        (
            "from torch import Tensor",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from __future__ import division",
            False,
        ),
        (
            "from __future__ import annotations, division",
            False,
        ),
        (
            "from __future__ import *",
            False,
        ),
        (
            "from . import current_training_model_assignment",
            False,
        ),
        (
            "from .current_training_model_assignment import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from .held_model_training_state_registry import HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import register_adopted_candidate_as_temporary_held_model",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import *",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment",
            False,
        ),
        ("from __future__ import annotations", True),
        ("from __future__ import annotations as postponed_annotations", True),
    ],
)
def test_temporary_model_id_allocation_rejects_every_import_except_annotations(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="learning/training/temporary_model_id_allocation.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import torch",
            False,
        ),
        (
            "from torch import float32",
            False,
        ),
        (
            "from torch import no_grad",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "from random import Random",
            False,
        ),
        (
            "from dataclasses import dataclass",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration import integrate_adopted_candidate_shared_features",
            False,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import initialize_model_and_class_loss_statistics_from_batch",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ModelTrainingSampleCollection",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingState",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.held_model_registration_confirmation import confirm_held_model_registration",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import _validate_registration_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import _select_active_shared_feature_extractor",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.adopted_candidate_initial_local_registration",
            False,
        ),
        (
            "from federated_learning_experiments.runtime import adopted_candidate_initial_local_registration",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_post_alarm_candidate_losses",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUpload",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import TemporaryModelIdAllocator",
            False,
        ),
        (
            "from torch import Tensor",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import ParameterOptimizerState",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import TemporaryModelIdAllocator",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import register_adopted_candidate_as_temporary_held_model",
            True,
        ),
        (
            "from .adopted_candidate_initial_local_registration import register_adopted_candidate_as_temporary_held_model as register",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_adopted_candidate_local_adoption_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/adopted_candidate_local_adoption.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import torch",
            False,
        ),
        (
            "from torch import Tensor",
            False,
        ),
        (
            "from torch import no_grad",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "from random import Random",
            False,
        ),
        (
            "from dataclasses import dataclass",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import TemporaryModelIdAllocator",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingState",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ModelTrainingSampleCollection",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import accumulate_bounded_loss_observation",
            False,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import _validate_classifier_bounded_loss_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_local_adoption import adopt_candidate_as_current_training_model",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import register_adopted_candidate_as_temporary_held_model",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.held_model_registration_confirmation import confirm_held_model_registration",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_store",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import ModelTrainingSampleStore",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "from ..learning.training.model_training_sample_store import ModelTrainingSampleStore as SampleStore",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_assigned_training_sample_absorption_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/assigned_training_sample_absorption.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import torch",
            False,
        ),
        (
            "from torch import float32",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "from random import Random",
            False,
        ),
        (
            "import dataclasses",
            False,
        ),
        (
            "from dataclasses import field",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import select_available_reference_within_historical_loss_tolerance",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import PostAlarmCandidateLossCollection",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights import FixedSharePredictionWeightController",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import register_adopted_candidate_as_temporary_held_model",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.held_model_registration_confirmation import confirm_held_model_registration",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_local_adoption import _validate_adoption_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            False,
        ),
        (
            "from federated_learning_experiments.evaluation.model_evaluation_sample_store import ModelEvaluationSampleStore",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.adopted_candidate_local_adoption",
            False,
        ),
        (
            "from federated_learning_experiments.runtime import adopted_candidate_local_adoption",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import *",
            False,
        ),
        (
            "from dataclasses import dataclass",
            True,
        ),
        (
            "from torch import Tensor",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import PostAlarmCandidateLossEvaluation",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import ParameterOptimizerState",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import TrainingModelAssignmentChange",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import TemporaryModelIdAllocator",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_local_adoption import adopt_candidate_as_current_training_model",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import absorb_assigned_training_samples_into_held_model",
            True,
        ),
        (
            "from .adopted_candidate_local_adoption import adopt_candidate_as_current_training_model as adopt",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_post_alarm_candidate_validation_resolution_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/post_alarm_candidate_validation_resolution.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import torch",
            False,
        ),
        (
            "from torch import no_grad",
            False,
        ),
        (
            "from torch import float32",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "from dataclasses import dataclass",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import PostAlarmCandidateLossCollectionState",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import PostAlarmCandidateLossEvaluation",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import apply_post_alarm_candidate_validation_resolution",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.adopted_candidate_local_adoption import adopt_candidate_as_current_training_model",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.assigned_training_sample_absorption import absorb_assigned_training_samples_into_held_model",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            False,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import _validate_classifier_bounded_loss_inputs",
            False,
        ),
        (
            "import federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation",
            False,
        ),
        (
            "from federated_learning_experiments.learning.prediction import classifier_bounded_loss_evaluation",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import *",
            False,
        ),
        (
            "from torch import Tensor",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import PostAlarmCandidateLossCollection",
            True,
        ),
        (
            "from ..learning.models.residual_adapter_classifier import ResidualAdapterClassifier as Classifier",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_post_alarm_candidate_validation_sample_observation_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/post_alarm_candidate_validation_sample_observation.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        (
            "import torch",
            False,
        ),
        (
            "from torch import Tensor",
            False,
        ),
        (
            "from torch import manual_seed",
            False,
        ),
        (
            "import numpy",
            False,
        ),
        (
            "import math",
            False,
        ),
        (
            "import random",
            False,
        ),
        (
            "import copy",
            False,
        ),
        (
            "from copy import deepcopy",
            False,
        ),
        (
            "import dataclasses",
            False,
        ),
        (
            "import federated_drift_experiment",
            False,
        ),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.torch_random_state_scope import isolated_cpu_torch_random_state",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.model_architecture_settings import ModelArchitectureSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import ParameterOptimizerState",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingState",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration import integrate_adopted_candidate_shared_features",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_loss_monitoring_baseline_mean_loss",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_alarm_interval_reuse_baseline_mean_loss",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import PostAlarmCandidateLossCollection",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation import observe_post_alarm_candidate_validation_sample",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import apply_post_alarm_candidate_validation_resolution",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import _validate_classifier_parameter_snapshot_inputs",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models import ResidualAdapterClassifier",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import *",
            False,
        ),
        (
            "from dataclasses import dataclass",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_post_alarm_reference_historical_mean_loss",
            True,
        ),
        (
            "from ..learning.models.residual_adapter_classifier import ResidualAdapterClassifier as Classifier",
            True,
        ),
        ("from __future__ import annotations", True),
    ],
)
def test_post_alarm_reference_model_fixation_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/post_alarm_reference_model_fixation.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import torch", False),
        ("import random", False),
        ("import math", False),
        ("import dataclasses", False),
        ("import numpy", False),
        ("from torch import rand", False),
        ("from torch import *", False),
        ("from torch.nn import Module", False),
        ("from __future__ import annotations", False),
        ("import federated_drift_experiment", False),
        ("from .post_alarm_reference_model_fixation import fix_reference_models_at_alarm", False),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import _private",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_state",
            False,
        ),
        ("from dataclasses import dataclass", True),
        ("from torch import Tensor, float32, isfinite, strided", True),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings, SgdParameterOptimizerSettings",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import ParameterOptimizerState",
            True,
        ),
        (
            "from ..learning.training.parameter_optimizer_state import ParameterOptimizerState as Manager",
            True,
        ),
    ],
)
def test_candidate_classifier_construction_exact_dependency_contract(
    source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/candidate_classifier_construction.py",
            source_text=source_text,
        )
    ) == expected_acceptance


def collect_module_allowed_dependency_names():
    """dependency_is_allowedのうち、moduleごとに`return imported_module_name in (...)`で書いた許可集合を読む。"""
    dependency_rule_function = next(
        statement
        for statement in ast.parse(Path(__file__).read_text(encoding="utf-8")).body
        if isinstance(statement, ast.FunctionDef) and statement.name == "dependency_is_allowed"
    )
    allowed_dependency_names_by_module = {}
    for rule_branch in ast.walk(dependency_rule_function):
        if not (
            isinstance(rule_branch, ast.If)
            and isinstance(rule_branch.test, ast.Compare)
            and isinstance(rule_branch.test.left, ast.Name)
            and rule_branch.test.left.id == "source_module_path"
            and isinstance(rule_branch.test.ops[0], ast.Eq)
            and isinstance(rule_branch.test.comparators[0], ast.Constant)
            and len(rule_branch.body) == 1
            and isinstance(rule_branch.body[0], ast.Return)
        ):
            continue
        returned_condition = rule_branch.body[0].value
        if (
            isinstance(returned_condition, ast.Compare)
            and isinstance(returned_condition.ops[0], ast.In)
            and isinstance(returned_condition.comparators[0], ast.Tuple)
            and all(
                isinstance(allowed_name, ast.Constant)
                for allowed_name in returned_condition.comparators[0].elts
            )
        ):
            allowed_dependency_names_by_module[rule_branch.test.comparators[0].value] = {
                allowed_name.value for allowed_name in returned_condition.comparators[0].elts
            }
    return allowed_dependency_names_by_module


def test_module_allowed_dependencies_are_all_imported_by_the_module():
    """許可集合が、実際のsourceのimportより広くなっていないこと（使っていない依存を許可したままにしない）。"""
    allowed_dependency_names_by_module = collect_module_allowed_dependency_names()
    # 読取りが空振りしていないこと（登録の書き方が変わったら、この読取りを直す）。
    assert len(allowed_dependency_names_by_module) >= 63
    unused_allowed_dependency_names = {}
    package_source_directory = (
        Path(__file__).resolve().parents[2] / "src" / "federated_learning_experiments"
    )
    for source_module_path, allowed_dependency_names in allowed_dependency_names_by_module.items():
        importing_package_name = "federated_learning_experiments." + source_module_path.rsplit(
            "/", 1
        )[0].replace("/", ".")
        imported_dependency_names = set()
        for import_statement in ast.walk(
            ast.parse((package_source_directory / source_module_path).read_text(encoding="utf-8"))
        ):
            imported_dependency_names.update(
                resolve_imported_module_names(
                    import_statement=import_statement,
                    importing_package_name=importing_package_name,
                )
            )
            if isinstance(import_statement, ast.ImportFrom):
                imported_base_module_name = resolve_name(
                    "." * import_statement.level + (import_statement.module or ""),
                    importing_package_name,
                )
                imported_dependency_names.add(imported_base_module_name)
                imported_dependency_names.update(
                    imported_base_module_name + "." + imported_module_alias.name
                    for imported_module_alias in import_statement.names
                )
        # `from __future__ import annotations`の許可は、依存先ではないので対象外。
        unused_names = sorted(
            allowed_name
            for allowed_name in allowed_dependency_names - imported_dependency_names
            if allowed_name.split(".")[0] != "__future__"
        )
        if unused_names:
            unused_allowed_dependency_names[source_module_path] = unused_names
    assert unused_allowed_dependency_names == {}


def test_single_run_package_boundaries_have_no_exports():
    """新しい境界は存在し、説明だけを持ち再exportしない。"""
    package_source_directory = (
        Path(__file__).resolve().parents[2] / "src" / "federated_learning_experiments"
    )
    package_boundary_paths = (
        "data",
        "data/concept_schedules",
        "data/sine",
        "execution",
        "runtime",
        "learning/prediction",
        "evaluation",
    )
    for package_boundary_path in package_boundary_paths:
        source_file_path = package_source_directory / package_boundary_path / "__init__.py"
        assert source_file_path.is_file(), package_boundary_path
        parsed_source_module = ast.parse(source_file_path.read_text(encoding="utf-8"))
        assert ast.get_docstring(parsed_source_module), package_boundary_path
        assert len(parsed_source_module.body) == 1, package_boundary_path


def test_single_run_layers_import_only_allowed_dependencies():
    """後続で追加されるモジュールも毎回全走査して境界を検査する。"""
    package_source_directory = (
        Path(__file__).resolve().parents[2] / "src" / "federated_learning_experiments"
    )
    for source_file_path in sorted(package_source_directory.rglob("*.py")):
        source_module_path = source_file_path.relative_to(package_source_directory).as_posix()
        dependency_boundary_violations = collect_dependency_boundary_violations(
            source_module_path=source_module_path,
            source_text=source_file_path.read_text(encoding="utf-8"),
        )
        assert not dependency_boundary_violations, (
            source_module_path,
            dependency_boundary_violations,
        )


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_imported_module_name",
    [
        *[
            (source_module_path, source_text, imported_module_name)
            for source_module_path, imported_module_name in (
                ("evaluation/adaptation_record_store.py", "dataclasses.dataclass"),
                ("evaluation/adaptation_record_store.py", "dataclasses.replace"),
                ("evaluation/adaptation_record_store.py", "typing.Literal"),
                ("runtime/alarm_adaptation_recording.py", "typing.cast"),
                (
                    "runtime/alarm_adaptation_recording.py",
                    "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationOutcome",
                ),
                (
                    "runtime/alarm_adaptation_recording.py",
                    "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecord",
                ),
                (
                    "runtime/alarm_adaptation_recording.py",
                    "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecordStore",
                ),
                (
                    "runtime/alarm_adaptation_recording.py",
                    "federated_learning_experiments.runtime.alarm_response_completion.AlarmResponseCompletion",
                ),
            )
            for source_text in (
                f"import {imported_module_name}",
                f"import {imported_module_name} as AcceptedDependency",
            )
        ],
        *[
            (source_module_path, f"import {imported_module_name}", imported_module_name)
            for source_module_path in (
                "evaluation/adaptation_record_store.py",
                "runtime/alarm_adaptation_recording.py",
            )
            for imported_module_name in (
                "dataclasses",
                "typing",
                "torch",
                "numpy",
                "random",
                "os",
                "federated_drift_experiment",
            )
        ],
        *[
            (
                source_module_path,
                f"from {imported_base_module_name} import {imported_module_alias}",
                f"{imported_base_module_name}.{imported_module_alias}",
            )
            for source_module_path, imported_base_module_name in (
                ("evaluation/adaptation_record_store.py", "dataclasses"),
                ("evaluation/adaptation_record_store.py", "typing"),
                ("runtime/alarm_adaptation_recording.py", "typing"),
                (
                    "runtime/alarm_adaptation_recording.py",
                    "federated_learning_experiments.evaluation.adaptation_record_store",
                ),
                (
                    "runtime/alarm_adaptation_recording.py",
                    "federated_learning_experiments.runtime.alarm_response_completion",
                ),
            )
            for imported_module_alias in ("*", "_private", "field")
        ],
        (
            "evaluation/adaptation_record_store.py",
            "from ..runtime.alarm_response_completion import AlarmResponseCompletion",
            "federated_learning_experiments.runtime.alarm_response_completion.AlarmResponseCompletion",
        ),
        (
            "runtime/alarm_adaptation_recording.py",
            "from .alarm_response_completion import complete_alarm_buffer_response",
            "federated_learning_experiments.runtime.alarm_response_completion.complete_alarm_buffer_response",
        ),
        (
            "runtime/alarm_adaptation_recording.py",
            "from .alarm_buffer_response import respond_to_alarm_with_buffered_samples",
            "federated_learning_experiments.runtime.alarm_buffer_response.respond_to_alarm_with_buffered_samples",
        ),
        (
            "runtime/alarm_adaptation_recording.py",
            "from federated_learning_experiments.evaluation import adaptation_record_store",
            "federated_learning_experiments.evaluation.adaptation_record_store",
        ),
        (
            "runtime/alarm_adaptation_recording.py",
            "from federated_learning_experiments.runtime import alarm_response_completion",
            "federated_learning_experiments.runtime.alarm_response_completion",
        ),
        *[
            (source_module_path, source_text, expected_imported_module_name)
            for source_module_path in (
                "learning/training/model_training_sample_records.py",
                "learning/training/held_model_training_batch_sampling.py",
            )
            for source_text, expected_imported_module_name in (
                ("import math", "math"),
                ("import numpy", "numpy"),
                ("import torch._C", "torch._C"),
                ("import torch.optim", "torch.optim"),
                ("from torch.nn import Module", "torch.nn"),
                (
                    "import federated_drift_experiment.clients.base",
                    "federated_drift_experiment.clients.base",
                ),
                (
                    "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
                    "federated_learning_experiments.learning.models.residual_adapter_classifier",
                ),
                (
                    "import federated_learning_experiments.runtime.single_run_execution",
                    "federated_learning_experiments.runtime.single_run_execution",
                ),
                (
                    "import federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer",
                    "federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer",
                ),
                (
                    "from .joint_model_parameter_update import perform_joint_model_parameter_update",
                    "federated_learning_experiments.learning.training.joint_model_parameter_update",
                ),
            )
        ],
        (
            "learning/training/model_training_sample_records.py",
            "from dataclasses import field",
            "dataclasses.field",
        ),
        (
            "learning/training/model_training_sample_records.py",
            "from torch import cat",
            "torch.cat",
        ),
        (
            "learning/training/model_training_sample_records.py",
            "from random import Random",
            "random",
        ),
        (
            "learning/training/model_training_sample_records.py",
            "from .held_model_training_batch_sampling import sample_training_batches_for_held_models",
            "federated_learning_experiments.learning.training.held_model_training_batch_sampling",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import sample",
            "random.sample",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import seed",
            "random.seed",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import getstate",
            "random.getstate",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import setstate",
            "random.setstate",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import SystemRandom",
            "random.SystemRandom",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from random import _inst",
            "random._inst",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "import random.child",
            "random.child",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from .model_training_sample_records import Tensor",
            "federated_learning_experiments.learning.training.model_training_sample_records.Tensor",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from .model_training_sample_records import _private",
            "federated_learning_experiments.learning.training.model_training_sample_records._private",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "import federated_learning_experiments.learning.training.model_training_sample_records.child",
            "federated_learning_experiments.learning.training.model_training_sample_records.child",
        ),
        *[
            (source_module_path, source_text, expected_imported_module_name)
            for source_module_path in (
                "learning/training/participating_model_training_batch.py",
                "learning/training/joint_model_parameter_update.py",
            )
            for source_text, expected_imported_module_name in (
                ("import math", "math"),
                ("import random", "random"),
                ("import numpy", "numpy"),
                ("import torch._C", "torch._C"),
                ("import torch.optim.lr_scheduler", "torch.optim.lr_scheduler"),
                ("from torch.optim import RMSprop", "torch.optim.RMSprop"),
                ("import federated_drift_experiment.models", "federated_drift_experiment.models"),
                (
                    "import federated_learning_experiments.runtime.single_run_execution",
                    "federated_learning_experiments.runtime.single_run_execution",
                ),
                (
                    "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization",
                    "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization",
                ),
                (
                    "from .parameter_optimizer_construction import create_parameter_optimizer",
                    "federated_learning_experiments.learning.training.parameter_optimizer_construction",
                ),
                (
                    "import federated_learning_experiments.learning.training",
                    "federated_learning_experiments.learning.training",
                ),
            )
        ],
        (
            "learning/training/participating_model_training_batch.py",
            "from dataclasses import field",
            "dataclasses.field",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from torch import cat",
            "torch.cat",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from torch.optim import Adam",
            "torch.optim.Adam",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from torch.nn import Parameter",
            "torch.nn.Parameter",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from .local_training_settings import LocalTrainingSettings",
            "federated_learning_experiments.learning.training.local_training_settings",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from ..models.shared_feature_extractor import SharedFeatureExtractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from ..models.residual_adapter_classifier import _validate_classifier_inputs",
            "federated_learning_experiments.learning.models.residual_adapter_classifier._validate_classifier_inputs",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from .participating_model_training_batch import Tensor",
            "federated_learning_experiments.learning.training.participating_model_training_batch.Tensor",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from .local_training_settings import validate_settings_field_values",
            "federated_learning_experiments.learning.training.local_training_settings.validate_settings_field_values",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from ..models.shared_feature_extractor import _validate_feature_tensor",
            "federated_learning_experiments.learning.models.shared_feature_extractor._validate_feature_tensor",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "import federated_learning_experiments.learning.training.participating_model_training_batch.child",
            "federated_learning_experiments.learning.training.participating_model_training_batch.child",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "import federated_learning_experiments.learning.models.residual_adapter_classifier.child",
            "federated_learning_experiments.learning.models.residual_adapter_classifier.child",
        ),
        *[
            (source_module_path, source_text, expected_imported_module_name)
            for source_module_path in (
                "learning/training/parameter_optimizer_settings.py",
                "learning/training/parameter_optimizer_construction.py",
            )
            for source_text, expected_imported_module_name in (
                ("import math", "math"),
                ("import random", "random"),
                ("import numpy", "numpy"),
                ("import torch._C", "torch._C"),
                ("import federated_drift_experiment.models", "federated_drift_experiment.models"),
                (
                    "from ..models import residual_adapter_classifier",
                    "federated_learning_experiments.learning.models.residual_adapter_classifier",
                ),
                (
                    "import federated_learning_experiments.data.sine.sine_sample_generation",
                    "federated_learning_experiments.data.sine.sine_sample_generation",
                ),
                (
                    "import federated_learning_experiments.runtime.single_run_execution",
                    "federated_learning_experiments.runtime.single_run_execution",
                ),
                (
                    "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization",
                    "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization",
                ),
            )
        ],
        ("learning/training/parameter_optimizer_settings.py", "import torch", "torch"),
        ("learning/training/parameter_optimizer_settings.py", "import torch.optim", "torch.optim"),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from dataclasses import asdict",
            "dataclasses.asdict",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from ...core.settings_field_validation import _private",
            "federated_learning_experiments.core.settings_field_validation._private",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from ...core.settings_field_validation import fields",
            "federated_learning_experiments.core.settings_field_validation.fields",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "import federated_learning_experiments.core.settings_field_validation.child",
            "federated_learning_experiments.core.settings_field_validation.child",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from ...core.configuration_errors import RunSettingsValidationError",
            "federated_learning_experiments.core.configuration_errors",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from torch.optim import RMSprop",
            "torch.optim.RMSprop",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "import torch.optim.lr_scheduler",
            "torch.optim.lr_scheduler",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "import torch.optim.optimizer",
            "torch.optim.optimizer",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from torch.nn import Module",
            "torch.nn.Module",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from .parameter_optimizer_settings import _private",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings._private",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from .parameter_optimizer_settings import validate_settings_field_values",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.validate_settings_field_values",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings.child",
            "federated_learning_experiments.learning.training.parameter_optimizer_settings.child",
        ),
        *[
            (source_module_path, source_text, expected_imported_module_name)
            for source_module_path in (
                "learning/models/shared_feature_extractor.py",
                "learning/models/nonlinear_residual_adapter.py",
                "learning/models/residual_adapter_classifier.py",
            )
            for source_text, expected_imported_module_name in (
                ("import numpy", "numpy"),
                ("import random", "random"),
                ("import torch.optim", "torch.optim"),
                ("import torch._C", "torch._C"),
                ("from torch.nn import Dropout", "torch.nn.Dropout"),
                ("from torch.nn.init import kaiming_uniform_", "torch.nn.init.kaiming_uniform_"),
                ("import federated_drift_experiment.models", "federated_drift_experiment.models"),
            )
        ],
        (
            "learning/models/shared_feature_extractor.py",
            "from .residual_adapter_classifier import ResidualAdapterClassifier",
            "federated_learning_experiments.learning.models.residual_adapter_classifier",
        ),
        (
            "learning/models/nonlinear_residual_adapter.py",
            "from .shared_feature_extractor import SharedFeatureExtractor",
            "federated_learning_experiments.learning.models.shared_feature_extractor",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from .shared_feature_extractor import _validate_feature_tensor",
            "federated_learning_experiments.learning.models.shared_feature_extractor._validate_feature_tensor",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from .model_architecture_settings import validate_settings_field_values",
            "federated_learning_experiments.learning.models.model_architecture_settings.validate_settings_field_values",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "import federated_learning_experiments.learning.models.shared_feature_extractor.child",
            "federated_learning_experiments.learning.models.shared_feature_extractor.child",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "import federated_learning_experiments.runtime.single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from ...methods.fedsda.candidate_model_selection import candidate_parameter_initialization",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import torch._C",
            "torch._C",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from torch import nn",
            "torch.nn",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from .candidate_parameter_initialization_settings import _private",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings._private",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from .candidate_parameter_initialization_settings import validate_settings_field_values",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.validate_settings_field_values",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.child",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.child",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from .post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import federated_drift_experiment.config",
            "federated_drift_experiment.config",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import federated_learning_experiments.runtime.single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "from federated_learning_experiments.core.settings_field_validation import fields",
            "federated_learning_experiments.core.settings_field_validation.fields",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "from federated_learning_experiments.core.settings_field_validation import _private",
            "federated_learning_experiments.core.settings_field_validation._private",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "from federated_learning_experiments.core import configuration_errors",
            "federated_learning_experiments.core.configuration_errors",
        ),
        ("learning/loss_statistics/server_loss_mean_aggregation.py", "import torch", "torch"),
        ("learning/loss_statistics/server_loss_mean_aggregation.py", "import numpy", "numpy"),
        ("learning/loss_statistics/server_loss_mean_aggregation.py", "import config", "config"),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from ...methods.fedsda.loss_statistics import loss_baseline_selection",
            "federated_learning_experiments.methods.fedsda.loss_statistics",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from .bounded_loss_moments import accumulate_bounded_loss_observation",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.accumulate_bounded_loss_observation",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from .bounded_loss_moments import estimate_loss_mean_and_sample_variance",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.estimate_loss_mean_and_sample_variance",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from .bounded_loss_moments import _validate_loss_moment_fields",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments._validate_loss_moment_fields",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "import federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.internal",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.internal",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from .model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from . import another_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "import torch",
            "torch",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "import numpy",
            "numpy",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "import config",
            "config",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from ...methods.fedsda.loss_statistics import loss_baseline_selection",
            "federated_learning_experiments.methods.fedsda.loss_statistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from .model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from .model_and_class_loss_statistics import _copy_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics._copy_loss_moments",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.internal",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.internal",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from .bounded_loss_moments import BoundedLossMoments",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from . import another_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from .model_and_class_loss_statistics import arbitrary_public_name",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.arbitrary_public_name",
        ),
        ("learning/prediction/class_probability_calculations.py", "import numpy", "numpy"),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.internal",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.internal",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from .candidate_model_training_and_acceptance_settings import UnsupportedSettings",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.UnsupportedSettings",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "import config",
            "config",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from federated_drift_experiment import provisional_model",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from ....runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from . import post_alarm_candidate_loss_evaluation",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from ..training_data_assignment import pending_training_assignment_buffer",
            "federated_learning_experiments.methods.fedsda.training_data_assignment",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from . import another_loss_collection",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "import config",
            "config",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from federated_drift_experiment import clients",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from ....runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from ..loss_change_detection import overall_and_true_class_loss_monitoring",
            "federated_learning_experiments.methods.fedsda.loss_change_detection",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from . import another_assignment_buffer",
            "federated_learning_experiments.methods.fedsda.training_data_assignment",
        ),
        ("learning/prediction/class_probability_calculations.py", "import config", "config"),
        (
            "learning/prediction/class_probability_calculations.py",
            "from federated_drift_experiment import expert_routing",
            "federated_drift_experiment",
        ),
        (
            "learning/prediction/class_probability_calculations.py",
            "from ... import runtime",
            "federated_learning_experiments.runtime",
        ),
        (
            "learning/prediction/class_probability_calculations.py",
            "from ...methods.fedsda.prediction_combination.fixed_share_prediction_weights import FixedSharePredictionWeightController",
            "federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights",
        ),
        ("learning/prediction/another_prediction.py", "import torch", "torch"),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "import config",
            "config",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "from federated_drift_experiment import provisional_model",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "from ....runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "from ..loss_change_detection import overall_and_true_class_loss_monitoring",
            "federated_learning_experiments.methods.fedsda.loss_change_detection",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "from ....learning.prediction import class_probability_calculations",
            "federated_learning_experiments.learning.prediction",
        ),
        ("methods/fedsda/candidate_model_selection/another_candidate.py", "import torch", "torch"),
        (
            "methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py",
            "from .loss_change_detection_settings import LossChangeDetectionSettings",
            "federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings",
        ),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "from ....runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "from ..prediction_combination import fixed_share_prediction_weights",
            "federated_learning_experiments.methods.fedsda.prediction_combination",
        ),
        ("methods/fedsda/loss_change_detection/another_detector.py", "import numpy", "numpy"),
        ("learning/loss_statistics/bounded_loss_moments.py", "import numpy", "numpy"),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "import numpy",
            "numpy",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "import torch._C",
            "torch._C",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "import config",
            "config",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from ...methods.fedsda.loss_statistics import loss_baseline_selection",
            "federated_learning_experiments.methods.fedsda.loss_statistics",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .bounded_loss_moments import accumulate_bounded_loss_observation",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.accumulate_bounded_loss_observation",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .bounded_loss_moments import _validate_loss_moment_fields",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments._validate_loss_moment_fields",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .bounded_loss_moments import estimate_loss_mean_and_sample_variance",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.estimate_loss_mean_and_sample_variance",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .model_and_class_loss_statistics import _copy_loss_moments",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics._copy_loss_moments",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.internal",
            "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.internal",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from . import another_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics",
        ),
        ("learning/loss_statistics/model_and_class_loss_statistics.py", "import numpy", "numpy"),
        ("learning/loss_statistics/model_and_class_loss_statistics.py", "import torch", "torch"),
        ("learning/loss_statistics/model_and_class_loss_statistics.py", "import config", "config"),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from ...methods.fedsda.loss_statistics import loss_baseline_selection",
            "federated_learning_experiments.methods.fedsda.loss_statistics",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from ...methods.fedsda.loss_change_detection import overall_and_true_class_loss_monitoring",
            "federated_learning_experiments.methods.fedsda.loss_change_detection",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from .bounded_loss_moments import _validate_loss_moment_fields",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments._validate_loss_moment_fields",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from .bounded_loss_moments import estimate_loss_mean_and_sample_variance",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.estimate_loss_mean_and_sample_variance",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "import federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.internal",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.internal",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from . import another_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics",
        ),
        ("learning/loss_statistics/bounded_loss_moments.py", "import torch", "torch"),
        ("learning/loss_statistics/bounded_loss_moments.py", "import config", "config"),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from ...methods.fedsda.loss_change_detection import overall_and_true_class_loss_monitoring",
            "federated_learning_experiments.methods.fedsda.loss_change_detection",
        ),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from ...methods.fedsda.candidate_model_selection import post_alarm_candidate_loss_evaluation",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection",
        ),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from ..prediction import class_probability_calculations",
            "federated_learning_experiments.learning.prediction",
        ),
        (
            "learning/loss_statistics/bounded_loss_moments.py",
            "from . import another_loss_statistics",
            "federated_learning_experiments.learning.loss_statistics",
        ),
        ("methods/fedsda/loss_statistics/loss_baseline_selection.py", "import numpy", "numpy"),
        ("methods/fedsda/loss_statistics/loss_baseline_selection.py", "import torch", "torch"),
        ("methods/fedsda/loss_statistics/loss_baseline_selection.py", "import config", "config"),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from federated_drift_experiment import config",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from ....runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from ..loss_change_detection import overall_and_true_class_loss_monitoring",
            "federated_learning_experiments.methods.fedsda.loss_change_detection",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from ..candidate_model_selection import post_alarm_candidate_loss_evaluation",
            "federated_learning_experiments.methods.fedsda.candidate_model_selection",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import _validate_loss_moment_fields",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments._validate_loss_moment_fields",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import estimate_loss_mean_and_sample_variance",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.estimate_loss_mean_and_sample_variance",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "import federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.another_module",
            "federated_learning_experiments.learning.loss_statistics.bounded_loss_moments.another_module",
        ),
        ("data/sine/sine_sample_generation.py", "import torch", "torch"),
        ("data/observed_streams.py", "import config", "config"),
        ("data/observed_streams.py", "from clients import fedsda", "clients"),
        (
            "data/observed_streams.py",
            "from federated_learning_experiments import runtime",
            "federated_learning_experiments.runtime",
        ),
        (
            "data/sine/sine_sample_generation.py",
            "from ... import runtime",
            "federated_learning_experiments.runtime",
        ),
        (
            "data/sine/sine_sample_generation.py",
            "from ...runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "data/observed_streams.py",
            "from federated_learning_experiments.learning.models import torch_random_state_scope",
            "federated_learning_experiments.learning.models.torch_random_state_scope",
        ),
        (
            "data/observed_streams.py",
            "from federated_learning_experiments.configuration import run_settings",
            "federated_learning_experiments.configuration.run_settings",
        ),
        ("data/concept_schedules/random_concept_schedule_settings.py", "import numpy", "numpy"),
        (
            "data/concept_schedules/random_concept_schedule_settings.py",
            "from .. import observed_streams",
            "federated_learning_experiments.data.observed_streams",
        ),
        ("execution/stream_protocol_execution_settings.py", "import numpy", "numpy"),
        (
            "execution/stream_protocol_execution_settings.py",
            "from federated_learning_experiments.data.sine import sine_sample_generation",
            "federated_learning_experiments.data.sine.sine_sample_generation",
        ),
        ("execution/run_random_sources.py", "import torch", "torch"),
        ("execution/run_participant_contracts.py", "import numpy", "numpy"),
        (
            "execution/stream_protocol_execution_loop.py",
            "from ..runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "execution/stream_protocol_execution_loop.py",
            "from federated_learning_experiments import methods",
            "federated_learning_experiments.methods",
        ),
        (
            "execution/run_execution_records.py",
            "from .. import learning",
            "federated_learning_experiments.learning",
        ),
        (
            "execution/stream_protocol_execution_loop.py",
            "from ..configuration.experiment_run_conditions import ExperimentRunConditions",
            "federated_learning_experiments.configuration.experiment_run_conditions",
        ),
        ("runtime/single_run_execution.py", "import torch", "torch"),
        ("runtime/single_run_execution.py", "import numpy", "numpy"),
        (
            "runtime/single_run_execution.py",
            "from ..learning.models import model_architecture_settings",
            "federated_learning_experiments.learning.models.model_architecture_settings",
        ),
        ("learning/models/torch_random_state_scope.py", "import numpy", "numpy"),
        (
            "learning/models/torch_random_state_scope.py",
            "from ... import runtime",
            "federated_learning_experiments.runtime",
        ),
        ("learning/models/another_model.py", "import torch", "torch"),
        ("core/configuration_errors.py", "import numpy", "numpy"),
        ("configuration/experiment_run_conditions.py", "import numpy", "numpy"),
        ("learning/models/model_architecture_settings.py", "import torch", "torch"),
        ("methods/fedsda/consolidation/model_consolidation_settings.py", "import numpy", "numpy"),
        ("data/observed_streams.py", "if False:\n    import torch", "torch"),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "import torch",
            "torch",
        ),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "import numpy",
            "numpy",
        ),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "from federated_drift_experiment import expert_routing",
            "federated_drift_experiment",
        ),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "from federated_learning_experiments.runtime import single_run_execution",
            "federated_learning_experiments.runtime.single_run_execution",
        ),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "from federated_learning_experiments.methods.fedsda.consolidation import model_consolidation_settings",
            "federated_learning_experiments.methods.fedsda.consolidation",
        ),
    ],
)
def test_single_run_dependency_checker_rejects_forbidden_imports(
    source_module_path,
    source_text,
    expected_imported_module_name,
):
    """productionを変更せず、実際の検査に禁止依存を注入する。"""
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path,
        source_text=source_text,
    )
    assert any(
        imported_module_name == expected_imported_module_name
        for imported_module_name, dependency_boundary_violation_reason in dependency_boundary_violations
    ), dependency_boundary_violations


@pytest.mark.parametrize(
    "source_module_path,source_text",
    [
        *[
            (source_module_path, source_text)
            for source_module_path, imported_module_name in (
                ("evaluation/adaptation_record_store.py", "dataclasses.dataclass"),
                ("evaluation/adaptation_record_store.py", "dataclasses.replace"),
                ("evaluation/adaptation_record_store.py", "typing.Literal"),
                ("runtime/alarm_adaptation_recording.py", "typing.cast"),
                (
                    "runtime/alarm_adaptation_recording.py",
                    "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationOutcome",
                ),
                (
                    "runtime/alarm_adaptation_recording.py",
                    "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecord",
                ),
                (
                    "runtime/alarm_adaptation_recording.py",
                    "federated_learning_experiments.evaluation.adaptation_record_store.AdaptationRecordStore",
                ),
                (
                    "runtime/alarm_adaptation_recording.py",
                    "federated_learning_experiments.runtime.alarm_response_completion.AlarmResponseCompletion",
                ),
            )
            for source_text in (
                f"from {imported_module_name.rsplit('.', 1)[0]} import {imported_module_name.rsplit('.', 1)[1]}",
                f"from {imported_module_name.rsplit('.', 1)[0]} import {imported_module_name.rsplit('.', 1)[1]} as AcceptedDependency",
            )
        ],
        ("evaluation/adaptation_record_store.py", "from dataclasses import dataclass, replace"),
        (
            "runtime/alarm_adaptation_recording.py",
            "from ..evaluation.adaptation_record_store import AdaptationOutcome, AdaptationRecord, AdaptationRecordStore",
        ),
        (
            "runtime/alarm_adaptation_recording.py",
            "from .alarm_response_completion import AlarmResponseCompletion",
        ),
        ("learning/training/model_training_sample_records.py", "from dataclasses import dataclass"),
        ("learning/training/model_training_sample_records.py", "from torch import Tensor"),
        (
            "learning/training/model_training_sample_records.py",
            "from __future__ import annotations",
        ),
        ("learning/training/held_model_training_batch_sampling.py", "from random import Random"),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from torch import Tensor, cat, isfinite, float32, strided",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from .model_training_sample_records import ObservedTrainingSample, ModelTrainingSampleCollection, SampledModelTrainingBatch",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample, ModelTrainingSampleCollection, SampledModelTrainingBatch",
        ),
        (
            "learning/training/held_model_training_batch_sampling.py",
            "from __future__ import annotations",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from dataclasses import dataclass",
        ),
        ("learning/training/participating_model_training_batch.py", "from torch import Tensor"),
        (
            "learning/training/participating_model_training_batch.py",
            "from torch.optim import Optimizer",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
        ),
        (
            "learning/training/participating_model_training_batch.py",
            "from __future__ import annotations",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from torch import Tensor, cat, isfinite, no_grad, is_grad_enabled, float32, strided",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from torch.nn import Parameter, BCELoss, CrossEntropyLoss",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from torch.optim import Optimizer, Adam, SGD",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from .local_training_settings import LocalTrainingSettings",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from .participating_model_training_batch import ParticipatingModelTrainingBatch",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from ..models.shared_feature_extractor import SharedFeatureExtractor",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from federated_learning_experiments.learning.training.local_training_settings import LocalTrainingSettings",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from federated_learning_experiments.learning.training.participating_model_training_batch import ParticipatingModelTrainingBatch",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
        ),
        (
            "learning/training/joint_model_parameter_update.py",
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
        ),
        ("learning/training/joint_model_parameter_update.py", "from __future__ import annotations"),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from dataclasses import dataclass, field",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from ...core.settings_field_validation import validate_settings_field_values",
        ),
        (
            "learning/training/parameter_optimizer_settings.py",
            "from federated_learning_experiments.core.settings_field_validation import validate_settings_field_values",
        ),
        ("learning/training/parameter_optimizer_settings.py", "from __future__ import annotations"),
        ("learning/training/parameter_optimizer_construction.py", "import torch"),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from torch import float32, strided",
        ),
        ("learning/training/parameter_optimizer_construction.py", "from torch.nn import Parameter"),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from torch.optim import Optimizer, Adam, SGD",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from .parameter_optimizer_settings import AdamParameterOptimizerSettings, SgdParameterOptimizerSettings",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings, SgdParameterOptimizerSettings",
        ),
        (
            "learning/training/parameter_optimizer_construction.py",
            "from __future__ import annotations",
        ),
        *[
            (source_module_path, source_text)
            for source_module_path in (
                "learning/models/shared_feature_extractor.py",
                "learning/models/nonlinear_residual_adapter.py",
                "learning/models/residual_adapter_classifier.py",
            )
            for source_text in (
                "import torch",
                "from torch import Tensor, float32, device, strided",
                "from torch.nn import Module, Sequential, Linear, ReLU, Sigmoid, Identity",
                "from torch.nn.init import zeros_",
                "from __future__ import annotations",
            )
        ],
        (
            "learning/models/residual_adapter_classifier.py",
            "from .model_architecture_settings import ModelArchitectureSettings",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from .shared_feature_extractor import SharedFeatureExtractor",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from .nonlinear_residual_adapter import NonlinearResidualAdapter",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from federated_learning_experiments.learning.models.model_architecture_settings import ModelArchitectureSettings",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
        ),
        (
            "learning/models/residual_adapter_classifier.py",
            "from federated_learning_experiments.learning.models.nonlinear_residual_adapter import NonlinearResidualAdapter",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import math",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "import torch",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from .candidate_parameter_initialization_settings import CandidateParameterInitializationSettings",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import CandidateParameterInitializationSettings",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "from dataclasses import dataclass, field",
        ),
        (
            "methods/fedsda/candidate_model_selection/candidate_parameter_initialization_settings.py",
            "from federated_learning_experiments.core.settings_field_validation import validate_settings_field_values",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from dataclasses import dataclass",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "import federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments",
        ),
        (
            "learning/loss_statistics/server_loss_mean_aggregation.py",
            "from .bounded_loss_moments import BoundedLossMoments",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from .model_and_class_loss_statistics import ModelAndClassLossStatistics",
        ),
        (
            "learning/loss_statistics/model_id_mapped_loss_statistics_selection.py",
            "from collections import defaultdict",
        ),
        ("learning/loss_statistics/bounded_loss_moments.py", "from dataclasses import dataclass"),
        ("learning/loss_statistics/batch_loss_statistics_initialization.py", "import torch"),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from torch import Tensor",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from dataclasses import dataclass",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .bounded_loss_moments import BoundedLossMoments",
        ),
        (
            "learning/loss_statistics/batch_loss_statistics_initialization.py",
            "from .model_and_class_loss_statistics import ModelAndClassLossStatistics",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from dataclasses import dataclass",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "import federated_learning_experiments.learning.loss_statistics.bounded_loss_moments",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments, accumulate_bounded_loss_observation",
        ),
        (
            "learning/loss_statistics/model_and_class_loss_statistics.py",
            "from .bounded_loss_moments import BoundedLossMoments, accumulate_bounded_loss_observation",
        ),
        ("learning/loss_statistics/bounded_loss_moments.py", "import math"),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments",
        ),
        (
            "methods/fedsda/loss_statistics/loss_baseline_selection.py",
            "from ....learning.loss_statistics.bounded_loss_moments import BoundedLossMoments",
        ),
        ("learning/prediction/class_probability_calculations.py", "import torch"),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from .candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py",
            "from dataclasses import dataclass",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from .training_data_assignment_settings import TrainingDataAssignmentSettings",
        ),
        (
            "methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py",
            "from collections import deque",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "import torch",
        ),
        (
            "methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py",
            "from .candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
        ),
        ("learning/prediction/class_probability_calculations.py", "from torch import Tensor"),
        (
            "learning/prediction/class_probability_calculations.py",
            "from collections.abc import Mapping",
        ),
        ("methods/fedsda/loss_change_detection/bounded_loss_e_sr_detection.py", "import numpy"),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "from .bounded_loss_e_sr_detection import BoundedLossESRDetector",
        ),
        (
            "methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py",
            "from .loss_change_detection_settings import LossChangeDetectionSettings",
        ),
        ("data/observed_streams.py", "from dataclasses import dataclass"),
        ("data/sine/sine_sample_generation.py", "import numpy as np"),
        ("data/sine/sine_sample_generation.py", "from ..observed_streams import ObservedSample"),
        ("data/sine/sine_sample_generation.py", "from .. import observed_streams"),
        (
            "data/concept_schedules/random_concept_schedule_generation.py",
            "from ...configuration.experiment_run_conditions import ExperimentRunConditions",
        ),
        (
            "data/concept_schedules/random_concept_schedule_settings.py",
            "from ...core.settings_field_validation import validate_settings_field_values",
        ),
        (
            "execution/stream_protocol_execution_settings.py",
            "from ..data.concept_schedules.random_concept_schedule_settings import RandomConceptScheduleSettings",
        ),
        (
            "execution/stream_protocol_execution_settings.py",
            "from ..configuration.experiment_run_conditions import ExperimentRunConditions",
        ),
        ("execution/run_random_sources.py", "from numpy.random import RandomState"),
        (
            "execution/run_participant_contracts.py",
            "from .run_random_sources import RunRandomSources",
        ),
        (
            "execution/run_participant_contracts.py",
            "from ..configuration.experiment_run_conditions import ExperimentRunConditions",
        ),
        ("execution/stream_protocol_execution_loop.py", "from . import run_participant_contracts"),
        (
            "execution/stream_protocol_execution_loop.py",
            "from ..data.observed_streams import ClientObservedStream",
        ),
        ("learning/models/torch_random_state_scope.py", "import torch"),
        ("learning/models/torch_random_state_scope.py", "from torch import random"),
        (
            "runtime/single_run_execution.py",
            "from ..learning.models.torch_random_state_scope import isolated_cpu_torch_random_state",
        ),
        (
            "runtime/single_run_execution.py",
            "from ..learning.models import torch_random_state_scope",
        ),
        ("runtime/single_run_execution.py", "from ..execution import run_random_sources"),
        ("methods/fedsda/prediction_combination/fixed_share_prediction_weights.py", "import math"),
        (
            "methods/fedsda/prediction_combination/fixed_share_prediction_weights.py",
            "from .prediction_combination_settings import PredictionCombinationSettings",
        ),
    ],
)
def test_single_run_dependency_checker_accepts_allowed_imports(source_module_path, source_text):
    """許可された型依存・NumPy例外・専用torch境界を拒否しない。"""
    assert (
        collect_dependency_boundary_violations(
            source_module_path=source_module_path,
            source_text=source_text,
        )
        == ()
    )


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        ("learning/training/held_model_training_binding.py", "import math", False),
        ("learning/training/held_model_training_binding.py", "from random import random", False),
        (
            "learning/training/held_model_training_binding.py",
            "from random import SystemRandom",
            False,
        ),
        ("learning/training/held_model_training_binding.py", "from torch import cat", False),
        ("learning/training/held_model_training_binding.py", "from torch.optim import Adam", False),
        (
            "learning/training/held_model_training_binding.py",
            "from .joint_model_parameter_update import _validate_training_tensor",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from .held_model_training_batch_sampling import Tensor",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from ..models.shared_feature_extractor import Module",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from ...runtime import single_run_execution",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "import federated_drift_experiment.clients.base",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from ...methods.fedsda.training_data_assignment import pending_training_assignment_buffer",
            False,
        ),
        ("learning/training/held_model_joint_training_iterations.py", "import math", False),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from random import random",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from random import SystemRandom",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from torch import cat",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from torch.optim import Adam",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .joint_model_parameter_update import _validate_training_tensor",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .held_model_training_batch_sampling import Tensor",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from ..models.shared_feature_extractor import Module",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from ...runtime import single_run_execution",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "import federated_drift_experiment.clients.base",
            False,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from ...methods.fedsda.training_data_assignment import pending_training_assignment_buffer",
            False,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from dataclasses import dataclass",
            True,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from torch.optim import Optimizer",
            True,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "learning/training/held_model_training_binding.py",
            "from __future__ import annotations",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from random import Random",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from torch.optim import Optimizer",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from ..models.shared_feature_extractor import SharedFeatureExtractor",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .held_model_training_binding import HeldModelTrainingBinding",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .model_training_sample_records import ModelTrainingSampleCollection",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .participating_model_training_batch import ParticipatingModelTrainingBatch",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .local_training_settings import LocalTrainingSettings",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .held_model_training_batch_sampling import sample_training_batches_for_held_models",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from .joint_model_parameter_update import perform_joint_model_parameter_update",
            True,
        ),
        (
            "learning/training/held_model_joint_training_iterations.py",
            "from __future__ import annotations",
            True,
        ),
    ],
)
def test_joint_training_iteration_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("import random", False),
        ("import copy", False),
        ("import dataclasses", False),
        ("import torch", False),
        ("import numpy", False),
        ("import federated_drift_experiment", False),
        ("from torch.optim import Adam", False),
        ("from torch.nn import Linear", False),
        ("from .parameter_optimizer_construction import _validate_optimizer_settings", False),
        ("from .parameter_optimizer_settings import field", False),
        ("from .parameter_optimizer_construction.child import create_parameter_optimizer", False),
        (
            "from .held_model_training_batch_sampling import sample_training_batches_for_held_models",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models import ResidualAdapterClassifier",
            False,
        ),
        ("from __future__ import annotations", True),
        ("from torch.nn import Parameter", True),
        ("from torch.optim import Optimizer", True),
        ("from .parameter_optimizer_settings import AdamParameterOptimizerSettings", True),
        ("from .parameter_optimizer_settings import SgdParameterOptimizerSettings", True),
        ("from .parameter_optimizer_construction import create_parameter_optimizer", True),
    ],
)
def test_parameter_optimizer_state_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/parameter_optimizer_state.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("import random", False),
        ("import dataclasses", False),
        ("import torch", False),
        ("import numpy", False),
        ("import federated_drift_experiment", False),
        (
            "from .held_model_training_batch_sampling import sample_training_batches_for_held_models",
            False,
        ),
        ("from .local_training_schedule_settings import LocalTrainingScheduleSettings", False),
        ("from .model_training_sample_records import Tensor", False),
        ("from .model_training_sample_records import SampledModelTrainingBatch", False),
        ("from .model_training_sample_records import _private", False),
        ("from .model_training_sample_records.child import ObservedTrainingSample", False),
        ("from federated_learning_experiments.runtime import run_experiment", False),
        (
            "from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor",
            False,
        ),
        ("from __future__ import annotations", True),
        ("from .model_training_sample_records import ObservedTrainingSample", True),
        ("from .model_training_sample_records import ModelTrainingSampleCollection", True),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
    ],
)
def test_model_training_sample_store_dependency_contract(source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path="learning/training/model_training_sample_store.py",
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        *[
            (source_module_path, source_text, False)
            for source_module_path in (
                "evaluation/model_evaluation_sample_records.py",
                "evaluation/model_evaluation_sample_store.py",
            )
            for source_text in (
                "import math",
                "import random",
                "from random import sample",
                "from random import SystemRandom",
                "import torch",
                "from torch import clone",
                "import numpy",
                "import federated_drift_experiment",
                "from dataclasses import asdict",
                "from torch import *",
                "from .model_evaluation_sample_records import _private",
                "from .model_evaluation_sample_records import Tensor",
                "from .model_evaluation_sample_records.child import ObservedEvaluationSample",
                "import federated_learning_experiments.evaluation.model_evaluation_sample_records",
                "from federated_learning_experiments.evaluation import model_evaluation_sample_records",
                "from federated_learning_experiments.runtime import run_experiment",
                "from ..learning.training.model_training_sample_store import ModelTrainingSampleStore",
                "from ..learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            )
        ],
        (
            "evaluation/model_evaluation_sample_records.py",
            "from dataclasses import dataclass",
            True,
        ),
        ("evaluation/model_evaluation_sample_records.py", "from torch import Tensor", True),
        ("evaluation/model_evaluation_sample_records.py", "from random import Random", False),
        ("evaluation/model_evaluation_sample_store.py", "from dataclasses import dataclass", False),
        ("evaluation/model_evaluation_sample_store.py", "from random import Random", True),
        ("evaluation/model_evaluation_sample_store.py", "from __future__ import annotations", True),
        (
            "evaluation/model_evaluation_sample_store.py",
            "from .model_evaluation_sample_records import ObservedEvaluationSample",
            True,
        ),
        (
            "evaluation/model_evaluation_sample_store.py",
            "from .model_evaluation_sample_records import ModelEvaluationSampleCollection",
            True,
        ),
    ],
)
def test_model_evaluation_sample_dependencies(source_module_path, source_text, expected_acceptance):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("import math", False),
        ("import random", False),
        ("from random import Random", False),
        ("from collections import Counter", False),
        ("import dataclasses", False),
        ("from dataclasses import asdict", False),
        ("from dataclasses import *", False),
        ("import numpy", False),
        ("import torch", False),
        ("from torch.optim import Optimizer", False),
        ("import federated_drift_experiment", False),
        ("from .local_training_request_schedule import LocalTrainingRequestSchedule", False),
        ("from .local_training_settings import LocalTrainingSettings", False),
        ("from .model_training_sample_store import ModelTrainingSampleStore", False),
        ("from federated_learning_experiments.runtime import run", False),
        ("from dataclasses import dataclass", True),
        ("from dataclasses import dataclass as Record", True),
        ("from __future__ import annotations", True),
    ],
)
@pytest.mark.parametrize(
    "source_module_path",
    [
        "learning/training/model_training_and_assignment_counts.py",
        "learning/training/current_training_model_assignment.py",
    ],
)
def test_training_state_owner_dataclass_only_dependency_contract(
    source_text, expected_acceptance, source_module_path
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path,
        source_text=source_text,
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        ("learning/training/local_training_schedule_settings.py", "import math", False),
        ("learning/training/local_training_schedule_settings.py", "import random", False),
        ("learning/training/local_training_schedule_settings.py", "import torch", False),
        (
            "learning/training/local_training_schedule_settings.py",
            "from dataclasses import asdict",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from ...core.settings_field_validation import fields",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from ...configuration.run_settings import ValidatedExperimentRunSettingsSubset",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from .held_model_joint_training_iterations import perform_held_model_joint_training_iterations",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from .local_training_schedule_settings import field",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "import federated_drift_experiment.config",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from .local_training_request_schedule import _private",
            False,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
            False,
        ),
        ("learning/training/local_training_schedule_settings.py", "import numpy", False),
        ("learning/training/local_training_request_schedule.py", "import math", False),
        ("learning/training/local_training_request_schedule.py", "import random", False),
        ("learning/training/local_training_request_schedule.py", "import torch", False),
        (
            "learning/training/local_training_request_schedule.py",
            "from dataclasses import asdict",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from ...core.settings_field_validation import fields",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from ...configuration.run_settings import ValidatedExperimentRunSettingsSubset",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from .held_model_joint_training_iterations import perform_held_model_joint_training_iterations",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from .local_training_schedule_settings import field",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "import federated_drift_experiment.config",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from .local_training_request_schedule import _private",
            False,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from ..models.residual_adapter_classifier import ResidualAdapterClassifier",
            False,
        ),
        ("learning/training/local_training_request_schedule.py", "import numpy", False),
        (
            "learning/training/local_training_schedule_settings.py",
            "from dataclasses import dataclass, field",
            True,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from ...core.settings_field_validation import validate_settings_field_values",
            True,
        ),
        (
            "learning/training/local_training_schedule_settings.py",
            "from __future__ import annotations",
            True,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from .local_training_schedule_settings import LocalTrainingScheduleSettings",
            True,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from federated_learning_experiments.learning.training.local_training_schedule_settings import LocalTrainingScheduleSettings",
            True,
        ),
        (
            "learning/training/local_training_request_schedule.py",
            "from __future__ import annotations",
            True,
        ),
    ],
)
def test_local_training_schedule_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    dependency_boundary_violations = collect_dependency_boundary_violations(
        source_module_path=source_module_path, source_text=source_text
    )
    assert (not dependency_boundary_violations) == expected_acceptance


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from dataclasses import dataclass",
            True,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from dataclasses import dataclass as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from dataclasses import field",
            True,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from dataclasses import field as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from federated_learning_experiments.core.settings_field_validation import validate_settings_field_values",
            True,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from federated_learning_experiments.core.settings_field_validation import validate_settings_field_values as AcceptedDependency",
            True,
        ),
        ("learning/training/candidate_epoch_training_settings.py", "import torch", False),
        ("learning/training/candidate_epoch_training_settings.py", "import math", False),
        ("learning/training/candidate_epoch_training_settings.py", "import dataclasses", False),
        ("learning/training/candidate_epoch_training_settings.py", "import random", False),
        ("learning/training/candidate_epoch_training_settings.py", "import numpy", False),
        ("learning/training/candidate_epoch_training_settings.py", "from torch import *", False),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from torch import manual_seed",
            False,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from torch import get_rng_state",
            False,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from torch import set_rng_state",
            False,
        ),
        ("learning/training/candidate_epoch_training_settings.py", "from torch import rand", False),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from torch.nn import Linear",
            False,
        ),
        ("learning/training/candidate_epoch_training_settings.py", "from math import sqrt", False),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from dataclasses import replace",
            False,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from federated_learning_experiments.runtime.candidate_classifier_construction import create_independent_candidate_training_state",
            False,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "import federated_drift_experiment",
            False,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import _validate_training_batch",
            False,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from federated_learning_experiments.learning.training import ParameterOptimizerState",
            False,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        (
            "learning/training/candidate_epoch_training_settings.py",
            "from .candidate_epoch_training_settings import CandidateEpochTrainingSettings",
            False,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from dataclasses import dataclass",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from dataclasses import dataclass as AcceptedDependency",
            True,
        ),
        ("learning/training/candidate_epoch_training.py", "from math import inf", True),
        (
            "learning/training/candidate_epoch_training.py",
            "from math import inf as AcceptedDependency",
            True,
        ),
        ("learning/training/candidate_epoch_training.py", "from torch import Tensor", True),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch import Tensor as AcceptedDependency",
            True,
        ),
        ("learning/training/candidate_epoch_training.py", "from torch import float32", True),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch import float32 as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch import is_grad_enabled",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch import is_grad_enabled as AcceptedDependency",
            True,
        ),
        ("learning/training/candidate_epoch_training.py", "from torch import isfinite", True),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch import isfinite as AcceptedDependency",
            True,
        ),
        ("learning/training/candidate_epoch_training.py", "from torch import randperm", True),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch import randperm as AcceptedDependency",
            True,
        ),
        ("learning/training/candidate_epoch_training.py", "from torch import strided", True),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch import strided as AcceptedDependency",
            True,
        ),
        ("learning/training/candidate_epoch_training.py", "from torch.optim import SGD", True),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch.optim import SGD as AcceptedDependency",
            True,
        ),
        ("learning/training/candidate_epoch_training.py", "from torch.optim import Adam", True),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch.optim import Adam as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch.utils.data import DataLoader",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch.utils.data import DataLoader as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch.utils.data import TensorDataset",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from torch.utils.data import TensorDataset as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import evaluate_classifier_per_sample_bounded_losses as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.local_training_settings import LocalTrainingSettings",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.local_training_settings import LocalTrainingSettings as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import ParameterOptimizerState",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import ParameterOptimizerState as AcceptedDependency",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.participating_model_training_batch import ParticipatingModelTrainingBatch",
            True,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.participating_model_training_batch import ParticipatingModelTrainingBatch as AcceptedDependency",
            True,
        ),
        ("learning/training/candidate_epoch_training.py", "import torch", False),
        ("learning/training/candidate_epoch_training.py", "import math", False),
        ("learning/training/candidate_epoch_training.py", "import dataclasses", False),
        ("learning/training/candidate_epoch_training.py", "import random", False),
        ("learning/training/candidate_epoch_training.py", "import numpy", False),
        ("learning/training/candidate_epoch_training.py", "from torch import *", False),
        ("learning/training/candidate_epoch_training.py", "from torch import manual_seed", False),
        ("learning/training/candidate_epoch_training.py", "from torch import get_rng_state", False),
        ("learning/training/candidate_epoch_training.py", "from torch import set_rng_state", False),
        ("learning/training/candidate_epoch_training.py", "from torch import rand", False),
        ("learning/training/candidate_epoch_training.py", "from torch.nn import Linear", False),
        ("learning/training/candidate_epoch_training.py", "from math import sqrt", False),
        ("learning/training/candidate_epoch_training.py", "from dataclasses import replace", False),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.runtime.candidate_classifier_construction import create_independent_candidate_training_state",
            False,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "import federated_drift_experiment",
            False,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training.joint_model_parameter_update import _validate_training_batch",
            False,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.learning.training import ParameterOptimizerState",
            False,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        (
            "learning/training/candidate_epoch_training.py",
            "from .candidate_epoch_training_settings import CandidateEpochTrainingSettings",
            True,
        ),
    ],
)
def test_candidate_epoch_training_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path=source_module_path, source_text=source_text
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_text,expected_acceptance",
    [
        ("from dataclasses import dataclass", True),
        ("from dataclasses import dataclass as AcceptedDependency", True),
        ("from torch import Tensor", True),
        ("from torch import Tensor as AcceptedDependency", True),
        ("from torch import float32", True),
        ("from torch import float32 as AcceptedDependency", True),
        ("from torch import is_grad_enabled", True),
        ("from torch import is_grad_enabled as AcceptedDependency", True),
        ("from torch import isfinite", True),
        ("from torch import isfinite as AcceptedDependency", True),
        ("from torch import strided", True),
        ("from torch import strided as AcceptedDependency", True),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from ..learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters",
            True,
        ),
        (
            "from ..learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.models.residual_adapter_classifier import ResidualAdapterClassifier",
            True,
        ),
        (
            "from ..learning.models.residual_adapter_classifier import ResidualAdapterClassifier as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training import CandidateEpochTrainingResult",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training import CandidateEpochTrainingResult as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.training.candidate_epoch_training import CandidateEpochTrainingResult",
            True,
        ),
        (
            "from ..learning.training.candidate_epoch_training import CandidateEpochTrainingResult as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training import train_candidate_classifier_epochs",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training import train_candidate_classifier_epochs as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.training.candidate_epoch_training import train_candidate_classifier_epochs",
            True,
        ),
        (
            "from ..learning.training.candidate_epoch_training import train_candidate_classifier_epochs as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings",
            True,
        ),
        (
            "from ..learning.training.candidate_epoch_training_settings import CandidateEpochTrainingSettings as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from ..learning.training.current_training_model_assignment import CurrentTrainingModelAssignment as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from ..learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import ObservedTrainingSample as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.training.model_training_sample_records import ObservedTrainingSample",
            True,
        ),
        (
            "from ..learning.training.model_training_sample_records import ObservedTrainingSample as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings",
            True,
        ),
        (
            "from ..learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import SgdParameterOptimizerSettings",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import SgdParameterOptimizerSettings as AcceptedDependency",
            True,
        ),
        (
            "from ..learning.training.parameter_optimizer_settings import SgdParameterOptimizerSettings",
            True,
        ),
        (
            "from ..learning.training.parameter_optimizer_settings import SgdParameterOptimizerSettings as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings as AcceptedDependency",
            True,
        ),
        (
            "from ..methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
            True,
        ),
        (
            "from ..methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import PostAlarmCandidateLossCollection",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import PostAlarmCandidateLossCollection as AcceptedDependency",
            True,
        ),
        (
            "from ..methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import PostAlarmCandidateLossCollection",
            True,
        ),
        (
            "from ..methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import PostAlarmCandidateLossCollection as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.candidate_classifier_construction import IndependentCandidateTrainingState",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.candidate_classifier_construction import IndependentCandidateTrainingState as AcceptedDependency",
            True,
        ),
        ("from .candidate_classifier_construction import IndependentCandidateTrainingState", True),
        (
            "from .candidate_classifier_construction import IndependentCandidateTrainingState as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.candidate_classifier_construction import create_independent_candidate_training_state",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.candidate_classifier_construction import create_independent_candidate_training_state as AcceptedDependency",
            True,
        ),
        (
            "from .candidate_classifier_construction import create_independent_candidate_training_state",
            True,
        ),
        (
            "from .candidate_classifier_construction import create_independent_candidate_training_state as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import FixedPostAlarmReferenceModels",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import FixedPostAlarmReferenceModels as AcceptedDependency",
            True,
        ),
        ("from .post_alarm_reference_model_fixation import FixedPostAlarmReferenceModels", True),
        (
            "from .post_alarm_reference_model_fixation import FixedPostAlarmReferenceModels as AcceptedDependency",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import fix_reference_models_at_alarm",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import fix_reference_models_at_alarm as AcceptedDependency",
            True,
        ),
        ("from .post_alarm_reference_model_fixation import fix_reference_models_at_alarm", True),
        (
            "from .post_alarm_reference_model_fixation import fix_reference_models_at_alarm as AcceptedDependency",
            True,
        ),
        ("import dataclasses", False),
        ("import dataclasses as AcceptedDependency", False),
        ("from dataclasses import *", False),
        ("from dataclasses import _private", False),
        ("import torch", False),
        ("import torch as AcceptedDependency", False),
        ("from torch import *", False),
        ("from torch import _private", False),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            False,
        ),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import _private",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.classifier_parameter_snapshot",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.classifier_parameter_snapshot as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import _private",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.residual_adapter_classifier import _private",
            False,
        ),
        ("import federated_learning_experiments.learning.training.candidate_epoch_training", False),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training import _private",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training_settings",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training_settings as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.candidate_epoch_training_settings import _private",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import _private",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import _private",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_records",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_records as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_records import _private",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import *",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_settings import _private",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import *",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import _private",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import *",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import _private",
            False,
        ),
        ("import federated_learning_experiments.runtime.candidate_classifier_construction", False),
        (
            "import federated_learning_experiments.runtime.candidate_classifier_construction as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.candidate_classifier_construction import *",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.candidate_classifier_construction import _private",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_reference_model_fixation",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_reference_model_fixation as AcceptedDependency",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import *",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import _private",
            False,
        ),
        ("from __future__ import annotations", False),
        ("import random", False),
        ("import numpy", False),
        ("import math", False),
        ("import pathlib", False),
        ("import os", False),
        ("from pathlib import Path", False),
        ("from copy import deepcopy", False),
        ("from dataclasses import replace", False),
        ("from torch import manual_seed", False),
        ("from torch import get_rng_state", False),
        ("from torch import set_rng_state", False),
        ("from torch import rand", False),
        ("from torch import randperm", False),
        ("from torch import save", False),
        ("from torch import load", False),
        ("from torch.nn import Linear", False),
        ("from torch.optim import Adam", False),
        ("import federated_drift_experiment", False),
        ("from federated_drift_experiment import config", False),
        (
            "from federated_learning_experiments.configuration.run_settings import RunSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.torch_random_state_scope import isolated_cpu_torch_random_state",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models.classifier_parameter_snapshot import _validate_classifier_parameter_snapshot_inputs",
            False,
        ),
        (
            "from federated_learning_experiments.learning.models import ResidualAdapterClassifier",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training import CandidateEpochTrainingSettings",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingState",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.parameter_optimizer_state import ParameterOptimizerState",
            False,
        ),
        (
            "from federated_learning_experiments.runtime import IndependentCandidateTrainingState",
            False,
        ),
        (
            "from .post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession",
            False,
        ),
        ("from ..models.residual_adapter_classifier import ResidualAdapterClassifier", False),
        ("from ..learning.models import residual_adapter_classifier", False),
        (
            "from ..learning.models.classifier_parameter_snapshot import snapshot_classifier_parameters, _private",
            False,
        ),
        ("from .candidate_classifier_construction import _private", False),
        ("from .post_alarm_reference_model_fixation import ResidualAdapterClassifier", False),
        (
            "from .post_alarm_candidate_validation_sample_observation import observe_post_alarm_candidate_validation_sample",
            False,
        ),
        (
            "from .post_alarm_candidate_validation_resolution import apply_post_alarm_candidate_validation_resolution",
            False,
        ),
        (
            "from ..methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot",
            False,
        ),
        (
            "from ..methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            False,
        ),
        ("from ..methods.fedsda.drift_detection import DriftDetectionSettings", False),
        (
            "from ..methods.fedsda.model_registration.pending_model_upload import PendingModelUpload",
            False,
        ),
        ("import dataclasses.dataclass", False),
        ("import dataclasses.dataclass as AcceptedDependency", False),
        ("import torch.Tensor", False),
        ("import torch.Tensor as AcceptedDependency", False),
        ("import torch.float32", False),
        ("import torch.float32 as AcceptedDependency", False),
        ("import torch.is_grad_enabled", False),
        ("import torch.is_grad_enabled as AcceptedDependency", False),
        ("import torch.isfinite", False),
        ("import torch.isfinite as AcceptedDependency", False),
        ("import torch.strided", False),
        ("import torch.strided as AcceptedDependency", False),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            False,
        ),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.classifier_parameter_snapshot.snapshot_classifier_parameters as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier",
            False,
        ),
        (
            "import federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training.CandidateEpochTrainingResult",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training.CandidateEpochTrainingResult as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training.train_candidate_classifier_epochs",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training.train_candidate_classifier_epochs as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection.PostAlarmCandidateLossCollection",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection.PostAlarmCandidateLossCollection as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.candidate_classifier_construction.IndependentCandidateTrainingState",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.candidate_classifier_construction.IndependentCandidateTrainingState as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.candidate_classifier_construction.create_independent_candidate_training_state",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.candidate_classifier_construction.create_independent_candidate_training_state as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_reference_model_fixation.FixedPostAlarmReferenceModels",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_reference_model_fixation.FixedPostAlarmReferenceModels as AcceptedDependency",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_reference_model_fixation.fix_reference_models_at_alarm",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_reference_model_fixation.fix_reference_models_at_alarm as AcceptedDependency",
            False,
        ),
    ],
)
def test_session_start_dependency_contract(source_text, expected_acceptance):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/post_alarm_candidate_validation_session_start.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    ("source_text", "expected_acceptance"),
    [
        ("from dataclasses import dataclass", True),
        ("from dataclasses import dataclass as AcceptedDependency", True),
        ("import dataclasses.dataclass", False),
        ("import dataclasses", False),
        ("from math import isfinite", True),
        ("from math import isfinite as AcceptedDependency", True),
        ("import math.isfinite", False),
        ("import math", False),
        ("from torch import Tensor", True),
        ("from torch import Tensor as AcceptedDependency", True),
        ("import torch.Tensor", False),
        ("import torch", False),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore",
            False,
        ),
        (
            "import federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.current_training_model_assignment import CurrentTrainingModelAssignment as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.current_training_model_assignment",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.held_model_training_state_registry import HeldModelTrainingStateRegistry as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.held_model_training_state_registry",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_and_assignment_counts import ModelTrainingAndAssignmentCountsStore as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_and_assignment_counts",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.model_training_sample_store import ModelTrainingSampleStore as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.model_training_sample_store",
            False,
        ),
        (
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import TemporaryModelIdAllocator",
            True,
        ),
        (
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation import TemporaryModelIdAllocator as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.learning.training.temporary_model_id_allocation.TemporaryModelIdAllocator",
            False,
        ),
        (
            "import federated_learning_experiments.learning.training.temporary_model_id_allocation",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import evaluate_candidate_using_post_alarm_losses as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation.evaluate_candidate_using_post_alarm_losses",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import PostAlarmCandidateValidationDecisionRecord",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import PostAlarmCandidateValidationDecisionRecord as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record.PostAlarmCandidateValidationDecisionRecord",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record",
            False,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import PendingModelUploadState as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload.PendingModelUploadState",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import PostAlarmCandidateValidationResolution",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import PostAlarmCandidateValidationResolution as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution.PostAlarmCandidateValidationResolution",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import apply_post_alarm_candidate_validation_resolution",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import apply_post_alarm_candidate_validation_resolution as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution.apply_post_alarm_candidate_validation_resolution",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation import observe_post_alarm_candidate_validation_sample",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation import observe_post_alarm_candidate_validation_sample as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation.observe_post_alarm_candidate_validation_sample",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession",
            True,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession",
            False,
        ),
        (
            "import federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start",
            False,
        ),
        ("from dataclasses import replace", False),
        ("from dataclasses import dataclass, replace", False),
        ("from __future__ import annotations", False),
        ("import math", False),
        ("import torch", False),
        ("from torch import rand", False),
        ("from pathlib import Path", False),
        ("from federated_drift_experiment import config", False),
        (
            "from federated_learning_experiments.runtime.candidate_classifier_construction import create_independent_candidate_training_state",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import fix_reference_models_at_alarm",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import _validate_resolution_inputs",
            False,
        ),
        ("from federated_learning_experiments.configuration import RunSettings", False),
        (
            "from .post_alarm_candidate_validation_session_start import PostAlarmCandidateValidationSession",
            True,
        ),
        ("from . import post_alarm_candidate_validation_resolution", False),
    ],
)
def test_validation_progress_dependency_contract(source_text, expected_acceptance):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="runtime/post_alarm_candidate_validation_progress.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    ("source_text", "expected_acceptance"),
    [
        ("from dataclasses import dataclass", True),
        ("from dataclasses import dataclass as AcceptedDependency", True),
        ("import dataclasses.dataclass", False),
        ("import dataclasses", False),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import PostAlarmCandidateLossEvaluation",
            True,
        ),
        (
            "from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import PostAlarmCandidateLossEvaluation as AcceptedDependency",
            True,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation.PostAlarmCandidateLossEvaluation",
            False,
        ),
        (
            "import federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation",
            False,
        ),
        ("from dataclasses import replace", False),
        ("from dataclasses import dataclass, replace", False),
        ("from __future__ import annotations", False),
        ("import math", False),
        ("import torch", False),
        ("from torch import rand", False),
        ("from pathlib import Path", False),
        ("from federated_drift_experiment import config", False),
        (
            "from federated_learning_experiments.runtime.candidate_classifier_construction import create_independent_candidate_training_state",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import fix_reference_models_at_alarm",
            False,
        ),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import _validate_resolution_inputs",
            False,
        ),
        ("from federated_learning_experiments.configuration import RunSettings", False),
        (
            "from .post_alarm_candidate_loss_evaluation import PostAlarmCandidateLossEvaluation",
            True,
        ),
        ("from . import post_alarm_candidate_loss_evaluation", False),
        ("from math import isfinite", False),
        (
            "from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import PostAlarmCandidateValidationProgress",
            False,
        ),
    ],
)
def test_validation_decision_record_dependency_contract(source_text, expected_acceptance):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="methods/fedsda/candidate_model_selection/post_alarm_candidate_validation_decision_record.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    ("source_text", "expected_acceptance"),
    [
        ("import math", True),
        ("import math as AcceptedDependency", True),
        ("from collections.abc import Iterable, Mapping", True),
        ("from collections.abc import Iterable as AcceptedDependency", True),
        ("from collections.abc import Mapping as AcceptedDependency", True),
        ("from math import exp", False),
        ("from math import inf", False),
        ("import math.exp", False),
        ("import math.child", False),
        ("import collections.abc", False),
        ("import collections.abc.Mapping", False),
        ("from collections.abc import Sequence", False),
        ("from collections.abc import Mapping, Sequence", False),
        ("from collections.abc.child import Mapping", False),
        ("from __future__ import annotations", False),
        ("import torch", False),
        ("import numpy", False),
        ("from federated_drift_experiment.expert_routing import AdaHedgeRouter", False),
        ("from federated_learning_experiments import runtime", False),
        ("from federated_learning_experiments.configuration import RunSettings", False),
        ("from ..runtime import single_run_execution", False),
        ("from . import adaptation_record_store", False),
    ],
)
def test_adahedge_diagnostic_evidence_exact_dependency_contract(source_text, expected_acceptance):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path="evaluation/adahedge_diagnostic_evidence.py",
            source_text=source_text,
        )
    ) == expected_acceptance


@pytest.mark.parametrize(
    "source_module_path,source_text,expected_acceptance",
    [
        (source_module_path, source_text, expected_acceptance)
        for source_module_path in (
            "evaluation/adahedge_diagnostic_evidence_collection.py",
            "runtime/training_assignment_diagnostic_notification.py",
        )
        for source_text, expected_acceptance in (
            (
                "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence "
                "import AdaHedgeDiagnosticEvidence",
                source_module_path == "evaluation/adahedge_diagnostic_evidence_collection.py",
            ),
            (
                "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence "
                "import AdaHedgeDiagnosticEvidence as AcceptedDependency",
                source_module_path == "evaluation/adahedge_diagnostic_evidence_collection.py",
            ),
            (
                "from ..evaluation.adahedge_diagnostic_evidence import AdaHedgeDiagnosticEvidence",
                source_module_path == "evaluation/adahedge_diagnostic_evidence_collection.py",
            ),
            (
                "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection "
                "import AdaHedgeDiagnosticEvidenceCollection",
                source_module_path == "runtime/training_assignment_diagnostic_notification.py",
            ),
            (
                "from ..evaluation.adahedge_diagnostic_evidence_collection "
                "import AdaHedgeDiagnosticEvidenceCollection as AcceptedDependency",
                source_module_path == "runtime/training_assignment_diagnostic_notification.py",
            ),
            (
                "from federated_learning_experiments.learning.training.current_training_model_assignment "
                "import TrainingModelAssignmentChange",
                source_module_path == "runtime/training_assignment_diagnostic_notification.py",
            ),
            (
                "from ..learning.training.current_training_model_assignment "
                "import TrainingModelAssignmentChange as AcceptedDependency",
                source_module_path == "runtime/training_assignment_diagnostic_notification.py",
            ),
            ("import math", False),
            ("from math import exp", False),
            ("from __future__ import annotations", False),
            ("import torch", False),
            ("import numpy", False),
            ("import random", False),
            ("from collections import defaultdict", False),
            ("from federated_drift_experiment.expert_routing import AdaHedgeRouter", False),
            ("from federated_learning_experiments.configuration import RunSettings", False),
            ("from . import single_run_execution", False),
            (
                "import federated_learning_experiments.evaluation.adahedge_diagnostic_evidence",
                False,
            ),
            (
                "import federated_learning_experiments.evaluation.adahedge_diagnostic_evidence"
                ".AdaHedgeDiagnosticEvidence",
                False,
            ),
            (
                "import federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection"
                ".AdaHedgeDiagnosticEvidenceCollection as AcceptedDependency",
                False,
            ),
            (
                "import federated_learning_experiments.learning.training.current_training_model_assignment"
                ".TrainingModelAssignmentChange",
                False,
            ),
            (
                "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence "
                "import AdaHedgeDiagnosticEvidence, OtherDependency",
                False,
            ),
            (
                "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence.child "
                "import AdaHedgeDiagnosticEvidence",
                False,
            ),
            (
                "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection "
                "import AdaHedgeDiagnosticEvidenceCollection, OtherDependency",
                False,
            ),
            (
                "from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection.child "
                "import AdaHedgeDiagnosticEvidenceCollection",
                False,
            ),
            (
                "from federated_learning_experiments.learning.training.current_training_model_assignment "
                "import TrainingModelAssignmentChange, CurrentTrainingModelAssignment",
                False,
            ),
            (
                "from federated_learning_experiments.learning.training.current_training_model_assignment.child "
                "import TrainingModelAssignmentChange",
                False,
            ),
            ("from ..evaluation import adahedge_diagnostic_evidence", False),
            ("from ..evaluation import adahedge_diagnostic_evidence_collection", False),
            ("from ..learning.training import current_training_model_assignment", False),
        )
    ],
)
def test_held_adahedge_diagnostic_exact_dependency_contract(
    source_module_path, source_text, expected_acceptance
):
    assert (
        not collect_dependency_boundary_violations(
            source_module_path=source_module_path, source_text=source_text
        )
    ) == expected_acceptance
