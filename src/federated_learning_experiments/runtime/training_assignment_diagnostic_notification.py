"""確定した学習帰属変更をglobal診断証拠だけへ通知する。"""

from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import (
    AdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    TrainingModelAssignmentChange,
)


def notify_diagnostics_of_training_assignment_change(
    *,
    assignment_change: TrainingModelAssignmentChange | None,
    diagnostic_evidence_collection: AdaHedgeDiagnosticEvidenceCollection,
) -> None:
    """全入力を先行検証し、異なるIDの通知一回につき一回再始動する。"""
    if type(diagnostic_evidence_collection) is not AdaHedgeDiagnosticEvidenceCollection:
        raise TypeError("診断証拠の保持集合はAdaHedgeDiagnosticEvidenceCollectionが必要です。")
    if assignment_change is None:
        return
    if type(assignment_change) is not TrainingModelAssignmentChange:
        raise TypeError("変更情報はTrainingModelAssignmentChangeまたはNoneが必要です。")
    for model_id in (assignment_change.previous_model_id, assignment_change.current_model_id):
        if type(model_id) is not int:
            raise TypeError("モデルIDはbool・派生型以外のbuiltin intが必要です。")
    if assignment_change.previous_model_id != assignment_change.current_model_id:
        diagnostic_evidence_collection.global_diagnostic_evidence.restart_evidence_after_concept_operation()
