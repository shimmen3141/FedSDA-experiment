"""確定した学習帰属変更を全標本のAdaHedgeの証拠だけへ通知する。"""

from federated_learning_experiments.evaluation.adahedge_weight_evidence_collection import (
    AdaHedgeWeightEvidenceCollection,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    TrainingModelAssignmentChange,
)


def restart_adahedge_weight_evidence_after_training_assignment_change(
    *,
    assignment_change: TrainingModelAssignmentChange | None,
    adahedge_weight_evidence_collection: AdaHedgeWeightEvidenceCollection,
) -> None:
    """全入力を先行検証し、異なるIDの通知一回につき一回再始動する。"""
    if type(adahedge_weight_evidence_collection) is not AdaHedgeWeightEvidenceCollection:
        raise TypeError("AdaHedgeの証拠の保持集合はAdaHedgeWeightEvidenceCollectionが必要です。")
    if assignment_change is None:
        return
    if type(assignment_change) is not TrainingModelAssignmentChange:
        raise TypeError("変更情報はTrainingModelAssignmentChangeまたはNoneが必要です。")
    for model_id in (assignment_change.previous_model_id, assignment_change.current_model_id):
        if type(model_id) is not int:
            raise TypeError("モデルIDはbool・派生型以外のbuiltin intが必要です。")
    # 予測の処理が証拠を更新しない構成では、使われない証拠を、始め直さない。
    if not adahedge_weight_evidence_collection.evidence_is_updated_by_prediction:
        return
    if assignment_change.previous_model_id != assignment_change.current_model_id:
        adahedge_weight_evidence_collection.all_sample_weight_evidence.restart_evidence_after_concept_operation()
