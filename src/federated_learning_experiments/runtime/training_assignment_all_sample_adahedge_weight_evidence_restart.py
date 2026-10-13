"""確定した学習帰属変更を全標本のAdaHedgeの証拠だけへ通知する。"""

from federated_learning_experiments.learning.training.current_training_model_assignment import (
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.all_sample_adahedge_weight_evidence_holder import (
    AllSampleAdaHedgeWeightEvidenceHolder,
)


def restart_all_sample_adahedge_weight_evidence_after_training_assignment_change(
    *,
    assignment_change: TrainingModelAssignmentChange | None,
    all_sample_adahedge_weight_evidence_holder: AllSampleAdaHedgeWeightEvidenceHolder,
) -> None:
    """全入力を先行検証し、異なるIDの通知一回につき一回再始動する。"""
    if (
        type(all_sample_adahedge_weight_evidence_holder)
        is not AllSampleAdaHedgeWeightEvidenceHolder
    ):
        raise TypeError(
            "全標本のAdaHedgeの証拠のownerはAllSampleAdaHedgeWeightEvidenceHolderが必要です。"
        )
    if assignment_change is None:
        return
    if type(assignment_change) is not TrainingModelAssignmentChange:
        raise TypeError("変更情報はTrainingModelAssignmentChangeまたはNoneが必要です。")
    for model_id in (assignment_change.previous_model_id, assignment_change.current_model_id):
        if type(model_id) is not int:
            raise TypeError("モデルIDはbool・派生型以外のbuiltin intが必要です。")
    # 予測の処理が証拠を更新しない構成では、使われない証拠を、始め直さない。
    if not all_sample_adahedge_weight_evidence_holder.evidence_is_updated_by_prediction:
        return
    if assignment_change.previous_model_id != assignment_change.current_model_id:
        all_sample_adahedge_weight_evidence_holder.all_sample_weight_evidence.restart_evidence_after_concept_operation()
