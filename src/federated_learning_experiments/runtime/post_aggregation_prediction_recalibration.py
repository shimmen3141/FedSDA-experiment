"""集約（と配布）の後で、client 1つの予測の重みと診断証拠を、現在の保有モデルに合わせて再較正する。"""

from dataclasses import dataclass

from torch import cat

from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import (
    AdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights import (
    FixedSharePredictionWeightController,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store import (
    PendingSampleObservationStore,
)


@dataclass(frozen=True, kw_only=True)
class PostAggregationPredictionRecalibration:
    """再較正1回の結果。"""

    # 再生した損失の列の長さ（保留中の標本の数）。保留中の標本がないか、保有モデルが1つ以下なら0。
    replayed_sample_count: int
    # 損失の列に含めたモデルのID（昇順）。列が空なら空。
    replayed_model_ids: tuple[int, ...]
    # 再始動した、真の概念別の診断証拠の概念ID（作成順）。
    restarted_true_concept_ids: tuple[int, ...]


def _compute_pending_sample_loss_sequence(
    *,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    pending_sample_observation_store: PendingSampleObservationStore,
) -> tuple[dict[int, float], ...]:
    """保留中の標本の、保有する全モデルの有界損失を、標本の観測順の「モデルIDから損失」の列にする。"""
    pending_sample_observations = (
        pending_sample_observation_store.snapshot_pending_sample_observations()
    )
    held_model_training_states = sorted(
        held_model_training_state_registry.snapshot_ordered_held_model_training_states(),
        key=lambda held_model_training_state: held_model_training_state.model_id,
    )
    if not pending_sample_observations or len(held_model_training_states) <= 1:
        return ()
    input_features = cat(
        [
            pending_observation.training_sample.input_features
            for pending_observation in pending_sample_observations
        ]
    )
    observed_class_labels = cat(
        [
            pending_observation.training_sample.observed_class_labels
            for pending_observation in pending_sample_observations
        ]
    )
    # 全モデルの損失を計算し終えてから、列を作る（どれかのモデルで失敗したら、何も返さない）。
    per_sample_losses_by_model_id = {
        held_model_training_state.model_id: evaluate_classifier_per_sample_bounded_losses(
            classifier=held_model_training_state.classifier,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
        )
        for held_model_training_state in held_model_training_states
    }
    return tuple(
        {
            model_id: float(per_sample_losses[sample_position].item())
            for model_id, per_sample_losses in per_sample_losses_by_model_id.items()
        }
        for sample_position in range(len(pending_sample_observations))
    )


def recalibrate_prediction_state_after_aggregation(
    *,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    pending_sample_observation_store: PendingSampleObservationStore,
    fixed_share_prediction_weight_controller: FixedSharePredictionWeightController,
    diagnostic_evidence_collection: AdaHedgeDiagnosticEvidenceCollection,
) -> PostAggregationPredictionRecalibration:
    """保留中の標本の損失を現在の保有モデルで計算し直し、その列で、診断証拠と予測の重みを作り直す。

    globalの診断証拠の再生→真の概念別の診断証拠の再始動→Fixed-Shareの重みの再生、の順に行う。
    損失の列が空（保留中の標本がないか、保有モデルが1つ以下）なら、再生は何も変えない。
    真の概念別の診断証拠は、列が空でも再始動する。モデルと、ほかのownerは変えない。
    """
    for owner_name, owner, required_type in (
        (
            "held_model_training_state_registry",
            held_model_training_state_registry,
            HeldModelTrainingStateRegistry,
        ),
        (
            "pending_sample_observation_store",
            pending_sample_observation_store,
            PendingSampleObservationStore,
        ),
        (
            "fixed_share_prediction_weight_controller",
            fixed_share_prediction_weight_controller,
            FixedSharePredictionWeightController,
        ),
        (
            "diagnostic_evidence_collection",
            diagnostic_evidence_collection,
            AdaHedgeDiagnosticEvidenceCollection,
        ),
    ):
        if type(owner) is not required_type:
            raise TypeError(f"{owner_name} must be exact {required_type.__name__}")
    observed_loss_sequence = _compute_pending_sample_loss_sequence(
        held_model_training_state_registry=held_model_training_state_registry,
        pending_sample_observation_store=pending_sample_observation_store,
    )
    restarted_true_concept_ids = diagnostic_evidence_collection.created_true_concept_ids
    recalibration = PostAggregationPredictionRecalibration(
        replayed_sample_count=len(observed_loss_sequence),
        replayed_model_ids=tuple(observed_loss_sequence[0]) if observed_loss_sequence else (),
        restarted_true_concept_ids=restarted_true_concept_ids,
    )
    diagnostic_evidence_collection.global_diagnostic_evidence.replay_observed_losses_after_aggregation(
        observed_loss_sequence=observed_loss_sequence
    )
    # 真の概念別の証拠は、集約の前の表現で得た比較なので、再生せずに捨てる。
    for true_concept_id in restarted_true_concept_ids:
        diagnostic_evidence_collection.get_true_concept_diagnostic_evidence(
            true_concept_id=true_concept_id
        ).restart_evidence_after_aggregation()
    fixed_share_prediction_weight_controller.replay_observed_losses_after_aggregation(
        observed_loss_sequence=observed_loss_sequence
    )
    return recalibration
