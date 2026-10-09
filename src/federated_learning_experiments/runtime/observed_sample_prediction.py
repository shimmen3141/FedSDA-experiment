"""観測した標本1件を予測する: 全モデルの評価→重み付き予測→記録→ラベル観測後の重みと診断証拠の更新。"""

from dataclasses import dataclass

from torch import Tensor, float32, isfinite, no_grad, strided

from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import (
    AdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.evaluation.sample_prediction_record_store import (
    SamplePredictionRecord,
    SamplePredictionRecordStore,
)
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    combine_model_prediction_probabilities,
    compute_model_mean_bounded_losses_after_label_observation,
    convert_model_outputs_to_prediction_probabilities,
    normalize_model_prediction_weights,
    predict_class_labels_from_prediction_scores,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights import (
    FixedSharePredictionWeightController,
)


@dataclass(frozen=True, kw_only=True)
class ObservedSamplePrediction:
    """標本1件の予測の結果。重みは、結合に使った正規化後の値（Fixed-Shareと、globalの診断）。"""

    sample_prediction_record: SamplePredictionRecord
    predicted_class_labels: Tensor
    combined_prediction_probabilities: Tensor
    prediction_probabilities_by_model_id: dict[int, Tensor]
    prediction_weights_by_model_id: dict[int, float]
    global_diagnostic_weights_by_model_id: dict[int, float]
    observed_losses_by_model_id: dict[int, float]


# 引数名から、予測が受け取るownerのexact型への対応（最初の状態更新より前に確かめる）。
_REQUIRED_OWNER_TYPES_BY_ARGUMENT_NAME = {
    "fixed_share_prediction_weight_controller": FixedSharePredictionWeightController,
    "diagnostic_evidence_collection": AdaHedgeDiagnosticEvidenceCollection,
    "sample_prediction_record_store": SamplePredictionRecordStore,
    "held_model_training_state_registry": HeldModelTrainingStateRegistry,
    "current_training_model_assignment": CurrentTrainingModelAssignment,
}


def _validate_observed_sample_prediction_inputs(
    *,
    indexed_observation: IndexedObservedTrainingSample,
    owners_by_argument_name: dict[str, object],
    sample_prediction_record_store: SamplePredictionRecordStore,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> None:
    for owner_argument_name, required_owner_type in _REQUIRED_OWNER_TYPES_BY_ARGUMENT_NAME.items():
        if type(owners_by_argument_name[owner_argument_name]) is not required_owner_type:
            raise TypeError(f"{owner_argument_name} must be exact {required_owner_type.__name__}")
    if type(indexed_observation) is not IndexedObservedTrainingSample:
        raise TypeError("indexed_observation must be exact IndexedObservedTrainingSample")
    sample_index = indexed_observation.sample_index
    if type(sample_index) is not int:
        raise TypeError("sample_index must be builtin int")
    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")
    if indexed_observation.observed_concept_id is not None and (
        type(indexed_observation.observed_concept_id) is not int
    ):
        raise TypeError("observed_concept_id must be builtin int or None")
    training_sample = indexed_observation.training_sample
    if type(training_sample) is not ObservedTrainingSample:
        raise TypeError("training_sample must be exact ObservedTrainingSample")
    input_features = training_sample.input_features
    observed_class_labels = training_sample.observed_class_labels
    if type(input_features) is not Tensor or type(observed_class_labels) is not Tensor:
        raise TypeError("input_features and observed_class_labels must be exact torch.Tensor")
    if input_features.ndim != 2 or input_features.shape[0] != 1:
        raise ValueError("input_features must hold exactly one sample as a 2D tensor")
    if tuple(observed_class_labels.shape) != (1, 1):
        raise ValueError("observed_class_labels must have shape (1, 1)")
    # 特徴の数とdtypeは共有特徴抽出部が、ラベルの範囲と整数値は損失の計算が確かめる（どちらも状態更新の前）。
    # ここでは、後の段（損失の監視、候補検証）だけが確かめる条件を、先に確かめる。
    if not isfinite(input_features).all().item():
        raise ValueError("input_features must be finite")
    if (
        observed_class_labels.device.type != "cpu"
        or observed_class_labels.dtype != float32
        or observed_class_labels.layout != strided
        or observed_class_labels.is_nested
    ):
        raise ValueError("observed_class_labels must be a CPU float32 strided tensor")
    # 位置の連続性: 記録の追加が同じ理由で拒否すると、重みと証拠の同期だけが済んだ状態が残るので、先に確かめる。
    last_recorded_sample_index = sample_prediction_record_store.last_recorded_sample_index
    if last_recorded_sample_index is not None and sample_index != last_recorded_sample_index + 1:
        raise ValueError("sample_index must follow the last recorded sample index by one")
    held_model_ids = tuple(
        held_model_training_state.model_id
        for held_model_training_state in held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )
    if not held_model_ids:
        raise ValueError("at least one held model is required for prediction")
    if current_training_model_assignment.current_training_model_id not in held_model_ids:
        raise ValueError("the current training model must be a held model")


def _combine_and_compare_with_observed_class(
    *,
    prediction_probabilities_by_model_id: dict[int, Tensor],
    normalized_weights_by_model_id: dict[int, float],
    class_count: int,
    observed_class_labels: Tensor,
) -> tuple[Tensor, Tensor, bool]:
    """重みで確率を結合し、（結合した確率、予測クラス、観測クラスと同じか）を返す。"""
    combined_prediction_probabilities = combine_model_prediction_probabilities(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        prediction_weights_by_model_id=normalized_weights_by_model_id,
        class_count=class_count,
    )
    predicted_class_labels = predict_class_labels_from_prediction_scores(
        prediction_scores=combined_prediction_probabilities, class_count=class_count
    )
    return (
        combined_prediction_probabilities,
        predicted_class_labels,
        bool(predicted_class_labels.view(-1)[0].item() == observed_class_labels.view(-1)[0].item()),
    )


def predict_observed_sample_and_update_prediction_weights(
    *,
    indexed_observation: IndexedObservedTrainingSample,
    fixed_share_prediction_weight_controller: FixedSharePredictionWeightController,
    diagnostic_evidence_collection: AdaHedgeDiagnosticEvidenceCollection,
    sample_prediction_record_store: SamplePredictionRecordStore,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> ObservedSamplePrediction:
    """保有する全モデルの確率をFixed-Shareの重みで結合して予測し、記録を足してから、重みを更新する。

    診断証拠（globalと、真の概念IDがあればその概念）も、同じ損失で更新する。旧の最終構成の予測と同じ数値になる。
    """
    _validate_observed_sample_prediction_inputs(
        indexed_observation=indexed_observation,
        owners_by_argument_name=dict(
            fixed_share_prediction_weight_controller=fixed_share_prediction_weight_controller,
            diagnostic_evidence_collection=diagnostic_evidence_collection,
            sample_prediction_record_store=sample_prediction_record_store,
            held_model_training_state_registry=held_model_training_state_registry,
            current_training_model_assignment=current_training_model_assignment,
        ),
        sample_prediction_record_store=sample_prediction_record_store,
        held_model_training_state_registry=held_model_training_state_registry,
        current_training_model_assignment=current_training_model_assignment,
    )
    input_features = indexed_observation.training_sample.input_features
    observed_class_labels = indexed_observation.training_sample.observed_class_labels
    observed_concept_id = indexed_observation.observed_concept_id
    current_training_model_id = current_training_model_assignment.current_training_model_id
    # (2) 全モデルの評価。状態を変えない。共有特徴は、ID昇順で最初のモデルで1回だけ計算する。
    classifiers_by_model_id = {
        held_model_training_state.model_id: held_model_training_state.classifier
        for held_model_training_state in held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    }
    model_ids = tuple(sorted(classifiers_by_model_id))
    class_count = classifiers_by_model_id[model_ids[0]].class_count
    with no_grad():
        shared_features = classifiers_by_model_id[model_ids[0]].extract_shared_features(
            input_features
        )
        model_outputs_by_model_id = {
            model_id: classifiers_by_model_id[model_id].forward_from_shared_features(
                shared_features
            )
            for model_id in model_ids
        }
    prediction_probabilities_by_model_id = convert_model_outputs_to_prediction_probabilities(
        model_outputs_by_model_id=model_outputs_by_model_id, class_count=class_count
    )
    observed_losses_by_model_id = compute_model_mean_bounded_losses_after_label_observation(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        observed_class_labels=observed_class_labels,
        class_count=class_count,
    )
    observed_class_value = observed_class_labels.view(-1)[0].item()
    model_prediction_is_correct_by_model_id: dict[int, bool] = {}
    prediction_confidences_by_model_id: dict[int, float] = {}
    for model_id in model_ids:
        model_prediction_probabilities = prediction_probabilities_by_model_id[model_id]
        model_prediction_is_correct_by_model_id[model_id] = bool(
            predict_class_labels_from_prediction_scores(
                prediction_scores=model_prediction_probabilities, class_count=class_count
            )
            .view(-1)[0]
            .item()
            == observed_class_value
        )
        # 確信度: 2値は確率と0.5の差、多クラスは最大のクラス確率。
        prediction_confidences_by_model_id[model_id] = float(
            (model_prediction_probabilities.view(-1) - 0.5).abs().mean().item()
            if class_count == 2
            else model_prediction_probabilities.max(dim=1).values.mean().item()
        )
    # (3) 観測前の重み。モデル集合が変わっていれば、それぞれのownerが初期化する（最初の状態更新）。
    global_diagnostic_evidence = diagnostic_evidence_collection.global_diagnostic_evidence
    global_diagnostic_weights_by_model_id = normalize_model_prediction_weights(
        prediction_weights_by_model_id=global_diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
            model_ids=model_ids
        )
    )
    prediction_weights_by_model_id = normalize_model_prediction_weights(
        prediction_weights_by_model_id=fixed_share_prediction_weight_controller.get_prediction_weights_before_label_observation(
            model_ids=model_ids
        )
    )
    true_concept_diagnostic_evidence = None
    unnormalized_true_concept_diagnostic_weights_by_model_id = None
    true_concept_diagnostic_prediction_is_correct = None
    if observed_concept_id is not None:
        true_concept_diagnostic_evidence = (
            diagnostic_evidence_collection.get_true_concept_diagnostic_evidence(
                true_concept_id=observed_concept_id
            )
        )
        unnormalized_true_concept_diagnostic_weights_by_model_id = (
            true_concept_diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
                model_ids=model_ids
            )
        )
        # (4) 結合、予測クラス、記録の項目。
        _, _, true_concept_diagnostic_prediction_is_correct = (
            _combine_and_compare_with_observed_class(
                prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
                normalized_weights_by_model_id=normalize_model_prediction_weights(
                    prediction_weights_by_model_id=unnormalized_true_concept_diagnostic_weights_by_model_id
                ),
                class_count=class_count,
                observed_class_labels=observed_class_labels,
            )
        )
    _, _, global_diagnostic_prediction_is_correct = _combine_and_compare_with_observed_class(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        normalized_weights_by_model_id=global_diagnostic_weights_by_model_id,
        class_count=class_count,
        observed_class_labels=observed_class_labels,
    )
    combined_prediction_probabilities, predicted_class_labels, combined_prediction_is_correct = (
        _combine_and_compare_with_observed_class(
            prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
            normalized_weights_by_model_id=prediction_weights_by_model_id,
            class_count=class_count,
            observed_class_labels=observed_class_labels,
        )
    )
    maximum_weight_model_id = FixedSharePredictionWeightController.select_maximum_weight_model_id(
        prediction_weights_by_model_id=prediction_weights_by_model_id,
        preferred_model_id=current_training_model_id,
    )
    maximum_prediction_confidence = max(prediction_confidences_by_model_id.values())
    # 確信度が同率なら、予測重みが大きいもの、次に現在の学習帰属のモデル、次にIDが小さいもの。
    highest_confidence_model_id = max(
        (
            model_id
            for model_id in model_ids
            if prediction_confidences_by_model_id[model_id] == maximum_prediction_confidence
        ),
        key=lambda model_id: (
            prediction_weights_by_model_id[model_id],
            model_id == current_training_model_id,
            -model_id,
        ),
    )
    sample_prediction_record = SamplePredictionRecord(
        sample_index=indexed_observation.sample_index,
        observed_concept_id=observed_concept_id,
        observed_class_id=int(observed_class_value),
        combined_prediction_is_correct=combined_prediction_is_correct,
        maximum_weight_model_id=maximum_weight_model_id,
        maximum_prediction_weight=max(prediction_weights_by_model_id.values()),
        effective_model_count=1.0
        / sum(
            prediction_weight * prediction_weight
            for prediction_weight in prediction_weights_by_model_id.values()
        ),
        any_model_or_combined_prediction_is_correct=(
            combined_prediction_is_correct or any(model_prediction_is_correct_by_model_id.values())
        ),
        maximum_weight_model_prediction_is_correct=model_prediction_is_correct_by_model_id[
            maximum_weight_model_id
        ],
        global_diagnostic_prediction_is_correct=global_diagnostic_prediction_is_correct,
        true_concept_diagnostic_prediction_is_correct=true_concept_diagnostic_prediction_is_correct,
        highest_confidence_model_prediction_is_correct=model_prediction_is_correct_by_model_id[
            highest_confidence_model_id
        ],
    )
    # (5) 記録を足す。
    sample_prediction_record_store.append_sample_prediction_record(
        sample_prediction_record=sample_prediction_record
    )
    # (6) ラベル観測後の更新。真の概念別の証拠だけは、旧と同じく、正規化前の重みで更新する。
    global_diagnostic_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id=observed_losses_by_model_id,
        diagnostic_weights_by_model_id=global_diagnostic_weights_by_model_id,
    )
    if (
        true_concept_diagnostic_evidence is not None
        and unnormalized_true_concept_diagnostic_weights_by_model_id is not None
    ):
        true_concept_diagnostic_evidence.update_evidence_after_loss_observation(
            observed_losses_by_model_id=observed_losses_by_model_id,
            diagnostic_weights_by_model_id=unnormalized_true_concept_diagnostic_weights_by_model_id,
        )
    fixed_share_prediction_weight_controller.update_weights_after_loss_observation(
        observed_losses_by_model_id=observed_losses_by_model_id,
        prediction_weights_by_model_id=prediction_weights_by_model_id,
    )
    return ObservedSamplePrediction(
        sample_prediction_record=sample_prediction_record,
        predicted_class_labels=predicted_class_labels,
        combined_prediction_probabilities=combined_prediction_probabilities,
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        prediction_weights_by_model_id=prediction_weights_by_model_id,
        global_diagnostic_weights_by_model_id=global_diagnostic_weights_by_model_id,
        observed_losses_by_model_id=observed_losses_by_model_id,
    )
