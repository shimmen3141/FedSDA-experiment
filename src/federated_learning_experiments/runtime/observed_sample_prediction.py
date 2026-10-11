"""観測した標本1件を予測する: 全モデルの評価→重み付き予測→記録→ラベル観測後の重みとAdaHedgeの証拠の更新。"""

from dataclasses import dataclass

from torch import Tensor, float32, isfinite, no_grad, strided

from federated_learning_experiments.evaluation.adahedge_weight_evidence_collection import (
    AdaHedgeWeightEvidenceCollection,
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
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
    FixedShareWeightController,
)


@dataclass(frozen=True, kw_only=True)
class ObservedSamplePrediction:
    """標本1件の予測の結果。重みは、正規化後の値。

    `prediction_weights_by_model_id`は、予測に使った重み（設定により、Fixed-Shareの重みか、AdaHedgeの重み）。
    """

    sample_prediction_record: SamplePredictionRecord
    predicted_class_labels: Tensor
    combined_prediction_probabilities: Tensor
    prediction_probabilities_by_model_id: dict[int, Tensor]
    prediction_weights_by_model_id: dict[int, float]
    fixed_share_weights_by_model_id: dict[int, float]
    adahedge_weights_by_model_id: dict[int, float]
    observed_losses_by_model_id: dict[int, float]


# 引数名から、予測が受け取るownerのexact型への対応（最初の状態更新より前に確かめる）。
_REQUIRED_OWNER_TYPES_BY_ARGUMENT_NAME = {
    "fixed_share_weight_controller": FixedShareWeightController,
    "adahedge_weight_evidence_collection": AdaHedgeWeightEvidenceCollection,
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
    """重みで確率を結合し、（結合した確率、予測クラス、観測クラスと同じか）を返す。

    確率は、出力から確率への変換（`convert_model_outputs_to_prediction_probabilities`）で検査済みのものを、
    重みは、正規化（`normalize_model_prediction_weights`）で検査済みのものを、渡すこと。
    """
    combined_prediction_probabilities = combine_model_prediction_probabilities(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        prediction_weights_by_model_id=normalized_weights_by_model_id,
        class_count=class_count,
        prediction_probabilities_are_validated=True,
    )
    # 検査済みの確率の、検査済みの重みでの結合なので、結果の中身は、検査し直さない。
    predicted_class_labels = predict_class_labels_from_prediction_scores(
        prediction_scores=combined_prediction_probabilities,
        class_count=class_count,
        prediction_scores_are_validated=True,
    )
    return (
        combined_prediction_probabilities,
        predicted_class_labels,
        bool(predicted_class_labels.view(-1)[0].item() == observed_class_labels.view(-1)[0].item()),
    )


def _predict_with_current_training_model_only(
    *,
    indexed_observation: IndexedObservedTrainingSample,
    sample_prediction_record_store: SamplePredictionRecordStore,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> ObservedSamplePrediction:
    """現在の学習帰属のモデルだけを評価して予測し、記録を足す。重みと証拠は、作らない・更新しない。

    記録と戻り値の、重みに関する項目は、「現在のモデルに重み1」の値にする: 最大重みのモデルは現在のモデル、
    最大の重みと実効モデル数は1、どの重みでの結合の正否も、現在のモデルの予測の正否。
    入力の検査は、呼出し元（`predict_observed_sample_and_update_prediction_weights`）が済ませている。
    """
    input_features = indexed_observation.training_sample.input_features
    observed_class_labels = indexed_observation.training_sample.observed_class_labels
    observed_concept_id = indexed_observation.observed_concept_id
    current_training_model_id = current_training_model_assignment.current_training_model_id
    current_classifier = held_model_training_state_registry.get_held_model_training_state(
        model_id=current_training_model_id
    ).classifier
    class_count = current_classifier.class_count
    with no_grad():
        model_outputs_by_model_id = {
            current_training_model_id: current_classifier.forward_from_shared_features(
                current_classifier.extract_shared_features(input_features)
            )
        }
    prediction_probabilities_by_model_id = convert_model_outputs_to_prediction_probabilities(
        model_outputs_by_model_id=model_outputs_by_model_id, class_count=class_count
    )
    observed_losses_by_model_id = compute_model_mean_bounded_losses_after_label_observation(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        observed_class_labels=observed_class_labels,
        class_count=class_count,
        prediction_probabilities_are_validated=True,
    )
    single_model_weights_by_model_id = {current_training_model_id: 1.0}
    combined_prediction_probabilities, predicted_class_labels, prediction_is_correct = (
        _combine_and_compare_with_observed_class(
            prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
            normalized_weights_by_model_id=single_model_weights_by_model_id,
            class_count=class_count,
            observed_class_labels=observed_class_labels,
        )
    )
    sample_prediction_record = SamplePredictionRecord(
        sample_index=indexed_observation.sample_index,
        observed_concept_id=observed_concept_id,
        observed_class_id=int(observed_class_labels.view(-1)[0].item()),
        combined_prediction_is_correct=prediction_is_correct,
        maximum_weight_model_id=current_training_model_id,
        maximum_prediction_weight=1.0,
        effective_model_count=1.0,
        any_model_or_combined_prediction_is_correct=prediction_is_correct,
        maximum_weight_model_prediction_is_correct=prediction_is_correct,
        adahedge_weighted_prediction_is_correct=prediction_is_correct,
        true_concept_diagnostic_prediction_is_correct=(
            None if observed_concept_id is None else prediction_is_correct
        ),
        highest_confidence_model_prediction_is_correct=prediction_is_correct,
        fixed_share_weighted_prediction_is_correct=prediction_is_correct,
        fixed_share_maximum_weight_model_id=current_training_model_id,
        fixed_share_effective_model_count=1.0,
    )
    sample_prediction_record_store.append_sample_prediction_record(
        sample_prediction_record=sample_prediction_record
    )
    return ObservedSamplePrediction(
        sample_prediction_record=sample_prediction_record,
        predicted_class_labels=predicted_class_labels,
        combined_prediction_probabilities=combined_prediction_probabilities,
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        prediction_weights_by_model_id=dict(single_model_weights_by_model_id),
        fixed_share_weights_by_model_id=dict(single_model_weights_by_model_id),
        adahedge_weights_by_model_id=dict(single_model_weights_by_model_id),
        observed_losses_by_model_id=observed_losses_by_model_id,
    )


def predict_observed_sample_and_update_prediction_weights(
    *,
    indexed_observation: IndexedObservedTrainingSample,
    fixed_share_weight_controller: FixedShareWeightController,
    adahedge_weight_evidence_collection: AdaHedgeWeightEvidenceCollection,
    sample_prediction_record_store: SamplePredictionRecordStore,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    prediction_combination_strategy: str,
) -> ObservedSamplePrediction:
    """保有する全モデルの確率を、予測に使う重みで結合して予測し、記録を足してから、重みを更新する。

    予測に使う重みは、`prediction_combination_strategy`で決まる: Fixed-Shareの重み（提案構成）か、AdaHedgeの重み。
    どちらの設定でも、両方の重みを更新し、使わなかった側の結合の結果も、比較用に記録する。
    `current_training_model_prediction`では、結合せず、現在の学習帰属のモデルだけを評価して予測する（下の
    `_predict_with_current_training_model_only`）。

    AdaHedgeの証拠（全標本のものと、真の概念IDがあればその概念の診断のもの）も、同じ損失で更新する。
    Fixed-Shareの重みで予測する設定は、旧の最終構成の予測と、AdaHedgeの重みで予測する設定は、旧の`global`の指定の予測と、同じ数値になる。
    """
    _validate_observed_sample_prediction_inputs(
        indexed_observation=indexed_observation,
        owners_by_argument_name=dict(
            fixed_share_weight_controller=fixed_share_weight_controller,
            adahedge_weight_evidence_collection=adahedge_weight_evidence_collection,
            sample_prediction_record_store=sample_prediction_record_store,
            held_model_training_state_registry=held_model_training_state_registry,
            current_training_model_assignment=current_training_model_assignment,
        ),
        sample_prediction_record_store=sample_prediction_record_store,
        held_model_training_state_registry=held_model_training_state_registry,
        current_training_model_assignment=current_training_model_assignment,
    )
    if type(prediction_combination_strategy) is not str:
        raise TypeError("prediction_combination_strategy must be builtin str")
    if prediction_combination_strategy not in (
        "fixed_share_weighted_prediction",
        "adahedge_weighted_prediction",
        "current_training_model_prediction",
    ):
        raise ValueError("prediction_combination_strategy must be a declared value")
    if prediction_combination_strategy == "current_training_model_prediction":
        return _predict_with_current_training_model_only(
            indexed_observation=indexed_observation,
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
    # モデルの出力の検査は、この変換の1回だけ。変換の結果（モデル別の確率）は、この関数の中で、何度も使うので、
    # 以後の呼出しには、検査済みと伝える（同じtensorを、検査し直さない）。
    prediction_probabilities_by_model_id = convert_model_outputs_to_prediction_probabilities(
        model_outputs_by_model_id=model_outputs_by_model_id, class_count=class_count
    )
    observed_losses_by_model_id = compute_model_mean_bounded_losses_after_label_observation(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        observed_class_labels=observed_class_labels,
        class_count=class_count,
        prediction_probabilities_are_validated=True,
    )
    observed_class_value = observed_class_labels.view(-1)[0].item()
    model_prediction_is_correct_by_model_id: dict[int, bool] = {}
    prediction_confidences_by_model_id: dict[int, float] = {}
    for model_id in model_ids:
        model_prediction_probabilities = prediction_probabilities_by_model_id[model_id]
        model_prediction_is_correct_by_model_id[model_id] = bool(
            predict_class_labels_from_prediction_scores(
                prediction_scores=model_prediction_probabilities,
                class_count=class_count,
                prediction_scores_are_validated=True,
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
    all_sample_weight_evidence = adahedge_weight_evidence_collection.all_sample_weight_evidence
    adahedge_weights_by_model_id = normalize_model_prediction_weights(
        prediction_weights_by_model_id=all_sample_weight_evidence.get_weights_before_loss_observation(
            model_ids=model_ids
        )
    )
    fixed_share_weights_by_model_id = normalize_model_prediction_weights(
        prediction_weights_by_model_id=fixed_share_weight_controller.get_weights_before_label_observation(
            model_ids=model_ids
        )
    )
    # 予測に使う重み。
    prediction_weights_by_model_id = (
        adahedge_weights_by_model_id
        if prediction_combination_strategy == "adahedge_weighted_prediction"
        else fixed_share_weights_by_model_id
    )
    true_concept_diagnostic_evidence = None
    unnormalized_true_concept_diagnostic_weights_by_model_id = None
    true_concept_diagnostic_prediction_is_correct = None
    if observed_concept_id is not None:
        true_concept_diagnostic_evidence = (
            adahedge_weight_evidence_collection.get_true_concept_diagnostic_evidence(
                true_concept_id=observed_concept_id
            )
        )
        unnormalized_true_concept_diagnostic_weights_by_model_id = (
            true_concept_diagnostic_evidence.get_weights_before_loss_observation(
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
    _, _, adahedge_weighted_prediction_is_correct = _combine_and_compare_with_observed_class(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        normalized_weights_by_model_id=adahedge_weights_by_model_id,
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
    # Fixed-Shareの重みでの結合（比較用。Fixed-Shareの重みで予測する設定では、実際の予測と同じ）。
    _, _, fixed_share_weighted_prediction_is_correct = _combine_and_compare_with_observed_class(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        normalized_weights_by_model_id=fixed_share_weights_by_model_id,
        class_count=class_count,
        observed_class_labels=observed_class_labels,
    )
    # 最大重みのモデルの選び方（同率なら、現在の学習帰属のモデル、次にIDが小さいもの）は、どちらの重みでも同じ。
    maximum_weight_model_id = FixedShareWeightController.select_maximum_weight_model_id(
        prediction_weights_by_model_id=prediction_weights_by_model_id,
        preferred_model_id=current_training_model_id,
    )
    fixed_share_maximum_weight_model_id = FixedShareWeightController.select_maximum_weight_model_id(
        prediction_weights_by_model_id=fixed_share_weights_by_model_id,
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
        adahedge_weighted_prediction_is_correct=adahedge_weighted_prediction_is_correct,
        true_concept_diagnostic_prediction_is_correct=true_concept_diagnostic_prediction_is_correct,
        highest_confidence_model_prediction_is_correct=model_prediction_is_correct_by_model_id[
            highest_confidence_model_id
        ],
        fixed_share_weighted_prediction_is_correct=fixed_share_weighted_prediction_is_correct,
        fixed_share_maximum_weight_model_id=fixed_share_maximum_weight_model_id,
        fixed_share_effective_model_count=1.0
        / sum(
            fixed_share_weight * fixed_share_weight
            for fixed_share_weight in fixed_share_weights_by_model_id.values()
        ),
    )
    # (5) 記録を足す。
    sample_prediction_record_store.append_sample_prediction_record(
        sample_prediction_record=sample_prediction_record
    )
    # (6) ラベル観測後の更新。真の概念別の証拠だけは、旧と同じく、正規化前の重みで更新する。
    all_sample_weight_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id=observed_losses_by_model_id,
        weights_by_model_id=adahedge_weights_by_model_id,
    )
    if (
        true_concept_diagnostic_evidence is not None
        and unnormalized_true_concept_diagnostic_weights_by_model_id is not None
    ):
        true_concept_diagnostic_evidence.update_evidence_after_loss_observation(
            observed_losses_by_model_id=observed_losses_by_model_id,
            weights_by_model_id=unnormalized_true_concept_diagnostic_weights_by_model_id,
        )
    fixed_share_weight_controller.update_weights_after_loss_observation(
        observed_losses_by_model_id=observed_losses_by_model_id,
        prediction_weights_by_model_id=fixed_share_weights_by_model_id,
    )
    return ObservedSamplePrediction(
        sample_prediction_record=sample_prediction_record,
        predicted_class_labels=predicted_class_labels,
        combined_prediction_probabilities=combined_prediction_probabilities,
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        prediction_weights_by_model_id=prediction_weights_by_model_id,
        fixed_share_weights_by_model_id=fixed_share_weights_by_model_id,
        adahedge_weights_by_model_id=adahedge_weights_by_model_id,
        observed_losses_by_model_id=observed_losses_by_model_id,
    )
