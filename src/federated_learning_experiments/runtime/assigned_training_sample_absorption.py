"""帰属先が確定した標本を、保有済みモデルの標本・割当概念計数・損失統計へ吸収する。"""

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
    validate_classifier_bounded_loss_inputs,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)


def _validate_absorption_inputs(
    *,
    model_id: int,
    assigned_training_samples: tuple[ObservedTrainingSample, ...],
    assigned_sample_concept_ids: tuple[int | None, ...],
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
) -> None:
    if type(model_id) is not int:
        raise TypeError("model_idはbool・派生型以外のbuiltin intが必要です。")
    for state_owner, expected_owner_type, owner_name in (
        (
            held_model_training_state_registry,
            HeldModelTrainingStateRegistry,
            "held_model_training_state_registry",
        ),
        (training_sample_store, ModelTrainingSampleStore, "training_sample_store"),
        (
            model_training_and_assignment_counts_store,
            ModelTrainingAndAssignmentCountsStore,
            "model_training_and_assignment_counts_store",
        ),
        (loss_statistics_store, ModelAndClassLossStatisticsStore, "loss_statistics_store"),
    ):
        if type(state_owner) is not expected_owner_type:
            raise TypeError(f"{owner_name}はexact {expected_owner_type.__name__}が必要です。")
    if type(assigned_training_samples) is not tuple:
        raise TypeError("assigned_training_samplesはexact tupleが必要です。")
    for training_sample in assigned_training_samples:
        if type(training_sample) is not ObservedTrainingSample:
            raise TypeError(
                "assigned_training_samplesの各要素はexact ObservedTrainingSampleが必要です。"
            )
    if type(assigned_sample_concept_ids) is not tuple:
        raise TypeError("assigned_sample_concept_idsはexact tupleが必要です。")
    if len(assigned_sample_concept_ids) != len(assigned_training_samples):
        raise ValueError("assigned_sample_concept_idsは標本列と同じ長さが必要です。")
    for observed_concept_id in assigned_sample_concept_ids:
        if observed_concept_id is not None and type(observed_concept_id) is not int:
            raise TypeError(
                "assigned_sample_concept_idsの各要素はNoneまたはbuiltin intが必要です。"
            )


def absorb_assigned_training_samples_into_held_model(
    *,
    model_id: int,
    assigned_training_samples: tuple[ObservedTrainingSample, ...],
    assigned_sample_concept_ids: tuple[int | None, ...],
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    evaluated_observed_losses: tuple[float, ...] | None = None,
) -> None:
    """全標本の検証と損失評価の後に、標本順で標本→割当概念→損失統計を更新する。

    `evaluated_observed_losses`を渡すと、損失を計算し直さずに、その値を使う。呼出し側が、同じモデルで、
    同じ標本の損失を、すでに計算しているとき（状態の更新より前の検査のため）に、順伝播の重複を避ける。
    """
    _validate_absorption_inputs(
        model_id=model_id,
        assigned_training_samples=assigned_training_samples,
        assigned_sample_concept_ids=assigned_sample_concept_ids,
        held_model_training_state_registry=held_model_training_state_registry,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        loss_statistics_store=loss_statistics_store,
    )
    classifier = held_model_training_state_registry.get_held_model_training_state(
        model_id=model_id
    ).classifier
    # 吸収中にモデルは変わらないため、先に評価しても各標本の損失は逐次評価と同じ値になる。
    observed_losses: list[float] = []
    observed_class_ids: list[int] = []
    if evaluated_observed_losses is not None:
        if type(evaluated_observed_losses) is not tuple:
            raise TypeError("evaluated_observed_lossesはexact tupleまたはNoneが必要です。")
        if len(evaluated_observed_losses) != len(assigned_training_samples):
            raise ValueError("evaluated_observed_lossesは標本列と同じ長さが必要です。")
        for evaluated_observed_loss in evaluated_observed_losses:
            if type(evaluated_observed_loss) is not float:
                raise TypeError("evaluated_observed_lossesの各要素はbuiltin floatが必要です。")
            if not 0.0 <= evaluated_observed_loss <= 1.0:
                raise ValueError("evaluated_observed_lossesの各要素は0以上1以下が必要です。")
        for training_sample in assigned_training_samples:
            # 損失を計算しない場合も、評価と同じ入力の検査を、順伝播なしで行う。
            validate_classifier_bounded_loss_inputs(
                classifier=classifier,
                input_features=training_sample.input_features,
                observed_class_labels=training_sample.observed_class_labels,
            )
            if training_sample.input_features.shape[0] != 1:
                raise ValueError("assigned_training_samplesの各要素は1標本ぶんが必要です。")
        observed_losses.extend(evaluated_observed_losses)
    for training_sample in assigned_training_samples:
        if evaluated_observed_losses is None:
            per_sample_bounded_losses = evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier,
                input_features=training_sample.input_features,
                observed_class_labels=training_sample.observed_class_labels,
            )
            if len(per_sample_bounded_losses) != 1:
                raise ValueError("assigned_training_samplesの各要素は1標本ぶんが必要です。")
            observed_losses.append(per_sample_bounded_losses[0].item())
        observed_class_ids.append(int(training_sample.observed_class_labels.reshape(-1)[0].item()))
    for training_sample, observed_concept_id, observed_loss, observed_class_id in zip(
        assigned_training_samples,
        assigned_sample_concept_ids,
        observed_losses,
        observed_class_ids,
    ):
        training_sample_store.append_model_training_samples(
            model_id=model_id, training_samples=(training_sample,)
        )
        model_training_and_assignment_counts_store.record_assigned_sample_concept(
            model_id=model_id, observed_concept_id=observed_concept_id
        )
        loss_statistics_store.record_assigned_loss(
            model_id=model_id, observed_loss=observed_loss, observed_class_id=observed_class_id
        )
