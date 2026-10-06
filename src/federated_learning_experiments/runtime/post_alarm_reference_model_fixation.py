"""警報時点の保有モデルの値で、候補検証の参照分類器と履歴平均損失を固定する。"""

from dataclasses import dataclass

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
    select_post_alarm_reference_historical_mean_loss,
)


@dataclass(frozen=True, kw_only=True)
class FixedPostAlarmReferenceModels:
    """保有一覧の順の参照分類器と、全体統計が2件以上あるモデルの履歴平均損失。"""

    reference_classifiers_by_model_id: dict[int, ResidualAdapterClassifier]
    reference_historical_mean_losses_by_model_id: dict[int, float]


def fix_reference_models_at_alarm(
    *,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
) -> FixedPostAlarmReferenceModels:
    """全モデルの値の複製と履歴平均を揃えた後に、一覧の順で独立した分類器を生成する。"""
    if type(held_model_training_state_registry) is not HeldModelTrainingStateRegistry:
        raise TypeError(
            "held_model_training_state_registryはexact HeldModelTrainingStateRegistryが必要です。"
        )
    if type(loss_statistics_store) is not ModelAndClassLossStatisticsStore:
        raise TypeError("loss_statistics_storeはexact ModelAndClassLossStatisticsStoreが必要です。")
    held_model_training_states = (
        held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )
    if not held_model_training_states:
        raise LookupError("保有モデルがないため参照モデルを固定できません。")
    # 値の複製と履歴平均の取得は乱数を消費しない。拒否はモデルの生成より前に起こす。
    parameter_snapshots_by_model_id = {
        held_model_training_state.model_id: snapshot_classifier_parameters(
            classifier=held_model_training_state.classifier
        )
        for held_model_training_state in held_model_training_states
    }
    reference_historical_mean_losses_by_model_id: dict[int, float] = {}
    for held_model_training_state in held_model_training_states:
        loss_statistics = loss_statistics_store.get_model_loss_statistics(
            model_id=held_model_training_state.model_id
        )
        if loss_statistics is None:
            continue
        historical_mean_loss = select_post_alarm_reference_historical_mean_loss(
            loss_moments=loss_statistics.overall_loss_moments
        )
        if historical_mean_loss is not None:
            reference_historical_mean_losses_by_model_id[held_model_training_state.model_id] = (
                historical_mean_loss
            )
    # モデルの生成は初期化でCPUのtorch乱数を消費する。一覧の順に1つずつ生成する。
    reference_classifiers_by_model_id: dict[int, ResidualAdapterClassifier] = {}
    for held_model_training_state in held_model_training_states:
        held_classifier = held_model_training_state.classifier
        reference_classifier = ResidualAdapterClassifier(
            model_architecture_settings=held_classifier.model_architecture_settings,
            input_feature_count=held_classifier.feature_extractor.input_feature_count,
            hidden_layer_widths=held_classifier.feature_extractor.hidden_layer_widths,
            class_count=held_classifier.class_count,
        )
        reference_classifier.load_state_dict(
            parameter_snapshots_by_model_id[held_model_training_state.model_id]
        )
        reference_classifiers_by_model_id[held_model_training_state.model_id] = reference_classifier
    return FixedPostAlarmReferenceModels(
        reference_classifiers_by_model_id=reference_classifiers_by_model_id,
        reference_historical_mean_losses_by_model_id=reference_historical_mean_losses_by_model_id,
    )
