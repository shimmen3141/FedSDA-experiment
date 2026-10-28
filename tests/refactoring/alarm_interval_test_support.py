"""警報区間と候補学習の旧実装照合で共有する入力を組み立てる。"""

from collections import defaultdict

import torch
from test_adopted_candidate_local_adoption import build_local_adoption_oracle

from federated_drift_experiment import config
from federated_drift_experiment.models import ResidualAdapterMLP
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)


def make_subclass_copy(original):
    """同じ属性を持つ、派生型の値（exact型の検査が拒否するべき値）。"""
    subclass = type(type(original).__name__ + "Subclass", (type(original),), {})
    subclass_copy = object.__new__(subclass)
    for attribute_name, attribute in vars(original).items():
        object.__setattr__(subclass_copy, attribute_name, attribute)
    return subclass_copy


def set_overall_loss_statistics_in_both_implementations(
    *, loss_statistics_store, legacy_client, statistics_by_model_id
):
    for model_id, statistics in statistics_by_model_id.items():
        if statistics is None:
            # 統計を両実装で保有していないIDへ退避し、未登録のモデルにする。
            loss_statistics_store.reassign_model_loss_statistics_id(
                original_model_id=model_id, reassigned_model_id=999
            )
            legacy_client.model_stats[999] = legacy_client.model_stats.pop(model_id)
            continue
        observed_loss_count, mean_loss = statistics
        loss_statistics_store.set_model_loss_statistics(
            model_id=model_id,
            loss_statistics=ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    observed_loss_count=observed_loss_count,
                    mean_loss=mean_loss,
                    sum_squared_loss_deviations=0.0,
                )
            ),
        )
        # 旧の比較helperが読むため、クラス別統計は空で持たせる（新の統計もクラス別は空）。
        legacy_client.model_stats[model_id] = {
            "n": observed_loss_count,
            "mean": mean_loss,
            "M2": 0.0,
            "class_stats": {},
        }


def build_held_models_oracle(
    *, class_count, monkeypatch, held_model_ids=(4, 9), optimizer_variant="standard"
):
    adoption_arguments, shared_optimizer_owners, legacy_client, _ = build_local_adoption_oracle(
        class_count=class_count,
        held_model_ids=held_model_ids,
        optimizer_variant=optimizer_variant,
        monkeypatch=monkeypatch,
    )
    # 実旧のモデル生成。実験では設定の既定値から同じ構造を作る。
    legacy_client.model_cls = lambda: ResidualAdapterMLP(input_dim=2, dataset="sine2")
    fixation_arguments = dict(
        held_model_training_state_registry=adoption_arguments["held_model_training_state_registry"],
        loss_statistics_store=adoption_arguments["loss_statistics_store"],
    )
    return fixation_arguments, adoption_arguments, shared_optimizer_owners, legacy_client


def build_candidate_training_oracle(
    *,
    class_count,
    optimizer_variant,
    monkeypatch,
    candidate_training_strategy="validation_loss_early_stopping",
    maximum_epoch_count=3,
    maximum_batch_sample_count=3,
    consecutive_non_improving_epoch_limit=1,
    minimum_validation_loss_decrease=10,
    held_model_ids=(4, 9),
):
    fixation_arguments, adoption_arguments, shared_optimizer_owners, legacy_client = (
        build_held_models_oracle(
            class_count=class_count,
            optimizer_variant=optimizer_variant,
            held_model_ids=held_model_ids,
            monkeypatch=monkeypatch,
        )
    )
    monkeypatch.setattr(config, "NEW_MODEL_CREATION_POLICY", "immediate")
    monkeypatch.setattr(
        config,
        "NEW_MODEL_TRAINING",
        {
            "fixed_epoch_training": "fixed",
            "validation_loss_early_stopping": "early_stopping",
            "skip_training": "none",
        }[candidate_training_strategy],
    )
    monkeypatch.setattr(config, "NEW_MODEL_EPOCHS", maximum_epoch_count)
    monkeypatch.setattr(config, "CLIENT_BATCH_SIZE", maximum_batch_sample_count)
    monkeypatch.setattr(config, "NEW_MODEL_VALIDATION_FRACTION", 0.2)
    monkeypatch.setattr(
        config, "NEW_MODEL_EARLY_STOPPING_PATIENCE", consecutive_non_improving_epoch_limit
    )
    monkeypatch.setattr(
        config, "NEW_MODEL_EARLY_STOPPING_MIN_DELTA", minimum_validation_loss_decrease
    )
    legacy_client.phase_seconds = defaultdict(float)
    session_start_arguments = dict(
        **fixation_arguments,
        architecture_reference_classifier=adoption_arguments["adopted_candidate_classifier"],
        # 構造参照と選択済みの値を別モデルにし、初期値選択の再実装を検出する。
        initial_candidate_parameter_snapshot=snapshot_classifier_parameters(
            classifier=fixation_arguments["held_model_training_state_registry"]
            .snapshot_ordered_held_model_training_states()[-1]
            .classifier
        ),
        parameter_optimizer_settings=(
            SgdParameterOptimizerSettings(learning_rate=0.01)
            if optimizer_variant == "sgd"
            else AdamParameterOptimizerSettings(
                learning_rate=0.01, weight_decay=0.001, adam_variant=optimizer_variant
            )
        ),
        candidate_epoch_training_settings=CandidateEpochTrainingSettings(
            candidate_training_strategy=candidate_training_strategy,
            maximum_epoch_count=maximum_epoch_count,
            maximum_batch_sample_count=maximum_batch_sample_count,
            validation_sample_fraction=0.2,
            consecutive_non_improving_epoch_limit=consecutive_non_improving_epoch_limit,
            minimum_validation_loss_decrease=minimum_validation_loss_decrease,
            candidate_shared_feature_update_policy="update_shared_features_with_candidate",
        ),
        current_training_model_assignment=adoption_arguments["current_training_model_assignment"],
        input_features=torch.arange(22, dtype=torch.float32).reshape(11, 2) / 7,
        observed_class_labels=(torch.arange(11) % class_count).float().reshape(11, 1),
        pending_assignment_training_samples=adoption_arguments[
            "pending_assignment_training_samples"
        ],
        proposal_sample_index=40,
        estimated_change_point_sample_index=35,
        detection_episode_id=7,
        detector_name=legacy_client._detector_label(),
    )
    return session_start_arguments, shared_optimizer_owners, legacy_client
