"""FedDriftの独立ownerへ統合と配布を適用する。"""

from random import Random

from torch import Tensor, isfinite

from federated_learning_experiments.evaluation.adaptation_record_store import (
    SERVER_REMAP_ADAPTATION_OUTCOME,
    AdaptationRecord,
    AdaptationRecordStore,
)
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import (
    select_loss_statistics_after_model_id_mapping,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingState,
    HeldModelTrainingStateRegistry,
    _validate_held_model_training_state_inputs,
)
from federated_learning_experiments.learning.training.independent_model_parameter_update import (
    _validate_independent_model_structure,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.runtime.client_model_cross_evaluation import (
    _validate_candidate_parameter_snapshot,
)


def apply_feddrift_model_distribution(
    *,
    model_id_mapping: dict[int, int],
    model_parameters_by_model_id: dict[int, dict[str, Tensor]],
    server_loss_statistics: tuple[tuple[int, ModelAndClassLossStatistics], ...],
    rebuilt_model_optimizer_settings: AdamParameterOptimizerSettings
    | SgdParameterOptimizerSettings,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    evaluation_sample_store: ModelEvaluationSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    adaptation_record_store: AdaptationRecordStore,
    processed_sample_count: int,
    python_random_generator: Random,
) -> None:
    """全入力を検査して正式モデルを再構築し、ownerへ反映する。"""
    for owner, owner_type in (
        (held_model_training_state_registry, HeldModelTrainingStateRegistry),
        (loss_statistics_store, ModelAndClassLossStatisticsStore),
        (training_sample_store, ModelTrainingSampleStore),
        (evaluation_sample_store, ModelEvaluationSampleStore),
        (model_training_and_assignment_counts_store, ModelTrainingAndAssignmentCountsStore),
        (current_training_model_assignment, CurrentTrainingModelAssignment),
        (adaptation_record_store, AdaptationRecordStore),
        (python_random_generator, Random),
    ):
        if type(owner) is not owner_type:
            raise TypeError(f"ownerはexact {owner_type.__name__}が必要です。")
    if type(processed_sample_count) is not int:
        raise TypeError("処理件数はbuiltin intが必要です。")
    if processed_sample_count < 0:
        raise ValueError("処理件数は非負が必要です。")
    if type(rebuilt_model_optimizer_settings) not in (
        AdamParameterOptimizerSettings,
        SgdParameterOptimizerSettings,
    ):
        raise TypeError("再構築optimizer設定の型が不正です。")
    rebuilt_model_optimizer_settings.__post_init__()
    if type(model_parameters_by_model_id) is not dict or type(model_id_mapping) is not dict:
        raise TypeError("配布parameterとID対応はdictが必要です。")
    if not model_parameters_by_model_id:
        raise ValueError("配布モデルは空にできません。")
    states = held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    for state in states:
        if type(state) is not HeldModelTrainingState:
            raise TypeError("保有記録はexact HeldModelTrainingStateが必要です。")
        _validate_held_model_training_state_inputs(
            model_id=state.model_id,
            classifier=state.classifier,
            concept_specific_parameter_optimizer_state=state.concept_specific_parameter_optimizer_state,
        )
        _validate_independent_model_structure(state.classifier)
        if any(not isfinite(value).all().item() for value in state.classifier.parameters()):
            raise ValueError("保有parameterは有限値が必要です。")
    if len({state.model_id for state in states}) != len(states):
        raise ValueError("保有IDは一意であることが必要です。")
    current_id = current_training_model_assignment.current_training_model_id
    if type(current_id) is not int:
        raise TypeError("現在IDはbuiltin intが必要です。")
    reference = held_model_training_state_registry.get_held_model_training_state(
        model_id=current_id
    ).classifier
    reference_snapshot = snapshot_classifier_parameters(classifier=reference)
    for model_id, parameters in model_parameters_by_model_id.items():
        if type(model_id) is not int:
            raise TypeError("配布IDはbuiltin intが必要です。")
        if model_id < 0:
            raise ValueError("配布IDは非負が必要です。")
        _validate_candidate_parameter_snapshot(
            candidate_parameter_snapshot=parameters, reference_parameter_snapshot=reference_snapshot
        )
        if any(not isfinite(value).all().item() for value in parameters.values()):
            raise ValueError("配布parameterは有限値が必要です。")
    held_ids = {state.model_id for state in states}
    for old_id, new_id in model_id_mapping.items():
        if type(old_id) is not int or type(new_id) is not int:
            raise TypeError("ID対応はbuiltin intが必要です。")
        if old_id < 0 or old_id not in held_ids or new_id not in model_parameters_by_model_id:
            raise ValueError("ID対応は保有正式IDから配布IDへの対応が必要です。")
    mapped_id = model_id_mapping.get(current_id, current_id)
    temporary = tuple(s for s in states if s.model_id < 0)
    if mapped_id not in model_parameters_by_model_id and mapped_id not in {
        s.model_id for s in temporary
    }:
        raise ValueError("現在IDの対応先が配布/保持モデルにありません。")
    selected_statistics = select_loss_statistics_after_model_id_mapping(
        local_model_loss_statistics=loss_statistics_store.get_state_snapshot(),
        model_id_mapping=model_id_mapping,
        server_model_loss_statistics=server_loss_statistics,
    )
    adaptation = (
        None
        if current_id == mapped_id
        else AdaptationRecord(
            adaptation_sample_index=processed_sample_count,
            detector_name="server",
            adaptation_outcome=SERVER_REMAP_ADAPTATION_OUTCOME,
            previous_training_model_id=current_id,
            current_training_model_id=mapped_id,
            estimated_change_point_sample_index=None,
            detection_episode_id=None,
        )
    )
    rebuilt = []
    for model_id, parameters in model_parameters_by_model_id.items():
        classifier = ModelArchitectureClassifier(
            model_architecture_settings=reference.model_architecture_settings,
            input_feature_count=reference.input_feature_count,
            hidden_layer_widths=reference.hidden_layer_widths,
            class_count=reference.class_count,
        )
        classifier.load_state_dict(parameters, strict=True)
        rebuilt.append(
            HeldModelTrainingState(
                model_id=model_id,
                classifier=classifier,
                concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                    parameters=tuple(classifier.parameters()),
                    optimizer_settings=rebuilt_model_optimizer_settings,
                ),
            )
        )
    replacing_states = tuple(rebuilt) + temporary
    if adaptation is not None:
        adaptation_record_store.append_adaptation_record(adaptation_record=adaptation)
    loss_statistics_store.replace_model_loss_statistics(
        loss_statistics_by_model_id=selected_statistics
    )
    evaluation_sample_store.remap_model_evaluation_sample_collections(
        model_id_mapping=model_id_mapping,
        python_random_generator=python_random_generator,
    )
    training_sample_store.remap_model_training_sample_collections(model_id_mapping=model_id_mapping)
    model_training_and_assignment_counts_store.remap_model_training_and_assignment_counts(
        model_id_mapping=model_id_mapping
    )
    held_model_training_state_registry.replace_held_model_training_states(
        held_model_training_states=replacing_states
    )
    current_training_model_assignment.remap_current_training_model_id(
        model_id_mapping=model_id_mapping
    )
