"""サーバから配布されたグローバルモデルと損失統計を、client 1つのownerへ反映する（受取り）。"""

from dataclasses import dataclass
from random import Random

from torch import Tensor, float32, strided

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
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.learning.training.held_model_shared_feature_reconnection import (
    HeldModelOptimizerBinding,
    reconnect_held_models_to_shared_feature_extractor,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingState,
    HeldModelTrainingStateRegistry,
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
from federated_learning_experiments.learning.training.shared_parameter_optimizer_state_holder import (
    SharedParameterOptimizerStateHolder,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)

# サーバによる付け替えの適応記録の検出器名（旧と同じ文字列）。
_SERVER_REMAP_DETECTOR_NAME = "server"


@dataclass(frozen=True, kw_only=True)
class GlobalModelDistributionApplication:
    """受取り1回の結果。"""

    # ID対応で、現在の学習帰属が付け替わった場合の、変更前後のID。変わらなければNone。
    training_assignment_change: TrainingModelAssignmentChange | None
    # 受取りの後の保有モデルのID（配布の順、その後に、残した一時IDのモデル）。
    held_model_ids: tuple[int, ...]


def _validate_owner_types(**owners_and_required_types: tuple[object, type]) -> None:
    for owner_name, (owner, required_type) in owners_and_required_types.items():
        if type(owner) is not required_type:
            raise TypeError(f"{owner_name} must be exact {required_type.__name__}")


def _validate_model_id_mapping(*, model_id_mapping: dict[int, int]) -> None:
    if type(model_id_mapping) is not dict:
        raise TypeError("model_id_mapping must be builtin dict")
    for original_model_id, mapped_model_id in model_id_mapping.items():
        if type(original_model_id) is not int or type(mapped_model_id) is not int:
            raise TypeError("model_id_mapping keys and values must be builtin int")


def _validate_distributed_model_id_sequence(
    *, distributed_values: tuple[tuple[int, object], ...], argument_name: str
) -> tuple[int, ...]:
    """（ID、値）の列の形と、IDが非負で重複しないことを確かめて、IDの列を返す。値そのものは見ない。"""
    if type(distributed_values) is not tuple:
        raise TypeError(f"{argument_name} must be exact tuple")
    distributed_model_ids: list[int] = []
    for distributed_value in distributed_values:
        if type(distributed_value) is not tuple:
            raise TypeError(f"{argument_name} elements must be exact tuple")
        if len(distributed_value) != 2:
            raise ValueError(f"{argument_name} elements must be (model_id, value) pairs")
        model_id = distributed_value[0]
        if type(model_id) is not int:
            raise TypeError(f"{argument_name} model_id must be builtin int")
        if model_id < 0:
            raise ValueError(f"{argument_name} model_id must be nonnegative")
        if model_id in distributed_model_ids:
            raise ValueError(f"{argument_name} model_id must be unique")
        distributed_model_ids.append(model_id)
    return tuple(distributed_model_ids)


def _validate_distributed_parameter_snapshot(
    *,
    model_id: int,
    parameter_snapshot: dict[str, Tensor],
    reference_parameter_snapshot: dict[str, Tensor],
) -> None:
    """配布されたパラメータが、保有している分類器と同じ名前・同じ形の、CPUのfloat32のtensorであること。"""
    if type(parameter_snapshot) is not dict:
        raise TypeError(f"distributed parameters of model {model_id} must be builtin dict")
    if any(type(parameter_name) is not str for parameter_name in parameter_snapshot):
        raise TypeError(f"distributed parameter names of model {model_id} must be builtin str")
    if parameter_snapshot.keys() != reference_parameter_snapshot.keys():
        raise ValueError(
            f"distributed parameter names of model {model_id} must match the held classifier"
        )
    for parameter_name, reference_parameter_values in reference_parameter_snapshot.items():
        parameter_values = parameter_snapshot[parameter_name]
        if type(parameter_values) is not Tensor:
            raise TypeError(
                f"distributed parameter {parameter_name} of model {model_id} must be exact torch.Tensor"
            )
        if (
            parameter_values.device.type != "cpu"
            or parameter_values.dtype != float32
            or parameter_values.layout != strided
            or parameter_values.is_nested
        ):
            raise ValueError(
                f"distributed parameter {parameter_name} of model {model_id} must be CPU float32 strided"
            )
        if parameter_values.shape != reference_parameter_values.shape:
            raise ValueError(
                f"distributed parameter {parameter_name} of model {model_id} must have the shape of the held classifier"
            )


def _validate_parameter_optimizer_settings(
    *,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    argument_name: str,
) -> None:
    if type(parameter_optimizer_settings) not in (
        AdamParameterOptimizerSettings,
        SgdParameterOptimizerSettings,
    ):
        raise TypeError(
            f"{argument_name} must be exact AdamParameterOptimizerSettings or SgdParameterOptimizerSettings"
        )
    # frozenを回避して組み立てた値も拒否できるよう、設定の検査をもう一度行う。
    parameter_optimizer_settings.__post_init__()


def _get_concept_specific_parameters(*, classifier: ResidualAdapterClassifier) -> tuple:
    return tuple(classifier.residual_adapter.parameters()) + tuple(
        classifier.classification_layer.parameters()
    )


def apply_global_model_distribution(
    *,
    model_id_mapping: dict[int, int],
    distributed_parameter_snapshots: tuple[tuple[int, dict[str, Tensor]], ...],
    distributed_loss_statistics: tuple[tuple[int, ModelAndClassLossStatistics], ...],
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    shared_parameter_optimizer_state_holder: SharedParameterOptimizerStateHolder,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    model_evaluation_sample_store: ModelEvaluationSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    adaptation_record_store: AdaptationRecordStore,
    pending_training_assignment_buffer: PendingTrainingAssignmentBuffer,
    rebuilt_model_parameter_optimizer_settings: AdamParameterOptimizerSettings
    | SgdParameterOptimizerSettings,
    reconnected_model_parameter_optimizer_settings: AdamParameterOptimizerSettings
    | SgdParameterOptimizerSettings,
    python_random_generator: Random,
) -> GlobalModelDistributionApplication:
    """配布されたモデルで正式IDの保有モデルを作り直し、ID対応を統計・標本・計数・現在の学習帰属へ適用する。

    検査と、新しい分類器・optimizerの状態の生成（torchの乱数を消費する）を、ownerの更新より前に行う。
    一時IDの保有モデルは、そのまま残す。全保有モデルを、非負のIDが最小のモデルの共有部へつなぎ直す。
    予測の重みと診断証拠には触れない。
    """
    # 段1: 検査（どの更新・乱数の消費より前）。
    _validate_owner_types(
        held_model_training_state_registry=(
            held_model_training_state_registry,
            HeldModelTrainingStateRegistry,
        ),
        shared_parameter_optimizer_state_holder=(
            shared_parameter_optimizer_state_holder,
            SharedParameterOptimizerStateHolder,
        ),
        loss_statistics_store=(loss_statistics_store, ModelAndClassLossStatisticsStore),
        training_sample_store=(training_sample_store, ModelTrainingSampleStore),
        model_evaluation_sample_store=(model_evaluation_sample_store, ModelEvaluationSampleStore),
        model_training_and_assignment_counts_store=(
            model_training_and_assignment_counts_store,
            ModelTrainingAndAssignmentCountsStore,
        ),
        current_training_model_assignment=(
            current_training_model_assignment,
            CurrentTrainingModelAssignment,
        ),
        adaptation_record_store=(adaptation_record_store, AdaptationRecordStore),
        pending_training_assignment_buffer=(
            pending_training_assignment_buffer,
            PendingTrainingAssignmentBuffer,
        ),
        python_random_generator=(python_random_generator, Random),
    )
    _validate_parameter_optimizer_settings(
        parameter_optimizer_settings=rebuilt_model_parameter_optimizer_settings,
        argument_name="rebuilt_model_parameter_optimizer_settings",
    )
    _validate_parameter_optimizer_settings(
        parameter_optimizer_settings=reconnected_model_parameter_optimizer_settings,
        argument_name="reconnected_model_parameter_optimizer_settings",
    )
    _validate_model_id_mapping(model_id_mapping=model_id_mapping)
    distributed_model_ids = _validate_distributed_model_id_sequence(
        distributed_values=distributed_parameter_snapshots,
        argument_name="distributed_parameter_snapshots",
    )
    if not distributed_model_ids:
        raise ValueError("distributed_parameter_snapshots must not be empty")
    _validate_distributed_model_id_sequence(
        distributed_values=distributed_loss_statistics,
        argument_name="distributed_loss_statistics",
    )
    previous_training_model_id = current_training_model_assignment.current_training_model_id
    architecture_reference_classifier = (
        held_model_training_state_registry.get_held_model_training_state(
            model_id=previous_training_model_id
        ).classifier
    )
    reference_parameter_snapshot = snapshot_classifier_parameters(
        classifier=architecture_reference_classifier
    )
    for distributed_model_id, distributed_parameter_snapshot in distributed_parameter_snapshots:
        _validate_distributed_parameter_snapshot(
            model_id=distributed_model_id,
            parameter_snapshot=distributed_parameter_snapshot,
            reference_parameter_snapshot=reference_parameter_snapshot,
        )
    kept_temporary_model_training_states = tuple(
        held_model_training_state
        for held_model_training_state in held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        if held_model_training_state.model_id < 0
    )
    remapped_training_model_id = model_id_mapping.get(
        previous_training_model_id, previous_training_model_id
    )
    if remapped_training_model_id not in distributed_model_ids and all(
        kept_state.model_id != remapped_training_model_id
        for kept_state in kept_temporary_model_training_states
    ):
        raise ValueError(
            "the current training model after model_id_mapping must be a distributed model "
            "or a held temporary model"
        )

    # 段2: 統計の選択と、適応記録の内容（ownerは変えない）。
    selected_loss_statistics = select_loss_statistics_after_model_id_mapping(
        local_model_loss_statistics=loss_statistics_store.get_state_snapshot(),
        model_id_mapping=model_id_mapping,
        server_model_loss_statistics=distributed_loss_statistics,
    )
    last_observed_sample_index = (
        pending_training_assignment_buffer.get_state_snapshot().last_observed_sample_index
    )
    server_remap_adaptation_record = (
        None
        if remapped_training_model_id == previous_training_model_id
        else AdaptationRecord(
            # 位置は、これまでに処理した標本数（旧と同じ）。
            adaptation_sample_index=(
                0 if last_observed_sample_index is None else last_observed_sample_index + 1
            ),
            detector_name=_SERVER_REMAP_DETECTOR_NAME,
            adaptation_outcome=SERVER_REMAP_ADAPTATION_OUTCOME,
            previous_training_model_id=previous_training_model_id,
            current_training_model_id=remapped_training_model_id,
            estimated_change_point_sample_index=None,
            detection_episode_id=None,
        )
    )

    # 段3: 配布の順に、新しい分類器（torchの乱数を消費）と、optimizerの状態を作る。
    # つなぎ先は、非負のIDが最小のモデル。ほかのモデルの概念固有部のoptimizerは、つなぎ直しで
    # その状態自身の設定から作り直されるので、最初から、つなぎ直しの設定で作る。
    shared_feature_source_model_id = min(distributed_model_ids)
    rebuilt_optimizer_bindings: list[HeldModelOptimizerBinding] = []
    for distributed_model_id, distributed_parameter_snapshot in distributed_parameter_snapshots:
        rebuilt_classifier = ResidualAdapterClassifier(
            model_architecture_settings=architecture_reference_classifier.model_architecture_settings,
            input_feature_count=architecture_reference_classifier.feature_extractor.input_feature_count,
            hidden_layer_widths=architecture_reference_classifier.feature_extractor.hidden_layer_widths,
            class_count=architecture_reference_classifier.class_count,
        )
        rebuilt_classifier.load_state_dict(distributed_parameter_snapshot, strict=True)
        rebuilt_optimizer_bindings.append(
            HeldModelOptimizerBinding(
                model_id=distributed_model_id,
                classifier=rebuilt_classifier,
                # つなぎ先以外の共有部と、そのoptimizerの状態は、つなぎ直しで使われなくなる。
                shared_parameter_optimizer_state=ParameterOptimizerState(
                    parameters=tuple(rebuilt_classifier.feature_extractor.parameters()),
                    optimizer_settings=rebuilt_model_parameter_optimizer_settings,
                ),
                concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                    parameters=_get_concept_specific_parameters(classifier=rebuilt_classifier),
                    optimizer_settings=(
                        rebuilt_model_parameter_optimizer_settings
                        if distributed_model_id == shared_feature_source_model_id
                        else reconnected_model_parameter_optimizer_settings
                    ),
                ),
            )
        )
    previous_shared_parameter_optimizer_state = (
        shared_parameter_optimizer_state_holder.held_shared_parameter_optimizer_state
    )
    held_model_optimizer_bindings = tuple(rebuilt_optimizer_bindings) + tuple(
        HeldModelOptimizerBinding(
            model_id=kept_state.model_id,
            classifier=kept_state.classifier,
            shared_parameter_optimizer_state=previous_shared_parameter_optimizer_state,
            concept_specific_parameter_optimizer_state=kept_state.concept_specific_parameter_optimizer_state,
        )
        for kept_state in kept_temporary_model_training_states
    )
    replacing_held_model_training_states = (
        tuple(
            HeldModelTrainingState(
                model_id=rebuilt_binding.model_id,
                classifier=rebuilt_binding.classifier,
                concept_specific_parameter_optimizer_state=rebuilt_binding.concept_specific_parameter_optimizer_state,
            )
            for rebuilt_binding in rebuilt_optimizer_bindings
        )
        + kept_temporary_model_training_states
    )

    # 段4: 全保有モデルを、1つの共有部へつなぎ直す。つなぎ直しの部品は、全部の対応を検査してから
    # つなぐので、ここで拒否されれば、ownerは何も変わらない。
    reconnect_held_models_to_shared_feature_extractor(
        held_model_optimizer_bindings=held_model_optimizer_bindings
    )
    # 段5: 現在の学習帰属が付け替わるなら、適応記録を足す。
    if server_remap_adaptation_record is not None:
        adaptation_record_store.append_adaptation_record(
            adaptation_record=server_remap_adaptation_record
        )
    # 段6: 統計→評価標本→学習データ→計数（旧と同じ順。評価標本が上限を超えると、借りた乱数で抜き出す）。
    loss_statistics_store.replace_model_loss_statistics(
        loss_statistics_by_model_id=selected_loss_statistics
    )
    model_evaluation_sample_store.remap_model_evaluation_sample_collections(
        model_id_mapping=model_id_mapping, python_random_generator=python_random_generator
    )
    training_sample_store.remap_model_training_sample_collections(model_id_mapping=model_id_mapping)
    model_training_and_assignment_counts_store.remap_model_training_and_assignment_counts(
        model_id_mapping=model_id_mapping
    )
    # 段7: 保有モデルと、共有部のoptimizerの状態を、つなぎ直した後のものへ置き換える。
    held_model_training_state_registry.replace_held_model_training_states(
        held_model_training_states=replacing_held_model_training_states
    )
    shared_parameter_optimizer_state_holder.replace_shared_parameter_optimizer_state(
        shared_parameter_optimizer_state=next(
            rebuilt_binding.shared_parameter_optimizer_state
            for rebuilt_binding in rebuilt_optimizer_bindings
            if rebuilt_binding.model_id == shared_feature_source_model_id
        )
    )
    # 段8: 現在の学習帰属を付け替える。
    training_assignment_change = current_training_model_assignment.remap_current_training_model_id(
        model_id_mapping=model_id_mapping
    )
    return GlobalModelDistributionApplication(
        training_assignment_change=training_assignment_change,
        held_model_ids=tuple(
            replacing_state.model_id for replacing_state in replacing_held_model_training_states
        ),
    )
