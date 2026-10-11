"""FedDrift clientを初期モデルから組み立てる。"""

from copy import deepcopy
from dataclasses import dataclass
from random import Random

from torch import float32, tensor

from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
    _copy_model_and_class_loss_statistics,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    INDEPENDENT_MODELS_ARCHITECTURE_NAME,
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
    _validate_held_model_training_state_inputs,
)
from federated_learning_experiments.learning.training.independent_model_parameter_update import (
    _validate_independent_model_structure,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    _validate_trainable_parameters,
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
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.runtime.feddrift_client_round_training import (
    train_feddrift_held_models_for_round,
)
from federated_learning_experiments.runtime.feddrift_client_stream_processing import (
    FeddriftClientStreamProcessor,
)
from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
    FeddriftDetectionBatchProcessor,
    FeddriftDetectionBatchTransition,
)
from federated_learning_experiments.runtime.feddrift_run_client_settings import (
    FeddriftRunClientSettings,
)


@dataclass(frozen=True, kw_only=True)
class FeddriftRunClientOwners:
    """既存ownerへの参照を固定し、可変状態はownerにだけ保持する。"""

    held_model_training_state_registry: HeldModelTrainingStateRegistry
    loss_statistics_store: ModelAndClassLossStatisticsStore
    current_training_model_assignment: CurrentTrainingModelAssignment
    training_sample_store: ModelTrainingSampleStore
    evaluation_sample_store: ModelEvaluationSampleStore
    temporary_model_id_allocator: TemporaryModelIdAllocator
    pending_model_upload_state: PendingModelUploadState
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore


class FeddriftRunClient:
    """assemblyが検査して接続したownerとprocessorへの操作入口。"""

    def __init__(
        self,
        *,
        client_id: int,
        owners: FeddriftRunClientOwners,
        run_client_settings: FeddriftRunClientSettings,
        python_random_generator: Random,
        detection_batch_processor: FeddriftDetectionBatchProcessor,
        stream_processor: FeddriftClientStreamProcessor,
    ) -> None:
        self._client_id = client_id
        self._owners = owners
        self._settings = run_client_settings
        self._random = python_random_generator
        self._batch_processor = detection_batch_processor
        self._stream_processor = stream_processor

    @property
    def client_id(self) -> int:
        return self._client_id

    @property
    def owners(self) -> FeddriftRunClientOwners:
        return self._owners

    @property
    def detection_batch_processor(self) -> FeddriftDetectionBatchProcessor:
        return self._batch_processor

    @property
    def stream_processor(self) -> FeddriftClientStreamProcessor:
        return self._stream_processor

    def has_model_ready_for_server_registration(self) -> bool:
        return self._owners.pending_model_upload_state.has_ready_model_upload()

    def process_observed_block(
        self,
        *,
        observed_samples: tuple[ObservedSample, ...],
        evaluation_concept_ids: tuple[int | None, ...],
    ) -> bool:
        """全blockを変換してからstreamへ渡し、概念は診断記録だけへ渡す。"""
        if type(observed_samples) is not tuple or type(evaluation_concept_ids) is not tuple:
            raise TypeError("観測と概念列はexact tupleが必要です。")
        if len(observed_samples) != len(evaluation_concept_ids):
            raise ValueError("観測と概念列は同じ件数が必要です。")
        indexed_observations: list[IndexedObservedTrainingSample] = []
        start = self._stream_processor.processed_sample_count
        for offset, (sample, concept_id) in enumerate(
            zip(observed_samples, evaluation_concept_ids, strict=True)
        ):
            if type(sample) is not ObservedSample:
                raise TypeError("観測はexact ObservedSampleが必要です。")
            sample.__post_init__()
            if concept_id is not None and type(concept_id) is not int:
                raise TypeError("概念IDはbuiltin intまたはNoneが必要です。")
            indexed_observations.append(
                IndexedObservedTrainingSample(
                    sample_index=start + offset,
                    observed_concept_id=concept_id,
                    training_sample=ObservedTrainingSample(
                        input_features=tensor((sample.feature_values,), dtype=float32),
                        observed_class_labels=tensor([[sample.class_label]], dtype=float32),
                    ),
                )
            )
        return self._stream_processor.process_observations(
            indexed_observations=tuple(indexed_observations)
        )

    def train_local_round(self) -> tuple[float, ...]:
        """旧の検出batch件数を倍率として、一roundだけ連続更新する。"""
        owners = self._owners
        return train_feddrift_held_models_for_round(
            held_model_training_state_registry=owners.held_model_training_state_registry,
            training_sample_store=owners.training_sample_store,
            model_training_and_assignment_counts_store=owners.model_training_and_assignment_counts_store,
            python_random_generator=self._random,
            batch_sample_count=self._settings.local_training_batch_sample_count,
            updates_per_sample=self._settings.updates_per_sample,
            k_steps=self._settings.detection_batch_sample_count,
        )

    def flush_detection_buffer(self) -> FeddriftDetectionBatchTransition | None:
        return self._stream_processor.flush_detection_buffer()


def assemble_feddrift_run_client(
    *,
    client_id: int,
    initial_classifier: ResidualAdapterClassifier,
    initial_parameter_optimizer_state: ParameterOptimizerState,
    initial_loss_statistics: ModelAndClassLossStatistics,
    run_client_settings: FeddriftRunClientSettings,
    python_random_generator: Random,
) -> FeddriftRunClient:
    """入力検査後だけ分類器/optimizerを同時に写し、同じownerへ接続する。"""
    if type(client_id) is not int or client_id < 0:
        raise ValueError("client IDは非負builtin intが必要です。")
    if type(run_client_settings) is not FeddriftRunClientSettings:
        raise TypeError("client設定はexact FeddriftRunClientSettingsが必要です。")
    run_client_settings.__post_init__()
    if type(python_random_generator) is not Random:
        raise TypeError("乱数はexact Randomが必要です。")
    _validate_held_model_training_state_inputs(
        model_id=0,
        classifier=initial_classifier,
        concept_specific_parameter_optimizer_state=initial_parameter_optimizer_state,
    )
    _validate_independent_model_structure(initial_classifier)
    if (
        initial_classifier.model_architecture_settings.model_architecture_name
        != INDEPENDENT_MODELS_ARCHITECTURE_NAME
    ):
        raise ValueError("FedDriftには独立モデル構造が必要です。")
    snapshot_classifier_parameters(classifier=initial_classifier)
    _validate_trainable_parameters(
        expected_parameters=tuple(initial_classifier.parameters()),
        require_trainable=True,
        tensor_name="initial_classifier",
    )
    statistics = _copy_model_and_class_loss_statistics(loss_statistics=initial_loss_statistics)
    if any(
        class_id >= initial_classifier.class_count
        for class_id, _ in statistics.class_loss_moments_by_class_id
    ):
        raise ValueError("初期統計のクラスIDが分類器の範囲を超えています。")
    classifier, optimizer_state = deepcopy((initial_classifier, initial_parameter_optimizer_state))
    registry = HeldModelTrainingStateRegistry()
    registry.register_held_model_training_state(
        model_id=0,
        classifier=classifier,
        concept_specific_parameter_optimizer_state=optimizer_state,
    )
    owners = FeddriftRunClientOwners(
        held_model_training_state_registry=registry,
        loss_statistics_store=ModelAndClassLossStatisticsStore(
            initial_loss_statistics_by_model_id={0: statistics}
        ),
        current_training_model_assignment=CurrentTrainingModelAssignment(initial_model_id=0),
        training_sample_store=ModelTrainingSampleStore(),
        evaluation_sample_store=ModelEvaluationSampleStore(
            maximum_stored_sample_count_per_model=run_client_settings.maximum_stored_evaluation_sample_count_per_model,
            added_batch_sample_count=run_client_settings.added_evaluation_batch_sample_count,
        ),
        temporary_model_id_allocator=TemporaryModelIdAllocator(client_id=client_id),
        pending_model_upload_state=PendingModelUploadState(),
        model_training_and_assignment_counts_store=ModelTrainingAndAssignmentCountsStore(),
    )
    processor = FeddriftDetectionBatchProcessor(
        held_model_training_state_registry=registry,
        loss_statistics_store=owners.loss_statistics_store,
        current_training_model_assignment=owners.current_training_model_assignment,
        training_sample_store=owners.training_sample_store,
        evaluation_sample_store=owners.evaluation_sample_store,
        temporary_model_id_allocator=owners.temporary_model_id_allocator,
        pending_model_upload_state=owners.pending_model_upload_state,
        python_random_generator=python_random_generator,
        new_model_optimizer_settings=run_client_settings.new_model_optimizer_settings,
        loss_increase_threshold=run_client_settings.loss_increase_threshold,
    )
    return FeddriftRunClient(
        client_id=client_id,
        owners=owners,
        run_client_settings=run_client_settings,
        python_random_generator=python_random_generator,
        detection_batch_processor=processor,
        stream_processor=FeddriftClientStreamProcessor(
            detection_batch_processor=processor,
            detection_batch_sample_count=run_client_settings.detection_batch_sample_count,
        ),
    )
