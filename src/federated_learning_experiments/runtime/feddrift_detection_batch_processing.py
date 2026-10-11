"""FedDriftの検出バッチを判定し、独立した状態ownerへ割当と登録を反映する。"""

from dataclasses import dataclass
from math import isfinite
from random import Random

from torch import cat

from federated_learning_experiments.evaluation.model_evaluation_sample_records import (
    ObservedEvaluationSample,
)
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import (
    initialize_model_and_class_loss_statistics_from_batch,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    INDEPENDENT_MODELS_ARCHITECTURE_NAME,
    ResidualAdapterClassifier,
    _validate_classifier_inputs,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
    validate_classifier_bounded_loss_inputs,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingState,
    HeldModelTrainingStateRegistry,
    _validate_held_model_training_state_inputs,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
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
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.feddrift.detection_batch_assessment import (
    assess_feddrift_detection_batch,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUpload,
    PendingModelUploadState,
)


def _validate_mean_loss(value: float) -> None:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("平均有界損失は有限な[0,1]のbuiltin floatが必要です。")


@dataclass(frozen=True, kw_only=True)
class FeddriftDetectionBatchTransition:
    """clientが検出位置・切替位置と結果種別を導くための、一バッチの不変記録。"""

    adaptation_kind: str
    assigned_model_id: int
    first_sample_index: int
    last_sample_index: int
    next_previous_minimum_mean_bounded_loss: float

    def __post_init__(self) -> None:
        if type(self.adaptation_kind) is not str or self.adaptation_kind not in (
            "current_model_retained",
            "held_model_selected",
            "new_model_created",
        ):
            raise ValueError("adaptation_kindはFedDriftの3分岐の名前が必要です。")
        if type(self.assigned_model_id) is not int:
            raise TypeError("assigned_model_idはbuiltin intが必要です。")
        if self.adaptation_kind == "new_model_created" and self.assigned_model_id >= 0:
            raise ValueError("新規モデルの割当IDは負が必要です。")
        if any(
            type(position) is not int
            for position in (self.first_sample_index, self.last_sample_index)
        ):
            raise TypeError("バッチ位置はbuiltin intが必要です。")
        if self.first_sample_index < 0 or self.last_sample_index < self.first_sample_index:
            raise ValueError("バッチ位置は非負で先頭から末尾の順が必要です。")
        _validate_mean_loss(self.next_previous_minimum_mean_bounded_loss)


class FeddriftDetectionBatchProcessor:
    """前回値のみを所有し、借用したownerへ完成バッチの処理を接続する。"""

    def __init__(
        self,
        *,
        held_model_training_state_registry: HeldModelTrainingStateRegistry,
        loss_statistics_store: ModelAndClassLossStatisticsStore,
        current_training_model_assignment: CurrentTrainingModelAssignment,
        training_sample_store: ModelTrainingSampleStore,
        evaluation_sample_store: ModelEvaluationSampleStore,
        temporary_model_id_allocator: TemporaryModelIdAllocator,
        pending_model_upload_state: PendingModelUploadState,
        python_random_generator: Random,
        new_model_optimizer_settings: AdamParameterOptimizerSettings
        | SgdParameterOptimizerSettings,
        loss_increase_threshold: float,
        previous_minimum_mean_bounded_loss: float | None = None,
    ) -> None:
        for owner, expected_type in (
            (held_model_training_state_registry, HeldModelTrainingStateRegistry),
            (loss_statistics_store, ModelAndClassLossStatisticsStore),
            (current_training_model_assignment, CurrentTrainingModelAssignment),
            (training_sample_store, ModelTrainingSampleStore),
            (evaluation_sample_store, ModelEvaluationSampleStore),
            (temporary_model_id_allocator, TemporaryModelIdAllocator),
            (pending_model_upload_state, PendingModelUploadState),
            (python_random_generator, Random),
        ):
            if type(owner) is not expected_type:
                raise TypeError(f"ownerはexact {expected_type.__name__}が必要です。")
        if type(new_model_optimizer_settings) not in (
            AdamParameterOptimizerSettings,
            SgdParameterOptimizerSettings,
        ):
            raise TypeError("新規optimizerはexact Adam/SGD設定が必要です。")
        new_model_optimizer_settings.__post_init__()
        if (
            type(loss_increase_threshold) is not float
            or not isfinite(loss_increase_threshold)
            or loss_increase_threshold < 0
        ):
            raise ValueError("閾値は非負有限のbuiltin floatが必要です。")
        if previous_minimum_mean_bounded_loss is not None:
            _validate_mean_loss(previous_minimum_mean_bounded_loss)
        self._registry = held_model_training_state_registry
        self._statistics = loss_statistics_store
        self._assignment = current_training_model_assignment
        self._training_samples = training_sample_store
        self._evaluation_samples = evaluation_sample_store
        self._allocator = temporary_model_id_allocator
        self._uploads = pending_model_upload_state
        self._random = python_random_generator
        self._optimizer_settings = new_model_optimizer_settings
        self._threshold = loss_increase_threshold
        self._previous_loss = previous_minimum_mean_bounded_loss

    @property
    def previous_minimum_mean_bounded_loss(self) -> float | None:
        return self._previous_loss

    def _validate_batch(
        self, *, training_samples: tuple[ObservedTrainingSample, ...], processed_sample_count: int
    ) -> tuple[HeldModelTrainingState, ...]:
        if type(training_samples) is not tuple or not training_samples:
            raise ValueError("training_samplesは非空exact tupleが必要です。")
        if type(processed_sample_count) is not int or processed_sample_count < len(
            training_samples
        ):
            raise ValueError("processed_sample_countは標本数以上のbuiltin intが必要です。")
        states = self._registry.snapshot_ordered_held_model_training_states()
        if not states:
            raise ValueError("保有モデルは非空が必要です。")
        current_id = self._assignment.current_training_model_id
        if type(current_id) is not int or current_id not in tuple(
            state.model_id for state in states
        ):
            raise ValueError("現在IDが保有モデルにありません。")
        self._optimizer_settings.__post_init__()
        for state in states:
            if type(state) is not HeldModelTrainingState:
                raise TypeError("保有記録はexact HeldModelTrainingStateが必要です。")
            _validate_held_model_training_state_inputs(
                model_id=state.model_id,
                classifier=state.classifier,
                concept_specific_parameter_optimizer_state=state.concept_specific_parameter_optimizer_state,
            )
            classifier = state.classifier
            _validate_classifier_inputs(
                model_architecture_settings=classifier.model_architecture_settings,
                input_feature_count=classifier.input_feature_count,
                hidden_layer_widths=classifier.hidden_layer_widths,
                class_count=classifier.class_count,
                shared_feature_extractor=None,
            )
            if (
                classifier.model_architecture_settings.model_architecture_name
                != INDEPENDENT_MODELS_ARCHITECTURE_NAME
            ):
                raise ValueError("FedDriftは独立モデル構造が必要です。")
            snapshot_classifier_parameters(classifier=classifier)
            for sample in training_samples:
                if type(sample) is not ObservedTrainingSample:
                    raise TypeError("標本はexact ObservedTrainingSampleが必要です。")
                validate_classifier_bounded_loss_inputs(
                    classifier=classifier,
                    input_features=sample.input_features,
                    observed_class_labels=sample.observed_class_labels,
                )
                if sample.input_features.shape[0] != 1:
                    raise ValueError("各標本のpayloadは1件が必要です。")
        return states

    def _validate_new_model_id(
        self, temporary_model_id: int, states: tuple[HeldModelTrainingState, ...]
    ) -> None:
        if type(temporary_model_id) is not int or temporary_model_id >= 0:
            raise ValueError("次の一時IDは負のbuiltin intが必要です。")
        used_ids = {state.model_id for state in states}
        used_ids.update(model_id for model_id, _ in self._statistics.get_state_snapshot())
        used_ids.update(
            collection.model_id
            for collection in self._training_samples.snapshot_ordered_model_training_samples()
        )
        used_ids.update(
            collection.model_id
            for collection in self._evaluation_samples.snapshot_ordered_model_evaluation_samples()
        )
        pending = self._uploads.get_pending_model_upload()
        if pending is not None:
            used_ids.add(pending.model_id)
        if temporary_model_id in used_ids:
            raise ValueError("次の一時IDが既存状態と衝突しています。")

    def process_detection_batch(
        self, *, training_samples: tuple[ObservedTrainingSample, ...], processed_sample_count: int
    ) -> FeddriftDetectionBatchTransition:
        """全入力と更新候補を準備してから、旧と同じ分岐のowner更新を行う。"""
        states = self._validate_batch(
            training_samples=training_samples, processed_sample_count=processed_sample_count
        )
        features = cat([sample.input_features for sample in training_samples])
        labels = cat([sample.observed_class_labels for sample in training_samples])
        assessment = assess_feddrift_detection_batch(
            held_models=tuple((state.model_id, state.classifier) for state in states),
            input_features=features,
            observed_class_labels=labels,
            previous_minimum_mean_bounded_loss=self._previous_loss,
            loss_increase_threshold=self._threshold,
        )
        previous_id = self._assignment.current_training_model_id
        created_state = None
        pending_upload = None
        if assessment.drift_detected:
            assigned_id = self._allocator.next_temporary_model_id
            self._validate_new_model_id(assigned_id, states)
            current = self._registry.get_held_model_training_state(model_id=previous_id).classifier
            # 最小損失モデルではなく、判定前の現在モデルから複写する。
            classifier = ResidualAdapterClassifier(
                model_architecture_settings=current.model_architecture_settings,
                input_feature_count=current.input_feature_count,
                hidden_layer_widths=current.hidden_layer_widths,
                class_count=current.class_count,
            )
            classifier.load_state_dict(snapshot_classifier_parameters(classifier=current))
            optimizer_state = ParameterOptimizerState(
                parameters=tuple(classifier.parameters()),
                optimizer_settings=self._optimizer_settings,
            )
            _validate_held_model_training_state_inputs(
                model_id=assigned_id,
                classifier=classifier,
                concept_specific_parameter_optimizer_state=optimizer_state,
            )
            created_state = HeldModelTrainingState(
                model_id=assigned_id,
                classifier=classifier,
                concept_specific_parameter_optimizer_state=optimizer_state,
            )
            losses = evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier, input_features=features, observed_class_labels=labels
            )
            statistics = initialize_model_and_class_loss_statistics_from_batch(
                per_sample_bounded_losses=losses,
                observed_class_labels=labels,
                class_count=classifier.class_count,
            )
            next_loss = statistics.overall_loss_moments.mean_loss
            pending_upload = PendingModelUpload(
                model_id=assigned_id,
                parameter_snapshot=snapshot_classifier_parameters(classifier=classifier),
            )
            kind = "new_model_created"
        else:
            assigned_id = assessment.minimum_loss_model_id
            classifier = self._registry.get_held_model_training_state(
                model_id=assigned_id
            ).classifier
            staged_statistics = ModelAndClassLossStatisticsStore(
                initial_loss_statistics_by_model_id=dict(self._statistics.get_state_snapshot())
            )
            for sample in training_samples:
                losses = evaluate_classifier_per_sample_bounded_losses(
                    classifier=classifier,
                    input_features=sample.input_features,
                    observed_class_labels=sample.observed_class_labels,
                )
                staged_statistics.record_assigned_loss(
                    model_id=assigned_id,
                    observed_loss=float(losses.mean().item()),
                    observed_class_id=int(sample.observed_class_labels.item()),
                )
            statistics = staged_statistics.get_model_loss_statistics(model_id=assigned_id)
            assert statistics is not None
            next_loss = assessment.minimum_mean_bounded_loss
            kind = "current_model_retained" if assigned_id == previous_id else "held_model_selected"
        transition = FeddriftDetectionBatchTransition(
            adaptation_kind=kind,
            assigned_model_id=assigned_id,
            first_sample_index=processed_sample_count - len(training_samples),
            last_sample_index=processed_sample_count - 1,
            next_previous_minimum_mean_bounded_loss=next_loss,
        )
        evaluation_samples = tuple(
            ObservedEvaluationSample(
                input_features=s.input_features, observed_class_labels=s.observed_class_labels
            )
            for s in training_samples
        )
        if created_state is not None:
            assert pending_upload is not None
            self._allocator.allocate_temporary_model_id()
            self._registry.register_held_model_training_state(
                model_id=assigned_id,
                classifier=created_state.classifier,
                concept_specific_parameter_optimizer_state=created_state.concept_specific_parameter_optimizer_state,
            )
            self._uploads.queue_ready_model_upload(
                model_id=assigned_id, parameter_snapshot=pending_upload.parameter_snapshot
            )
        self._assignment.assign_model_for_training(model_id=assigned_id)
        self._training_samples.append_model_training_samples(
            model_id=assigned_id, training_samples=training_samples
        )
        self._statistics.set_model_loss_statistics(model_id=assigned_id, loss_statistics=statistics)
        self._previous_loss = next_loss
        self._evaluation_samples.sample_and_append_model_evaluation_samples(
            model_id=assigned_id,
            evaluation_samples=evaluation_samples,
            python_random_generator=self._random,
        )
        return transition
