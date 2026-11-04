"""FedDrift clientの時刻をまたぐ観測列を処理する。"""

from dataclasses import dataclass, replace

from torch import no_grad

from federated_learning_experiments.evaluation.held_model_count_record_store import (
    HeldModelCountRecordStore,
)
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    predict_class_labels_from_prediction_scores,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.timestep_training_sample_store import (
    TimestepTrainingSampleStore,
)
from federated_learning_experiments.runtime.fedccfa_classifier_clustering import FedccfaClientState
from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
    FeddriftDetectionBatchProcessor,
    FeddriftDetectionBatchTransition,
)
from federated_learning_experiments.runtime.fedem_mixture import FedemClientMixture


@dataclass(frozen=True, kw_only=True)
class FeddriftSamplePrediction:
    """切替前の予測と、通常観測で発火したバッチの結果を結び付ける。"""

    sample_index: int
    observed_concept_id: int | None
    prediction_model_id: int
    predicted_class_id: int
    observed_class_id: int
    prediction_is_correct: bool
    detection_batch_transition: FeddriftDetectionBatchTransition | None = None
    training_model_id: int | None = None

    def __post_init__(self) -> None:
        for value in (
            self.sample_index,
            self.prediction_model_id,
            self.predicted_class_id,
            self.observed_class_id,
        ):
            if type(value) is not int:
                raise TypeError("位置・モデルID・クラスはbuiltin intが必要です。")
        if min(self.sample_index, self.predicted_class_id, self.observed_class_id) < 0:
            raise ValueError("位置・クラスは非負が必要です。")
        if self.observed_concept_id is not None and type(self.observed_concept_id) is not int:
            raise TypeError("概念IDはbuiltin intまたはNoneが必要です。")
        if type(self.prediction_is_correct) is not bool:
            raise TypeError("正否はbuiltin boolが必要です。")
        if self.prediction_is_correct != (self.predicted_class_id == self.observed_class_id):
            raise ValueError("正否と予測/観測クラスが対応していません。")
        if self.training_model_id is not None and (
            type(self.training_model_id) is not int or self.training_model_id < 0
        ):
            raise ValueError("学習の行き先は非負のbuiltin intが必要です。")
        if self.detection_batch_transition is not None:
            if type(self.detection_batch_transition) is not FeddriftDetectionBatchTransition:
                raise TypeError("バッチ結果はexact FeddriftDetectionBatchTransitionが必要です。")
            self.detection_batch_transition.__post_init__()
            if self.detection_batch_transition.last_sample_index != self.sample_index:
                raise ValueError("バッチ末尾と予測位置が対応していません。")


class FeddriftClientStreamProcessor:
    """予測→蓄積→満杯バッチの順に観測を進め、局所学習は行わない。"""

    def __init__(
        self,
        *,
        detection_batch_processor: FeddriftDetectionBatchProcessor,
        detection_batch_sample_count: int,
        model_assignment_policy: str = "detection_batch_assignment",
        training_sample_window: str = "all_timesteps",
    ) -> None:
        if type(detection_batch_processor) is not FeddriftDetectionBatchProcessor:
            raise TypeError("バッチ処理はexact FeddriftDetectionBatchProcessorが必要です。")
        if type(detection_batch_sample_count) is not int or detection_batch_sample_count < 1:
            raise ValueError("検出件数は正のbuiltin intが必要です。")
        self._processor = detection_batch_processor
        if model_assignment_policy not in (
            "detection_batch_assignment",
            "true_concept_assignment",
            "minimum_loss_cluster_assignment",
            "single_global_model_assignment",
            "driftsurf_global_role_assignment",
            "cfl_bipartition_assignment",
            "fedem_responsibility_assignment",
            "fedccfa_classifier_clustering",
        ):
            raise ValueError("未対応の割当方式です。")
        self._driftsurf_assignment = model_assignment_policy == "driftsurf_global_role_assignment"
        self._cfl_assignment = model_assignment_policy == "cfl_bipartition_assignment"
        self._cfl_assignments: list[tuple[int, int, int]] = []
        self._driftsurf_pending_samples: list[ObservedTrainingSample] = []
        self._driftsurf_timestep_samples = TimestepTrainingSampleStore()
        self._single_global_model_assignment = (
            model_assignment_policy == "single_global_model_assignment"
        )
        self._true_concept_assignment = model_assignment_policy == "true_concept_assignment"
        self._minimum_loss_cluster_assignment = (
            model_assignment_policy == "minimum_loss_cluster_assignment"
        )
        if training_sample_window not in (
            "all_timesteps",
            "current_timestep",
            "role_timestep_window",
            "all_observed_history",
            "recent_observed_samples",
        ):
            raise ValueError("未対応の学習窓です。")
        self._current_timestep_window = training_sample_window == "current_timestep"
        self._fedem = (
            FedemClientMixture(
                registry=detection_batch_processor._registry,
                training_samples=detection_batch_processor._training_samples,
                current_timestep_window=self._current_timestep_window,
            )
            if model_assignment_policy == "fedem_responsibility_assignment"
            else None
        )
        self._fedccfa = (
            FedccfaClientState(
                classifier=detection_batch_processor.current_training_model_state.classifier,
                recent_window=training_sample_window == "recent_observed_samples",
            )
            if model_assignment_policy == "fedccfa_classifier_clustering"
            else None
        )
        self._indexed_buffer: list[IndexedObservedTrainingSample] = []
        self._current_timestep_assigned_model_ids: set[int] = set()
        self._batch_size = detection_batch_sample_count
        self._processed_count = 0
        self._buffer: list[ObservedTrainingSample] = []
        self._predictions: list[FeddriftSamplePrediction] = []
        self._transitions: list[FeddriftDetectionBatchTransition] = []
        self.held_model_count_record_store = HeldModelCountRecordStore()

    @property
    def processed_sample_count(self) -> int:
        return self._processed_count

    @property
    def buffered_training_samples(self) -> tuple[ObservedTrainingSample, ...]:
        """標本Tensorは借用したまま、列の変更だけを切り離す。"""
        return tuple(self._buffer)

    @property
    def sample_predictions(self) -> tuple[FeddriftSamplePrediction, ...]:
        return tuple(self._predictions)

    @property
    def completed_detection_batch_transitions(self) -> tuple[FeddriftDetectionBatchTransition, ...]:
        return tuple(self._transitions)

    def process_observations(
        self, *, indexed_observations: tuple[IndexedObservedTrainingSample, ...]
    ) -> bool:
        """全ブロックの入力を先に検査し、満杯検出が一度でもあればtrueを返す。"""
        if type(indexed_observations) is not tuple:
            raise TypeError("観測列はexact tupleが必要です。")
        if not indexed_observations:
            return False
        if type(self.held_model_count_record_store) is not HeldModelCountRecordStore:
            raise TypeError("保有数の記録にはexact HeldModelCountRecordStoreが必要です。")
        for offset, observation in enumerate(indexed_observations):
            if type(observation) is not IndexedObservedTrainingSample:
                raise TypeError("観測はexact IndexedObservedTrainingSampleが必要です。")
            if type(observation.sample_index) is not int:
                raise TypeError("位置はbuiltin intが必要です。")
            if observation.sample_index != self._processed_count + offset:
                raise ValueError("位置は処理済み件数から連続している必要があります。")
            if (
                observation.observed_concept_id is not None
                and type(observation.observed_concept_id) is not int
            ):
                raise TypeError("概念IDはbuiltin intまたはNoneが必要です。")
        self._processor.validate_detection_batch_inputs(
            training_samples=tuple(self._buffer)
            + tuple(x.training_sample for x in indexed_observations),
            processed_sample_count=self._processed_count + len(indexed_observations),
        )
        if self._true_concept_assignment:
            self._processor._validate_true_concept_observations(indexed_observations)
        fired = False
        for observation in indexed_observations:
            if (
                self._fedccfa is not None
                and not self._fedccfa.recent_window
                and self._processed_count % self._batch_size == 0
            ):
                self._fedccfa.samples.clear()
                self._fedccfa.sample_indices.clear()
                self._processor._training_samples.clear_model_training_samples()
            if self._cfl_assignment and self._processed_count % self._batch_size == 0:
                # 前の分割後の割当を先に写し、それから前時刻の学習窓を空にする。
                self._cfl_assignments.append(
                    (
                        self._processed_count // self._batch_size,
                        self._processed_count,
                        self._processor._assignment.current_training_model_id,
                    )
                )
                self._processor._training_samples.clear_model_training_samples()
            held_model_count = self._processor.held_model_count
            if type(held_model_count) is not int:
                raise TypeError("予測前保有数はbuiltin intが必要です。")
            if held_model_count < 1:
                raise ValueError("予測前保有数は正が必要です。")
            current = self._processor.current_training_model_state
            sample = observation.training_sample
            with no_grad():
                scores = (
                    self._fedccfa.predict(sample.input_features)
                    if self._fedccfa is not None
                    else self._fedem.predict(sample.input_features)
                    if self._fedem is not None
                    else current.classifier(sample.input_features)
                )
                predicted_class_id = int(
                    predict_class_labels_from_prediction_scores(
                        prediction_scores=scores, class_count=current.classifier.class_count
                    )
                    .view(-1)[0]
                    .item()
                )
            observed_class_id = int(sample.observed_class_labels.view(-1)[0].item())
            prediction = FeddriftSamplePrediction(
                sample_index=observation.sample_index,
                observed_concept_id=observation.observed_concept_id,
                prediction_model_id=current.model_id,
                predicted_class_id=predicted_class_id,
                observed_class_id=observed_class_id,
                prediction_is_correct=predicted_class_id == observed_class_id,
                training_model_id=(
                    observation.observed_concept_id if self._true_concept_assignment else None
                ),
            )
            self._processed_count += 1
            if self._fedccfa is not None:
                self._fedccfa.append_timestep_samples(
                    (sample,), self._processed_count, self._batch_size
                )
            self._buffer.append(sample)
            if self._driftsurf_assignment:
                self._driftsurf_pending_samples.append(sample)
            if self._true_concept_assignment:
                self._indexed_buffer.append(observation)
                self._processor._observe_true_concept(observation)
            if len(self._buffer) >= self._batch_size:
                transition = self.flush_detection_buffer()
                prediction = replace(prediction, detection_batch_transition=transition)
                fired = True
            self._predictions.append(prediction)
            self.held_model_count_record_store.append_held_model_count(
                held_model_count=held_model_count
            )
        return fired

    def flush_detection_buffer(self) -> FeddriftDetectionBatchTransition | None:
        """残りを一度処理し、既存予測記録へは結果を付加しない。"""
        if not self._buffer:
            return None
        if self._fedccfa is not None:
            store = self._processor._training_samples
            store.clear_model_training_samples()
            store.append_model_training_samples(
                model_id=0, training_samples=tuple(self._fedccfa.samples)
            )
            self._buffer.clear()
            return None
        if self._fedem is not None:
            self._fedem.append_timestep_samples(
                samples=tuple(self._buffer), processed_sample_count=self._processed_count
            )
            self._buffer.clear()
            return None
        if self._cfl_assignment:
            selected = self._processor._assignment.current_training_model_id
            self._processor._training_samples.append_model_training_samples(
                model_id=selected, training_samples=tuple(self._buffer)
            )
            self._buffer.clear()
            return None
        if self._driftsurf_assignment:
            # 役割の窓を決めるまで、モデルの学習storeへは追加しない。
            self._buffer.clear()
            return None
        if self._single_global_model_assignment:
            self._processor._assign_single_global_model_samples(
                training_samples=tuple(self._buffer), processed_sample_count=self._processed_count
            )
            self._buffer.clear()
            return None
        if self._minimum_loss_cluster_assignment:
            self._processor._assign_minimum_loss_cluster_samples(
                training_samples=tuple(self._buffer),
                processed_sample_count=self._processed_count,
                current_timestep_window=self._current_timestep_window,
            )
            self._buffer.clear()
            return None
        if self._true_concept_assignment:
            self._processor._assign_true_concept_samples(tuple(self._indexed_buffer))
            self._current_timestep_assigned_model_ids.update(
                x.observed_concept_id
                for x in self._indexed_buffer
                if x.observed_concept_id is not None
            )
            self._indexed_buffer.clear()
            self._buffer.clear()
            return None
        transition = self._processor.process_detection_batch(
            training_samples=tuple(self._buffer), processed_sample_count=self._processed_count
        )
        self._transitions.append(transition)
        self._buffer.clear()
        return transition
