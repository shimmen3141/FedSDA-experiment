"""全到着標本を保持して学習する単一モデルclient。"""

from copy import deepcopy
from dataclasses import dataclass
from random import Random

from torch import Tensor, cat, float32, isfinite, no_grad, tensor

from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
    validate_classifier_bounded_loss_inputs,
)
from federated_learning_experiments.learning.training.independent_model_parameter_update import (
    update_independent_model_parameters,
    validate_independent_model_parameter_update_inputs,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)


@dataclass(frozen=True, kw_only=True)
class FedavgSamplePrediction:
    """学習前の予測と診断専用の概念。モデルIDは常に0。"""

    sample_index: int
    predicted_class_id: int
    observed_class_id: int
    observed_concept_id: int
    prediction_is_correct: bool
    prediction_model_id: int = 0


class FedavgClient:
    """モデル/optimizerの結合と、全保持データ・統計・予算を所有する。"""

    def __init__(
        self,
        *,
        client_id: int,
        initial_classifier: ModelArchitectureClassifier,
        initial_parameter_optimizer_state: ParameterOptimizerState,
        initial_loss_statistics: ModelAndClassLossStatistics,
        local_training_batch_sample_count: int,
        updates_per_sample: int,
        local_update_interval_per_client_samples: int,
        python_random_generator: Random,
    ) -> None:
        for count, minimum in (
            (client_id, 0),
            (local_training_batch_sample_count, 1),
            (updates_per_sample, 0),
            (local_update_interval_per_client_samples, 1),
        ):
            if type(count) is not int or count < minimum:
                raise ValueError("件数/IDには範囲内のbuiltin intが必要です。")
        if type(python_random_generator) is not Random:
            raise TypeError("exact Randomが必要です。")
        if type(initial_parameter_optimizer_state) is not ParameterOptimizerState:
            raise TypeError("exact ParameterOptimizerStateが必要です。")
        snapshot_classifier_parameters(classifier=initial_classifier)
        x = tensor([[0.0] * initial_classifier.input_feature_count], dtype=float32)
        y = tensor([[0.0]], dtype=float32)
        validate_independent_model_parameter_update_inputs(
            classifier=initial_classifier,
            parameter_optimizer=initial_parameter_optimizer_state.parameter_optimizer,
            input_features=x,
            observed_class_labels=y,
        )
        statistics = ModelAndClassLossStatisticsStore(
            initial_loss_statistics_by_model_id={0: initial_loss_statistics}
        )
        if any(
            k >= initial_classifier.class_count
            for k, _ in initial_loss_statistics.class_loss_moments_by_class_id
        ):
            raise ValueError("統計のクラスIDが分類器の範囲外です。")
        self.classifier, self.parameter_optimizer_state = deepcopy(
            (initial_classifier, initial_parameter_optimizer_state)
        )
        self.client_id = client_id
        self.loss_statistics_store = statistics
        self.python_random_generator = python_random_generator
        self._batch = local_training_batch_sample_count
        self._updates = updates_per_sample
        self._interval = local_update_interval_per_client_samples
        self.training_samples: list[tuple[Tensor, Tensor]] = []
        self.sample_predictions: list[FedavgSamplePrediction] = []
        self.pending_update_sample_count = 0
        self.trained_sample_count = 0
        self.parameter_update_step_count = 0
        self.prediction_forward_call_count = 0
        self.statistics_forward_call_count = 0

    def process_observed_sample(
        self, *, observed_sample: ObservedSample, sample_index: int, evaluation_concept_id: int
    ) -> None:
        """検査後、予測→全体損失統計→保持→更新の順に進める。"""
        if type(observed_sample) is not ObservedSample:
            raise TypeError("exact ObservedSampleが必要です。")
        observed_sample.__post_init__()
        if type(sample_index) is not int or sample_index != len(self.sample_predictions):
            raise ValueError("連続するbuiltin標本位置が必要です。")
        if type(evaluation_concept_id) is not int or evaluation_concept_id < 0:
            raise ValueError("診断概念IDには非負builtin intが必要です。")
        x = tensor([observed_sample.feature_values], dtype=float32)
        y = tensor([[float(observed_sample.class_label)]], dtype=float32)
        validate_classifier_bounded_loss_inputs(
            classifier=self.classifier, input_features=x, observed_class_labels=y
        )
        validate_independent_model_parameter_update_inputs(
            classifier=self.classifier,
            parameter_optimizer=self.parameter_optimizer_state.parameter_optimizer,
            input_features=x,
            observed_class_labels=y,
        )
        with no_grad():
            scores = self.classifier(x)
            predicted = (
                int((scores > 0.5).item())
                if self.classifier.class_count == 2
                else int(scores.argmax(dim=1).item())
            )
        loss = float(
            evaluate_classifier_per_sample_bounded_losses(
                classifier=self.classifier, input_features=x, observed_class_labels=y
            )[0].item()
        )
        prediction = FedavgSamplePrediction(
            sample_index=sample_index,
            predicted_class_id=predicted,
            observed_class_id=observed_sample.class_label,
            observed_concept_id=evaluation_concept_id,
            prediction_is_correct=predicted == observed_sample.class_label,
        )
        self.sample_predictions.append(prediction)
        self.prediction_forward_call_count += 1
        self.statistics_forward_call_count += 1
        # 旧baselineのオンライン統計は全体だけ。初期class統計を維持する。
        self.loss_statistics_store.record_assigned_loss(model_id=0, observed_loss=loss)
        self.training_samples.append((x, y))
        self.pending_update_sample_count += 1
        if self.pending_update_sample_count >= self._interval:
            self.flush_pending_local_updates(round_index=0)

    def flush_pending_local_updates(self, *, round_index: int) -> None:
        """十分なbatchがあるときだけ保留件数×予算回学習し、保留を消化する。"""
        if type(round_index) is not int or round_index < 0:
            raise ValueError("round_indexは非負builtin intが必要です。")
        if len(self.training_samples) >= self._batch:
            for _ in range(self.pending_update_sample_count * self._updates):
                batch = self.python_random_generator.sample(self.training_samples, self._batch)
                update_independent_model_parameters(
                    classifier=self.classifier,
                    parameter_optimizer=self.parameter_optimizer_state.parameter_optimizer,
                    input_features=cat([x for x, _ in batch]),
                    observed_class_labels=cat([y for _, y in batch]),
                )
                self.trained_sample_count += self._batch
                self.parameter_update_step_count += 1
        self.pending_update_sample_count = 0

    def has_model_ready_for_server_registration(self) -> bool:
        return False

    def advance_new_model_upload_wait_after_synchronization(self, *, round_index: int) -> None:
        pass

    def apply_global_parameters(self, *, parameter_snapshot: dict[str, Tensor]) -> None:
        """配布値を検査してから新モデルと新optimizerへ交換する。"""
        current = snapshot_classifier_parameters(classifier=self.classifier)
        if type(parameter_snapshot) is not dict or parameter_snapshot.keys() != current.keys():
            raise ValueError("配布parameterのキーが分類器と一致しません。")
        for key, value in parameter_snapshot.items():
            reference = current[key]
            if (
                type(value) is not Tensor
                or value.shape != reference.shape
                or value.dtype != reference.dtype
                or value.device != reference.device
                or not isfinite(value).all().item()
            ):
                raise ValueError("配布parameterは同じshape/device/dtypeの有限値が必要です。")
        old = self.classifier
        classifier = ModelArchitectureClassifier(
            model_architecture_settings=old.model_architecture_settings,
            input_feature_count=old.input_feature_count,
            hidden_layer_widths=old.hidden_layer_widths,
            class_count=old.class_count,
        )
        classifier.load_state_dict(parameter_snapshot)
        optimizer = ParameterOptimizerState(
            parameters=tuple(classifier.parameters()),
            optimizer_settings=self.parameter_optimizer_state._optimizer_settings,
        )
        self.classifier = classifier
        self.parameter_optimizer_state = optimizer
