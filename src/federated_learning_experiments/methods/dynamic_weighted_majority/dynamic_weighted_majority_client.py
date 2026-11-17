"""`fedavg`と同じ学習器を専門家にしたDynamic Weighted Majority（Kolter & Maloof, JMLR 8, 2007の図1）のclient。

標本ごとに、(1) 学習の前の重みで投票して評価の予測を記録し、(2) 周期の標本（`sample_index + 1`がpの倍数）でだけ
図1の重みの更新・削除・追加を行い、(3) 全専門家に標本を保持させて学習する。学習は`FedavgClient`と同じ部品を
同じ順で呼ぶ（保持数がbatch以上なら`python_random_generator.sample(保持, batch)`で抜いたbatchで
`update_independent_model_parameters`を`updates_per_sample`回）。新しい専門家は、受け取った初期状態
（事前学習の直後）の複製。公式の実装と違い、何も学習していない学習器ではない。
"""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from random import Random

from torch import Tensor, cat, float32, no_grad, tensor

from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    validate_classifier_bounded_loss_inputs,
)
from federated_learning_experiments.learning.training.independent_model_parameter_update import (
    update_independent_model_parameters,
    validate_independent_model_parameter_update_inputs,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.methods.dynamic_weighted_majority.dynamic_weighted_majority_rules import (
    update_weights_at_period,
    vote_weighted_majority,
)


@dataclass(kw_only=True, eq=False)
class DynamicWeightedMajorityExpert:
    """専門家1つの状態。分類器とoptimizerは、初期状態の組の1回の`deepcopy`で作る。"""

    expert_id: int
    classifier: ModelArchitectureClassifier
    parameter_optimizer_state: ParameterOptimizerState
    weight: float
    training_samples: list[tuple[Tensor, Tensor]] = field(default_factory=list)
    trained_sample_count: int = 0
    parameter_update_step_count: int = 0
    prediction_forward_call_count: int = 0


@dataclass(frozen=True, kw_only=True)
class DynamicWeightedMajoritySamplePrediction:
    """学習の前の重みによる評価の予測。`prediction_model_id`は予測の時点で重みが最大の専門家（同点は最小のID）。"""

    sample_index: int
    predicted_class_id: int
    observed_class_id: int
    observed_concept_id: int
    prediction_is_correct: bool
    prediction_model_id: int
    held_model_count_at_prediction: int


@dataclass(frozen=True, kw_only=True)
class DynamicWeightedMajorityExpertLifetime:
    """専門家ごとの作られた`sample_index`（初期の専門家は-1）と消された`sample_index`（残ったものは-1）。"""

    expert_id: int
    created_sample_index: int
    removed_sample_index: int


@dataclass(frozen=True, kw_only=True)
class DynamicWeightedMajorityExpertTrainingCounts:
    """専門家ごとの学習と予測の計数（消された専門家は消される時点の値）。"""

    expert_id: int
    trained_sample_count: int
    parameter_update_step_count: int
    prediction_forward_call_count: int


def _validate_unit_interval(value: object, *, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name}にはbuiltin floatが必要です。")
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name}は開区間(0, 1)が必要です: {value}")
    return value


class DynamicWeightedMajorityClient:
    """専門家・重み・保持標本・記録を所有し、標本ごとの予測と学習を行う。"""

    def __init__(
        self,
        *,
        client_id: int,
        initial_classifier: ModelArchitectureClassifier,
        initial_parameter_optimizer_state: ParameterOptimizerState,
        local_training_batch_sample_count: int,
        updates_per_sample: int,
        weight_decrease_factor: float,
        removal_weight_threshold: float,
        period_sample_count: int,
        python_random_generator: Random,
    ) -> None:
        for count, minimum in (
            (client_id, 0),
            (local_training_batch_sample_count, 1),
            (updates_per_sample, 0),
            (period_sample_count, 1),
        ):
            if type(count) is not int or count < minimum:
                raise ValueError("件数/IDには範囲内のbuiltin intが必要です。")
        _validate_unit_interval(weight_decrease_factor, name="weight_decrease_factor")
        _validate_unit_interval(removal_weight_threshold, name="removal_weight_threshold")
        if type(python_random_generator) is not Random:
            raise TypeError("exact Randomが必要です。")
        if type(initial_parameter_optimizer_state) is not ParameterOptimizerState:
            raise TypeError("exact ParameterOptimizerStateが必要です。")
        snapshot_classifier_parameters(classifier=initial_classifier)
        validate_independent_model_parameter_update_inputs(
            classifier=initial_classifier,
            parameter_optimizer=initial_parameter_optimizer_state.parameter_optimizer,
            input_features=tensor([[0.0] * initial_classifier.input_feature_count], dtype=float32),
            observed_class_labels=tensor([[0.0]], dtype=float32),
        )
        # 以後の学習の影響を受けない元。分類器とoptimizerの組を1回のdeepcopyで複製する
        # （別々に複製すると、optimizerが複製でない方の分類器のパラメータを持ったままになる）。
        self._initial_state = deepcopy((initial_classifier, initial_parameter_optimizer_state))
        self.client_id = client_id
        self.python_random_generator = python_random_generator
        self.weight_decrease_factor = weight_decrease_factor
        self.removal_weight_threshold = removal_weight_threshold
        self.period_sample_count = period_sample_count
        self._batch = local_training_batch_sample_count
        self._updates = updates_per_sample
        self.class_count = initial_classifier.class_count
        self.sample_predictions: list[DynamicWeightedMajoritySamplePrediction] = []
        self._lifetimes: dict[int, DynamicWeightedMajorityExpertLifetime] = {}
        self._removed_training_counts: dict[int, DynamicWeightedMajorityExpertTrainingCounts] = {}
        self.experts: list[DynamicWeightedMajorityExpert] = []
        self._add_expert(created_sample_index=-1)

    def _add_expert(self, *, created_sample_index: int) -> None:
        classifier, optimizer_state = deepcopy(self._initial_state)
        expert_id = len(self._lifetimes)
        self.experts.append(
            DynamicWeightedMajorityExpert(
                expert_id=expert_id,
                classifier=classifier,
                parameter_optimizer_state=optimizer_state,
                weight=1.0,
            )
        )
        self._lifetimes[expert_id] = DynamicWeightedMajorityExpertLifetime(
            expert_id=expert_id, created_sample_index=created_sample_index, removed_sample_index=-1
        )

    @property
    def expert_lifetimes(self) -> tuple[DynamicWeightedMajorityExpertLifetime, ...]:
        """作られた順の全専門家（消されたものを含む）の寿命の記録。"""
        return tuple(self._lifetimes[expert_id] for expert_id in sorted(self._lifetimes))

    @property
    def expert_training_counts(self) -> tuple[DynamicWeightedMajorityExpertTrainingCounts, ...]:
        """作られた順の全専門家（消されたものは消される時点）の学習と予測の計数。"""
        current = {expert.expert_id: self._training_counts(expert) for expert in self.experts}
        merged = {**self._removed_training_counts, **current}
        return tuple(merged[expert_id] for expert_id in sorted(merged))

    @staticmethod
    def _training_counts(
        expert: DynamicWeightedMajorityExpert,
    ) -> DynamicWeightedMajorityExpertTrainingCounts:
        return DynamicWeightedMajorityExpertTrainingCounts(
            expert_id=expert.expert_id,
            trained_sample_count=expert.trained_sample_count,
            parameter_update_step_count=expert.parameter_update_step_count,
            prediction_forward_call_count=expert.prediction_forward_call_count,
        )

    def _predict_class_id(self, classifier: ModelArchitectureClassifier, x: Tensor) -> int:
        with no_grad():
            scores = classifier(x)
        return (
            int((scores > 0.5).item())
            if self.class_count == 2
            else int(scores.argmax(dim=1).item())
        )

    def process_observed_sample(
        self, *, observed_sample: ObservedSample, sample_index: int, evaluation_concept_id: int
    ) -> None:
        """検査→評価の予測（学習の前の重み）→周期の更新→全専門家の保持と学習。"""
        if type(observed_sample) is not ObservedSample:
            raise TypeError("exact ObservedSampleが必要です。")
        observed_sample.__post_init__()
        if type(sample_index) is not int or sample_index != len(self.sample_predictions):
            raise ValueError("連続するbuiltin標本位置が必要です。")
        if type(evaluation_concept_id) is not int or evaluation_concept_id < 0:
            raise ValueError("診断概念IDには非負builtin intが必要です。")
        x = tensor([observed_sample.feature_values], dtype=float32)
        y = tensor([[float(observed_sample.class_label)]], dtype=float32)
        for expert in self.experts:
            validate_classifier_bounded_loss_inputs(
                classifier=expert.classifier, input_features=x, observed_class_labels=y
            )
            validate_independent_model_parameter_update_inputs(
                classifier=expert.classifier,
                parameter_optimizer=expert.parameter_optimizer_state.parameter_optimizer,
                input_features=x,
                observed_class_labels=y,
            )
        # (1) 評価の予測。真のコンセプトは使わない。
        predictions = []
        for expert in self.experts:
            predictions.append(self._predict_class_id(expert.classifier, x))
            expert.prediction_forward_call_count += 1
        weights = tuple(expert.weight for expert in self.experts)
        predicted = vote_weighted_majority(
            expert_predicted_class_ids=tuple(predictions),
            expert_weights=weights,
            class_count=self.class_count,
        )
        heaviest = max(self.experts, key=lambda e: (e.weight, -e.expert_id))
        self.sample_predictions.append(
            DynamicWeightedMajoritySamplePrediction(
                sample_index=sample_index,
                predicted_class_id=predicted,
                observed_class_id=observed_sample.class_label,
                observed_concept_id=evaluation_concept_id,
                prediction_is_correct=predicted == observed_sample.class_label,
                prediction_model_id=heaviest.expert_id,
                held_model_count_at_prediction=len(self.experts),
            )
        )
        # (2) 周期の標本での重みの更新・削除・追加（図1の8〜9行目、12〜19行目）。
        if (sample_index + 1) % self.period_sample_count == 0:
            result = update_weights_at_period(
                expert_predicted_class_ids=tuple(predictions),
                expert_weights=weights,
                observed_class_id=observed_sample.class_label,
                class_count=self.class_count,
                weight_decrease_factor=self.weight_decrease_factor,
                removal_weight_threshold=self.removal_weight_threshold,
            )
            kept = set(result.kept_expert_positions)
            for position, expert in enumerate(self.experts):
                if position not in kept:
                    self._removed_training_counts[expert.expert_id] = self._training_counts(expert)
                    self._lifetimes[expert.expert_id] = replace(
                        self._lifetimes[expert.expert_id], removed_sample_index=sample_index
                    )
            self.experts = [self.experts[position] for position in result.kept_expert_positions]
            for expert, weight in zip(self.experts, result.updated_weights, strict=True):
                expert.weight = weight
            if result.new_expert_is_added:
                self._add_expert(created_sample_index=sample_index)
        # (3) 全専門家（足したばかりの専門家を含む）の保持と学習。作られた順。
        for expert in self.experts:
            expert.training_samples.append((x, y))
            if len(expert.training_samples) < self._batch:
                continue
            for _ in range(self._updates):
                batch = self.python_random_generator.sample(expert.training_samples, self._batch)
                update_independent_model_parameters(
                    classifier=expert.classifier,
                    parameter_optimizer=expert.parameter_optimizer_state.parameter_optimizer,
                    input_features=cat([bx for bx, _ in batch]),
                    observed_class_labels=cat([by for _, by in batch]),
                )
                expert.trained_sample_count += self._batch
                expert.parameter_update_step_count += 1

    def flush_pending_local_updates(self, *, round_index: int) -> None:
        """学習は標本ごとに済ませるので、何もしない（実行の枠の契約）。"""
        if type(round_index) is not int or round_index < 0:
            raise ValueError("round_indexは非負builtin intが必要です。")

    def has_model_ready_for_server_registration(self) -> bool:
        return False

    def advance_new_model_upload_wait_after_synchronization(self, *, round_index: int) -> None:
        pass
