"""FIFOから放す時点の局所モデルで責任度を求める接続。"""

from collections.abc import Callable

from torch import no_grad

from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    convert_model_outputs_to_prediction_probabilities,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.responsibility_weighted_training import (
    calculate_training_responsibilities,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
    FixedShareWeightController,
)


def build_released_sample_responsibility_provider(
    *,
    registry: HeldModelTrainingStateRegistry,
    weight_controller: FixedShareWeightController,
    own_assignment_sample_store: ModelTrainingSampleStore | None = None,
) -> Callable[[int, ObservedTrainingSample], dict[int, float]]:
    """動くownerを借用し、解放時の集合（採用直後の候補も含む）を毎回読む。

    `own_assignment_sample_store`を与えたときは、重みと確率を全保有モデルで求めた後で、責任度の対象を、
    そのstoreで固い割り当ての標本を持つモデルと、いま割り当てる先（`fallback_model_id`）に絞る。
    """
    if own_assignment_sample_store is not None and type(own_assignment_sample_store) is not (
        ModelTrainingSampleStore
    ):
        raise TypeError("own_assignment_sample_storeはexact ModelTrainingSampleStoreが必要です。")

    def evaluate(fallback_model_id: int, sample: ObservedTrainingSample) -> dict[int, float]:
        states = registry.snapshot_ordered_held_model_training_states()
        classifiers = {state.model_id: state.classifier for state in states}
        # 集合が変わった場合は、次の予測と同じFixed-Shareの一様重みへの同期を行う。
        weights = weight_controller.get_weights_before_label_observation(model_ids=classifiers)
        reference = states[0].classifier
        with no_grad():
            shared = reference.extract_shared_features(sample.input_features)
            outputs = {
                m: classifier.forward_from_shared_features(shared)
                for m, classifier in classifiers.items()
            }
        probabilities = convert_model_outputs_to_prediction_probabilities(
            model_outputs_by_model_id=outputs, class_count=reference.class_count
        )
        label = int(sample.observed_class_labels.item())
        true_probabilities = {
            m: (float(p[0, 0]) if label == 1 else 1.0 - float(p[0, 0]))
            if reference.class_count == 2
            else float(p[0, label])
            for m, p in probabilities.items()
        }
        if own_assignment_sample_store is not None:
            own_model_ids = own_assignment_sample_store.snapshot_model_ids_with_assigned_samples()
            weights = {
                m: w for m, w in weights.items() if m in own_model_ids or m == fallback_model_id
            }
            true_probabilities = {
                m: p
                for m, p in true_probabilities.items()
                if m in own_model_ids or m == fallback_model_id
            }
        return calculate_training_responsibilities(
            prediction_weights=weights,
            true_class_probabilities=true_probabilities,
            fallback_model_id=fallback_model_id,
        )

    return evaluate
