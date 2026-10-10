"""client 1つが、サーバから渡されたモデル（候補）を、対象のモデルの手元の標本で評価する。"""

from dataclasses import dataclass
from random import Random

from torch import Tensor, cat, float32, no_grad, strided, unique

from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    predict_class_labels_from_prediction_scores,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)

# 評価に使う標本の件数の条件（旧の固定値）。
# 対象のモデルの評価標本を使うのに要る件数。
_MINIMUM_STORED_EVALUATION_SAMPLE_COUNT = 6
# 評価標本が足りないとき、現在の学習帰属のモデルの学習データを使うのに要る件数。
_MINIMUM_CURRENT_MODEL_TRAINING_SAMPLE_COUNT = 11
# 評価を行うのに要る標本の件数。
_MINIMUM_EVALUATED_SAMPLE_COUNT = 5


@dataclass(frozen=True, kw_only=True)
class ModelPairCorrectnessCounts:
    """同じ標本での、候補のモデルと対象のモデルの正誤の数。4つの数の和は、評価した標本数。"""

    evaluated_sample_count: int
    candidate_only_correct_count: int
    target_only_correct_count: int
    both_correct_count: int
    both_wrong_count: int


@dataclass(frozen=True, kw_only=True)
class ClientModelCrossEvaluation:
    """clientの評価1回の結果。標本が足りないときは、件数0・和0で、正誤の数を持たない。"""

    evaluated_sample_count: int
    bounded_loss_sum: float
    squared_bounded_loss_sum: float
    # 正誤を比べなかったらNone。
    correctness_counts: ModelPairCorrectnessCounts | None
    # 標本に現れた観測クラスごとの正誤の数（クラスの昇順）。
    class_correctness_counts: tuple[tuple[int, ModelPairCorrectnessCounts], ...]


_NOT_EVALUATED = ClientModelCrossEvaluation(
    evaluated_sample_count=0,
    bounded_loss_sum=0.0,
    squared_bounded_loss_sum=0.0,
    correctness_counts=None,
    class_correctness_counts=(),
)


def _validate_candidate_parameter_snapshot(
    *,
    candidate_parameter_snapshot: dict[str, Tensor],
    reference_parameter_snapshot: dict[str, Tensor],
) -> None:
    """渡されたパラメータが、保有している分類器と同じ名前・同じ形の、CPUのfloat32のtensorであること。"""
    if type(candidate_parameter_snapshot) is not dict:
        raise TypeError("candidate_parameter_snapshot must be builtin dict")
    if any(type(parameter_name) is not str for parameter_name in candidate_parameter_snapshot):
        raise TypeError("candidate_parameter_snapshot keys must be builtin str")
    if candidate_parameter_snapshot.keys() != reference_parameter_snapshot.keys():
        raise ValueError("candidate_parameter_snapshot names must match the held classifier")
    for parameter_name, reference_parameter_values in reference_parameter_snapshot.items():
        parameter_values = candidate_parameter_snapshot[parameter_name]
        if type(parameter_values) is not Tensor:
            raise TypeError(
                f"candidate_parameter_snapshot.{parameter_name} must be exact torch.Tensor"
            )
        if (
            parameter_values.device.type != "cpu"
            or parameter_values.dtype != float32
            or parameter_values.layout != strided
            or parameter_values.is_nested
        ):
            raise ValueError(
                f"candidate_parameter_snapshot.{parameter_name} must be CPU float32 strided"
            )
        if parameter_values.shape != reference_parameter_values.shape:
            raise ValueError(
                f"candidate_parameter_snapshot.{parameter_name} must have the shape of the held classifier"
            )


def _count_model_pair_correctness(
    *, candidate_is_correct: Tensor, target_is_correct: Tensor
) -> ModelPairCorrectnessCounts:
    evaluated_sample_count = int(candidate_is_correct.numel())
    candidate_only_correct_count = int((candidate_is_correct & ~target_is_correct).sum().item())
    target_only_correct_count = int((~candidate_is_correct & target_is_correct).sum().item())
    both_correct_count = int((candidate_is_correct & target_is_correct).sum().item())
    return ModelPairCorrectnessCounts(
        evaluated_sample_count=evaluated_sample_count,
        candidate_only_correct_count=candidate_only_correct_count,
        target_only_correct_count=target_only_correct_count,
        both_correct_count=both_correct_count,
        both_wrong_count=(
            evaluated_sample_count
            - candidate_only_correct_count
            - target_only_correct_count
            - both_correct_count
        ),
    )


@no_grad()
def evaluate_candidate_model_on_target_model_samples(
    *,
    candidate_parameter_snapshot: dict[str, Tensor],
    target_model_id: int,
    compare_correctness_with_held_target_model: bool,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    model_evaluation_sample_store: ModelEvaluationSampleStore,
    training_sample_store: ModelTrainingSampleStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    maximum_evaluation_sample_count: int,
    python_random_generator: Random,
) -> ClientModelCrossEvaluation:
    """対象のモデルの標本を選び、渡されたモデルの有界損失の件数・和・2乗和を返す。

    求められたら、同じ標本で、渡されたモデルと、保有する対象のモデルの正誤も比べる。
    標本は、対象のモデルの評価標本（足りなければ、対象が現在の学習帰属のときだけ、その学習データ）。
    上限を超えたら、借りた乱数生成器で抜き出す。どのownerも変えない。
    """
    for owner_name, owner, required_type in (
        (
            "held_model_training_state_registry",
            held_model_training_state_registry,
            HeldModelTrainingStateRegistry,
        ),
        (
            "model_evaluation_sample_store",
            model_evaluation_sample_store,
            ModelEvaluationSampleStore,
        ),
        ("training_sample_store", training_sample_store, ModelTrainingSampleStore),
        (
            "current_training_model_assignment",
            current_training_model_assignment,
            CurrentTrainingModelAssignment,
        ),
        ("python_random_generator", python_random_generator, Random),
    ):
        if type(owner) is not required_type:
            raise TypeError(f"{owner_name} must be exact {required_type.__name__}")
    if type(target_model_id) is not int:
        raise TypeError("target_model_id must be builtin int")
    if type(compare_correctness_with_held_target_model) is not bool:
        raise TypeError("compare_correctness_with_held_target_model must be builtin bool")
    if type(maximum_evaluation_sample_count) is not int:
        raise TypeError("maximum_evaluation_sample_count must be builtin int")
    if maximum_evaluation_sample_count < 1:
        raise ValueError("maximum_evaluation_sample_count must be positive")
    current_training_model_id = current_training_model_assignment.current_training_model_id
    architecture_reference_classifier = (
        held_model_training_state_registry.get_held_model_training_state(
            model_id=current_training_model_id
        ).classifier
    )
    _validate_candidate_parameter_snapshot(
        candidate_parameter_snapshot=candidate_parameter_snapshot,
        reference_parameter_snapshot=snapshot_classifier_parameters(
            classifier=architecture_reference_classifier
        ),
    )

    # 標本の選択（旧と同じ順: 評価標本→現在の学習帰属の学習データ→なし。上限を超えたら抜き出す）。
    stored_evaluation_samples = next(
        (
            evaluation_sample_collection.evaluation_samples
            for evaluation_sample_collection in model_evaluation_sample_store.snapshot_ordered_model_evaluation_samples()
            if evaluation_sample_collection.model_id == target_model_id
        ),
        (),
    )
    if len(stored_evaluation_samples) >= _MINIMUM_STORED_EVALUATION_SAMPLE_COUNT:
        selected_samples = stored_evaluation_samples
    else:
        current_model_training_samples = (
            next(
                (
                    training_sample_collection.training_samples
                    for training_sample_collection in training_sample_store.snapshot_ordered_model_training_samples()
                    if training_sample_collection.model_id == target_model_id
                ),
                (),
            )
            if target_model_id == current_training_model_id
            else ()
        )
        selected_samples = (
            current_model_training_samples
            if len(current_model_training_samples) >= _MINIMUM_CURRENT_MODEL_TRAINING_SAMPLE_COUNT
            else ()
        )
    if len(selected_samples) > maximum_evaluation_sample_count:
        selected_samples = tuple(
            python_random_generator.sample(selected_samples, maximum_evaluation_sample_count)
        )
    held_target_model_training_state = next(
        (
            held_model_training_state
            for held_model_training_state in held_model_training_state_registry.snapshot_ordered_held_model_training_states()
            if held_model_training_state.model_id == target_model_id
        ),
        None,
    )
    if len(selected_samples) < _MINIMUM_EVALUATED_SAMPLE_COUNT or (
        compare_correctness_with_held_target_model and held_target_model_training_state is None
    ):
        return _NOT_EVALUATED

    input_features = cat([selected_sample.input_features for selected_sample in selected_samples])
    observed_class_labels = cat(
        [selected_sample.observed_class_labels for selected_sample in selected_samples]
    )
    # 評価用の分類器を新しく作る（torchの乱数を消費する）。値は、渡されたパラメータで上書きする。
    candidate_classifier = ResidualAdapterClassifier(
        model_architecture_settings=architecture_reference_classifier.model_architecture_settings,
        input_feature_count=architecture_reference_classifier.feature_extractor.input_feature_count,
        hidden_layer_widths=architecture_reference_classifier.feature_extractor.hidden_layer_widths,
        class_count=architecture_reference_classifier.class_count,
    )
    candidate_classifier.load_state_dict(candidate_parameter_snapshot, strict=True)
    # 和と2乗和は、float32の配列のまま求める（旧と同じ演算。Pythonのfloatへは、和を取ってから直す）。
    per_sample_losses = evaluate_classifier_per_sample_bounded_losses(
        classifier=candidate_classifier,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    ).numpy()
    evaluated_sample_count = len(per_sample_losses)
    bounded_loss_sum = float(per_sample_losses.sum())
    squared_bounded_loss_sum = float((per_sample_losses**2).sum())
    if not compare_correctness_with_held_target_model or held_target_model_training_state is None:
        return ClientModelCrossEvaluation(
            evaluated_sample_count=evaluated_sample_count,
            bounded_loss_sum=bounded_loss_sum,
            squared_bounded_loss_sum=squared_bounded_loss_sum,
            correctness_counts=None,
            class_correctness_counts=(),
        )

    class_count = architecture_reference_classifier.class_count
    flat_observed_class_labels = observed_class_labels.reshape(-1)
    candidate_is_correct = (
        predict_class_labels_from_prediction_scores(
            prediction_scores=candidate_classifier(input_features), class_count=class_count
        ).reshape(-1)
        == flat_observed_class_labels
    )
    target_is_correct = (
        predict_class_labels_from_prediction_scores(
            prediction_scores=held_target_model_training_state.classifier(input_features),
            class_count=class_count,
        ).reshape(-1)
        == flat_observed_class_labels
    )
    integer_class_labels = flat_observed_class_labels.long()
    class_correctness_counts = []
    for observed_class_id in unique(integer_class_labels).tolist():
        class_sample_mask = integer_class_labels == int(observed_class_id)
        class_correctness_counts.append(
            (
                int(observed_class_id),
                _count_model_pair_correctness(
                    candidate_is_correct=candidate_is_correct[class_sample_mask],
                    target_is_correct=target_is_correct[class_sample_mask],
                ),
            )
        )
    return ClientModelCrossEvaluation(
        evaluated_sample_count=evaluated_sample_count,
        bounded_loss_sum=bounded_loss_sum,
        squared_bounded_loss_sum=squared_bounded_loss_sum,
        correctness_counts=_count_model_pair_correctness(
            candidate_is_correct=candidate_is_correct, target_is_correct=target_is_correct
        ),
        class_correctness_counts=tuple(class_correctness_counts),
    )
