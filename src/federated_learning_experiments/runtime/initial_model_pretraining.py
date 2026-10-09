"""runの最初に、初期モデルを概念0の標本で学習し、その標本での損失統計を求める。"""

from dataclasses import dataclass
from random import Random

from numpy.random import RandomState
from torch import float32, tensor

from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.participating_model_training_batch import (
    ParticipatingModelTrainingBatch,
)

# 観測標本（ObservedSample）が持つ特徴の数と、事前学習に使う概念。
_OBSERVED_SAMPLE_FEATURE_COUNT = 2
_PRETRAINING_CONCEPT_ID = 0
# 統計を求めるための一時的なownerの中で、初期モデルを指すID（結果には残らない）。
_STATISTICS_MODEL_ID = 0


@dataclass(frozen=True, kw_only=True)
class PretrainedInitialModel:
    """事前学習の結果。clientの組立て（assemble_fedsda_run_client）の、初期モデルと統計の引数に対応する。"""

    classifier: ResidualAdapterClassifier
    concept_specific_parameter_optimizer_state: ParameterOptimizerState
    shared_parameter_optimizer_state: ParameterOptimizerState
    loss_statistics: ModelAndClassLossStatistics


def _validate_initial_model_pretraining_inputs(
    *,
    initial_model_pretraining_settings: InitialModelPretrainingSettings,
    model_architecture_settings: ModelArchitectureSettings,
    hidden_layer_widths: tuple[int, ...],
    class_count: int,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    sample_generator: SineSampleGenerator,
    python_random_generator: Random,
) -> None:
    if type(initial_model_pretraining_settings) is not InitialModelPretrainingSettings:
        raise TypeError(
            "initial_model_pretraining_settings must be exact InitialModelPretrainingSettings"
        )
    if type(model_architecture_settings) is not ModelArchitectureSettings:
        raise TypeError("model_architecture_settings must be exact ModelArchitectureSettings")
    if type(parameter_optimizer_settings) not in (
        AdamParameterOptimizerSettings,
        SgdParameterOptimizerSettings,
    ):
        raise TypeError(
            "parameter_optimizer_settings must be exact AdamParameterOptimizerSettings "
            "or SgdParameterOptimizerSettings"
        )
    # frozenを回避して組み立てた設定も拒否できるよう、各設定の検査をもう一度行う。
    initial_model_pretraining_settings.__post_init__()
    model_architecture_settings.__post_init__()
    parameter_optimizer_settings.__post_init__()
    if type(sample_generator) is not SineSampleGenerator:
        raise TypeError("sample_generator must be exact SineSampleGenerator")
    if type(sample_generator.numpy_random_generator) is not RandomState:
        raise TypeError("sample_generator must hold exact numpy.random.RandomState")
    if type(python_random_generator) is not Random:
        raise TypeError("python_random_generator must be exact random.Random")
    # 分類器の生成が拒否する寸法とクラス数を、乱数を使わない検査で先に確かめる。
    SharedFeatureExtractor.validate_feature_dimensions(
        input_feature_count=_OBSERVED_SAMPLE_FEATURE_COUNT, hidden_layer_widths=hidden_layer_widths
    )
    if type(class_count) is not int:
        raise TypeError("class_count must be builtin int")
    if class_count < 2:
        raise ValueError("class_count must be at least 2")


def pretrain_initial_model(
    *,
    initial_model_pretraining_settings: InitialModelPretrainingSettings,
    model_architecture_settings: ModelArchitectureSettings,
    hidden_layer_widths: tuple[int, ...],
    class_count: int,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    sample_generator: SineSampleGenerator,
    python_random_generator: Random,
) -> PretrainedInitialModel:
    """分類器を作り、概念0の標本で学習して、最後の並びの順に求めた損失統計とともに返す。

    旧の事前学習と同じ順に乱数を消費する: 分類器の初期化（torchの全体の乱数）→標本の生成（標本生成器の
    乱数）→epochごとのshuffle（借りた乱数生成器）。
    """
    _validate_initial_model_pretraining_inputs(
        initial_model_pretraining_settings=initial_model_pretraining_settings,
        model_architecture_settings=model_architecture_settings,
        hidden_layer_widths=hidden_layer_widths,
        class_count=class_count,
        parameter_optimizer_settings=parameter_optimizer_settings,
        sample_generator=sample_generator,
        python_random_generator=python_random_generator,
    )
    # (1) 分類器と、概念固有部（アダプタ→分類層）・共有部のoptimizerの状態。
    classifier = ResidualAdapterClassifier(
        model_architecture_settings=model_architecture_settings,
        input_feature_count=_OBSERVED_SAMPLE_FEATURE_COUNT,
        hidden_layer_widths=hidden_layer_widths,
        class_count=class_count,
    )
    concept_specific_parameter_optimizer_state = ParameterOptimizerState(
        parameters=tuple(classifier.residual_adapter.parameters())
        + tuple(classifier.classification_layer.parameters()),
        optimizer_settings=parameter_optimizer_settings,
    )
    shared_parameter_optimizer_state = ParameterOptimizerState(
        parameters=tuple(classifier.feature_extractor.parameters()),
        optimizer_settings=parameter_optimizer_settings,
    )
    # (2) 概念0の標本。
    pretraining_samples: list[ObservedSample] = [
        sample_generator.generate_sample(concept_id=_PRETRAINING_CONCEPT_ID)
        for _ in range(initial_model_pretraining_settings.pretraining_sample_count)
    ]
    # (3) epochごとに並びをshuffleし、batchごとに、共有部と概念固有部を1回更新する。
    local_training_settings = LocalTrainingSettings(
        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
    )
    batch_sample_count = initial_model_pretraining_settings.pretraining_batch_sample_count
    for _ in range(initial_model_pretraining_settings.pretraining_epoch_count):
        python_random_generator.shuffle(pretraining_samples)
        for batch_start in range(0, len(pretraining_samples), batch_sample_count):
            batch_samples = pretraining_samples[batch_start : batch_start + batch_sample_count]
            perform_joint_model_parameter_update(
                local_training_settings=local_training_settings,
                shared_feature_extractor=classifier.feature_extractor,
                shared_parameter_optimizer=shared_parameter_optimizer_state.parameter_optimizer,
                participating_training_batches=(
                    ParticipatingModelTrainingBatch(
                        classifier=classifier,
                        concept_specific_parameter_optimizer=concept_specific_parameter_optimizer_state.parameter_optimizer,
                        input_features=tensor(
                            [list(batch_sample.feature_values) for batch_sample in batch_samples],
                            dtype=float32,
                        ),
                        observed_class_labels=tensor(
                            [[float(batch_sample.class_label)] for batch_sample in batch_samples],
                            dtype=float32,
                        ),
                    ),
                ),
                update_shared_features=True,
            )
    # (4) 最後の並びの順に、1件ずつ評価して、全体と観測クラス別の統計を逐次に更新する。
    loss_statistics_store = ModelAndClassLossStatisticsStore()
    for pretraining_sample in pretraining_samples:
        loss_statistics_store.record_assigned_loss(
            model_id=_STATISTICS_MODEL_ID,
            observed_loss=evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier,
                input_features=tensor([list(pretraining_sample.feature_values)], dtype=float32),
                observed_class_labels=tensor(
                    [[float(pretraining_sample.class_label)]], dtype=float32
                ),
            )[0].item(),
            observed_class_id=pretraining_sample.class_label,
        )
    loss_statistics = loss_statistics_store.get_model_loss_statistics(model_id=_STATISTICS_MODEL_ID)
    if loss_statistics is None:
        raise RuntimeError("pretraining loss statistics were not recorded")
    return PretrainedInitialModel(
        classifier=classifier,
        concept_specific_parameter_optimizer_state=concept_specific_parameter_optimizer_state,
        shared_parameter_optimizer_state=shared_parameter_optimizer_state,
        loss_statistics=loss_statistics,
    )
