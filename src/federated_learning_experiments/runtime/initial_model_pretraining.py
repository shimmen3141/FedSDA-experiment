"""runの最初に、初期モデルを概念0の標本で学習し、その標本での損失統計を求める。

統計の推定は2つ: 学習に使った標本で学習の後に測る再代入の推定（`pretrain_initial_model`が返す統計）と、
事前学習の標本のk分割交差検証で、各標本を学習に使っていないモデルで測る推定
（`estimate_cross_validated_loss_statistics`）。再代入の推定は、学習に使った標本なので損失が低く出る。
"""

from dataclasses import dataclass, replace
from random import Random

from numpy.random import RandomState
from torch import float32, is_grad_enabled, tensor

from federated_learning_experiments.data.observed_sample_generation import (
    OBSERVED_SAMPLE_GENERATOR_TYPES,
    ObservedSampleGenerator,
)
from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)
from federated_learning_experiments.learning.models.torch_random_state_scope import (
    isolated_cpu_torch_random_state,
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
_PRETRAINING_CONCEPT_ID = 0
# 統計を求めるための一時的なownerの中で、初期モデルを指すID（結果には残らない）。
_STATISTICS_MODEL_ID = 0

# 初期モデルの損失の統計の推定の仕方（FedSDAの参加者の設定`initial_model_loss_statistics_estimation`の値）。
# 学習に使った事前学習の標本で、学習の後に測る（基準版10までの扱い）。
RESUBSTITUTION_ESTIMATION_NAME = "resubstitution"
# 事前学習の標本の5分割交差検証で、各標本を学習に使っていないモデルで測る（基準版11の既定）。
FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME = "five_fold_cross_validation"
INITIAL_MODEL_LOSS_STATISTICS_ESTIMATION_NAMES: tuple[str, ...] = (
    RESUBSTITUTION_ESTIMATION_NAME,
    FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME,
)
# `five_fold_cross_validation`の分割の数。
INITIAL_MODEL_LOSS_STATISTICS_CROSS_VALIDATION_FOLD_COUNT = 5


@dataclass(frozen=True, kw_only=True)
class PretrainedInitialModel:
    """事前学習の結果。clientの組立て（assemble_fedsda_run_client）の、初期モデルと統計の引数に対応する。"""

    classifier: ModelArchitectureClassifier
    concept_specific_parameter_optimizer_state: ParameterOptimizerState
    shared_parameter_optimizer_state: ParameterOptimizerState
    loss_statistics: ModelAndClassLossStatistics
    # 事前学習の標本（生成の順。学習の並べ替えの前）。交差検証の推定に使う。
    pretraining_samples: tuple[ObservedSample, ...]


def _validate_initial_model_pretraining_inputs(
    *,
    initial_model_pretraining_settings: InitialModelPretrainingSettings,
    model_architecture_settings: ModelArchitectureSettings,
    input_feature_count: int,
    hidden_layer_widths: tuple[int, ...],
    class_count: int,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    sample_generator: ObservedSampleGenerator,
    python_random_generator: Random,
) -> None:
    _validate_training_settings(
        initial_model_pretraining_settings=initial_model_pretraining_settings,
        model_architecture_settings=model_architecture_settings,
        parameter_optimizer_settings=parameter_optimizer_settings,
    )
    if type(sample_generator) not in OBSERVED_SAMPLE_GENERATOR_TYPES:
        raise TypeError("sample_generator must be a generator of a supported dataset")
    if type(getattr(sample_generator, "numpy_random_generator", None)) is not RandomState:
        raise TypeError("sample_generator must hold exact numpy.random.RandomState")
    if type(python_random_generator) is not Random:
        raise TypeError("python_random_generator must be exact random.Random")
    _validate_classifier_shape_and_gradient(
        initial_model_pretraining_settings=initial_model_pretraining_settings,
        input_feature_count=input_feature_count,
        hidden_layer_widths=hidden_layer_widths,
        class_count=class_count,
    )


def _validate_training_settings(
    *,
    initial_model_pretraining_settings: InitialModelPretrainingSettings,
    model_architecture_settings: ModelArchitectureSettings,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
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


def _validate_classifier_shape_and_gradient(
    *,
    initial_model_pretraining_settings: InitialModelPretrainingSettings,
    input_feature_count: int,
    hidden_layer_widths: tuple[int, ...],
    class_count: int,
) -> None:
    # 分類器の生成が拒否する寸法とクラス数を、乱数を使わない検査で先に確かめる。
    SharedFeatureExtractor.validate_feature_dimensions(
        input_feature_count=input_feature_count, hidden_layer_widths=hidden_layer_widths
    )
    # 隠れ層がないと共有部にパラメータがなく、共有部のoptimizerを作れない（分類器の生成の後に拒否される）。
    if not hidden_layer_widths:
        raise ValueError("hidden_layer_widths must hold at least one hidden layer")
    if type(class_count) is not int:
        raise TypeError("class_count must be builtin int")
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    # 更新は勾配の計算を要る。無効のままだと、分類器と標本の生成の後で、共同更新が拒否する。
    if initial_model_pretraining_settings.pretraining_epoch_count > 0 and not is_grad_enabled():
        raise ValueError(
            "pretraining with one or more epochs requires gradient calculation enabled"
        )


def _create_classifier_and_optimizer_states(
    *,
    model_architecture_settings: ModelArchitectureSettings,
    input_feature_count: int,
    hidden_layer_widths: tuple[int, ...],
    class_count: int,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
) -> tuple[ModelArchitectureClassifier, ParameterOptimizerState, ParameterOptimizerState]:
    """分類器（torchの全体の乱数で初期化）と、概念固有部（アダプタ→分類層）・共有部のoptimizerの状態。"""
    classifier = ModelArchitectureClassifier(
        model_architecture_settings=model_architecture_settings,
        input_feature_count=input_feature_count,
        hidden_layer_widths=hidden_layer_widths,
        class_count=class_count,
    )
    concept_specific_parameter_optimizer_state = ParameterOptimizerState(
        parameters=tuple(classifier.model_owned_feature_transform.parameters())
        + tuple(classifier.classification_layer.parameters()),
        optimizer_settings=parameter_optimizer_settings,
    )
    shared_parameter_optimizer_state = ParameterOptimizerState(
        parameters=tuple(classifier.feature_extractor.parameters()),
        optimizer_settings=parameter_optimizer_settings,
    )
    return classifier, concept_specific_parameter_optimizer_state, shared_parameter_optimizer_state


def _train_classifier_on_pretraining_samples(
    *,
    classifier: ModelArchitectureClassifier,
    concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    shared_parameter_optimizer_state: ParameterOptimizerState,
    pretraining_samples: list[ObservedSample],
    initial_model_pretraining_settings: InitialModelPretrainingSettings,
    python_random_generator: Random,
) -> None:
    """epochごとに`pretraining_samples`の並びをshuffleし（その場で並べ替える）、batchごとに、共有部と
    概念固有部を1回更新する。"""
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
                shared_parameter_optimizer=shared_parameter_optimizer_state.parameter_optimizer_if_any,
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


def _record_bounded_loss(
    *,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    classifier: ModelArchitectureClassifier,
    sample: ObservedSample,
) -> None:
    loss_statistics_store.record_assigned_loss(
        model_id=_STATISTICS_MODEL_ID,
        observed_loss=evaluate_classifier_per_sample_bounded_losses(
            classifier=classifier,
            input_features=tensor([list(sample.feature_values)], dtype=float32),
            observed_class_labels=tensor([[float(sample.class_label)]], dtype=float32),
        )[0].item(),
        observed_class_id=sample.class_label,
    )


def pretrain_initial_model(
    *,
    initial_model_pretraining_settings: InitialModelPretrainingSettings,
    model_architecture_settings: ModelArchitectureSettings,
    input_feature_count: int,
    hidden_layer_widths: tuple[int, ...],
    class_count: int,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    sample_generator: ObservedSampleGenerator,
    python_random_generator: Random,
) -> PretrainedInitialModel:
    """分類器を作り、概念0の標本で学習して、最後の並びの順に求めた損失統計（再代入の推定）とともに返す。

    旧の事前学習と同じ順に乱数を消費する: 分類器の初期化（torchの全体の乱数）→標本の生成（標本生成器の
    乱数）→epochごとのshuffle（借りた乱数生成器）。標本は、生成の順でも返す。
    """
    _validate_initial_model_pretraining_inputs(
        initial_model_pretraining_settings=initial_model_pretraining_settings,
        model_architecture_settings=model_architecture_settings,
        input_feature_count=input_feature_count,
        hidden_layer_widths=hidden_layer_widths,
        class_count=class_count,
        parameter_optimizer_settings=parameter_optimizer_settings,
        sample_generator=sample_generator,
        python_random_generator=python_random_generator,
    )
    # (1) 分類器と、概念固有部（アダプタ→分類層）・共有部のoptimizerの状態。
    (
        classifier,
        concept_specific_parameter_optimizer_state,
        shared_parameter_optimizer_state,
    ) = _create_classifier_and_optimizer_states(
        model_architecture_settings=model_architecture_settings,
        input_feature_count=input_feature_count,
        hidden_layer_widths=hidden_layer_widths,
        class_count=class_count,
        parameter_optimizer_settings=parameter_optimizer_settings,
    )
    # (2) 概念0の標本。
    pretraining_samples: list[ObservedSample] = [
        sample_generator.generate_sample(concept_id=_PRETRAINING_CONCEPT_ID)
        for _ in range(initial_model_pretraining_settings.pretraining_sample_count)
    ]
    samples_in_generation_order = tuple(pretraining_samples)
    # (3) epochごとに並びをshuffleし、batchごとに、共有部と概念固有部を1回更新する。
    _train_classifier_on_pretraining_samples(
        classifier=classifier,
        concept_specific_parameter_optimizer_state=concept_specific_parameter_optimizer_state,
        shared_parameter_optimizer_state=shared_parameter_optimizer_state,
        pretraining_samples=pretraining_samples,
        initial_model_pretraining_settings=initial_model_pretraining_settings,
        python_random_generator=python_random_generator,
    )
    # (4) 最後の並びの順に、1件ずつ評価して、全体と観測クラス別の統計を逐次に更新する。
    loss_statistics_store = ModelAndClassLossStatisticsStore()
    for pretraining_sample in pretraining_samples:
        _record_bounded_loss(
            loss_statistics_store=loss_statistics_store,
            classifier=classifier,
            sample=pretraining_sample,
        )
    loss_statistics = loss_statistics_store.get_model_loss_statistics(model_id=_STATISTICS_MODEL_ID)
    if loss_statistics is None:
        raise RuntimeError("pretraining loss statistics were not recorded")
    return PretrainedInitialModel(
        classifier=classifier,
        concept_specific_parameter_optimizer_state=concept_specific_parameter_optimizer_state,
        shared_parameter_optimizer_state=shared_parameter_optimizer_state,
        loss_statistics=loss_statistics,
        pretraining_samples=samples_in_generation_order,
    )


def estimate_cross_validated_loss_statistics(
    *,
    pretraining_samples: tuple[ObservedSample, ...],
    fold_count: int,
    initial_model_pretraining_settings: InitialModelPretrainingSettings,
    model_architecture_settings: ModelArchitectureSettings,
    input_feature_count: int,
    hidden_layer_widths: tuple[int, ...],
    class_count: int,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
) -> ModelAndClassLossStatistics:
    """事前学習の標本のk分割交差検証で、out-of-foldの有界の損失から、全体と観測クラス別の統計を作る。

    標本を生成の順のまま、位置をkで割った余りでk組に分ける（組iは、位置がi, i+k, …の標本）。組i（i＝0〜k−1の
    順）ごとに、組i以外の標本（生成の順）で、初期モデルと同じ事前学習の手順で一時的な分類器を学習し、組iの標本を
    位置の順に1件ずつ評価して統計に足す。組iの分類器の初期化と並べ替えは、runの乱数から切り離した乱数（torchは
    CPUの乱数をseed iで、Pythonは`Random(i)`）で行い、runの乱数を消費しない。一時的な分類器は残さない。
    """
    _validate_training_settings(
        initial_model_pretraining_settings=initial_model_pretraining_settings,
        model_architecture_settings=model_architecture_settings,
        parameter_optimizer_settings=parameter_optimizer_settings,
    )
    _validate_classifier_shape_and_gradient(
        initial_model_pretraining_settings=initial_model_pretraining_settings,
        input_feature_count=input_feature_count,
        hidden_layer_widths=hidden_layer_widths,
        class_count=class_count,
    )
    if type(pretraining_samples) is not tuple:
        raise TypeError("pretraining_samples must be builtin tuple")
    if any(type(sample) is not ObservedSample for sample in pretraining_samples):
        raise TypeError("pretraining_samples must hold exact ObservedSample")
    if type(fold_count) is not int:
        raise TypeError("fold_count must be builtin int")
    if not 2 <= fold_count <= len(pretraining_samples):
        raise ValueError("fold_count must be at least 2 and at most the pretraining sample count")
    loss_statistics_store = ModelAndClassLossStatisticsStore()
    for fold_index in range(fold_count):
        training_samples = [
            sample
            for position, sample in enumerate(pretraining_samples)
            if position % fold_count != fold_index
        ]
        with isolated_cpu_torch_random_state(random_seed=fold_index):
            (
                classifier,
                concept_specific_parameter_optimizer_state,
                shared_parameter_optimizer_state,
            ) = _create_classifier_and_optimizer_states(
                model_architecture_settings=model_architecture_settings,
                input_feature_count=input_feature_count,
                hidden_layer_widths=hidden_layer_widths,
                class_count=class_count,
                parameter_optimizer_settings=parameter_optimizer_settings,
            )
            _train_classifier_on_pretraining_samples(
                classifier=classifier,
                concept_specific_parameter_optimizer_state=concept_specific_parameter_optimizer_state,
                shared_parameter_optimizer_state=shared_parameter_optimizer_state,
                pretraining_samples=training_samples,
                initial_model_pretraining_settings=replace(
                    initial_model_pretraining_settings,
                    pretraining_sample_count=len(training_samples),
                ),
                python_random_generator=Random(fold_index),
            )
            for held_out_sample in pretraining_samples[fold_index::fold_count]:
                _record_bounded_loss(
                    loss_statistics_store=loss_statistics_store,
                    classifier=classifier,
                    sample=held_out_sample,
                )
    loss_statistics = loss_statistics_store.get_model_loss_statistics(model_id=_STATISTICS_MODEL_ID)
    if loss_statistics is None:
        raise RuntimeError("cross-validated loss statistics were not recorded")
    return loss_statistics
