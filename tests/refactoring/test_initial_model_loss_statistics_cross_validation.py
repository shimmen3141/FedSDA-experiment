"""初期モデルの損失の統計の、事前学習の標本でのk分割交差検証を検証する。

照合の対照は、組ごとに、組以外の標本だけを生成する標本生成器を与えて`pretrain_initial_model`（実旧の事前学習と
一致することを、test_initial_model_pretraining.pyが確かめている）を呼び、除いた組の標本を1件ずつ評価する形
（研究ツールの関数は使わない）。
"""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_initial_model_pretraining import (
    VALID_PRETRAINING_VALUES,
    assert_numpy_random_states_equal,
    make_pretraining_arguments,
)

from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.torch_random_state_scope import (
    isolated_cpu_torch_random_state,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.runtime.initial_model_pretraining import (
    FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME,
    INITIAL_MODEL_LOSS_STATISTICS_CROSS_VALIDATION_FOLD_COUNT,
    INITIAL_MODEL_LOSS_STATISTICS_ESTIMATION_NAMES,
    RESUBSTITUTION_ESTIMATION_NAME,
    estimate_cross_validated_loss_statistics,
    pretrain_initial_model,
)


def _pretrain(**arguments):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return pretrain_initial_model(**arguments)


def _cross_validation_arguments(arguments, pretrained, fold_count):
    return dict(
        pretraining_samples=pretrained.pretraining_samples,
        fold_count=fold_count,
        initial_model_pretraining_settings=arguments["initial_model_pretraining_settings"],
        model_architecture_settings=arguments["model_architecture_settings"],
        input_feature_count=arguments["input_feature_count"],
        hidden_layer_widths=arguments["hidden_layer_widths"],
        class_count=arguments["class_count"],
        parameter_optimizer_settings=arguments["parameter_optimizer_settings"],
    )


def _expected_out_of_fold_statistics(arguments, samples, fold_count):
    store = ModelAndClassLossStatisticsStore()
    for fold_index in range(fold_count):
        held_out = samples[fold_index::fold_count]
        training = [s for position, s in enumerate(samples) if position % fold_count != fold_index]
        replay = iter(training)
        generator = SineSampleGenerator(numpy_random_generator=np.random.RandomState(0))
        generator.generate_sample = lambda *, concept_id, replay=replay: next(replay)
        with isolated_cpu_torch_random_state(random_seed=fold_index):
            fold_model = pretrain_initial_model(
                **arguments
                | dict(
                    initial_model_pretraining_settings=replace(
                        arguments["initial_model_pretraining_settings"],
                        pretraining_sample_count=len(training),
                    ),
                    sample_generator=generator,
                    python_random_generator=random.Random(fold_index),
                )
            )
            for sample in held_out:
                store.record_assigned_loss(
                    model_id=0,
                    observed_loss=evaluate_classifier_per_sample_bounded_losses(
                        classifier=fold_model.classifier,
                        input_features=torch.tensor(
                            [list(sample.feature_values)], dtype=torch.float32
                        ),
                        observed_class_labels=torch.tensor(
                            [[float(sample.class_label)]], dtype=torch.float32
                        ),
                    )[0].item(),
                    observed_class_id=sample.class_label,
                )
    return store.get_model_loss_statistics(model_id=0)


def test_estimation_names_and_fold_count():
    assert INITIAL_MODEL_LOSS_STATISTICS_ESTIMATION_NAMES == (
        RESUBSTITUTION_ESTIMATION_NAME,
        FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME,
    )
    assert RESUBSTITUTION_ESTIMATION_NAME == "resubstitution"
    assert FIVE_FOLD_CROSS_VALIDATION_ESTIMATION_NAME == "five_fold_cross_validation"
    assert INITIAL_MODEL_LOSS_STATISTICS_CROSS_VALIDATION_FOLD_COUNT == 5


def test_pretraining_returns_the_samples_in_generation_order():
    arguments = make_pretraining_arguments()
    pretrained = _pretrain(**arguments)
    generator = SineSampleGenerator(numpy_random_generator=np.random.RandomState(7))
    expected = tuple(
        generator.generate_sample(concept_id=0)
        for _ in range(VALID_PRETRAINING_VALUES["pretraining_sample_count"])
    )
    assert type(pretrained.pretraining_samples) is tuple
    assert pretrained.pretraining_samples == expected


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("pretraining_sizes", ((24, 2, 8), (23, 2, 5), (13, 0, 4), (30, 3, 7)))
@pytest.mark.parametrize("optimizer_variant", ("amsgrad", "sgd"))
@pytest.mark.parametrize("fold_count", (5, 3))
def test_cross_validation_trains_each_fold_on_the_other_samples(
    class_count, pretraining_sizes, optimizer_variant, fold_count
):
    pretraining_values = dict(zip(VALID_PRETRAINING_VALUES, pretraining_sizes, strict=True))
    arguments = make_pretraining_arguments(
        class_count=class_count,
        optimizer_variant=optimizer_variant,
        pretraining_values=pretraining_values,
    )
    pretrained = _pretrain(**arguments)
    statistics = estimate_cross_validated_loss_statistics(
        **_cross_validation_arguments(arguments, pretrained, fold_count)
    )
    expected = _expected_out_of_fold_statistics(
        make_pretraining_arguments(
            class_count=class_count,
            optimizer_variant=optimizer_variant,
            pretraining_values=pretraining_values,
        ),
        list(pretrained.pretraining_samples),
        fold_count,
    )
    assert type(statistics) is ModelAndClassLossStatistics
    assert statistics == expected
    assert statistics.overall_loss_moments.observed_loss_count == pretraining_sizes[0]
    assert statistics != pretrained.loss_statistics


def test_cross_validation_does_not_consume_run_randomness_or_change_the_initial_model():
    arguments = make_pretraining_arguments()
    pretrained = _pretrain(**arguments)
    classifier_state = {
        name: value.clone() for name, value in pretrained.classifier.state_dict().items()
    }
    random.seed(3)
    np.random.seed(4)
    torch.manual_seed(5)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    estimate_cross_validated_loss_statistics(
        **_cross_validation_arguments(arguments, pretrained, 5)
    )
    assert random.getstate() == python_state
    assert_numpy_random_states_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert all(
        torch.equal(classifier_state[name], value)
        for name, value in pretrained.classifier.state_dict().items()
    )


def test_cross_validation_rejects_invalid_fold_counts_and_samples():
    arguments = make_pretraining_arguments()
    pretrained = _pretrain(**arguments)
    base = _cross_validation_arguments(arguments, pretrained, 5)
    for fold_count in (1, 0, len(pretrained.pretraining_samples) + 1):
        with pytest.raises(ValueError):
            estimate_cross_validated_loss_statistics(**base | dict(fold_count=fold_count))
    with pytest.raises(TypeError):
        estimate_cross_validated_loss_statistics(**base | dict(fold_count=True))
    with pytest.raises(TypeError):
        estimate_cross_validated_loss_statistics(
            **base | dict(pretraining_samples=list(pretrained.pretraining_samples))
        )
    with pytest.raises(TypeError):
        estimate_cross_validated_loss_statistics(
            **base | dict(pretraining_samples=(None,) * len(pretrained.pretraining_samples))
        )
