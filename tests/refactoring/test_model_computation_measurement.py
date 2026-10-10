"""モデルの計算の計測: 計数と積和演算の数（手計算との照合）、区間の後始末、差、結果を変えないこと。"""

import random
from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch
from torch.nn.modules import module as torch_module_internals
from torch.optim import optimizer as torch_optimizer_internals

from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
    ModelComputationMeter,
    measure_model_computation,
    subtract_model_computation_counts,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)

INPUT_FEATURE_COUNT = 3
HIDDEN_LAYER_WIDTHS = (5, 4)
ADAPTER_RANK = 2
# 標本1件あたりの、全結合層の積和演算の数（入力の次元×出力の次元の合計）。
SHARED_PART_MACS_PER_EXAMPLE = 3 * 5 + 5 * 4
ADAPTER_MACS_PER_EXAMPLE = 4 * 2 + 2 * 4
# 逆伝播の見積り（標本1件あたり）。共有部の最初の層は、入力へ勾配を戻さない（重みの勾配だけ）。
SHARED_PART_BACKWARD_MACS_PER_EXAMPLE = 3 * 5 + 2 * (5 * 4)
ZERO_COUNTS = ModelComputationCounts(
    **{count_field.name: 0 for count_field in fields(ModelComputationCounts)}
)


def make_classifier(*, class_count=2, shared_feature_extractor=None, seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ResidualAdapterClassifier(
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=ADAPTER_RANK,
            ),
            input_feature_count=INPUT_FEATURE_COUNT,
            hidden_layer_widths=HIDDEN_LAYER_WIDTHS,
            class_count=class_count,
            shared_feature_extractor=shared_feature_extractor,
        )


def concept_specific_macs_per_example(class_count):
    """アダプタの2層と、分類層（2クラスは出力1、多クラスはクラス数）。"""
    return ADAPTER_MACS_PER_EXAMPLE + 4 * (1 if class_count == 2 else class_count)


def make_inputs(sample_count):
    return torch.arange(sample_count * INPUT_FEATURE_COUNT, dtype=torch.float32).reshape(
        sample_count, INPUT_FEATURE_COUNT
    ) / (sample_count * INPUT_FEATURE_COUNT)


def count_global_hooks():
    return (
        len(torch_module_internals._global_forward_hooks),
        len(torch_module_internals._global_forward_pre_hooks),
        len(torch_optimizer_internals._global_optimizer_post_hooks),
    )


def test_counts_are_zero_without_model_computation_and_meter_is_readable_after_exit():
    with measure_model_computation() as model_computation_meter:
        assert type(model_computation_meter) is ModelComputationMeter
        assert model_computation_meter.get_model_computation_counts() == ZERO_COUNTS
    assert model_computation_meter.get_model_computation_counts() == ZERO_COUNTS


@pytest.mark.parametrize("class_count", [2, 3])
def test_inference_with_shared_features_counts_shared_part_once(class_count):
    """共有部の特徴を1回だけ計算し、3つの概念固有部で使うと、共有部は標本数、概念固有部は標本数×3。"""
    first_classifier = make_classifier(class_count=class_count)
    classifiers = [
        first_classifier,
        *(
            make_classifier(
                class_count=class_count,
                shared_feature_extractor=first_classifier.feature_extractor,
                seed=seed,
            )
            for seed in (1, 2)
        ),
    ]
    input_features = make_inputs(7)
    with measure_model_computation() as model_computation_meter, torch.no_grad():
        shared_features = first_classifier.extract_shared_features(input_features)
        for classifier in classifiers:
            classifier.forward_from_shared_features(shared_features)
    assert model_computation_meter.get_model_computation_counts() == replace(
        ZERO_COUNTS,
        shared_part_inference_example_count=7,
        concept_specific_part_inference_example_count=21,
        shared_part_inference_forward_multiply_accumulate_count=7 * SHARED_PART_MACS_PER_EXAMPLE,
        concept_specific_part_inference_forward_multiply_accumulate_count=(
            21 * concept_specific_macs_per_example(class_count)
        ),
    )


@pytest.mark.parametrize("class_count", [2, 3])
def test_training_forward_counts_examples_forward_and_estimated_backward(class_count):
    """勾配つきの順伝播は学習として数え、逆伝播の見積りは、実際に勾配が付く層と対応する。"""
    classifier = make_classifier(class_count=class_count)
    input_features = make_inputs(4)
    with measure_model_computation() as model_computation_meter:
        classifier_outputs = classifier(input_features)
    classifier_outputs.sum().backward()
    # 実際の逆伝播: 全部の重みに勾配が付き、入力には付かない（入力は勾配を求めない）。
    assert all(parameter.grad is not None for parameter in classifier.parameters())
    assert input_features.grad is None
    concept_specific_macs = concept_specific_macs_per_example(class_count)
    assert model_computation_meter.get_model_computation_counts() == replace(
        ZERO_COUNTS,
        shared_part_training_example_count=4,
        concept_specific_part_training_example_count=4,
        shared_part_training_forward_multiply_accumulate_count=4 * SHARED_PART_MACS_PER_EXAMPLE,
        concept_specific_part_training_forward_multiply_accumulate_count=4 * concept_specific_macs,
        # 概念固有部の3層は、どれも、重みの勾配と、入力へ戻す勾配。
        shared_part_estimated_backward_multiply_accumulate_count=(
            4 * SHARED_PART_BACKWARD_MACS_PER_EXAMPLE
        ),
        concept_specific_part_estimated_backward_multiply_accumulate_count=(
            4 * 2 * concept_specific_macs
        ),
    )


def test_estimated_backward_follows_which_tensors_require_gradients():
    """勾配を止めた重みは、重みの分を数えない。入力が勾配を求めるなら、最初の層も、入力の分を数える。"""
    classifier = make_classifier()
    first_layer, second_layer = (
        layer
        for layer in classifier.feature_extractor.hidden_layers
        if type(layer) is torch.nn.Linear
    )
    first_layer.weight.requires_grad_(False)
    first_layer.bias.requires_grad_(False)
    with measure_model_computation() as model_computation_meter:
        classifier_outputs = classifier(make_inputs(4))
    classifier_outputs.sum().backward()
    assert first_layer.weight.grad is None
    assert second_layer.weight.grad is not None
    # 最初の層: 重みも入力も勾配なし（0）。2層め: 重みの勾配だけ（入力は、勾配のない層の出力）。
    assert (
        model_computation_meter.get_model_computation_counts().shared_part_estimated_backward_multiply_accumulate_count
        == 4 * (5 * 4)
    )
    first_layer.weight.requires_grad_(True)
    input_features = make_inputs(4).requires_grad_(True)
    with measure_model_computation() as model_computation_meter:
        classifier_outputs = classifier(input_features)
    classifier_outputs.sum().backward()
    assert input_features.grad is not None
    assert (
        model_computation_meter.get_model_computation_counts().shared_part_estimated_backward_multiply_accumulate_count
        == 4 * 2 * SHARED_PART_MACS_PER_EXAMPLE
    )


def test_single_vector_input_counts_one_example():
    """1次元の入力（標本1件）は、標本数1として数える。"""
    classifier = make_classifier()
    with measure_model_computation() as model_computation_meter, torch.no_grad():
        classifier(make_inputs(1).reshape(-1))
    counts = model_computation_meter.get_model_computation_counts()
    assert counts.shared_part_inference_example_count == 1
    assert counts.concept_specific_part_inference_example_count == 1
    assert counts.shared_part_inference_forward_multiply_accumulate_count == (
        SHARED_PART_MACS_PER_EXAMPLE
    )
    assert counts.concept_specific_part_inference_forward_multiply_accumulate_count == (
        concept_specific_macs_per_example(2)
    )


def test_optimizer_steps_are_counted_by_shared_or_concept_specific_parameters():
    """共有部のパラメータを持つoptimizerの更新は共有部、ほかは概念固有部として数える。"""
    classifier = make_classifier()
    other_classifier = make_classifier(
        shared_feature_extractor=classifier.feature_extractor, seed=1
    )
    shared_parameter_optimizer = torch.optim.Adam(classifier.feature_extractor.parameters())
    concept_specific_parameter_optimizers = [
        torch.optim.Adam(
            [
                *each_classifier.residual_adapter.parameters(),
                *each_classifier.classification_layer.parameters(),
            ]
        )
        for each_classifier in (classifier, other_classifier)
    ]
    with measure_model_computation() as model_computation_meter:
        shared_features = classifier.extract_shared_features(make_inputs(6))
        (
            classifier.forward_from_shared_features(shared_features).sum()
            + other_classifier.forward_from_shared_features(shared_features).sum()
        ).backward()
        shared_parameter_optimizer.step()
        for concept_specific_parameter_optimizer in concept_specific_parameter_optimizers:
            concept_specific_parameter_optimizer.step()
        intermediate_counts = model_computation_meter.get_model_computation_counts()
        concept_specific_parameter_optimizers[0].step()
    assert intermediate_counts.shared_parameter_optimizer_step_count == 1
    assert intermediate_counts.concept_specific_parameter_optimizer_step_count == 2
    assert intermediate_counts.shared_part_training_example_count == 6
    assert intermediate_counts.concept_specific_part_training_example_count == 12
    final_counts = model_computation_meter.get_model_computation_counts()
    assert final_counts == replace(
        intermediate_counts, concept_specific_parameter_optimizer_step_count=3
    )
    # 途中の読取りは、後からの計算で変わらない。
    assert intermediate_counts.concept_specific_parameter_optimizer_step_count == 2


def test_optimizer_step_before_any_shared_forward_in_the_interval_counts_as_concept_specific():
    """共有部かどうかは、区間の中で順伝播した共有部のパラメータで見分ける（制約）。

    区間の中で、共有部を1度も順伝播せずに、そのoptimizerを更新すると、概念固有部として数える。
    学習では、更新の前に、必ず順伝播があるので、この場合は起きない。
    """
    classifier = make_classifier()
    shared_parameter_optimizer = torch.optim.Adam(classifier.feature_extractor.parameters())
    with measure_model_computation() as model_computation_meter:
        shared_parameter_optimizer.step()
        counts_before_forward = model_computation_meter.get_model_computation_counts()
        with torch.no_grad():
            classifier(make_inputs(1))
        shared_parameter_optimizer.step()
    assert counts_before_forward.shared_parameter_optimizer_step_count == 0
    assert counts_before_forward.concept_specific_parameter_optimizer_step_count == 1
    counts = model_computation_meter.get_model_computation_counts()
    assert counts.shared_parameter_optimizer_step_count == 1
    assert counts.concept_specific_parameter_optimizer_step_count == 1


def test_keyword_only_and_non_tensor_inputs_are_not_counted():
    """入力を位置引数のtensorで渡さない順伝播は、数えない（制約。新実装のモデルは、位置引数で呼ぶ）。"""
    classifier = make_classifier()
    with measure_model_computation() as model_computation_meter, torch.no_grad():
        classifier.feature_extractor(input_features=make_inputs(2))
    counts = model_computation_meter.get_model_computation_counts()
    assert counts.shared_part_inference_example_count == 0
    # 中の全結合層は、位置引数で呼ばれるので、積和演算は数える。
    assert counts.shared_part_inference_forward_multiply_accumulate_count == (
        2 * SHARED_PART_MACS_PER_EXAMPLE
    )


def test_measurement_removes_hooks_after_exit_and_after_exception():
    """区間を出た後と、例外で出た後は、登録が残らず、区間の外の計算は数えない。"""
    classifier = make_classifier()
    hook_counts_before = count_global_hooks()
    with measure_model_computation() as model_computation_meter:
        assert count_global_hooks() != hook_counts_before
        with torch.no_grad():
            classifier(make_inputs(2))
    assert count_global_hooks() == hook_counts_before
    counts_after_exit = model_computation_meter.get_model_computation_counts()
    with torch.no_grad():
        classifier(make_inputs(5))
    torch.optim.Adam(classifier.parameters()).step()
    assert model_computation_meter.get_model_computation_counts() == counts_after_exit
    assert counts_after_exit.shared_part_inference_example_count == 2

    class ExpectedFailure(Exception):
        pass

    with pytest.raises(ExpectedFailure):
        with measure_model_computation() as failing_meter:
            with torch.no_grad():
                classifier(make_inputs(3))
            raise ExpectedFailure
    assert count_global_hooks() == hook_counts_before
    assert failing_meter.get_model_computation_counts().shared_part_inference_example_count == 3


def test_failed_forward_inside_shared_part_does_not_break_later_attribution():
    """共有部の順伝播が例外で終わっても、その後の全結合層を、共有部の中として数えない。"""
    classifier = make_classifier()
    with measure_model_computation() as model_computation_meter, torch.no_grad():
        with pytest.raises(ValueError):
            # 特徴数が合わない入力（共有部の順伝播が、入力の検査で失敗する）。
            classifier.feature_extractor(torch.zeros(2, INPUT_FEATURE_COUNT + 1))
        counts_after_failure = model_computation_meter.get_model_computation_counts()
        classifier.forward_from_shared_features(torch.zeros(2, HIDDEN_LAYER_WIDTHS[-1]))
    # 失敗した順伝播は、標本数にも、積和演算にも数えない。
    assert counts_after_failure == ZERO_COUNTS
    counts = model_computation_meter.get_model_computation_counts()
    assert counts.shared_part_inference_forward_multiply_accumulate_count == 0
    assert counts.concept_specific_part_inference_forward_multiply_accumulate_count == (
        2 * concept_specific_macs_per_example(2)
    )


def test_nested_measurements_count_independently():
    """入れ子の区間は、それぞれが、自分の区間の計算を数える。"""
    classifier = make_classifier()
    with measure_model_computation() as outer_meter, torch.no_grad():
        classifier(make_inputs(2))
        with measure_model_computation() as inner_meter:
            classifier(make_inputs(3))
        classifier(make_inputs(4))
    assert outer_meter.get_model_computation_counts().shared_part_inference_example_count == 9
    assert inner_meter.get_model_computation_counts().shared_part_inference_example_count == 3


def test_measurement_does_not_change_outputs_gradients_or_random_state():
    """計測の有無で、出力・勾配・乱数が変わらない。"""
    results = []
    for measured in (False, True):
        classifier = make_classifier(class_count=3)
        input_features = make_inputs(5)
        torch.manual_seed(11)
        random.seed(11)
        if measured:
            with measure_model_computation():
                classifier_outputs = classifier(input_features)
                classifier_outputs.sum().backward()
        else:
            classifier_outputs = classifier(input_features)
            classifier_outputs.sum().backward()
        results.append(
            (
                classifier_outputs.detach().clone(),
                [parameter.grad.clone() for parameter in classifier.parameters()],
                torch.get_rng_state().clone(),
                random.getstate(),
            )
        )
    assert torch.equal(results[0][0], results[1][0])
    assert all(
        torch.equal(gradient, measured_gradient)
        for gradient, measured_gradient in zip(results[0][1], results[1][1], strict=True)
    )
    assert torch.equal(results[0][2], results[1][2])
    assert results[0][3] == results[1][3]


def test_counts_subtraction_returns_field_differences_and_rejects_invalid_pairs():
    later_counts = ModelComputationCounts(
        **{
            count_field.name: 10 * (position + 1)
            for position, count_field in enumerate(fields(ModelComputationCounts))
        }
    )
    earlier_counts = ModelComputationCounts(
        **{
            count_field.name: position
            for position, count_field in enumerate(fields(ModelComputationCounts))
        }
    )
    difference = subtract_model_computation_counts(
        later_counts=later_counts, earlier_counts=earlier_counts
    )
    assert difference == ModelComputationCounts(
        **{
            count_field.name: 10 * (position + 1) - position
            for position, count_field in enumerate(fields(ModelComputationCounts))
        }
    )
    assert len(fields(ModelComputationCounts)) == 12
    assert (
        subtract_model_computation_counts(later_counts=later_counts, earlier_counts=later_counts)
        == ZERO_COUNTS
    )
    with pytest.raises(ValueError):
        subtract_model_computation_counts(later_counts=earlier_counts, earlier_counts=later_counts)
    for invalid_counts in (None, {}, 3):
        with pytest.raises(TypeError):
            subtract_model_computation_counts(
                later_counts=invalid_counts, earlier_counts=earlier_counts
            )
        with pytest.raises(TypeError):
            subtract_model_computation_counts(
                later_counts=later_counts, earlier_counts=invalid_counts
            )
    with pytest.raises(TypeError):
        subtract_model_computation_counts(later_counts, earlier_counts)
    with pytest.raises(FrozenInstanceError):
        later_counts.shared_part_training_example_count = 0
