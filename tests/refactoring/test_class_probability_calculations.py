"""モデル出力から予測・観測後損失への数値を旧基準と照合する。"""

import inspect
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from federated_drift_experiment.clients.fedsda import _AdaHedgeRoutingFedSDAClientMixin
from federated_drift_experiment.expert_routing import SwitchingExpertRouter
from test_run_settings_validation import valid_run_settings_mapping
from test_fixed_share_prediction_weights import (
    assert_fixed_share_state_matches_reference,
    capture_fixed_share_controller_state,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights import (
    FixedSharePredictionWeightController,
)
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    convert_model_outputs_to_prediction_probabilities,
    normalize_model_prediction_weights,
    combine_model_prediction_probabilities,
    predict_class_labels_from_prediction_scores,
    compute_model_mean_bounded_losses_after_label_observation,
)


@pytest.mark.parametrize("prediction_weights_by_model_id", [
    {7: 0.2, -3: 0.8}, {2: 1.0}, {7: 0.0, -3: 1.0},
    {model_id: 0.1 for model_id in reversed(range(10))},
    {7: 0.5 + 5e-13, -3: 0.5},
])
def test_class_probability_normalization_matches_reference_without_input_mutation(
    prediction_weights_by_model_id,
):
    """全poolでも通常加算で正規化し、昇順の独立辞書を返す。"""
    reference_prediction_mixin = _AdaHedgeRoutingFedSDAClientMixin
    reference_prediction_weights = dict(prediction_weights_by_model_id)
    normalized_prediction_weights_by_model_id = normalize_model_prediction_weights(
        prediction_weights_by_model_id=prediction_weights_by_model_id,
    )
    assert normalized_prediction_weights_by_model_id == reference_prediction_mixin._restrict_routing_probabilities(
        prediction_weights_by_model_id, sorted(prediction_weights_by_model_id),
    )
    assert tuple(normalized_prediction_weights_by_model_id) == tuple(sorted(prediction_weights_by_model_id))
    assert normalized_prediction_weights_by_model_id is not prediction_weights_by_model_id
    assert prediction_weights_by_model_id == reference_prediction_weights


@pytest.mark.parametrize("class_count", [2, 3, 4])
def test_class_probability_model_outputs_become_probabilities_before_combination(class_count):
    """二値はsigmoid済み値を保持し、多クラスはモデルごとにsoftmaxする。"""
    model_outputs_by_model_id = {
        7: torch.tensor([[0.0], [0.5], [1.0]], requires_grad=True) if class_count == 2 else
        torch.arange(3 * class_count, dtype=torch.float32).reshape(3, class_count).requires_grad_(),
        -3: torch.tensor([[1.0], [0.5], [0.0]], requires_grad=True) if class_count == 2 else
        -torch.arange(3 * class_count, dtype=torch.float32).reshape(3, class_count).requires_grad_(),
    }
    input_tensors_before_call = {model_id: prediction_tensor.detach().clone() for model_id, prediction_tensor in model_outputs_by_model_id.items()}
    prediction_probabilities_by_model_id = convert_model_outputs_to_prediction_probabilities(
        model_outputs_by_model_id=model_outputs_by_model_id, class_count=class_count,
    )
    assert tuple(prediction_probabilities_by_model_id) == (-3, 7)
    for model_id, prediction_tensor in prediction_probabilities_by_model_id.items():
        assert torch.equal(prediction_tensor, model_outputs_by_model_id[model_id] if class_count == 2 else torch.softmax(model_outputs_by_model_id[model_id], dim=1))
        assert prediction_tensor.data_ptr() != model_outputs_by_model_id[model_id].data_ptr()
        assert not prediction_tensor.requires_grad and prediction_tensor.grad_fn is None
        assert torch.equal(model_outputs_by_model_id[model_id], input_tensors_before_call[model_id])
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("class_count", [2, 3, 4])
@pytest.mark.parametrize("prediction_weights_by_model_id", [
    {7: 0.2, -3: 0.8}, {7: 0, -3: 1}, {-3: 1},
    {-3: 0.5000000298023224 + 1e-13, 7: 0.4999999701976776 + 4e-13},
])
def test_class_probability_combination_matches_reference_in_sorted_model_order(
    class_count, prediction_weights_by_model_id,
):
    """ゼロ・単一モデル・未補正snapshotを昇順に混合する。"""
    reference_prediction_mixin = _AdaHedgeRoutingFedSDAClientMixin
    prediction_probabilities_by_model_id = {
        model_id: torch.tensor([[1.0], [0.0]]) if class_count == 2 else
        torch.softmax(torch.arange(2 * class_count, dtype=torch.float32).reshape(2, class_count) * model_id, dim=1)
        for model_id in reversed(sorted(prediction_weights_by_model_id))
    }
    if class_count == 2 and 7 in prediction_probabilities_by_model_id:
        prediction_probabilities_by_model_id[7] = torch.tensor([[0.0], [1.0]])
    reference_prediction_scores = reference_prediction_mixin._weighted_routing_scores(
        prediction_probabilities_by_model_id, dict(sorted(prediction_weights_by_model_id.items())),
    )
    combined_prediction_probabilities = combine_model_prediction_probabilities(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        prediction_weights_by_model_id=prediction_weights_by_model_id, class_count=class_count,
    )
    assert torch.equal(combined_prediction_probabilities, reference_prediction_scores)
    for prediction_tensor in prediction_probabilities_by_model_id.values():
        assert combined_prediction_probabilities.data_ptr() != prediction_tensor.data_ptr()


@pytest.mark.parametrize("class_count,prediction_scores", [
    (2, torch.tensor([[0.0], [0.49999997], [0.5], [0.50000006], [1.0]])),
    (3, torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.2, 0.2, 0.6]])),
    (4, torch.tensor([[0.25, 0.25, 0.25, 0.25]])),
])
def test_class_probability_class_predictions_preserve_threshold_and_tie_rules(class_count, prediction_scores):
    """境界隣接と同率を旧閾値・argmaxへ完全一致させる。"""
    predicted_class_labels = predict_class_labels_from_prediction_scores(prediction_scores=prediction_scores, class_count=class_count)
    assert torch.equal(predicted_class_labels, _AdaHedgeRoutingFedSDAClientMixin._routing_prediction(prediction_scores, class_count))
    assert predicted_class_labels.shape == (prediction_scores.shape[0], 1)
    assert predicted_class_labels.dtype == torch.float32


@pytest.mark.parametrize("class_count", [2, 3])
def test_class_probability_class_predictions_preserve_threshold_and_tie_rules_after_mixture_rounding(class_count):
    """モデル側で許容した値と混合丸めを、判定で再拒否・補正しない。"""
    prediction_probabilities_by_model_id = {
        model_id: torch.tensor([[1.0]]) if class_count == 2 else
        torch.tensor([[1.0, 9.536743e-7, 0.0]])
        for model_id in range(10)
    }
    normalized_prediction_weights_by_model_id = normalize_model_prediction_weights(
        prediction_weights_by_model_id={model_id: 0.1 for model_id in range(10)},
    )
    combined_prediction_probabilities = combine_model_prediction_probabilities(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        prediction_weights_by_model_id=normalized_prediction_weights_by_model_id,
        class_count=class_count,
    )
    assert combined_prediction_probabilities[0, 0].item() > 1.0
    assert torch.equal(predict_class_labels_from_prediction_scores(
        prediction_scores=combined_prediction_probabilities, class_count=class_count,
    ), _AdaHedgeRoutingFedSDAClientMixin._routing_prediction(combined_prediction_probabilities, class_count))


@pytest.mark.parametrize("class_count", [2, 3, 4])
@pytest.mark.parametrize("observed_class_labels", [
    torch.tensor([0, 1, 0]), torch.tensor([[0], [1], [0]], dtype=torch.int32),
    torch.tensor([0.0, 1.0, 0.0]),
])
def test_class_probability_observed_model_losses_match_reference(class_count, observed_class_labels):
    """複数標本をfloat32平均し、モデルごとの損失を旧staticへ照合する。"""
    prediction_probabilities_by_model_id = {
        7: torch.tensor([[0.1], [0.7], [0.3]]) if class_count == 2 else
        torch.softmax(torch.arange(3 * class_count, dtype=torch.float32).reshape(3, class_count), dim=1),
        -3: torch.tensor([[0.9], [0.3], [0.7]]) if class_count == 2 else
        torch.softmax(-torch.arange(3 * class_count, dtype=torch.float32).reshape(3, class_count), dim=1),
    }
    observed_losses_by_model_id = compute_model_mean_bounded_losses_after_label_observation(
        prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
        observed_class_labels=observed_class_labels, class_count=class_count,
    )
    assert tuple(observed_losses_by_model_id) == (-3, 7)
    assert observed_losses_by_model_id == {
        model_id: _AdaHedgeRoutingFedSDAClientMixin._routing_score_loss(prediction_tensor, observed_class_labels, class_count)
        for model_id, prediction_tensor in prediction_probabilities_by_model_id.items()
    }


@pytest.mark.parametrize("observed_class_labels", [
    [0], torch.tensor([True]), torch.tensor([0.0], dtype=torch.float64),
    torch.tensor([0], dtype=torch.int16), torch.tensor([0], device='meta'),
    torch.tensor([0, 1]), torch.tensor([[0, 1]]), torch.tensor(0),
    torch.tensor([-1]), torch.tensor([2]), torch.tensor([0.5]),
    torch.tensor([float('nan')]), torch.tensor([float('inf')]),
    torch.tensor([[0.0]]).to_sparse(),
])
def test_class_probability_invalid_inputs_are_rejected_without_mutation_for_labels(observed_class_labels):
    """不正な観測ラベルで予測時の確率を変更しない。"""
    prediction_probabilities_by_model_id = {0: torch.tensor([[0.5]], requires_grad=True)}
    input_tensors_before_call = prediction_probabilities_by_model_id[0].detach().clone()
    with pytest.raises((TypeError, ValueError)) as exception_info:
        compute_model_mean_bounded_losses_after_label_observation(
            prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
            observed_class_labels=observed_class_labels, class_count=2,
        )
    assert str(exception_info.value)
    assert torch.equal(prediction_probabilities_by_model_id[0], input_tensors_before_call)
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("invalid_input_name,invalid_input_value,class_count", [
    ('probabilities', {}, 2), ('probabilities', {True: torch.tensor([[0.5]])}, 2),
    ('probabilities', {0: torch.tensor([[1.1]])}, 2),
    ('probabilities', {0: torch.tensor([[0.1, 0.2, 0.3]])}, 3),
    ('probabilities', {0: torch.tensor([[float('nan')]])}, 2),
    ('probabilities', {0: torch.tensor([[0.5]], dtype=torch.float64)}, 2),
    ('weights', {1: 1}, 2), ('weights', {0: 0}, 2), ('weights', {}, 2),
    ('weights', {0: float('nan')}, 2),
    ('scores', torch.tensor([[float('nan')]]), 2),
    ('scores', torch.tensor([[0.5]], dtype=torch.float64), 2),
    ('scores', torch.empty((0, 1)), 2), ('scores', torch.tensor([0.5]), 2),
    ('scores', torch.tensor([[0.2, 0.8]]), 2),
])
def test_class_probability_invalid_inputs_are_rejected_without_mutation_for_prediction(
    invalid_input_name, invalid_input_value, class_count,
):
    """混合前の確率・重み対応とクラス判定の入力形を検査する。"""
    with pytest.raises((TypeError, ValueError)) as exception_info:
        if invalid_input_name == 'scores':
            predict_class_labels_from_prediction_scores(prediction_scores=invalid_input_value, class_count=class_count)
        else:
            combine_model_prediction_probabilities(
                prediction_probabilities_by_model_id=invalid_input_value if invalid_input_name == 'probabilities' else {0: torch.tensor([[0.5]])},
                prediction_weights_by_model_id=invalid_input_value if invalid_input_name == 'weights' else {0: 1},
                class_count=class_count,
            )
    assert str(exception_info.value)
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("class_count", [2, 3])
def test_class_probability_outputs_are_independent_and_have_no_gradient_history(class_count):
    """勾配と入力値を保存し、返却を変更しても入力に波及させない。"""
    model_outputs_by_model_id = {0: torch.tensor([[0.5]]) if class_count == 2 else torch.tensor([[0.1, 0.2, 0.3]])}
    model_outputs_by_model_id[0].requires_grad_()
    model_outputs_by_model_id[0].grad = torch.ones_like(model_outputs_by_model_id[0])
    input_tensors_before_call = model_outputs_by_model_id[0].detach().clone()
    prediction_probabilities_by_model_id = convert_model_outputs_to_prediction_probabilities(model_outputs_by_model_id=model_outputs_by_model_id, class_count=class_count)
    observed_class_labels = torch.tensor([0.0], requires_grad=True)
    observed_class_labels.grad = torch.ones_like(observed_class_labels)
    with torch.no_grad():
        combined_prediction_probabilities = combine_model_prediction_probabilities(prediction_probabilities_by_model_id=prediction_probabilities_by_model_id, prediction_weights_by_model_id={0: 1}, class_count=class_count)
        predicted_class_labels = predict_class_labels_from_prediction_scores(prediction_scores=combined_prediction_probabilities, class_count=class_count)
        observed_losses_by_model_id = compute_model_mean_bounded_losses_after_label_observation(prediction_probabilities_by_model_id=prediction_probabilities_by_model_id, observed_class_labels=observed_class_labels, class_count=class_count)
        assert not torch.is_grad_enabled()
    for prediction_tensor in (prediction_probabilities_by_model_id[0], combined_prediction_probabilities, predicted_class_labels):
        assert not prediction_tensor.requires_grad and prediction_tensor.grad_fn is None
        assert prediction_tensor.data_ptr() != model_outputs_by_model_id[0].data_ptr()
    assert torch.equal(model_outputs_by_model_id[0], input_tensors_before_call)
    assert torch.equal(model_outputs_by_model_id[0].grad, torch.ones_like(model_outputs_by_model_id[0]))
    assert torch.equal(observed_class_labels.grad, torch.ones_like(observed_class_labels))
    assert type(observed_losses_by_model_id[0]) is float
    combined_prediction_probabilities.zero_()
    assert prediction_probabilities_by_model_id[0].sum().item() > 0
    prediction_probabilities_by_model_id[0].zero_()
    assert torch.equal(model_outputs_by_model_id[0], input_tensors_before_call)
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("invalid_input_name,invalid_input_value", [
    ("weights", {}), ("weights", {True: 1.0}), ("weights", {0.0: 1.0}),
    ("weights", {0: True}), ("weights", {0: float('nan')}),
    ("weights", {0: float('inf')}), ("weights", {0: -0.1, 1: 1.1}),
    ("weights", {0: 0.0}), ("weights", {0: 0.5}), ("weights", []),
    ("weights", {0: 10 ** 500}),
    ("class_count", True), ("class_count", 1), ("class_count", 2.0),
    ("outputs", {}), ("outputs", {True: torch.tensor([[0.5]])}),
    ("outputs", {0: [[0.5]]}),
    ("outputs", {0: torch.tensor([[0.5]], dtype=torch.float64)}),
    ("outputs", {0: torch.tensor([[0.5]], device='meta')}),
    ("outputs", {0: torch.tensor([[0.5]]).to_sparse()}),
    ("outputs", {0: torch.empty((0, 1))}),
    ("outputs", {0: torch.tensor([0.5])}),
    ("outputs", {0: torch.tensor([[0.2, 0.8]])}),
    ("outputs", {0: torch.tensor([[float('nan')]])}),
    ("outputs", {0: torch.tensor([[1.1]])}),
    ("outputs", {0: torch.tensor([[0.5]]), 1: torch.tensor([[0.2], [0.8]])}),
])
def test_class_probability_invalid_inputs_are_rejected_without_mutation(invalid_input_name, invalid_input_value):
    """前処理の不正値を理由付きで拒否し、勾配モードを保つ。"""
    with pytest.raises((TypeError, ValueError)) as exception_info:
        if invalid_input_name == 'weights':
            normalize_model_prediction_weights(prediction_weights_by_model_id=invalid_input_value)
        else:
            convert_model_outputs_to_prediction_probabilities(
                model_outputs_by_model_id=invalid_input_value if invalid_input_name == 'outputs' else {0: torch.tensor([[0.5]])},
                class_count=invalid_input_value if invalid_input_name == 'class_count' else 2,
            )
    assert str(exception_info.value)
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("class_count", [2, 3, 10])
def test_class_probability_fixed_share_prediction_and_observation_match_reference(
    valid_run_settings_mapping, class_count,
):
    """ラベル前のsnapshotを共有し、30標本の予測と更新後全状態を照合する。"""
    prediction_combination_settings = replace(
        valid_run_settings_mapping['prediction_combination_settings'],
        fixed_share_weight_redistribution_time_scale_samples=30,
    )
    controller = FixedSharePredictionWeightController(prediction_combination_settings=prediction_combination_settings)
    reference_router = SwitchingExpertRouter(30)
    reference_prediction_mixin = _AdaHedgeRoutingFedSDAClientMixin
    for observation_index in range(30):
        model_ids = (7, -3, 0) if observation_index < 10 else (
            (-3, 7) if observation_index < 15 else ((7,) if observation_index < 20 else tuple(reversed(range(10))))
        )
        prediction_weights_by_model_id = controller.get_prediction_weights_before_label_observation(model_ids=model_ids)
        reference_prediction_weights = reference_router.probabilities(model_ids)
        assert prediction_weights_by_model_id == reference_prediction_weights
        assert_fixed_share_state_matches_reference(controller=controller, reference_router=reference_router)
        controller_state_before_call = capture_fixed_share_controller_state(controller=controller)
        model_outputs_by_model_id = {
            model_id: torch.tensor([[(observation_index + model_id) % 11 / 10], [(observation_index - model_id) % 11 / 10]])
            if class_count == 2 else torch.arange(2 * class_count, dtype=torch.float32).reshape(2, class_count) * (
                model_id if observation_index % 2 == 0 else -model_id
            )
            for model_id in model_ids
        }
        prediction_probabilities_by_model_id = convert_model_outputs_to_prediction_probabilities(model_outputs_by_model_id=model_outputs_by_model_id, class_count=class_count)
        reference_prediction_probabilities = {
            model_id: model_outputs_by_model_id[model_id] if class_count == 2 else torch.softmax(model_outputs_by_model_id[model_id], dim=1)
            for model_id in sorted(model_ids)
        }
        for model_id in model_ids:
            assert torch.equal(prediction_probabilities_by_model_id[model_id], reference_prediction_probabilities[model_id])
        normalized_prediction_weights_by_model_id = normalize_model_prediction_weights(prediction_weights_by_model_id=prediction_weights_by_model_id)
        reference_prediction_weights = reference_prediction_mixin._restrict_routing_probabilities(reference_prediction_weights, sorted(model_ids))
        assert normalized_prediction_weights_by_model_id == reference_prediction_weights
        combined_prediction_probabilities = combine_model_prediction_probabilities(prediction_probabilities_by_model_id=prediction_probabilities_by_model_id, prediction_weights_by_model_id=normalized_prediction_weights_by_model_id, class_count=class_count)
        reference_prediction_scores = reference_prediction_mixin._weighted_routing_scores(reference_prediction_probabilities, reference_prediction_weights)
        assert torch.equal(combined_prediction_probabilities, reference_prediction_scores)
        predicted_class_labels = predict_class_labels_from_prediction_scores(prediction_scores=combined_prediction_probabilities, class_count=class_count)
        reference_predicted_class_labels = reference_prediction_mixin._routing_prediction(reference_prediction_scores, class_count)
        assert torch.equal(predicted_class_labels, reference_predicted_class_labels)
        assert capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
        with pytest.raises(ValueError):
            compute_model_mean_bounded_losses_after_label_observation(prediction_probabilities_by_model_id=prediction_probabilities_by_model_id, observed_class_labels=torch.tensor([class_count, 0]), class_count=class_count)
        assert capture_fixed_share_controller_state(controller=controller) == controller_state_before_call
        observed_class_labels = torch.tensor([observation_index % class_count, (observation_index + 1) % class_count])
        observed_losses_by_model_id = compute_model_mean_bounded_losses_after_label_observation(prediction_probabilities_by_model_id=prediction_probabilities_by_model_id, observed_class_labels=observed_class_labels, class_count=class_count)
        reference_observed_losses = {
            model_id: reference_prediction_mixin._routing_score_loss(reference_prediction_probabilities[model_id], observed_class_labels, class_count)
            for model_id in sorted(model_ids)
        }
        assert observed_losses_by_model_id == reference_observed_losses
        assert torch.equal(predict_class_labels_from_prediction_scores(prediction_scores=combined_prediction_probabilities, class_count=class_count), predicted_class_labels)
        controller.update_weights_after_loss_observation(observed_losses_by_model_id=observed_losses_by_model_id, prediction_weights_by_model_id=normalized_prediction_weights_by_model_id)
        reference_router.update(reference_observed_losses, reference_prediction_weights)
        assert_fixed_share_state_matches_reference(controller=controller, reference_router=reference_router)


@pytest.mark.parametrize("prediction_operation", [
    convert_model_outputs_to_prediction_probabilities, normalize_model_prediction_weights,
    combine_model_prediction_probabilities, predict_class_labels_from_prediction_scores,
    compute_model_mean_bounded_losses_after_label_observation,
])
def test_class_probability_functions_require_explicit_keyword_arguments(prediction_operation):
    """予測前APIへラベルを持ち込まず、引数の意味を呼出し側で明示する。"""
    assert str(inspect.signature(prediction_operation)).startswith('(*,')
    with pytest.raises(TypeError):
        prediction_operation(object())
    if prediction_operation is not compute_model_mean_bounded_losses_after_label_observation:
        assert 'observed_class_labels' not in inspect.signature(prediction_operation).parameters


def test_class_probability_numeric_calls_preserve_caller_random_states():
    """数値APIはPython・NumPy・torchの共有乱数を消費しない。"""
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    global_torch_random_state = torch.random.get_rng_state().clone()
    prediction_probabilities_by_model_id = convert_model_outputs_to_prediction_probabilities(model_outputs_by_model_id={0: torch.tensor([[0.2, 0.3, 0.5]])}, class_count=3)
    normalized_prediction_weights_by_model_id = normalize_model_prediction_weights(prediction_weights_by_model_id={0: 1})
    combined_prediction_probabilities = combine_model_prediction_probabilities(prediction_probabilities_by_model_id=prediction_probabilities_by_model_id, prediction_weights_by_model_id=normalized_prediction_weights_by_model_id, class_count=3)
    predict_class_labels_from_prediction_scores(prediction_scores=combined_prediction_probabilities, class_count=3)
    compute_model_mean_bounded_losses_after_label_observation(prediction_probabilities_by_model_id=prediction_probabilities_by_model_id, observed_class_labels=torch.tensor([2]), class_count=3)
    assert random.getstate() == global_python_random_state
    assert np.random.get_state()[0] == global_numpy_random_state[0]
    assert np.array_equal(np.random.get_state()[1], global_numpy_random_state[1])
    assert np.random.get_state()[2:] == global_numpy_random_state[2:]
    assert torch.equal(torch.random.get_rng_state(), global_torch_random_state)
