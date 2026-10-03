"""モデル出力から予測・観測後損失への数値を旧基準と照合する。"""

import pytest
import torch

from federated_drift_experiment.clients.fedsda import _AdaHedgeRoutingFedSDAClientMixin
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    convert_model_outputs_to_prediction_probabilities,
    normalize_model_prediction_weights,
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
