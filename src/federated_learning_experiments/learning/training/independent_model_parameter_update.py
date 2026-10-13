"""独立モデルのparameterを平均損失から一回更新する。"""

from torch import Tensor, is_grad_enabled
from torch.nn import BCELoss, CrossEntropyLoss, Identity, Linear, ReLU, Sequential, Sigmoid
from torch.optim import Optimizer

from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    INDEPENDENT_MODELS_ARCHITECTURE_NAME,
    ModelArchitectureClassifier,
    _validate_classifier_inputs,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.nonlinear_residual_adapter import (
    ModelOwnedHiddenLayers,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    validate_classifier_bounded_loss_inputs,
)

from .joint_model_parameter_update import (
    _validate_parameter_optimizer,
    _validate_trainable_parameters,
)


def _validate_linear_layer(layer: object, input_width: int, output_width: int) -> None:
    if type(layer) is not Linear or (
        layer.in_features != input_width
        or layer.out_features != output_width
        or tuple(layer.weight.shape) != (output_width, input_width)
        or layer.bias is None
        or tuple(layer.bias.shape) != (output_width,)
    ):
        raise ValueError("Linearの宣言幅とparameter shapeが一致しません。")


def _validate_independent_model_structure(classifier: ModelArchitectureClassifier) -> None:
    _validate_classifier_inputs(
        model_architecture_settings=classifier.model_architecture_settings,
        input_feature_count=classifier.input_feature_count,
        hidden_layer_widths=classifier.hidden_layer_widths,
        class_count=classifier.class_count,
        shared_feature_extractor=None,
    )
    if type(classifier.feature_extractor) is not SharedFeatureExtractor:
        raise ValueError("独立モデルの特徴抽出部が不正です。")
    classifier.feature_extractor.validate_structure(
        input_feature_count=classifier.input_feature_count, hidden_layer_widths=()
    )
    adapter = classifier.model_owned_feature_transform
    if type(adapter) is not ModelOwnedHiddenLayers or (
        adapter.feature_count != classifier.input_feature_count
        or adapter.hidden_layer_widths != classifier.hidden_layer_widths
        or type(adapter.hidden_layers) is not Sequential
        or len(adapter.hidden_layers) != 2 * len(classifier.hidden_layer_widths)
    ):
        raise ValueError("独立モデルの隠れ層が不正です。")
    width = classifier.input_feature_count
    for index, next_width in enumerate(classifier.hidden_layer_widths):
        _validate_linear_layer(adapter.hidden_layers[2 * index], width, next_width)
        if type(adapter.hidden_layers[2 * index + 1]) is not ReLU:
            raise ValueError("隠れ層の活性化はexact ReLUが必要です。")
        width = next_width
    if adapter.output_feature_count != width:
        raise ValueError("隠れ層の出力幅が不正です。")
    _validate_linear_layer(
        classifier.classification_layer,
        width,
        1 if classifier.class_count == 2 else classifier.class_count,
    )
    if type(classifier.output_activation) is not (
        Sigmoid if classifier.class_count == 2 else Identity
    ):
        raise ValueError("分類出力の活性化がクラス数と対応していません。")


def validate_independent_model_parameter_update_inputs(
    *,
    classifier: ModelArchitectureClassifier,
    parameter_optimizer: Optimizer,
    input_features: Tensor,
    observed_class_labels: Tensor,
) -> None:
    """学習に使うモデル・標本を、forward/zero_gradより前に検査する。"""
    if type(classifier) is not ModelArchitectureClassifier:
        raise TypeError("分類器はexact ModelArchitectureClassifierが必要です。")
    if type(classifier.model_architecture_settings) is not ModelArchitectureSettings or (
        classifier.model_architecture_settings.model_architecture_name
        != INDEPENDENT_MODELS_ARCHITECTURE_NAME
    ):
        raise ValueError("独立モデル構造が必要です。")
    if not is_grad_enabled():
        raise ValueError("学習にはgrad有効が必要です。")
    _validate_independent_model_structure(classifier)
    validate_classifier_bounded_loss_inputs(
        classifier=classifier,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )
    snapshot_classifier_parameters(classifier=classifier)
    parameters = tuple(classifier.parameters())
    _validate_trainable_parameters(
        expected_parameters=parameters, require_trainable=True, tensor_name="parameters"
    )
    _validate_parameter_optimizer(
        parameter_optimizer=parameter_optimizer,
        expected_parameters=parameters,
        tensor_name="parameter_optimizer",
    )


def update_independent_model_parameters(
    *,
    classifier: ModelArchitectureClassifier,
    parameter_optimizer: Optimizer,
    input_features: Tensor,
    observed_class_labels: Tensor,
) -> float:
    """旧単一モデルと同じloss演算順で、借用optimizerを一回進める。"""
    validate_independent_model_parameter_update_inputs(
        classifier=classifier,
        parameter_optimizer=parameter_optimizer,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )
    parameter_optimizer.zero_grad()
    predictions = classifier(input_features)
    loss = (
        BCELoss()(predictions, observed_class_labels)
        if classifier.class_count == 2
        else CrossEntropyLoss()(predictions, observed_class_labels.view(-1).long())
    )
    loss.backward()
    parameter_optimizer.step()
    return float(loss.item())
