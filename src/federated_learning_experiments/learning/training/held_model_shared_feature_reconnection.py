"""保有モデルを選択した共有部へ接続し、現在のoptimizer対応を返す。"""

from dataclasses import dataclass

from torch import float32, strided
from torch.nn import Parameter
from torch.optim import SGD, Adam

from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)

from .parameter_optimizer_state import ParameterOptimizerState


@dataclass(frozen=True, kw_only=True)
class HeldModelOptimizerBinding:
    """分類器とoptimizer管理器の対応を借用する。所有者は生成しない。"""

    model_id: int
    classifier: ResidualAdapterClassifier
    shared_parameter_optimizer_state: ParameterOptimizerState
    concept_specific_parameter_optimizer_state: ParameterOptimizerState


def _validate_optimizer_parameter_binding(
    *, optimizer_state: ParameterOptimizerState, expected_parameters: tuple[Parameter, ...]
) -> None:
    if type(optimizer_state) is not ParameterOptimizerState:
        raise ValueError("optimizer_stateはexact ParameterOptimizerStateが必要です。")
    parameter_optimizer = optimizer_state.parameter_optimizer
    if type(parameter_optimizer) not in (Adam, SGD):
        raise ValueError("現在optimizerはexact Adam/SGDが必要です。")
    optimizer_parameters = tuple(
        parameter
        for parameter_group in parameter_optimizer.param_groups
        for parameter in parameter_group["params"]
    )
    if (
        not expected_parameters
        or len(optimizer_parameters) != len(expected_parameters)
        or any(
            parameter is not expected_parameter
            for parameter, expected_parameter in zip(optimizer_parameters, expected_parameters)
        )
    ):
        raise ValueError("現在optimizerとモデルのparameter列が対応していません。")
    for parameter in expected_parameters:
        if type(parameter) is not Parameter or (
            parameter.device.type != "cpu"
            or parameter.dtype != float32
            or parameter.layout != strided
            or parameter.is_nested
        ):
            raise ValueError("parameterはexact Parameter/CPU float32 strided/notnestedが必要です。")


def _validate_held_model_optimizer_bindings(
    *, held_model_optimizer_bindings: tuple[HeldModelOptimizerBinding, ...]
) -> HeldModelOptimizerBinding | None:
    if type(held_model_optimizer_bindings) is not tuple:
        raise ValueError("held_model_optimizer_bindingsはexact tupleが必要です。")
    model_ids: set[int] = set()
    classifier_ids: set[int] = set()
    concept_parameter_ids: set[int] = set()
    for binding in held_model_optimizer_bindings:
        if type(binding) is not HeldModelOptimizerBinding:
            raise ValueError("bindingはexact HeldModelOptimizerBindingが必要です。")
        if type(binding.model_id) is not int or binding.model_id in model_ids:
            raise ValueError("model_idはbool/派生型以外の一意なintが必要です。")
        model_ids.add(binding.model_id)
        if (
            type(binding.classifier) is not ResidualAdapterClassifier
            or id(binding.classifier) in classifier_ids
        ):
            raise ValueError("classifierは重複しないexact ResidualAdapterClassifierが必要です。")
        classifier_ids.add(id(binding.classifier))
        if type(binding.classifier.feature_extractor) is not SharedFeatureExtractor:
            raise ValueError("共有部はexact SharedFeatureExtractorが必要です。")
        binding.classifier.feature_extractor.validate_structure(
            input_feature_count=binding.classifier.feature_extractor.input_feature_count,
            hidden_layer_widths=binding.classifier.feature_extractor.hidden_layer_widths,
        )
        _validate_optimizer_parameter_binding(
            optimizer_state=binding.shared_parameter_optimizer_state,
            expected_parameters=tuple(binding.classifier.feature_extractor.parameters()),
        )
        parameters = tuple(binding.classifier.residual_adapter.parameters()) + tuple(
            binding.classifier.classification_layer.parameters()
        )
        _validate_optimizer_parameter_binding(
            optimizer_state=binding.concept_specific_parameter_optimizer_state,
            expected_parameters=parameters,
        )
        current_concept_parameter_ids = {id(parameter) for parameter in parameters}
        if (
            len(current_concept_parameter_ids) != len(parameters)
            or current_concept_parameter_ids & concept_parameter_ids
        ):
            raise ValueError("概念固有parameterはモデル間で独立している必要があります。")
        concept_parameter_ids.update(current_concept_parameter_ids)
    if not held_model_optimizer_bindings:
        return None
    nonnegative_model_ids = [model_id for model_id in model_ids if model_id >= 0]
    source_binding = (
        next(
            binding
            for binding in held_model_optimizer_bindings
            if binding.model_id == min(nonnegative_model_ids)
        )
        if nonnegative_model_ids
        else held_model_optimizer_bindings[0]
    )
    for binding in held_model_optimizer_bindings:
        source_binding.classifier.feature_extractor.validate_structure(
            input_feature_count=binding.classifier.feature_extractor.input_feature_count,
            hidden_layer_widths=binding.classifier.feature_extractor.hidden_layer_widths,
        )
    return source_binding


def reconnect_held_models_to_shared_feature_extractor(
    *, held_model_optimizer_bindings: tuple[HeldModelOptimizerBinding, ...]
) -> tuple[HeldModelOptimizerBinding, ...]:
    """事前検証後、旧順序で接続/resetし、成功時の現在対応だけを返す。"""
    source_binding = _validate_held_model_optimizer_bindings(
        held_model_optimizer_bindings=held_model_optimizer_bindings
    )
    if source_binding is None:
        return ()
    reconnected_bindings = []
    for binding in held_model_optimizer_bindings:
        if binding is source_binding:
            reconnected_bindings.append(binding)
            continue
        binding.classifier.attach_shared_feature_extractor(
            shared_feature_extractor=source_binding.classifier.feature_extractor
        )
        binding.concept_specific_parameter_optimizer_state.reset_parameter_optimizer()
        reconnected_bindings.append(
            HeldModelOptimizerBinding(
                model_id=binding.model_id,
                classifier=binding.classifier,
                shared_parameter_optimizer_state=source_binding.shared_parameter_optimizer_state,
                concept_specific_parameter_optimizer_state=binding.concept_specific_parameter_optimizer_state,
            )
        )
    return tuple(reconnected_bindings)
