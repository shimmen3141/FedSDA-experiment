"""採用済み候補の共有学習を反映し、接続後に個別学習状態を初期化する。"""

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


def _validate_adopted_candidate_integration_inputs(
    *,
    adopted_candidate_classifier: ResidualAdapterClassifier,
    candidate_concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    active_shared_feature_extractor: SharedFeatureExtractor,
) -> None:
    if type(adopted_candidate_classifier) is not ResidualAdapterClassifier:
        raise ValueError("採用候補はexact ResidualAdapterClassifierが必要です。")
    if type(candidate_concept_specific_parameter_optimizer_state) is not ParameterOptimizerState:
        raise ValueError("候補個別管理器はexact ParameterOptimizerStateが必要です。")
    source_feature_extractor = adopted_candidate_classifier.feature_extractor
    if (
        type(source_feature_extractor) is not SharedFeatureExtractor
        or type(active_shared_feature_extractor) is not SharedFeatureExtractor
    ):
        raise ValueError("双方の共有部はexact SharedFeatureExtractorが必要です。")
    source_feature_extractor.validate_structure(
        input_feature_count=source_feature_extractor.input_feature_count,
        hidden_layer_widths=source_feature_extractor.hidden_layer_widths,
    )
    active_shared_feature_extractor.validate_structure(
        input_feature_count=source_feature_extractor.input_feature_count,
        hidden_layer_widths=source_feature_extractor.hidden_layer_widths,
    )
    candidate_concept_parameters = tuple(
        adopted_candidate_classifier.residual_adapter.parameters()
    ) + tuple(adopted_candidate_classifier.classification_layer.parameters())
    candidate_parameter_optimizer = (
        candidate_concept_specific_parameter_optimizer_state.parameter_optimizer
    )
    if type(candidate_parameter_optimizer) not in (Adam, SGD):
        raise ValueError("現在の個別optimizerはexact Adam/SGDが必要です。")
    optimizer_parameters = tuple(
        parameter
        for parameter_group in candidate_parameter_optimizer.param_groups
        for parameter in parameter_group["params"]
    )
    if (
        not candidate_concept_parameters
        or len(optimizer_parameters) != len(candidate_concept_parameters)
        or any(
            parameter is not expected_parameter
            for parameter, expected_parameter in zip(
                optimizer_parameters, candidate_concept_parameters
            )
        )
    ):
        raise ValueError("個別optimizerと候補parameter列が対応していません。")
    shared_parameter_ids = {
        id(parameter)
        for parameter in tuple(source_feature_extractor.parameters())
        + tuple(active_shared_feature_extractor.parameters())
    }
    if len({id(parameter) for parameter in candidate_concept_parameters}) != len(
        candidate_concept_parameters
    ):
        raise ValueError("候補の概念固有parameterは一意である必要があります。")
    for parameter in candidate_concept_parameters:
        if type(parameter) is not Parameter or (
            parameter.device.type != "cpu"
            or parameter.dtype != float32
            or parameter.layout != strided
            or parameter.is_nested
            or id(parameter) in shared_parameter_ids
        ):
            raise ValueError("候補個別parameterは独立したCPU float32 strided Parameterが必要です。")


def integrate_adopted_candidate_shared_features(
    *,
    adopted_candidate_classifier: ResidualAdapterClassifier,
    candidate_concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    active_shared_feature_extractor: SharedFeatureExtractor,
) -> None:
    """全入力検証後に値反映・接続・個別resetを行う。登録は上位で行う。"""
    _validate_adopted_candidate_integration_inputs(
        adopted_candidate_classifier=adopted_candidate_classifier,
        candidate_concept_specific_parameter_optimizer_state=candidate_concept_specific_parameter_optimizer_state,
        active_shared_feature_extractor=active_shared_feature_extractor,
    )
    active_shared_feature_extractor.copy_parameter_values_from(
        source_feature_extractor=adopted_candidate_classifier.feature_extractor
    )
    adopted_candidate_classifier.attach_shared_feature_extractor(
        shared_feature_extractor=active_shared_feature_extractor
    )
    candidate_concept_specific_parameter_optimizer_state.reset_parameter_optimizer()
