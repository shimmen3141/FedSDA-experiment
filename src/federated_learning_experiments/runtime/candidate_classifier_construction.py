"""選択済み初期値から独立候補と専用optimizer管理器を生成する。"""

from dataclasses import dataclass

from torch import Tensor, float32, isfinite, strided

from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)


@dataclass(frozen=True, kw_only=True)
class IndependentCandidateTrainingState:
    """候補と二つの管理器を借用で返し、bindingの交換を禁止する。"""

    candidate_classifier: ResidualAdapterClassifier
    candidate_shared_parameter_optimizer_state: ParameterOptimizerState
    candidate_concept_specific_parameter_optimizer_state: ParameterOptimizerState


def _validate_candidate_construction_inputs(
    *,
    architecture_reference_classifier: ResidualAdapterClassifier,
    initial_candidate_parameter_snapshot: dict[str, Tensor],
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
) -> AdamParameterOptimizerSettings | SgdParameterOptimizerSettings:
    expected_parameter_snapshot = snapshot_classifier_parameters(
        classifier=architecture_reference_classifier
    )
    if not tuple(architecture_reference_classifier.feature_extractor.parameters()):
        raise ValueError("architecture_reference_classifierの共有部にはparameterが必要です。")
    if type(initial_candidate_parameter_snapshot) is not dict:
        raise TypeError("initial_candidate_parameter_snapshotはexact dictが必要です。")
    if any(
        type(parameter_name) is not str for parameter_name in initial_candidate_parameter_snapshot
    ):
        raise TypeError("initial_candidate_parameter_snapshotのキーはexact strが必要です。")
    if initial_candidate_parameter_snapshot.keys() != expected_parameter_snapshot.keys():
        raise ValueError("initial_candidate_parameter_snapshotのキーが参照と一致しません。")
    for parameter_name, expected_parameter_values in expected_parameter_snapshot.items():
        parameter_values = initial_candidate_parameter_snapshot[parameter_name]
        if type(parameter_values) is not Tensor:
            raise TypeError(
                f"initial_candidate_parameter_snapshot.{parameter_name}はexact Tensorが必要です。"
            )
        if (
            parameter_values.device.type != "cpu"
            or parameter_values.dtype != float32
            or parameter_values.layout != strided
            or parameter_values.is_nested
        ):
            raise ValueError(
                f"initial_candidate_parameter_snapshot.{parameter_name}はCPU float32 stridedが必要です。"
            )
        if parameter_values.shape != expected_parameter_values.shape:
            raise ValueError(
                f"initial_candidate_parameter_snapshot.{parameter_name}のshapeが一致しません。"
            )
        if not isfinite(parameter_values).all().item():
            raise ValueError(
                f"initial_candidate_parameter_snapshot.{parameter_name}は有限値が必要です。"
            )
    if type(parameter_optimizer_settings) is AdamParameterOptimizerSettings:
        return AdamParameterOptimizerSettings(
            learning_rate=parameter_optimizer_settings.learning_rate,
            weight_decay=parameter_optimizer_settings.weight_decay,
            adam_variant=parameter_optimizer_settings.adam_variant,
        )
    if type(parameter_optimizer_settings) is SgdParameterOptimizerSettings:
        return SgdParameterOptimizerSettings(
            learning_rate=parameter_optimizer_settings.learning_rate
        )
    raise TypeError("parameter_optimizer_settingsはexact Adam/SGD設定型が必要です。")


def create_independent_candidate_training_state(
    *,
    architecture_reference_classifier: ResidualAdapterClassifier,
    initial_candidate_parameter_snapshot: dict[str, Tensor],
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
) -> IndependentCandidateTrainingState:
    """生成前に検査し、一つの候補へ値をコピーして空の学習状態を返す。"""
    validated_parameter_optimizer_settings = _validate_candidate_construction_inputs(
        architecture_reference_classifier=architecture_reference_classifier,
        initial_candidate_parameter_snapshot=initial_candidate_parameter_snapshot,
        parameter_optimizer_settings=parameter_optimizer_settings,
    )
    candidate_classifier = ResidualAdapterClassifier(
        model_architecture_settings=architecture_reference_classifier.model_architecture_settings,
        input_feature_count=architecture_reference_classifier.feature_extractor.input_feature_count,
        hidden_layer_widths=architecture_reference_classifier.feature_extractor.hidden_layer_widths,
        class_count=architecture_reference_classifier.class_count,
    )
    candidate_classifier.load_state_dict(initial_candidate_parameter_snapshot, strict=True)
    candidate_shared_parameter_optimizer_state = ParameterOptimizerState(
        parameters=tuple(candidate_classifier.feature_extractor.parameters()),
        optimizer_settings=validated_parameter_optimizer_settings,
    )
    concept_specific_parameters = tuple(candidate_classifier.residual_adapter.parameters()) + tuple(
        candidate_classifier.classification_layer.parameters()
    )
    candidate_concept_specific_parameter_optimizer_state = ParameterOptimizerState(
        parameters=concept_specific_parameters,
        optimizer_settings=validated_parameter_optimizer_settings,
    )
    return IndependentCandidateTrainingState(
        candidate_classifier=candidate_classifier,
        candidate_shared_parameter_optimizer_state=candidate_shared_parameter_optimizer_state,
        candidate_concept_specific_parameter_optimizer_state=candidate_concept_specific_parameter_optimizer_state,
    )
