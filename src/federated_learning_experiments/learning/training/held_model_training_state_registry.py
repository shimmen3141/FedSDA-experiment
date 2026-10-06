"""モデルID別に分類器と現在の個別optimizer管理器を保持する。"""

from dataclasses import dataclass

from torch import float32, strided
from torch.nn import Parameter
from torch.optim import SGD, Adam

from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)

from .held_model_training_binding import HeldModelTrainingBinding
from .parameter_optimizer_state import ParameterOptimizerState


@dataclass(frozen=True, kw_only=True)
class HeldModelTrainingState:
    """フィールドは固定し、分類器とoptimizer管理器をliveな参照で保持する。"""

    model_id: int
    classifier: ResidualAdapterClassifier
    concept_specific_parameter_optimizer_state: ParameterOptimizerState


def _validate_model_id(*, model_id: int, parameter_name: str = "model_id") -> None:
    if type(model_id) is not int:
        raise ValueError(f"{parameter_name}はbool・派生型以外のbuiltin intが必要です。")


def _validate_held_model_training_state_inputs(
    *,
    model_id: int,
    classifier: ResidualAdapterClassifier,
    concept_specific_parameter_optimizer_state: ParameterOptimizerState,
) -> None:
    _validate_model_id(model_id=model_id)
    if type(classifier) is not ResidualAdapterClassifier:
        raise ValueError("classifierはexact ResidualAdapterClassifierが必要です。")
    if type(concept_specific_parameter_optimizer_state) is not ParameterOptimizerState:
        raise ValueError("個別optimizer管理器はexact ParameterOptimizerStateが必要です。")
    concept_specific_parameters = tuple(classifier.residual_adapter.parameters()) + tuple(
        classifier.classification_layer.parameters()
    )
    if not concept_specific_parameters or len(
        {id(parameter) for parameter in concept_specific_parameters}
    ) != len(concept_specific_parameters):
        raise ValueError("概念固有parameter列は非空かつ一意である必要があります。")
    shared_parameter_ids = {
        id(parameter) for parameter in classifier.feature_extractor.parameters()
    }
    for parameter in concept_specific_parameters:
        if type(parameter) is not Parameter or (
            parameter.device.type != "cpu"
            or parameter.dtype != float32
            or parameter.layout != strided
            or parameter.is_nested
            or id(parameter) in shared_parameter_ids
        ):
            raise ValueError(
                "個別parameterは共有部から独立したCPU float32 strided Parameterが必要です。"
            )
    parameter_optimizer = concept_specific_parameter_optimizer_state.parameter_optimizer
    if type(parameter_optimizer) not in (Adam, SGD):
        raise ValueError("現在の個別optimizerはexact Adam/SGDが必要です。")
    optimizer_parameters = tuple(
        parameter
        for parameter_group in parameter_optimizer.param_groups
        for parameter in parameter_group["params"]
    )
    if len(optimizer_parameters) != len(concept_specific_parameters) or any(
        parameter is not expected_parameter
        for parameter, expected_parameter in zip(optimizer_parameters, concept_specific_parameters)
    ):
        raise ValueError(
            "個別optimizerとモデルの概念固有parameter列のidentity・順序が一致しません。"
        )


class HeldModelTrainingStateRegistry:
    """一覧構造を所有し、上位で生成された分類器と管理器の参照を保持する。"""

    def __init__(self) -> None:
        self._held_model_training_states_by_model_id: dict[int, HeldModelTrainingState] = {}

    def register_held_model_training_state(
        self,
        *,
        model_id: int,
        classifier: ResidualAdapterClassifier,
        concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    ) -> None:
        """対応検証後に登録する。同ID置換は初出位置と旧取得記録を保持する。"""
        _validate_held_model_training_state_inputs(
            model_id=model_id,
            classifier=classifier,
            concept_specific_parameter_optimizer_state=concept_specific_parameter_optimizer_state,
        )
        self._held_model_training_states_by_model_id[model_id] = HeldModelTrainingState(
            model_id=model_id,
            classifier=classifier,
            concept_specific_parameter_optimizer_state=concept_specific_parameter_optimizer_state,
        )

    def reassign_held_model_training_state_id(
        self,
        *,
        original_model_id: int,
        reassigned_model_id: int,
    ) -> None:
        """IDだけを移し、分類器と現在optimizer管理器の借用参照を維持する。"""
        _validate_model_id(model_id=original_model_id, parameter_name="original_model_id")
        _validate_model_id(model_id=reassigned_model_id, parameter_name="reassigned_model_id")
        if original_model_id not in self._held_model_training_states_by_model_id:
            return
        held_model_training_state = self._held_model_training_states_by_model_id[original_model_id]
        reassigned_held_model_training_state = HeldModelTrainingState(
            model_id=reassigned_model_id,
            classifier=held_model_training_state.classifier,
            concept_specific_parameter_optimizer_state=(
                held_model_training_state.concept_specific_parameter_optimizer_state
            ),
        )
        self._held_model_training_states_by_model_id.pop(original_model_id)
        self._held_model_training_states_by_model_id[reassigned_model_id] = (
            reassigned_held_model_training_state
        )

    def get_held_model_training_state(self, *, model_id: int) -> HeldModelTrainingState:
        """登録済みrecordを借用する。未登録はKeyError、取得時の自動生成はない。"""
        _validate_model_id(model_id=model_id)
        return self._held_model_training_states_by_model_id[model_id]

    def snapshot_ordered_held_model_training_states(self) -> tuple[HeldModelTrainingState, ...]:
        """一覧構造を分離する。record内の分類器と管理器はliveな参照である。"""
        return tuple(self._held_model_training_states_by_model_id.values())

    def snapshot_ordered_held_model_training_bindings(self) -> tuple[HeldModelTrainingBinding, ...]:
        """取得時の現在optimizerを借用する。以前のbindingは交換しない。"""
        return tuple(
            HeldModelTrainingBinding(
                model_id=held_model_training_state.model_id,
                classifier=held_model_training_state.classifier,
                concept_specific_parameter_optimizer=(
                    held_model_training_state.concept_specific_parameter_optimizer_state.parameter_optimizer
                ),
            )
            for held_model_training_state in self._held_model_training_states_by_model_id.values()
        )
