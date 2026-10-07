"""候補の区間検査と固定エポック反復。"""

from dataclasses import dataclass

from torch import Tensor, float32, is_grad_enabled, isfinite, strided
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset

from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)

from .candidate_epoch_training_settings import CandidateEpochTrainingSettings
from .joint_model_parameter_update import perform_joint_model_parameter_update
from .local_training_settings import LocalTrainingSettings
from .parameter_optimizer_state import ParameterOptimizerState
from .participating_model_training_batch import ParticipatingModelTrainingBatch


@dataclass(frozen=True, kw_only=True)
class CandidateEpochTrainingResult:
    """呼出し内で実行した量だけを保持する。"""

    completed_epoch_count: int
    candidate_trained_sample_count: int
    candidate_parameter_update_step_count: int
    validation_evaluated_sample_count: int


def _validate_candidate_epoch_training_inputs(
    *,
    candidate_classifier: ResidualAdapterClassifier,
    candidate_shared_parameter_optimizer_state: ParameterOptimizerState,
    candidate_concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    input_features: Tensor,
    observed_class_labels: Tensor,
    candidate_epoch_training_settings: CandidateEpochTrainingSettings,
) -> None:
    if type(candidate_epoch_training_settings) is not CandidateEpochTrainingSettings:
        raise TypeError("exact CandidateEpochTrainingSettingsが必要です。")
    CandidateEpochTrainingSettings(
        candidate_training_strategy=candidate_epoch_training_settings.candidate_training_strategy,
        maximum_epoch_count=candidate_epoch_training_settings.maximum_epoch_count,
        maximum_batch_sample_count=candidate_epoch_training_settings.maximum_batch_sample_count,
        validation_sample_fraction=candidate_epoch_training_settings.validation_sample_fraction,
        consecutive_non_improving_epoch_limit=candidate_epoch_training_settings.consecutive_non_improving_epoch_limit,
        minimum_validation_loss_decrease=candidate_epoch_training_settings.minimum_validation_loss_decrease,
    )
    snapshot_classifier_parameters(classifier=candidate_classifier)
    shared_parameters = tuple(candidate_classifier.feature_extractor.parameters())
    concept_specific_parameters = tuple(candidate_classifier.residual_adapter.parameters()) + tuple(
        candidate_classifier.classification_layer.parameters()
    )
    if not shared_parameters:
        raise ValueError("候補の共有parameterは非空が必要です。")
    for classifier_parameter in shared_parameters + concept_specific_parameters:
        if not classifier_parameter.requires_grad:
            raise ValueError("候補の全parameterはrequires_grad=Trueが必要です。")
    for parameter_optimizer_state, expected_parameters in (
        (candidate_shared_parameter_optimizer_state, shared_parameters),
        (candidate_concept_specific_parameter_optimizer_state, concept_specific_parameters),
    ):
        if type(parameter_optimizer_state) is not ParameterOptimizerState:
            raise TypeError("exact ParameterOptimizerStateが必要です。")
        parameter_optimizer = parameter_optimizer_state.parameter_optimizer
        if type(parameter_optimizer) not in (Adam, SGD):
            raise TypeError("exact Adam/SGDが必要です。")
        optimizer_parameters = tuple(
            classifier_parameter
            for optimizer_parameter_group in parameter_optimizer.param_groups
            for classifier_parameter in optimizer_parameter_group["params"]
        )
        if len(optimizer_parameters) != len(expected_parameters) or any(
            classifier_parameter is not expected_parameters[parameter_index]
            for parameter_index, classifier_parameter in enumerate(optimizer_parameters)
        ):
            raise ValueError("optimizerは候補parameterへnative順で結合してください。")
    if (
        candidate_shared_parameter_optimizer_state.parameter_optimizer
        is candidate_concept_specific_parameter_optimizer_state.parameter_optimizer
    ):
        raise ValueError("二つのoptimizerは専用の別物が必要です。")
    for training_tensor, tensor_name in (
        (input_features, "input_features"),
        (observed_class_labels, "observed_class_labels"),
    ):
        if not isinstance(training_tensor, Tensor):
            raise TypeError(f"{tensor_name}はTensorが必要です。")
        if (
            training_tensor.device.type != "cpu"
            or training_tensor.dtype != float32
            or training_tensor.layout != strided
            or training_tensor.is_nested
        ):
            raise ValueError(f"{tensor_name}はCPU float32 stridedが必要です。")
        if training_tensor.ndim != 2 or len(training_tensor) < 1:
            raise ValueError(f"{tensor_name}は非空の2次元区間が必要です。")
        if not isfinite(training_tensor).all().item():
            raise ValueError(f"{tensor_name}は有限値が必要です。")
    if (
        input_features.shape[1] != candidate_classifier.feature_extractor.input_feature_count
        or observed_class_labels.shape != (len(input_features), 1)
    ):
        raise ValueError("区間shapeが分類器と一致しません。")
    if (
        not (
            (observed_class_labels == observed_class_labels.round())
            & (observed_class_labels >= 0)
            & (observed_class_labels < candidate_classifier.class_count)
        )
        .all()
        .item()
    ):
        raise ValueError("観測クラスは分類器の整数クラス範囲が必要です。")
    if (
        candidate_epoch_training_settings.candidate_training_strategy != "skip_training"
        and candidate_epoch_training_settings.maximum_epoch_count > 0
        and not is_grad_enabled()
    ):
        raise ValueError("学習更新にはgrad有効が必要です。")


def _train_candidate_dataset_epochs(
    *,
    candidate_classifier: ResidualAdapterClassifier,
    candidate_shared_parameter_optimizer_state: ParameterOptimizerState,
    candidate_concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    training_dataset: TensorDataset,
    epoch_count: int,
    maximum_batch_sample_count: int,
) -> CandidateEpochTrainingResult:
    training_data_loader = DataLoader(
        training_dataset,
        batch_size=min(maximum_batch_sample_count, len(training_dataset)),
        shuffle=True,
    )
    local_training_settings = LocalTrainingSettings(
        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
    )
    completed_epoch_count = 0
    candidate_trained_sample_count = 0
    candidate_parameter_update_step_count = 0
    for _ in range(epoch_count):
        for batch_input_features, batch_observed_class_labels in training_data_loader:
            participating_training_batch = ParticipatingModelTrainingBatch(
                classifier=candidate_classifier,
                concept_specific_parameter_optimizer=candidate_concept_specific_parameter_optimizer_state.parameter_optimizer,
                input_features=batch_input_features,
                observed_class_labels=batch_observed_class_labels,
            )
            perform_joint_model_parameter_update(
                local_training_settings=local_training_settings,
                shared_feature_extractor=candidate_classifier.feature_extractor,
                shared_parameter_optimizer=candidate_shared_parameter_optimizer_state.parameter_optimizer,
                participating_training_batches=(participating_training_batch,),
                update_shared_features=True,
            )
            candidate_trained_sample_count += len(batch_input_features)
            candidate_parameter_update_step_count += 1
        completed_epoch_count += 1
    return CandidateEpochTrainingResult(
        completed_epoch_count=completed_epoch_count,
        candidate_trained_sample_count=candidate_trained_sample_count,
        candidate_parameter_update_step_count=candidate_parameter_update_step_count,
        validation_evaluated_sample_count=0,
    )
