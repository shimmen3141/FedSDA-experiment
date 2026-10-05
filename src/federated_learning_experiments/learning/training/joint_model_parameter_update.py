"""明示参加batch列の標本数加重共同更新を一回だけ実行する。"""

from torch import Tensor, cat, isfinite, no_grad, is_grad_enabled, float32, strided
from torch.nn import Parameter, BCELoss, CrossEntropyLoss
from torch.optim import Optimizer, Adam, SGD

from .local_training_settings import LocalTrainingSettings
from .participating_model_training_batch import ParticipatingModelTrainingBatch
from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor
from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier


def _validate_training_tensor(*, input_features: Tensor, tensor_name: str, expected_shape: tuple[int, int]) -> None:
    if not isinstance(input_features, Tensor):
        raise ValueError(f"{tensor_name}はTensorが必要です。")
    if (input_features.device.type != "cpu" or input_features.dtype != float32
            or input_features.layout != strided or input_features.is_nested):
        raise ValueError(f"{tensor_name}はCPU float32 strided/notnestedが必要です。")
    if expected_shape[0] < 1 or input_features.dim() != 2 or tuple(input_features.shape) != expected_shape:
        raise ValueError(f"{tensor_name}は正の標本数と指定のshape {expected_shape}が必要です。")
    if not isfinite(input_features).all().item():
        raise ValueError(f"{tensor_name}は有限値が必要です。")


def _validate_trainable_parameters(
    *, expected_parameters: tuple[Parameter, ...], require_trainable: bool, tensor_name: str,
) -> None:
    for parameter_index, parameter in enumerate(expected_parameters):
        if type(parameter) is not Parameter:
            raise ValueError(f"{tensor_name}[{parameter_index}]はexact Parameterが必要です。")
        if (parameter.device.type != "cpu" or parameter.dtype != float32
                or parameter.layout != strided or parameter.is_nested):
            raise ValueError(f"{tensor_name}[{parameter_index}]はCPU float32 strided/notnestedが必要です。")
        if require_trainable and not parameter.requires_grad:
            raise ValueError(f"{tensor_name}[{parameter_index}]はrequires_grad=Trueが必要です。")


def _validate_parameter_optimizer(
    *, parameter_optimizer: Optimizer, expected_parameters: tuple[Parameter, ...], tensor_name: str,
) -> None:
    if type(parameter_optimizer) not in (Adam, SGD):
        raise ValueError(f"{tensor_name}はexact Adam/SGDが必要です。")
    optimizer_parameters = tuple(parameter for parameter_group in parameter_optimizer.param_groups
        for parameter in parameter_group["params"])
    if (len(optimizer_parameters) != len(expected_parameters)
            or any(parameter is not expected_parameter
                for parameter, expected_parameter in zip(optimizer_parameters, expected_parameters))):
        raise ValueError(f"{tensor_name}のParameter参照列と順序が一致しません。")


def _validate_joint_update_inputs(
    *, local_training_settings: LocalTrainingSettings,
    shared_feature_extractor: SharedFeatureExtractor,
    shared_parameter_optimizer: Optimizer | None,
    participating_training_batches: tuple[ParticipatingModelTrainingBatch, ...],
    update_shared_features: bool,
) -> None:
    if type(local_training_settings) is not LocalTrainingSettings:
        raise ValueError("local_training_settingsはexact LocalTrainingSettingsが必要です。")
    try:
        validated_local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy=local_training_settings.local_model_parameter_update_strategy,
            shared_backbone_gradient_combination_strategy=local_training_settings.shared_backbone_gradient_combination_strategy)
    except (TypeError, ValueError) as validation_error:
        raise ValueError(f"local_training_settings: {validation_error}") from validation_error
    if type(shared_feature_extractor) is not SharedFeatureExtractor:
        raise ValueError("shared_feature_extractorはexact SharedFeatureExtractorが必要です。")
    if type(participating_training_batches) is not tuple:
        raise ValueError("participating_training_batchesはexact tupleが必要です。")
    if type(update_shared_features) is not bool:
        raise ValueError("update_shared_featuresはexact boolが必要です。")
    for training_batch_index, training_batch in enumerate(participating_training_batches):
        if type(training_batch) is not ParticipatingModelTrainingBatch:
            raise ValueError(f"participating_training_batches[{training_batch_index}]はexact ParticipatingModelTrainingBatchが必要です。")
    if not participating_training_batches:
        return
    if not is_grad_enabled():
        raise ValueError("participating_training_batchesが非空の更新にはgrad有効が必要です。")
    shared_feature_extractor.validate_structure(input_feature_count=shared_feature_extractor.input_feature_count,
        hidden_layer_widths=shared_feature_extractor.hidden_layer_widths)
    shared_parameters = tuple(shared_feature_extractor.parameters())
    _validate_trainable_parameters(expected_parameters=shared_parameters, require_trainable=update_shared_features,
        tensor_name="shared_parameters")
    if shared_parameters:
        _validate_parameter_optimizer(parameter_optimizer=shared_parameter_optimizer,
            expected_parameters=shared_parameters, tensor_name="shared_parameter_optimizer")
    elif shared_parameter_optimizer is not None:
        raise ValueError("空のshared_parametersにはshared_parameter_optimizer=Noneが必要です。")
    seen_classifier_ids = set()
    seen_optimizer_ids = {id(shared_parameter_optimizer)} if shared_parameter_optimizer is not None else set()
    seen_parameter_ids = {id(parameter) for parameter in shared_parameters}
    for training_batch_index, training_batch in enumerate(participating_training_batches):
        tensor_name = f"participating_training_batches[{training_batch_index}]"
        classifier = training_batch.classifier
        if type(classifier) is not ResidualAdapterClassifier:
            raise ValueError(f"{tensor_name}.classifierはexact ResidualAdapterClassifierが必要です。")
        if classifier.feature_extractor is not shared_feature_extractor:
            raise ValueError(f"{tensor_name}.classifierは同じshared_feature_extractor参照が必要です。")
        if type(classifier.class_count) is not int or classifier.class_count < 2:
            raise ValueError(f"{tensor_name}.classifier.class_countはexact int>=2が必要です。")
        if id(classifier) in seen_classifier_ids:
            raise ValueError(f"{tensor_name}.classifierが重複しています。")
        seen_classifier_ids.add(id(classifier))
        sample_count = (training_batch.input_features.shape[0]
            if isinstance(training_batch.input_features, Tensor) and not training_batch.input_features.is_nested
                and training_batch.input_features.dim() == 2 else 0)
        _validate_training_tensor(input_features=training_batch.input_features, tensor_name=f"{tensor_name}.input_features",
            expected_shape=(sample_count, shared_feature_extractor.input_feature_count))
        _validate_training_tensor(input_features=training_batch.observed_class_labels,
            tensor_name=f"{tensor_name}.observed_class_labels", expected_shape=(sample_count, 1))
        observed_class_labels = training_batch.observed_class_labels
        if classifier.class_count == 2:
            if not ((observed_class_labels >= 0) & (observed_class_labels <= 1)).all().item():
                raise ValueError(f"{tensor_name}.observed_class_labelsは二値の0～1が必要です。")
        elif not ((observed_class_labels >= 0) & (observed_class_labels < classifier.class_count)
                & (observed_class_labels == observed_class_labels.floor())).all().item():
            raise ValueError(f"{tensor_name}.observed_class_labelsは多クラスの範囲内整数値が必要です。")
        concept_specific_parameters = tuple(classifier.residual_adapter.parameters()) + tuple(classifier.classification_layer.parameters())
        _validate_trainable_parameters(expected_parameters=concept_specific_parameters, require_trainable=True,
            tensor_name=f"{tensor_name}.concept_specific_parameters")
        for parameter in concept_specific_parameters:
            if id(parameter) in seen_parameter_ids:
                raise ValueError(f"{tensor_name}.concept_specific_parametersが共有部または他の個別部と重複しています。")
            seen_parameter_ids.add(id(parameter))
        if id(training_batch.concept_specific_parameter_optimizer) in seen_optimizer_ids:
            raise ValueError(f"{tensor_name}.concept_specific_parameter_optimizerが重複しています。")
        seen_optimizer_ids.add(id(training_batch.concept_specific_parameter_optimizer))
        _validate_parameter_optimizer(parameter_optimizer=training_batch.concept_specific_parameter_optimizer,
            expected_parameters=concept_specific_parameters, tensor_name=f"{tensor_name}.concept_specific_parameter_optimizer")


def perform_joint_model_parameter_update(
    *, local_training_settings: LocalTrainingSettings,
    shared_feature_extractor: SharedFeatureExtractor,
    shared_parameter_optimizer: Optimizer | None,
    participating_training_batches: tuple[ParticipatingModelTrainingBatch, ...],
    update_shared_features: bool,
) -> float | None:
    """全入力検査後に借用Parameterを更新し、更新前共同損失を返す。"""
    _validate_joint_update_inputs(local_training_settings=local_training_settings,
        shared_feature_extractor=shared_feature_extractor, shared_parameter_optimizer=shared_parameter_optimizer,
        participating_training_batches=participating_training_batches, update_shared_features=update_shared_features)
    if not participating_training_batches:
        return None
    if shared_parameter_optimizer is not None:
        shared_parameter_optimizer.zero_grad()
    for training_batch in participating_training_batches:
        training_batch.concept_specific_parameter_optimizer.zero_grad()
    combined_input_features = cat([training_batch.input_features for training_batch in participating_training_batches])
    if update_shared_features:
        combined_shared_features = shared_feature_extractor(combined_input_features)
    else:
        with no_grad():
            combined_shared_features = shared_feature_extractor(combined_input_features)
    weighted_model_losses = []
    feature_offset = 0
    total_sample_count = 0
    for training_batch in participating_training_batches:
        sample_count = len(training_batch.input_features)
        shared_features = combined_shared_features[feature_offset:feature_offset + sample_count]
        classifier_predictions = training_batch.classifier.forward_from_shared_features(shared_features)
        if training_batch.classifier.class_count == 2:
            model_loss = BCELoss()(classifier_predictions, training_batch.observed_class_labels)
        else:
            target_class_indices = training_batch.observed_class_labels.view(-1).long()
            model_loss = CrossEntropyLoss()(classifier_predictions, target_class_indices)
        weighted_model_losses.append(model_loss * sample_count)
        feature_offset += sample_count
        total_sample_count += sample_count
    joint_loss = sum(weighted_model_losses) / total_sample_count
    joint_loss.backward()
    if update_shared_features and shared_parameter_optimizer is not None:
        shared_parameter_optimizer.step()
    for training_batch in participating_training_batches:
        training_batch.concept_specific_parameter_optimizer.step()
    return float(joint_loss.item())
