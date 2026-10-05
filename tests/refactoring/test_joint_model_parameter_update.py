"""固定参加batchから実旧共同更新へ値・勾配・操作順を照合する。"""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import DEFAULT, Mock

import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.data.specs import DatasetSpec
from federated_drift_experiment.models import ResidualAdapterMLP
from federated_drift_experiment.clients.shared_backbone import _SharedRepresentationFedSDAClientMixin
from federated_learning_experiments.learning.models.model_architecture_settings import ModelArchitectureSettings
from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier
from federated_learning_experiments.learning.training.local_training_settings import LocalTrainingSettings
from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings, SgdParameterOptimizerSettings
from federated_learning_experiments.learning.training.parameter_optimizer_construction import create_parameter_optimizer
from federated_learning_experiments.learning.training.participating_model_training_batch import ParticipatingModelTrainingBatch
from federated_learning_experiments.learning.training.joint_model_parameter_update import perform_joint_model_parameter_update


def build_joint_update_oracle_pair(*, class_count, batch_sample_counts, optimizer_variant, monkeypatch):
    monkeypatch.setattr(config, "dataset_spec", lambda dataset: DatasetSpec(
        input_dim=2, num_concepts=2, num_classes=class_count, hidden_dims=(5, 4)))
    monkeypatch.setattr(config, "SHARED_ADAPTER_RANK", 2)
    monkeypatch.setattr(config, "OPTIMIZER", "sgd" if optimizer_variant == "sgd" else "adam")
    monkeypatch.setattr(config, "BASE_LR", 0.01)
    monkeypatch.setattr(config, "WEIGHT_DECAY", 0.001)
    monkeypatch.setattr(config, "AMSGRAD", optimizer_variant == "amsgrad")
    monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
    legacy_models = []
    new_classifiers = []
    for training_batch_index in range(len(batch_sample_counts)):
        legacy_models.append(ResidualAdapterMLP(input_dim=2, dataset="sine2",
            backbone=legacy_models[0].backbone if legacy_models else None))
        classifier = ResidualAdapterClassifier(model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter", residual_adapter_requested_rank=2),
            input_feature_count=2, hidden_layer_widths=(5, 4), class_count=class_count,
            shared_feature_extractor=new_classifiers[0].feature_extractor if new_classifiers else None)
        classifier.load_state_dict({
            parameter_name.replace("backbone.net.", "feature_extractor.hidden_layers.")
                .replace("adapter.down.", "residual_adapter.feature_compression.")
                .replace("adapter.up.", "residual_adapter.feature_expansion.")
                .replace("head.", "classification_layer."): parameter_value
            for parameter_name, parameter_value in legacy_models[-1].state_dict().items()})
        new_classifiers.append(classifier)
    optimizer_settings = (SgdParameterOptimizerSettings(learning_rate=0.01) if optimizer_variant == "sgd"
        else AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0.001, adam_variant=optimizer_variant))
    shared_parameter_optimizer = create_parameter_optimizer(
        parameters=tuple(new_classifiers[0].feature_extractor.parameters()), optimizer_settings=optimizer_settings)
    participating_training_batches = []
    legacy_training_batches = []
    for training_batch_index, sample_count in enumerate(batch_sample_counts):
        input_features = (torch.arange(sample_count * 2, dtype=torch.float32).reshape(sample_count, 2)
            - training_batch_index * 2) / 7
        observed_class_labels = ((torch.arange(sample_count) + training_batch_index) % class_count).float().reshape(-1, 1)
        classifier = new_classifiers[training_batch_index]
        participating_training_batches.append(ParticipatingModelTrainingBatch(classifier=classifier,
            concept_specific_parameter_optimizer=create_parameter_optimizer(
                parameters=tuple(classifier.residual_adapter.parameters()) + tuple(classifier.classification_layer.parameters()),
                optimizer_settings=optimizer_settings), input_features=input_features,
            observed_class_labels=observed_class_labels))
        # IDを非昇順にし、診断のsortが更新順を変えないことも照合する。
        legacy_training_batches.append(((4, -7)[training_batch_index], input_features.clone(), observed_class_labels.clone()))
    legacy_client = SimpleNamespace(updates_per_sample=1,
        models={(4, -7)[training_batch_index]: model for training_batch_index, model in enumerate(legacy_models)},
        _sample_training_batches=lambda: legacy_training_batches,
        _shared_backbone=lambda: legacy_models[0].backbone,
        backbone_gradient_diagnostics=defaultdict(float), compute_counters=defaultdict(int),
        model_training_examples=defaultdict(int), model_optimizer_steps=defaultdict(int),
        phase_seconds=defaultdict(float), _record_model_compute=lambda *args, **kwargs: None)
    return tuple(participating_training_batches), shared_parameter_optimizer, legacy_client


def run_legacy_joint_update(*, legacy_client, update_shared_features):
    weighted_model_losses = []
    loss_hooks = [legacy_client.models[model_id].loss_fn.register_forward_hook(
        lambda model_module, input_features, model_loss, sample_count=sample_count:
            weighted_model_losses.append(model_loss.detach().clone() * sample_count))
        for model_id, input_features, observed_class_labels in legacy_client._sample_training_batches()
        for sample_count in (len(input_features),)]
    try:
        _SharedRepresentationFedSDAClientMixin._train_heads_together(
            legacy_client, count_multiplier=1, update_backbone=update_shared_features)
    finally:
        for loss_hook in loss_hooks:
            loss_hook.remove()
    return float((sum(weighted_model_losses) / sum(len(input_features)
        for _, input_features, _ in legacy_client._sample_training_batches())).item())


def assert_nested_state_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert tuple(actual) == tuple(expected)
        for parameter_name in actual:
            assert_nested_state_equal(actual[parameter_name], expected[parameter_name])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for parameter_value, expected_parameter_value in zip(actual, expected):
            assert_nested_state_equal(parameter_value, expected_parameter_value)
    else:
        assert actual == expected


def assert_joint_update_states_equal(*, participating_training_batches, shared_parameter_optimizer, legacy_client):
    for training_batch, legacy_model in zip(participating_training_batches, legacy_client.models.values()):
        legacy_parameters = dict(legacy_model.named_parameters())
        for parameter_name, parameter in training_batch.classifier.named_parameters():
            old_parameter_name = parameter_name.replace("feature_extractor.hidden_layers.", "backbone.net.")
            old_parameter_name = old_parameter_name.replace("residual_adapter.feature_compression.", "adapter.down.")
            old_parameter_name = old_parameter_name.replace("residual_adapter.feature_expansion.", "adapter.up.")
            old_parameter_name = old_parameter_name.replace("classification_layer.", "head.")
            assert torch.equal(parameter, legacy_parameters[old_parameter_name])
            assert_nested_state_equal(parameter.grad, legacy_parameters[old_parameter_name].grad)
        assert_nested_state_equal(training_batch.concept_specific_parameter_optimizer.state_dict(), legacy_model.head_optimizer.state_dict())
    assert_nested_state_equal(shared_parameter_optimizer.state_dict(), legacy_client._shared_backbone().optimizer.state_dict())


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("batch_sample_counts", [(3,), (2, 5)])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_joint_model_parameter_update_matches_legacy(
    class_count, batch_sample_counts, optimizer_variant, update_shared_features, monkeypatch,
):
    global_torch_random_state = torch.get_rng_state()
    try:
        torch.manual_seed(137)
        participating_training_batches, shared_parameter_optimizer, legacy_client = build_joint_update_oracle_pair(
            class_count=class_count, batch_sample_counts=batch_sample_counts,
            optimizer_variant=optimizer_variant, monkeypatch=monkeypatch)
        local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients")
        for _ in range(3):
            expected_joint_loss = run_legacy_joint_update(legacy_client=legacy_client, update_shared_features=update_shared_features)
            actual_joint_loss = perform_joint_model_parameter_update(local_training_settings=local_training_settings,
                shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
                shared_parameter_optimizer=shared_parameter_optimizer,
                participating_training_batches=participating_training_batches, update_shared_features=update_shared_features)
            assert actual_joint_loss == expected_joint_loss
            assert_joint_update_states_equal(participating_training_batches=participating_training_batches,
                shared_parameter_optimizer=shared_parameter_optimizer, legacy_client=legacy_client)
    finally:
        torch.set_rng_state(global_torch_random_state)


@pytest.mark.parametrize("update_shared_features", [True, False])
def test_joint_model_parameter_update_observes_operation_order(update_shared_features, monkeypatch):
    participating_training_batches, shared_parameter_optimizer, legacy_client = build_joint_update_oracle_pair(
        class_count=2, batch_sample_counts=(2, 5), optimizer_variant="amsgrad", monkeypatch=monkeypatch)
    operation_events = []
    for parameter_optimizer, training_batch_index in (
        (shared_parameter_optimizer, "shared"),
        *((training_batch.concept_specific_parameter_optimizer, training_batch_index)
          for training_batch_index, training_batch in enumerate(participating_training_batches)),
        (legacy_client._shared_backbone().optimizer, "shared"),
        *((legacy_model.head_optimizer, training_batch_index)
          for training_batch_index, legacy_model in enumerate(legacy_client.models.values())),
    ):
        monkeypatch.setattr(parameter_optimizer, "zero_grad", Mock(wraps=parameter_optimizer.zero_grad,
            side_effect=lambda *args, training_batch_index=training_batch_index, **kwargs:
                (operation_events.append(("zero", training_batch_index)), DEFAULT)[1]))
        parameter_optimizer.register_step_pre_hook(lambda *args, training_batch_index=training_batch_index:
            operation_events.append(("step", training_batch_index)))
    for model_module in (participating_training_batches[0].classifier.feature_extractor, legacy_client._shared_backbone()):
        model_module.register_forward_pre_hook(lambda *args: operation_events.append(("forward", "shared")))
    original_backward = torch.Tensor.backward
    monkeypatch.setattr(torch.Tensor, "backward", lambda self, *args, **kwargs:
        (operation_events.append(("backward", None)), original_backward(self, *args, **kwargs))[1])
    actual_joint_loss = perform_joint_model_parameter_update(local_training_settings=LocalTrainingSettings(
        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients"),
        shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
        shared_parameter_optimizer=shared_parameter_optimizer, participating_training_batches=participating_training_batches,
        update_shared_features=update_shared_features)
    expected_operation_events = [("zero", "shared"), ("zero", 0), ("zero", 1),
        ("forward", "shared"), ("backward", None)]
    if update_shared_features:
        expected_operation_events.append(("step", "shared"))
    expected_operation_events.extend([("step", 0), ("step", 1)])
    assert operation_events == expected_operation_events
    operation_events.clear()
    expected_joint_loss = run_legacy_joint_update(legacy_client=legacy_client, update_shared_features=update_shared_features)
    assert operation_events == expected_operation_events
    assert actual_joint_loss == expected_joint_loss
    assert_joint_update_states_equal(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer, legacy_client=legacy_client)
