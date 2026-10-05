"""固定参加batchから実旧共同更新へ値・勾配・操作順を照合する。"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import FrozenInstanceError, MISSING, fields
import random
import warnings
from types import SimpleNamespace
from unittest.mock import DEFAULT, Mock

import pytest
import numpy as np
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


def capture_joint_update_state(*, participating_training_batches, shared_parameter_optimizer):
    """モデル全値・gradと借用入力・optimizerを独立した値として保存する。"""
    snapshot_before_update = []
    parameter_optimizers = tuple(training_batch.concept_specific_parameter_optimizer
        for training_batch in participating_training_batches) + ((shared_parameter_optimizer,) if shared_parameter_optimizer is not None else ())
    # 不正な参照列ではstate_dictのindex再対応が失敗するため、実stateとgroupを直接保存する。
    optimizer_states = [{
        "state": deepcopy({id(parameter): parameter_value for parameter, parameter_value in parameter_optimizer.state.items()}),
        "param_groups": [{parameter_name: [id(parameter) for parameter in parameter_value]
            if parameter_name == "params" else deepcopy(parameter_value)
            for parameter_name, parameter_value in parameter_group.items()}
            for parameter_group in parameter_optimizer.param_groups],
    } for parameter_optimizer in parameter_optimizers]
    for training_batch_index, training_batch in enumerate(participating_training_batches):
        classifier = training_batch.classifier
        snapshot_before_update.append({
            "parameters": tuple((id(parameter), parameter.detach().clone(),
                id(parameter.grad) if parameter.grad is not None else None,
                None if parameter.grad is None else parameter.grad.detach().clone(), parameter.requires_grad)
                for parameter in classifier.parameters()),
            "modules": tuple((model_module_name, id(model_module)) for model_module_name, model_module in classifier.named_modules()),
            "optimizer": optimizer_states[training_batch_index],
            "input_features": (id(training_batch.input_features), training_batch.input_features.detach().clone()),
            "observed_class_labels": (id(training_batch.observed_class_labels), training_batch.observed_class_labels.detach().clone()),
        })
    return snapshot_before_update, None if shared_parameter_optimizer is None else optimizer_states[-1]


@pytest.mark.parametrize("input_contract_case", [
    "settings_type", "settings_pcgrad", "settings_unknown_update", "outer_list", "update_bool",
    "record_type", "model_type", "shared_type", "shared_reference", "class_count",
    "empty_n", "wrong_d", "rank", "label_shape", "input_nan", "label_inf", "input_type",
    "input_dtype", "label_dtype", "input_sparse", "input_meta", "input_nested", "label_nested",
    "binary_below", "binary_above", "multi_fractional", "multi_outside",
    "classifier_duplicate", "optimizer_duplicate", "individual_parameter_duplicate",
    "individual_shared_parameter", "optimizer_type", "optimizer_reverse", "optimizer_missing", "optimizer_extra",
    "shared_optimizer_reverse", "shared_optimizer_none_true", "shared_optimizer_none_false",
    "individual_requires_grad", "shared_requires_grad", "grad_disabled",
])
def test_joint_model_parameter_update_rejects_invalid_inputs_before_mutation(input_contract_case, monkeypatch):
    class_count = 4 if input_contract_case.startswith("multi_") else 2
    participating_training_batches, shared_parameter_optimizer, legacy_client = build_joint_update_oracle_pair(
        class_count=class_count, batch_sample_counts=(2, 5), optimizer_variant="amsgrad", monkeypatch=monkeypatch)
    local_training_settings = LocalTrainingSettings(
        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients")
    invalid_inputs = dict(local_training_settings=local_training_settings,
        shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
        shared_parameter_optimizer=shared_parameter_optimizer, participating_training_batches=participating_training_batches,
        update_shared_features=True)
    perform_joint_model_parameter_update(**invalid_inputs)
    for training_batch in participating_training_batches:
        for parameter in training_batch.classifier.parameters():
            parameter.grad = torch.full_like(parameter, 0.37)
    operation_events = []
    parameter_optimizers = (shared_parameter_optimizer,) + tuple(
        training_batch.concept_specific_parameter_optimizer for training_batch in participating_training_batches)
    for parameter_optimizer in parameter_optimizers:
        monkeypatch.setattr(parameter_optimizer, "zero_grad", Mock(wraps=parameter_optimizer.zero_grad))
        monkeypatch.setattr(parameter_optimizer, "step", Mock(wraps=parameter_optimizer.step))
    training_batch = participating_training_batches[-1]
    input_features = training_batch.input_features
    observed_class_labels = training_batch.observed_class_labels
    expected_error_field = r"participating_training_batches\[1\]"
    if input_contract_case == "settings_type":
        invalid_inputs["local_training_settings"] = None
        expected_error_field = "local_training_settings"
    elif input_contract_case in ("settings_pcgrad", "settings_unknown_update"):
        parameter_name = ("shared_backbone_gradient_combination_strategy" if input_contract_case == "settings_pcgrad"
            else "local_model_parameter_update_strategy")
        object.__setattr__(local_training_settings, parameter_name, "pcgrad")
        expected_error_field = parameter_name
    elif input_contract_case == "outer_list":
        invalid_inputs["participating_training_batches"] = list(participating_training_batches)
        expected_error_field = "participating_training_batches"
    elif input_contract_case == "update_bool":
        invalid_inputs["update_shared_features"] = 1
        expected_error_field = "update_shared_features"
    elif input_contract_case == "record_type":
        invalid_inputs["participating_training_batches"] = (participating_training_batches[0], object())
    elif input_contract_case == "shared_type":
        invalid_inputs["shared_feature_extractor"] = torch.nn.Identity()
        expected_error_field = "shared_feature_extractor"
    elif input_contract_case == "shared_reference":
        invalid_inputs["shared_feature_extractor"] = deepcopy(training_batch.classifier.feature_extractor)
        invalid_inputs["shared_parameter_optimizer"] = torch.optim.Adam(invalid_inputs["shared_feature_extractor"].parameters())
        expected_error_field = r"participating_training_batches\[0\]"
    elif input_contract_case == "model_type":
        invalid_inputs["participating_training_batches"] = (participating_training_batches[0],
            ParticipatingModelTrainingBatch(classifier=None, concept_specific_parameter_optimizer=training_batch.concept_specific_parameter_optimizer,
                input_features=input_features, observed_class_labels=observed_class_labels))
    elif input_contract_case == "class_count":
        training_batch.classifier.class_count = True
    elif input_contract_case in ("classifier_duplicate", "optimizer_duplicate"):
        invalid_inputs["participating_training_batches"] = (participating_training_batches[0],
            ParticipatingModelTrainingBatch(classifier=participating_training_batches[0].classifier if input_contract_case == "classifier_duplicate"
                else training_batch.classifier,
                concept_specific_parameter_optimizer=participating_training_batches[0].concept_specific_parameter_optimizer,
                input_features=input_features, observed_class_labels=observed_class_labels))
    elif input_contract_case in ("individual_parameter_duplicate", "individual_shared_parameter"):
        training_batch.classifier.residual_adapter.feature_compression.weight = (
            participating_training_batches[0].classifier.residual_adapter.feature_compression.weight
            if input_contract_case == "individual_parameter_duplicate" else next(training_batch.classifier.feature_extractor.parameters()))
    elif input_contract_case.startswith("optimizer_") or input_contract_case == "shared_optimizer_reverse":
        parameter_optimizer = (shared_parameter_optimizer if input_contract_case == "shared_optimizer_reverse"
            else training_batch.concept_specific_parameter_optimizer)
        if input_contract_case == "optimizer_type":
            invalid_inputs["participating_training_batches"] = (participating_training_batches[0],
                ParticipatingModelTrainingBatch(classifier=training_batch.classifier, concept_specific_parameter_optimizer=None,
                    input_features=input_features, observed_class_labels=observed_class_labels))
        elif input_contract_case.endswith("reverse"):
            parameter_optimizer.param_groups[0]["params"].reverse()
        elif input_contract_case == "optimizer_missing":
            parameter_optimizer.param_groups[0]["params"].pop()
        else:
            parameter_optimizer.param_groups[0]["params"].append(next(training_batch.classifier.feature_extractor.parameters()))
        if input_contract_case == "shared_optimizer_reverse":
            expected_error_field = "shared_parameter_optimizer"
    elif input_contract_case.startswith("shared_optimizer_none"):
        invalid_inputs["shared_parameter_optimizer"] = None
        invalid_inputs["update_shared_features"] = input_contract_case.endswith("true")
        expected_error_field = "shared_parameter_optimizer"
    elif input_contract_case in ("individual_requires_grad", "shared_requires_grad"):
        parameter = (training_batch.classifier.classification_layer.weight if input_contract_case == "individual_requires_grad"
            else next(training_batch.classifier.feature_extractor.parameters()))
        parameter.requires_grad_(False)
        if input_contract_case == "shared_requires_grad":
            expected_error_field = "shared_parameters"
    elif input_contract_case == "grad_disabled":
        expected_error_field = "grad"
    else:
        if input_contract_case == "empty_n":
            input_features = input_features[:0]
        elif input_contract_case == "wrong_d":
            input_features = torch.ones(5, 3)
        elif input_contract_case == "rank":
            input_features = input_features.unsqueeze(0)
        elif input_contract_case == "label_shape":
            observed_class_labels = observed_class_labels.view(-1)
        elif input_contract_case == "input_nan":
            input_features = input_features.clone()
            input_features[0, 0] = float("nan")
        elif input_contract_case == "label_inf":
            observed_class_labels = torch.full_like(observed_class_labels, float("inf"))
        elif input_contract_case == "input_type":
            input_features = [[1.0, 2.0]]
        elif input_contract_case == "input_dtype":
            input_features = input_features.double()
        elif input_contract_case == "label_dtype":
            observed_class_labels = observed_class_labels.long()
        elif input_contract_case == "input_sparse":
            input_features = input_features.to_sparse()
        elif input_contract_case == "input_meta":
            input_features = input_features.to("meta")
        elif input_contract_case in ("input_nested", "label_nested"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                training_tensor = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)])
            if input_contract_case == "input_nested":
                input_features = training_tensor
            else:
                observed_class_labels = training_tensor
        else:
            observed_class_labels = torch.full_like(observed_class_labels, {
                "binary_below": -0.1, "binary_above": 1.1, "multi_fractional": 1.5, "multi_outside": 4.0,
            }[input_contract_case])
        invalid_inputs["participating_training_batches"] = (participating_training_batches[0],
            ParticipatingModelTrainingBatch(classifier=training_batch.classifier,
                concept_specific_parameter_optimizer=training_batch.concept_specific_parameter_optimizer,
                input_features=input_features, observed_class_labels=observed_class_labels))
    snapshot_before_update = capture_joint_update_state(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer)
    global_torch_random_state = torch.get_rng_state()
    expected_settings = dict(vars(local_training_settings))
    # 不正入力と同じ参照へ書かないことも確認する（meta/nestedは値照合不可）。
    expected_input_features = input_features.detach().clone() if isinstance(input_features, torch.Tensor) else deepcopy(input_features)
    expected_observed_class_labels = observed_class_labels.detach().clone()
    with torch.set_grad_enabled(input_contract_case != "grad_disabled"):
        with pytest.raises(ValueError, match=expected_error_field):
            perform_joint_model_parameter_update(**invalid_inputs)
    for parameter_optimizer in parameter_optimizers:
        parameter_optimizer.zero_grad.assert_not_called()
        parameter_optimizer.step.assert_not_called()
    snapshot_after_update = capture_joint_update_state(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer)
    assert_nested_state_equal(snapshot_after_update, snapshot_before_update)
    assert torch.equal(torch.get_rng_state(), global_torch_random_state)
    assert vars(local_training_settings) == expected_settings
    if not isinstance(input_features, torch.Tensor):
        assert input_features == expected_input_features
    elif input_features.is_nested:
        for parameter_value, expected_parameter_value in zip(input_features.unbind(), expected_input_features.unbind()):
            assert torch.equal(parameter_value, expected_parameter_value)
    elif input_features.device.type == "meta":
        assert input_features.shape == expected_input_features.shape
    elif input_features.layout == torch.sparse_coo:
        assert torch.equal(input_features.to_dense(), expected_input_features.to_dense())
    else:
        torch.testing.assert_close(input_features, expected_input_features, rtol=0, atol=0, equal_nan=True)
    if observed_class_labels.is_nested:
        for parameter_value, expected_parameter_value in zip(observed_class_labels.unbind(), expected_observed_class_labels.unbind()):
            assert torch.equal(parameter_value, expected_parameter_value)
    else:
        assert torch.equal(observed_class_labels, expected_observed_class_labels)


def test_joint_model_parameter_update_empty_participation_preserves_state(monkeypatch):
    participating_training_batches, shared_parameter_optimizer, legacy_client = build_joint_update_oracle_pair(
        class_count=2, batch_sample_counts=(3,), optimizer_variant="amsgrad", monkeypatch=monkeypatch)
    local_training_settings = LocalTrainingSettings(
        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients")
    perform_joint_model_parameter_update(local_training_settings=local_training_settings,
        shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
        shared_parameter_optimizer=shared_parameter_optimizer, participating_training_batches=participating_training_batches,
        update_shared_features=True)
    snapshot_before_update = capture_joint_update_state(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer)
    monkeypatch.setattr(shared_parameter_optimizer, "zero_grad", Mock(wraps=shared_parameter_optimizer.zero_grad))
    monkeypatch.setattr(shared_parameter_optimizer, "step", Mock(wraps=shared_parameter_optimizer.step))
    with torch.no_grad():
        assert perform_joint_model_parameter_update(local_training_settings=local_training_settings,
            shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
            shared_parameter_optimizer=shared_parameter_optimizer, participating_training_batches=(),
            update_shared_features=True) is None
        # 空列は共有optimizerの詳細や外側gradを要求しない。
        assert perform_joint_model_parameter_update(local_training_settings=local_training_settings,
            shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
            shared_parameter_optimizer=None, participating_training_batches=(), update_shared_features=False) is None
    shared_parameter_optimizer.zero_grad.assert_not_called()
    shared_parameter_optimizer.step.assert_not_called()
    assert_nested_state_equal(capture_joint_update_state(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer), snapshot_before_update)


@pytest.mark.parametrize("update_shared_features", [True, False])
def test_joint_model_parameter_update_without_shared_parameters(update_shared_features, monkeypatch):
    classifier = ResidualAdapterClassifier(model_architecture_settings=ModelArchitectureSettings(
        model_architecture_name="shared_backbone_residual_adapter", residual_adapter_requested_rank=2),
        input_feature_count=2, hidden_layer_widths=(), class_count=2)
    expected_classifier = deepcopy(classifier)
    optimizer_settings = AdamParameterOptimizerSettings(learning_rate=0.01, weight_decay=0.001, adam_variant="amsgrad")
    concept_specific_parameter_optimizer = create_parameter_optimizer(
        parameters=tuple(classifier.residual_adapter.parameters()) + tuple(classifier.classification_layer.parameters()),
        optimizer_settings=optimizer_settings)
    expected_parameter_optimizer = create_parameter_optimizer(parameters=tuple(expected_classifier.parameters()), optimizer_settings=optimizer_settings)
    training_batch = ParticipatingModelTrainingBatch(classifier=classifier,
        concept_specific_parameter_optimizer=concept_specific_parameter_optimizer,
        input_features=torch.tensor([[0.0, 1.0], [2.0, -1.0]]), observed_class_labels=torch.tensor([[0.25], [0.75]]))
    local_training_settings = LocalTrainingSettings(
        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients")
    assert tuple(classifier.feature_extractor.parameters()) == ()
    for _ in range(2):
        # 旧optimizer空列失敗の範囲とは区別し、独立モデルの標準loss/stepで期待値を作る。
        expected_parameter_optimizer.zero_grad()
        model_loss = torch.nn.BCELoss()(expected_classifier(training_batch.input_features), training_batch.observed_class_labels)
        model_loss.backward()
        expected_parameter_optimizer.step()
        actual_joint_loss = perform_joint_model_parameter_update(local_training_settings=local_training_settings,
            shared_feature_extractor=classifier.feature_extractor, shared_parameter_optimizer=None,
            participating_training_batches=(training_batch,), update_shared_features=update_shared_features)
        assert actual_joint_loss == model_loss.item()
        for parameter, expected_parameter in zip(classifier.parameters(), expected_classifier.parameters()):
            assert torch.equal(parameter, expected_parameter)
            assert torch.equal(parameter.grad, expected_parameter.grad)
        assert_nested_state_equal(concept_specific_parameter_optimizer.state_dict(), expected_parameter_optimizer.state_dict())
    snapshot_before_update = capture_joint_update_state(participating_training_batches=(training_batch,), shared_parameter_optimizer=None)
    monkeypatch.setattr(concept_specific_parameter_optimizer, "zero_grad", Mock(wraps=concept_specific_parameter_optimizer.zero_grad))
    monkeypatch.setattr(concept_specific_parameter_optimizer, "step", Mock(wraps=concept_specific_parameter_optimizer.step))
    with pytest.raises(ValueError, match="shared_parameter_optimizer=None"):
        perform_joint_model_parameter_update(local_training_settings=local_training_settings,
            shared_feature_extractor=classifier.feature_extractor, shared_parameter_optimizer=concept_specific_parameter_optimizer,
            participating_training_batches=(training_batch,), update_shared_features=update_shared_features)
    concept_specific_parameter_optimizer.zero_grad.assert_not_called()
    concept_specific_parameter_optimizer.step.assert_not_called()
    assert_nested_state_equal(capture_joint_update_state(participating_training_batches=(training_batch,), shared_parameter_optimizer=None),
        snapshot_before_update)


def test_joint_model_parameter_update_accepts_soft_binary_labels_and_noncontiguous_batches(monkeypatch):
    participating_training_batches, shared_parameter_optimizer, legacy_client = build_joint_update_oracle_pair(
        class_count=2, batch_sample_counts=(2, 5), optimizer_variant="amsgrad", monkeypatch=monkeypatch)
    for training_batch, (model_id, _, _) in zip(participating_training_batches, legacy_client._sample_training_batches()):
        input_features = training_batch.input_features.t().contiguous().t()
        observed_class_labels = torch.linspace(0.0, 1.0, len(input_features)).reshape(-1, 1)
        assert not input_features.is_contiguous()
        object.__setattr__(training_batch, "input_features", input_features)
        object.__setattr__(training_batch, "observed_class_labels", observed_class_labels)
        legacy_client._sample_training_batches()[0 if model_id == 4 else 1] = (model_id, input_features.clone(), observed_class_labels.clone())
    expected_joint_loss = run_legacy_joint_update(legacy_client=legacy_client, update_shared_features=True)
    actual_joint_loss = perform_joint_model_parameter_update(local_training_settings=LocalTrainingSettings(
        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients"),
        shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
        shared_parameter_optimizer=shared_parameter_optimizer, participating_training_batches=participating_training_batches,
        update_shared_features=True)
    assert actual_joint_loss == expected_joint_loss
    assert_joint_update_states_equal(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer, legacy_client=legacy_client)


def test_joint_model_parameter_update_accepts_ordered_multiple_optimizer_groups(monkeypatch):
    participating_training_batches, shared_parameter_optimizer, legacy_client = build_joint_update_oracle_pair(
        class_count=4, batch_sample_counts=(2, 5), optimizer_variant="sgd", monkeypatch=monkeypatch)
    parameter_optimizers = (shared_parameter_optimizer,) + tuple(
        training_batch.concept_specific_parameter_optimizer for training_batch in participating_training_batches)
    for parameter_optimizer in parameter_optimizers:
        parameter_groups = dict(parameter_optimizer.param_groups[0])
        expected_parameters = tuple(parameter_groups["params"])
        parameter_optimizer.param_groups[0]["params"] = list(expected_parameters[:2])
        parameter_groups["params"] = list(expected_parameters[2:])
        parameter_optimizer.add_param_group(parameter_groups)
    expected_joint_loss = run_legacy_joint_update(legacy_client=legacy_client, update_shared_features=True)
    actual_joint_loss = perform_joint_model_parameter_update(local_training_settings=LocalTrainingSettings(
        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients"),
        shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
        shared_parameter_optimizer=shared_parameter_optimizer, participating_training_batches=participating_training_batches,
        update_shared_features=True)
    assert actual_joint_loss == expected_joint_loss
    for training_batch, legacy_model in zip(participating_training_batches, legacy_client.models.values()):
        for parameter, expected_parameter in zip(training_batch.classifier.parameters(), legacy_model.parameters()):
            assert torch.equal(parameter, expected_parameter)
            assert torch.equal(parameter.grad, expected_parameter.grad)
    assert all(len(parameter_optimizer.param_groups) == 2 for parameter_optimizer in parameter_optimizers)


@pytest.mark.parametrize("require_trainable", [True, False])
def test_joint_model_parameter_update_preserves_frozen_shared_optimizer_state(require_trainable, monkeypatch):
    participating_training_batches, shared_parameter_optimizer, legacy_client = build_joint_update_oracle_pair(
        class_count=2, batch_sample_counts=(2, 5), optimizer_variant="amsgrad", monkeypatch=monkeypatch)
    invalid_inputs = dict(local_training_settings=LocalTrainingSettings(
        local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
        shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients"),
        shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
        shared_parameter_optimizer=shared_parameter_optimizer, participating_training_batches=participating_training_batches,
        update_shared_features=True)
    perform_joint_model_parameter_update(**invalid_inputs)
    shared_parameters_before_update = tuple(parameter.detach().clone() for parameter in invalid_inputs["shared_feature_extractor"].parameters())
    snapshot_before_update = capture_joint_update_state(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer)
    for parameter in invalid_inputs["shared_feature_extractor"].parameters():
        parameter.requires_grad_(require_trainable)
        parameter.grad = torch.full_like(parameter, 0.37)
    invalid_inputs["update_shared_features"] = False
    perform_joint_model_parameter_update(**invalid_inputs)
    snapshot_after_update = capture_joint_update_state(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer)
    assert_nested_state_equal(snapshot_after_update[1], snapshot_before_update[1])
    for parameter, expected_parameter_value in zip(invalid_inputs["shared_feature_extractor"].parameters(), shared_parameters_before_update):
        assert torch.equal(parameter, expected_parameter_value)
        assert parameter.grad is None
        assert parameter.requires_grad is require_trainable
    for training_batch_index in range(len(participating_training_batches)):
        for parameter_id, parameter_value in snapshot_before_update[0][training_batch_index]["optimizer"]["state"].items():
            assert snapshot_after_update[0][training_batch_index]["optimizer"]["state"][parameter_id]["step"].item() == parameter_value["step"].item() + 1


def test_participating_model_training_batch_is_frozen_and_explicit(monkeypatch):
    participating_training_batches, shared_parameter_optimizer, legacy_client = build_joint_update_oracle_pair(
        class_count=2, batch_sample_counts=(2,), optimizer_variant="sgd", monkeypatch=monkeypatch)
    training_batch = participating_training_batches[0]
    for training_batch_field in fields(training_batch):
        assert training_batch_field.kw_only
        assert training_batch_field.default is MISSING
        assert training_batch_field.default_factory is MISSING
    with pytest.raises(FrozenInstanceError):
        training_batch.classifier = None
    with pytest.raises(TypeError):
        ParticipatingModelTrainingBatch()
    with pytest.raises(TypeError):
        ParticipatingModelTrainingBatch(training_batch.classifier, training_batch.concept_specific_parameter_optimizer,
            training_batch.input_features, training_batch.observed_class_labels)


@pytest.mark.parametrize("update_shared_features", [True, False])
def test_joint_model_parameter_update_preserves_environment_and_borrowed_inputs(update_shared_features, monkeypatch):
    participating_training_batches, shared_parameter_optimizer, legacy_client = build_joint_update_oracle_pair(
        class_count=4, batch_sample_counts=(2, 5), optimizer_variant="sgd", monkeypatch=monkeypatch)
    snapshot_before_update = capture_joint_update_state(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer)
    global_torch_random_state = torch.get_rng_state()
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    original_default_dtype = torch.get_default_dtype()
    original_default_device = torch.get_default_device()
    expected_joint_loss = run_legacy_joint_update(legacy_client=legacy_client, update_shared_features=update_shared_features)
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            # モデル構築もambientに依存せず明示CPU32。生成分のCPU乱数だけ復元する。
            classifier = ResidualAdapterClassifier(model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="shared_backbone_residual_adapter", residual_adapter_requested_rank=2),
                input_feature_count=2, hidden_layer_widths=(4,), class_count=4)
            assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in classifier.parameters())
            torch.set_rng_state(global_torch_random_state)
            actual_joint_loss = perform_joint_model_parameter_update(local_training_settings=LocalTrainingSettings(
                local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients"),
                shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
                shared_parameter_optimizer=shared_parameter_optimizer,
                participating_training_batches=participating_training_batches, update_shared_features=update_shared_features)
            assert torch.is_grad_enabled()
            assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_device() == torch.device("meta")
            assert torch.equal(torch.get_rng_state(), global_torch_random_state)
    finally:
        torch.set_default_dtype(original_default_dtype)
        torch.set_rng_state(global_torch_random_state)
    assert actual_joint_loss == expected_joint_loss
    assert_joint_update_states_equal(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer, legacy_client=legacy_client)
    snapshot_after_update = capture_joint_update_state(participating_training_batches=participating_training_batches,
        shared_parameter_optimizer=shared_parameter_optimizer)
    for training_batch_index in range(len(participating_training_batches)):
        for parameter_name in ("input_features", "observed_class_labels", "modules"):
            assert_nested_state_equal(snapshot_after_update[0][training_batch_index][parameter_name],
                snapshot_before_update[0][training_batch_index][parameter_name])
        assert_nested_state_equal(snapshot_after_update[0][training_batch_index]["optimizer"]["param_groups"],
            snapshot_before_update[0][training_batch_index]["optimizer"]["param_groups"])
    assert_nested_state_equal(snapshot_after_update[1]["param_groups"], snapshot_before_update[1]["param_groups"])
    assert torch.get_default_dtype() == original_default_dtype
    assert torch.get_default_device() == original_default_device
    assert random.getstate() == global_python_random_state
    expected_numpy_random_state = np.random.get_state()
    assert expected_numpy_random_state[0] == global_numpy_random_state[0]
    assert np.array_equal(expected_numpy_random_state[1], global_numpy_random_state[1])
    assert expected_numpy_random_state[2:] == global_numpy_random_state[2:]
