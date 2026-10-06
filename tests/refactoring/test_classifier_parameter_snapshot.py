"""分類器snapshotの独立性と固定旧モデルとの対応を確認する。"""

import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from test_joint_model_parameter_update import (
    assert_joint_update_states_equal,
    assert_nested_state_equal,
    build_joint_update_oracle_pair,
    run_legacy_joint_update,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    build_attachment_classifier,
    snapshot_parameter_values_and_gradients,
)

from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import (
    select_candidate_initial_parameter_snapshot,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)


class ResidualAdapterClassifierSubclass(ResidualAdapterClassifier):
    pass


@pytest.mark.parametrize("class_count", [2, 4, 10])
def test_snapshot_matches_legacy_values_and_native_key_order(class_count, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        training_batches, _, legacy_client = build_joint_update_oracle_pair(
            class_count=class_count,
            batch_sample_counts=(3,),
            optimizer_variant="standard",
            monkeypatch=monkeypatch,
        )
        classifier = training_batches[0].classifier
        legacy_parameter_snapshot = legacy_client.models[4].get_params()
        parameter_snapshot = snapshot_classifier_parameters(classifier=classifier)
        prefix_pairs = (
            ("backbone.net.", "feature_extractor.hidden_layers."),
            ("adapter.down.", "residual_adapter.feature_compression."),
            ("adapter.up.", "residual_adapter.feature_expansion."),
            ("head.", "classification_layer."),
        )
        expected_parameter_values = {}
        for legacy_parameter_name, parameter_values in legacy_parameter_snapshot.items():
            native_parameter_name = legacy_parameter_name
            for legacy_prefix, native_prefix in prefix_pairs:
                native_parameter_name = native_parameter_name.replace(legacy_prefix, native_prefix)
            expected_parameter_values[native_parameter_name] = parameter_values
        assert type(parameter_snapshot) is dict
        assert tuple(parameter_snapshot) == tuple(classifier.state_dict())
        assert tuple(parameter_snapshot) == tuple(expected_parameter_values)
        assert not hasattr(parameter_snapshot, "_metadata")
        for parameter_name, parameter_values in classifier.named_parameters():
            assert torch.equal(
                parameter_snapshot[parameter_name], expected_parameter_values[parameter_name]
            )
            assert parameter_snapshot[parameter_name].shape == parameter_values.shape
            assert parameter_snapshot[parameter_name].data_ptr() != parameter_values.data_ptr()
            assert not parameter_snapshot[parameter_name].requires_grad
            assert parameter_snapshot[parameter_name].grad_fn is None
        with torch.no_grad():
            first_parameter = next(legacy_client.models[4].parameters())
            first_parameter.add_(1)
        assert not torch.equal(first_parameter, next(iter(legacy_parameter_snapshot.values())))


@pytest.mark.parametrize("hidden_layer_widths", [(), (3,), (5, 4)])
def test_snapshot_is_independent_in_both_directions_and_across_calls(hidden_layer_widths):
    classifier = build_attachment_classifier(hidden_layer_widths=hidden_layer_widths)
    first_parameter = classifier.residual_adapter.feature_compression.weight
    # 非連続strided parameterでも値とshapeを保持する。
    first_parameter.data = first_parameter.detach().t().contiguous().t()
    first_parameter_snapshot = snapshot_classifier_parameters(classifier=classifier)
    second_parameter_snapshot = snapshot_classifier_parameters(classifier=classifier)
    for parameter_name in first_parameter_snapshot:
        assert (
            first_parameter_snapshot[parameter_name].data_ptr()
            != second_parameter_snapshot[parameter_name].data_ptr()
        )
    expected_parameter_values = first_parameter.detach().clone()
    with torch.no_grad():
        first_parameter.add_(2)
    parameter_name = "residual_adapter.feature_compression.weight"
    assert torch.equal(first_parameter_snapshot[parameter_name], expected_parameter_values)
    first_parameter_snapshot[parameter_name].add_(3)
    assert torch.equal(first_parameter, expected_parameter_values + 2)
    assert torch.equal(second_parameter_snapshot[parameter_name], expected_parameter_values)
    current_parameter_snapshot = snapshot_classifier_parameters(classifier=classifier)
    assert torch.equal(current_parameter_snapshot[parameter_name], first_parameter)
    first_parameter_snapshot.clear()
    assert parameter_name in current_parameter_snapshot


@pytest.mark.parametrize("previous_grad_mode", [True, False])
def test_snapshot_preserves_model_state_and_caller_environment(previous_grad_mode):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier()
        second_classifier = ResidualAdapterClassifier(
            model_architecture_settings=classifier.model_architecture_settings,
            input_feature_count=2,
            hidden_layer_widths=(5, 4),
            class_count=2,
            shared_feature_extractor=classifier.feature_extractor,
        )
        for parameter_values in classifier.parameters():
            parameter_values.grad = torch.ones_like(parameter_values)
        next(classifier.parameters()).grad = None
        classifier.train(False)
        classifier.residual_adapter.train(True)
        previous_training_flags = tuple(module.training for module in classifier.modules())
        previous_parameter_state = snapshot_parameter_values_and_gradients(classifier.parameters())
        previous_feature_extractor = classifier.feature_extractor
        previous_random_states = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
        )
        previous_default_dtype = torch.get_default_dtype()
        previous_default_device = torch.get_default_device()
        forward_calls = []
        forward_hook = classifier.register_forward_hook(lambda *args: forward_calls.append(True))
        try:
            torch.set_default_dtype(torch.float64)
            with torch.device("meta"), torch.set_grad_enabled(previous_grad_mode):
                parameter_snapshot = snapshot_classifier_parameters(classifier=classifier)
                assert torch.is_grad_enabled() is previous_grad_mode
                assert torch.get_default_device().type == "meta"
                assert torch.get_default_dtype() == torch.float64
            assert not forward_calls
            assert classifier.feature_extractor is previous_feature_extractor
            assert second_classifier.feature_extractor is previous_feature_extractor
            assert (
                tuple(module.training for module in classifier.modules()) == previous_training_flags
            )
            assert_parameter_values_and_gradients_unchanged(previous_parameter_state)
            assert random.getstate() == previous_random_states[0]
            assert np.random.get_state()[0] == previous_random_states[1][0]
            assert np.array_equal(np.random.get_state()[1], previous_random_states[1][1])
            assert np.random.get_state()[2:] == previous_random_states[1][2:]
            assert torch.equal(torch.get_rng_state(), previous_random_states[2])
            assert all(
                parameter_values.device.type == "cpu"
                for parameter_values in parameter_snapshot.values()
            )
            with torch.inference_mode():
                parameter_snapshot = snapshot_classifier_parameters(classifier=classifier)
                assert torch.is_inference_mode_enabled()
            assert all(
                not parameter_values.requires_grad
                for parameter_values in parameter_snapshot.values()
            )
        finally:
            forward_hook.remove()
            torch.set_default_dtype(previous_default_dtype)
        assert torch.get_default_device() == previous_default_device


@pytest.mark.parametrize("invalid_classifier", [None, object(), torch.nn.Linear(2, 1)])
def test_snapshot_rejects_invalid_classifier(invalid_classifier):
    with pytest.raises(TypeError, match="classifier"):
        snapshot_classifier_parameters(classifier=invalid_classifier)
    classifier = build_attachment_classifier()
    classifier.__class__ = ResidualAdapterClassifierSubclass
    with pytest.raises(TypeError, match="classifier"):
        snapshot_classifier_parameters(classifier=classifier)


@pytest.mark.parametrize("invalid_parameter_kind", ["nan", "inf", "float64", "meta", "sparse"])
def test_snapshot_rejects_invalid_parameter_values(invalid_parameter_kind):
    classifier = build_attachment_classifier()
    first_parameter = classifier.classification_layer.weight
    parameter_values = first_parameter.detach().clone()
    if invalid_parameter_kind in ("nan", "inf"):
        parameter_values[0, 0] = float(invalid_parameter_kind)
    elif invalid_parameter_kind == "float64":
        parameter_values = parameter_values.double()
    elif invalid_parameter_kind == "meta":
        parameter_values = parameter_values.to("meta")
    else:
        parameter_values = parameter_values.to_sparse()
    classifier.classification_layer.weight = torch.nn.Parameter(parameter_values)
    first_parameter = classifier.classification_layer.weight
    first_parameter.grad = first_parameter.detach().clone()
    expected_parameter_values = first_parameter.detach().clone()
    previous_parameter_state = snapshot_parameter_values_and_gradients(
        parameter_values
        for parameter_values in classifier.parameters()
        if parameter_values is not first_parameter
    )
    previous_parameter_gradients = first_parameter.grad
    with (
        torch.set_grad_enabled(True),
        pytest.raises(ValueError, match="classification_layer.weight"),
    ):
        snapshot_classifier_parameters(classifier=classifier)
    assert torch.is_grad_enabled()
    assert classifier.classification_layer.weight is first_parameter
    assert first_parameter.grad is previous_parameter_gradients
    assert_parameter_values_and_gradients_unchanged(previous_parameter_state)
    if invalid_parameter_kind != "meta":
        torch.testing.assert_close(
            first_parameter.to_dense(),
            expected_parameter_values.to_dense(),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            first_parameter.grad.to_dense(),
            expected_parameter_values.to_dense(),
            rtol=0,
            atol=0,
            equal_nan=True,
        )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [False, True])
def test_training_snapshot_and_candidate_initialization_match_legacy(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        training_batches, shared_parameter_optimizer, legacy_client = (
            build_joint_update_oracle_pair(
                class_count=class_count,
                batch_sample_counts=(3, 5),
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        )
        for _ in range(3):
            assert perform_joint_model_parameter_update(
                local_training_settings=local_training_settings,
                shared_feature_extractor=training_batches[0].classifier.feature_extractor,
                shared_parameter_optimizer=shared_parameter_optimizer,
                participating_training_batches=training_batches,
                update_shared_features=update_shared_features,
            ) == run_legacy_joint_update(
                legacy_client=legacy_client, update_shared_features=update_shared_features
            )
            assert_joint_update_states_equal(
                participating_training_batches=training_batches,
                shared_parameter_optimizer=shared_parameter_optimizer,
                legacy_client=legacy_client,
            )
        previous_optimizer_state = (
            deepcopy(shared_parameter_optimizer.state_dict()),
            tuple(
                deepcopy(training_batch.concept_specific_parameter_optimizer.state_dict())
                for training_batch in training_batches
            ),
        )
        previous_torch_random_state = torch.get_rng_state().clone()
        available_parameter_snapshots_by_model_id = {}
        for model_id, training_batch in zip(legacy_client.models, training_batches):
            classifier = training_batch.classifier
            previous_parameter_state = snapshot_parameter_values_and_gradients(
                classifier.parameters()
            )
            available_parameter_snapshots_by_model_id[model_id] = snapshot_classifier_parameters(
                classifier=classifier
            )
            legacy_parameter_snapshot = legacy_client.models[model_id].get_params()
            expected_parameter_values = {
                legacy_parameter_name.replace("backbone.net.", "feature_extractor.hidden_layers.")
                .replace("adapter.down.", "residual_adapter.feature_compression.")
                .replace("adapter.up.", "residual_adapter.feature_expansion.")
                .replace("head.", "classification_layer."): parameter_values
                for legacy_parameter_name, parameter_values in legacy_parameter_snapshot.items()
            }
            assert_nested_state_equal(
                available_parameter_snapshots_by_model_id[model_id], expected_parameter_values
            )
            assert_parameter_values_and_gradients_unchanged(previous_parameter_state)
        assert torch.equal(torch.get_rng_state(), previous_torch_random_state)
        initialization_settings = CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="assigned_training_model"
        )
        for model_id, training_batch in zip(legacy_client.models, training_batches):
            initialized_parameter_snapshot = select_candidate_initial_parameter_snapshot(
                settings=initialization_settings,
                available_parameter_snapshots_by_model_id=available_parameter_snapshots_by_model_id,
                current_training_model_id=model_id,
                evaluated_mean_losses_by_model_id=(),
            )
            assert initialized_parameter_snapshot is not None
            assert_nested_state_equal(
                initialized_parameter_snapshot, available_parameter_snapshots_by_model_id[model_id]
            )
            restored_classifier = build_attachment_classifier(class_count=class_count)
            restored_classifier.load_state_dict(initialized_parameter_snapshot)
            assert_nested_state_equal(
                dict(restored_classifier.state_dict()),
                available_parameter_snapshots_by_model_id[model_id],
            )
            with torch.no_grad():
                assert torch.equal(
                    restored_classifier(training_batch.input_features),
                    training_batch.classifier(training_batch.input_features),
                )
            for parameter_name, parameter_values in initialized_parameter_snapshot.items():
                assert (
                    parameter_values.data_ptr()
                    != available_parameter_snapshots_by_model_id[model_id][
                        parameter_name
                    ].data_ptr()
                )
                parameter_values.add_(1)
            assert_nested_state_equal(
                dict(restored_classifier.state_dict()),
                available_parameter_snapshots_by_model_id[model_id],
            )
        assert_nested_state_equal(
            shared_parameter_optimizer.state_dict(), previous_optimizer_state[0]
        )
        for training_batch, expected_parameter_values in zip(
            training_batches, previous_optimizer_state[1]
        ):
            assert_nested_state_equal(
                training_batch.concept_specific_parameter_optimizer.state_dict(),
                expected_parameter_values,
            )
        assert_joint_update_states_equal(
            participating_training_batches=training_batches,
            shared_parameter_optimizer=shared_parameter_optimizer,
            legacy_client=legacy_client,
        )
