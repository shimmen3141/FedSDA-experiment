"""実旧の共同学習反復と新しい借用状態の接続を照合する。"""

import random
from copy import deepcopy
from dataclasses import MISSING, FrozenInstanceError, fields
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_joint_model_parameter_update import (
    assert_joint_update_states_equal,
    assert_nested_state_equal,
    build_joint_update_oracle_pair,
)

from federated_drift_experiment.clients.shared_backbone import (
    _SharedRepresentationFedSDAClientMixin,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training import (
    held_model_joint_training_iterations as training_iteration_module,
)
from federated_learning_experiments.learning.training.held_model_joint_training_iterations import (
    perform_held_model_joint_training_iterations,
)
from federated_learning_experiments.learning.training.held_model_training_binding import (
    HeldModelTrainingBinding,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ModelTrainingSampleCollection,
    ObservedTrainingSample,
)


def build_training_iteration_oracle_pair(*, class_count, optimizer_variant, monkeypatch):
    participating_training_batches, shared_parameter_optimizer, legacy_client = (
        build_joint_update_oracle_pair(
            class_count=class_count,
            batch_sample_counts=(9, 100),
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
    )
    ordered_model_training_samples = tuple(
        ModelTrainingSampleCollection(
            model_id=model_id,
            training_samples=tuple(
                ObservedTrainingSample(
                    input_features=training_batch.input_features[
                        training_sample_index : training_sample_index + 1
                    ],
                    observed_class_labels=training_batch.observed_class_labels[
                        training_sample_index : training_sample_index + 1
                    ],
                )
                for training_sample_index in range(len(training_batch.input_features))
            ),
        )
        for model_id, training_batch in zip((4, -7), participating_training_batches)
    )
    legacy_client.train_data_store = {
        model_training_samples.model_id: [
            (training_sample.input_features, training_sample.observed_class_labels, 0)
            for training_sample in model_training_samples.training_samples
        ]
        for model_training_samples in ordered_model_training_samples
    }
    held_model_training_bindings = tuple(
        reversed(
            tuple(
                HeldModelTrainingBinding(
                    model_id=model_id,
                    classifier=training_batch.classifier,
                    concept_specific_parameter_optimizer=training_batch.concept_specific_parameter_optimizer,
                )
                for model_id, training_batch in zip((4, -7), participating_training_batches)
            )
        )
    )
    return (
        participating_training_batches,
        shared_parameter_optimizer,
        legacy_client,
        ordered_model_training_samples,
        held_model_training_bindings,
    )


def run_legacy_training_iterations(
    *, legacy_client, iteration_count, update_shared_features, initial_random_state
):
    sampled_batch_history = []
    legacy_weighted_loss_history = []

    def sample_legacy_training_batches():
        legacy_training_batches = _SharedRepresentationFedSDAClientMixin._sample_training_batches(
            legacy_client
        )
        sampled_batch_history.append(
            tuple(
                (model_id, input_features.clone(), observed_class_labels.clone())
                for model_id, input_features, observed_class_labels in legacy_training_batches
            )
        )
        return legacy_training_batches

    legacy_client._sample_training_batches = Mock(side_effect=sample_legacy_training_batches)
    loss_hooks = [
        classifier.loss_fn.register_forward_hook(
            lambda classifier, training_inputs, joint_update_loss: (
                legacy_weighted_loss_history.append(
                    joint_update_loss.detach().clone() * len(training_inputs[1])
                )
            )
        )
        for classifier in legacy_client.models.values()
    ]
    global_python_random_state = random.getstate()
    try:
        random.setstate(initial_random_state)
        _SharedRepresentationFedSDAClientMixin._train_heads_together(
            legacy_client, count_multiplier=iteration_count, update_backbone=update_shared_features
        )
        expected_random_state = random.getstate()
    finally:
        random.setstate(global_python_random_state)
        for loss_hook in loss_hooks:
            loss_hook.remove()
    expected_losses = []
    loss_offset = 0
    for legacy_training_batches in sampled_batch_history:
        participating_model_count = len(legacy_training_batches)
        if not participating_model_count:
            continue
        total_batch_sample_count = sum(
            len(input_features) for _, input_features, _ in legacy_training_batches
        )
        expected_losses.append(
            float(
                (
                    sum(
                        legacy_weighted_loss_history[
                            loss_offset : loss_offset + participating_model_count
                        ]
                    )
                    / total_batch_sample_count
                ).item()
            )
        )
        loss_offset += participating_model_count
    return tuple(expected_losses), expected_random_state, sampled_batch_history


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
@pytest.mark.parametrize("iteration_count", [0, 1, 4])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_joint_training_iterations_match_legacy(
    class_count, optimizer_variant, update_shared_features, iteration_count, batch_size, monkeypatch
):
    global_python_random_state = random.getstate()
    global_torch_random_state = torch.get_rng_state()
    try:
        (
            participating_training_batches,
            shared_parameter_optimizer,
            legacy_client,
            ordered_model_training_samples,
            held_model_training_bindings,
        ) = build_training_iteration_oracle_pair(
            class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
        )
        legacy_client.batch_size = batch_size
        python_random_generator = random.Random(731)
        expected_losses, expected_random_state, sampled_batch_history = (
            run_legacy_training_iterations(
                legacy_client=legacy_client,
                iteration_count=iteration_count,
                update_shared_features=update_shared_features,
                initial_random_state=python_random_generator.getstate(),
            )
        )
        original_sampling_function = (
            training_iteration_module.sample_training_batches_for_held_models
        )
        new_sampled_batch_history = []
        monkeypatch.setattr(
            training_iteration_module,
            "sample_training_batches_for_held_models",
            Mock(
                side_effect=lambda **sampling_request: (
                    new_sampled_batch_history.append(original_sampling_function(**sampling_request))
                    or new_sampled_batch_history[-1]
                )
            ),
        )
        actual_losses = perform_held_model_joint_training_iterations(
            requested_joint_update_iteration_count=iteration_count,
            held_model_training_bindings=held_model_training_bindings,
            ordered_model_training_samples=ordered_model_training_samples,
            batch_sample_count=batch_size,
            python_random_generator=python_random_generator,
            local_training_settings=LocalTrainingSettings(
                local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
            ),
            shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
            shared_parameter_optimizer=shared_parameter_optimizer,
            update_shared_features=update_shared_features,
        )
        assert actual_losses == expected_losses
        assert python_random_generator.getstate() == expected_random_state
        assert len(new_sampled_batch_history) == len(sampled_batch_history) == iteration_count
        for sampled_training_batches, legacy_training_batches in zip(
            new_sampled_batch_history, sampled_batch_history
        ):
            assert tuple(batch.model_id for batch in sampled_training_batches) == (4, -7)
            for sampled_training_batch, (model_id, input_features, observed_class_labels) in zip(
                sampled_training_batches, legacy_training_batches
            ):
                assert sampled_training_batch.model_id == model_id
                assert torch.equal(sampled_training_batch.input_features, input_features)
                assert torch.equal(
                    sampled_training_batch.observed_class_labels, observed_class_labels
                )
        assert_joint_update_states_equal(
            participating_training_batches=participating_training_batches,
            shared_parameter_optimizer=shared_parameter_optimizer,
            legacy_client=legacy_client,
        )
        assert random.getstate() == global_python_random_state
    finally:
        random.setstate(global_python_random_state)
        torch.set_rng_state(global_torch_random_state)


@pytest.fixture
def keyword_arguments(monkeypatch):
    global_torch_random_state = torch.get_rng_state()
    try:
        (
            participating_training_batches,
            shared_parameter_optimizer,
            _,
            ordered_model_training_samples,
            held_model_training_bindings,
        ) = build_training_iteration_oracle_pair(
            class_count=2, optimizer_variant="amsgrad", monkeypatch=monkeypatch
        )
        yield dict(
            requested_joint_update_iteration_count=4,
            held_model_training_bindings=held_model_training_bindings,
            ordered_model_training_samples=ordered_model_training_samples,
            batch_sample_count=3,
            python_random_generator=random.Random(731),
            local_training_settings=LocalTrainingSettings(
                local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
            ),
            shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
            shared_parameter_optimizer=shared_parameter_optimizer,
            update_shared_features=True,
        )
    finally:
        torch.set_rng_state(global_torch_random_state)


@pytest.mark.parametrize(
    "invalid_iteration_count", [True, False, -1, 1.5, None, type("DerivedInteger", (int,), {})(1)]
)
def test_joint_training_iterations_reject_before_sampling(
    keyword_arguments, monkeypatch, invalid_iteration_count
):
    generator_state_before = keyword_arguments["python_random_generator"].getstate()
    sampling_calls = Mock(wraps=training_iteration_module.sample_training_batches_for_held_models)
    update_calls = Mock(wraps=training_iteration_module.perform_joint_model_parameter_update)
    monkeypatch.setattr(
        training_iteration_module, "sample_training_batches_for_held_models", sampling_calls
    )
    monkeypatch.setattr(
        training_iteration_module, "perform_joint_model_parameter_update", update_calls
    )
    keyword_arguments["requested_joint_update_iteration_count"] = invalid_iteration_count
    with pytest.raises(ValueError, match="requested_joint_update_iteration_count"):
        perform_held_model_joint_training_iterations(**keyword_arguments)
    sampling_calls.assert_not_called()
    update_calls.assert_not_called()
    assert keyword_arguments["python_random_generator"].getstate() == generator_state_before


@pytest.mark.parametrize(
    "binding_contract_case", ["list", "record", "bool_id", "derived_id", "duplicate"]
)
def test_joint_training_bindings_reject_before_sampling(
    keyword_arguments, monkeypatch, binding_contract_case
):
    training_binding = keyword_arguments["held_model_training_bindings"][0]
    invalid_bindings = {
        "list": [training_binding],
        "record": (None,),
        "bool_id": (
            HeldModelTrainingBinding(
                model_id=True, classifier=None, concept_specific_parameter_optimizer=None
            ),
        ),
        "derived_id": (
            HeldModelTrainingBinding(
                model_id=type("DerivedInteger", (int,), {})(4),
                classifier=None,
                concept_specific_parameter_optimizer=None,
            ),
        ),
        "duplicate": (training_binding, training_binding),
    }[binding_contract_case]
    generator_state_before = keyword_arguments["python_random_generator"].getstate()
    sampling_calls = Mock(wraps=training_iteration_module.sample_training_batches_for_held_models)
    update_calls = Mock(wraps=training_iteration_module.perform_joint_model_parameter_update)
    monkeypatch.setattr(
        training_iteration_module, "sample_training_batches_for_held_models", sampling_calls
    )
    monkeypatch.setattr(
        training_iteration_module, "perform_joint_model_parameter_update", update_calls
    )
    keyword_arguments["held_model_training_bindings"] = invalid_bindings
    with pytest.raises(ValueError, match="held_model_training_bindings"):
        perform_held_model_joint_training_iterations(**keyword_arguments)
    sampling_calls.assert_not_called()
    update_calls.assert_not_called()
    assert keyword_arguments["python_random_generator"].getstate() == generator_state_before


def test_joint_training_iterations_zero_count_ignores_training_inputs(
    keyword_arguments, monkeypatch
):
    sampling_calls, update_calls = Mock(), Mock()
    monkeypatch.setattr(
        training_iteration_module, "sample_training_batches_for_held_models", sampling_calls
    )
    monkeypatch.setattr(
        training_iteration_module, "perform_joint_model_parameter_update", update_calls
    )
    keyword_arguments = {record_field: None for record_field in keyword_arguments}
    keyword_arguments["requested_joint_update_iteration_count"] = 0
    assert perform_held_model_joint_training_iterations(**keyword_arguments) == ()
    sampling_calls.assert_not_called()
    update_calls.assert_not_called()


@pytest.mark.parametrize("binding_contract_case", ["empty", "unheld", "insufficient"])
def test_joint_training_iterations_skip_ineligible_models(
    keyword_arguments, monkeypatch, binding_contract_case
):
    # 未参加payloadと未使用の共有部/設定は検査しない。
    keyword_arguments["held_model_training_bindings"] = (
        HeldModelTrainingBinding(
            model_id=-7, classifier=None, concept_specific_parameter_optimizer=None
        ),
    )
    invalid_sample = ObservedTrainingSample(input_features=None, observed_class_labels=None)
    keyword_arguments["ordered_model_training_samples"] = {
        "empty": (),
        "unheld": (
            ModelTrainingSampleCollection(model_id=4, training_samples=(invalid_sample,) * 3),
        ),
        "insufficient": (
            ModelTrainingSampleCollection(model_id=-7, training_samples=(invalid_sample,)),
        ),
    }[binding_contract_case]
    keyword_arguments.update(
        shared_feature_extractor=None, shared_parameter_optimizer=None, local_training_settings=None
    )
    generator_state_before = keyword_arguments["python_random_generator"].getstate()
    sampling_calls = Mock(wraps=training_iteration_module.sample_training_batches_for_held_models)
    update_calls = Mock(wraps=training_iteration_module.perform_joint_model_parameter_update)
    monkeypatch.setattr(
        training_iteration_module, "sample_training_batches_for_held_models", sampling_calls
    )
    monkeypatch.setattr(
        training_iteration_module, "perform_joint_model_parameter_update", update_calls
    )
    assert perform_held_model_joint_training_iterations(**keyword_arguments) == ()
    assert sampling_calls.call_count == 4
    update_calls.assert_not_called()
    assert keyword_arguments["python_random_generator"].getstate() == generator_state_before


@pytest.mark.parametrize("update_shared_features", [True, False])
def test_joint_training_iterations_preserve_environment(keyword_arguments, update_shared_features):
    # Adamの初回step scalar生成はTorch自身がambient deviceへ依存する。
    # 上流specと同じSGDでexecutorの環境保持を検証する。
    keyword_arguments["shared_parameter_optimizer"] = torch.optim.SGD(
        keyword_arguments["shared_feature_extractor"].parameters(), lr=0.01
    )
    keyword_arguments["held_model_training_bindings"] = tuple(
        HeldModelTrainingBinding(
            model_id=binding.model_id,
            classifier=binding.classifier,
            concept_specific_parameter_optimizer=torch.optim.SGD(
                tuple(binding.classifier.residual_adapter.parameters())
                + tuple(binding.classifier.classification_layer.parameters()),
                lr=0.01,
            ),
        )
        for binding in keyword_arguments["held_model_training_bindings"]
    )
    global_python_random_state, global_torch_random_state = random.getstate(), torch.get_rng_state()
    global_numpy_random_state = np.random.get_state()
    previous_default_dtype, previous_default_device = (
        torch.get_default_dtype(),
        torch.get_default_device(),
    )
    previous_grad_mode = torch.is_grad_enabled()
    sample_values_before = tuple(
        (training_sample.input_features.clone(), training_sample.observed_class_labels.clone())
        for model_training_samples in keyword_arguments["ordered_model_training_samples"]
        for training_sample in model_training_samples.training_samples
    )
    classifier_parameters_before = deepcopy(
        keyword_arguments["shared_feature_extractor"].state_dict()
    )
    optimizer_state_before = deepcopy(keyword_arguments["shared_parameter_optimizer"].state_dict())
    keyword_arguments["update_shared_features"] = update_shared_features
    local_training_settings = keyword_arguments["local_training_settings"]
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            assert len(perform_held_model_joint_training_iterations(**keyword_arguments)) == 4
            assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_device() == torch.device("meta")
            assert torch.is_grad_enabled() == previous_grad_mode
    finally:
        torch.set_default_dtype(previous_default_dtype)
    assert torch.get_default_device() == previous_default_device
    assert keyword_arguments["local_training_settings"] is local_training_settings
    for training_sample, (input_features, observed_class_labels) in zip(
        (
            training_sample
            for model_training_samples in keyword_arguments["ordered_model_training_samples"]
            for training_sample in model_training_samples.training_samples
        ),
        sample_values_before,
    ):
        assert torch.equal(training_sample.input_features, input_features)
        assert torch.equal(training_sample.observed_class_labels, observed_class_labels)
    if not update_shared_features:
        assert_nested_state_equal(
            keyword_arguments["shared_feature_extractor"].state_dict(), classifier_parameters_before
        )
        assert_nested_state_equal(
            keyword_arguments["shared_parameter_optimizer"].state_dict(), optimizer_state_before
        )
        assert all(
            parameter.grad is None
            for parameter in keyword_arguments["shared_feature_extractor"].parameters()
        )
    assert random.getstate() == global_python_random_state
    assert torch.equal(torch.get_rng_state(), global_torch_random_state)
    assert np.random.get_state()[0] == global_numpy_random_state[0]
    assert np.array_equal(np.random.get_state()[1], global_numpy_random_state[1])
    assert np.random.get_state()[2:] == global_numpy_random_state[2:]


def test_joint_training_iterations_reject_invalid_batch_before_draw(keyword_arguments):
    invalid_sample = ObservedTrainingSample(
        input_features=torch.tensor([[float("nan"), 0.0]]),
        observed_class_labels=torch.tensor([[0.0]]),
    )
    keyword_arguments["ordered_model_training_samples"] = (
        ModelTrainingSampleCollection(model_id=4, training_samples=(invalid_sample,) * 3),
    )
    generator_state_before = keyword_arguments["python_random_generator"].getstate()
    classifier_parameters_before = tuple(
        deepcopy(binding.classifier.state_dict())
        for binding in keyword_arguments["held_model_training_bindings"]
    )
    with pytest.raises(ValueError, match="input_features"):
        perform_held_model_joint_training_iterations(**keyword_arguments)
    assert keyword_arguments["python_random_generator"].getstate() == generator_state_before
    for binding, initial_parameters in zip(
        keyword_arguments["held_model_training_bindings"], classifier_parameters_before
    ):
        assert_nested_state_equal(binding.classifier.state_dict(), initial_parameters)
        assert binding.concept_specific_parameter_optimizer.state == {}
        assert all(parameter.grad is None for parameter in binding.classifier.parameters())


def test_held_model_training_binding_is_frozen_and_explicit(keyword_arguments):
    training_binding = keyword_arguments["held_model_training_bindings"][0]
    for record_field in fields(HeldModelTrainingBinding):
        assert record_field.default is MISSING
        assert record_field.default_factory is MISSING
        assert record_field.kw_only
    with pytest.raises(FrozenInstanceError):
        training_binding.model_id = 10
    binding = HeldModelTrainingBinding(
        model_id=-(10**30),
        classifier=training_binding.classifier,
        concept_specific_parameter_optimizer=training_binding.concept_specific_parameter_optimizer,
    )
    assert binding.classifier is training_binding.classifier
    assert (
        binding.concept_specific_parameter_optimizer
        is training_binding.concept_specific_parameter_optimizer
    )


def test_joint_training_iterations_preserve_completed_updates_on_failure(monkeypatch):
    global_torch_random_state = torch.get_rng_state()
    try:
        (
            participating_training_batches,
            shared_parameter_optimizer,
            legacy_client,
            ordered_model_training_samples,
            held_model_training_bindings,
        ) = build_training_iteration_oracle_pair(
            class_count=2, optimizer_variant="amsgrad", monkeypatch=monkeypatch
        )
        legacy_client.batch_size = 3
        python_random_generator = random.Random(731)
        run_legacy_training_iterations(
            legacy_client=legacy_client,
            iteration_count=1,
            update_shared_features=True,
            initial_random_state=python_random_generator.getstate(),
        )
        original_update_function = training_iteration_module.perform_joint_model_parameter_update
        update_calls = Mock()

        def failing_update(**keyword_arguments):
            update_calls()
            if update_calls.call_count == 2:
                # 抽出batchだけを改変し、本物の更新検査を失敗させる。
                keyword_arguments["participating_training_batches"][0].observed_class_labels.fill_(
                    9.0
                )
            return original_update_function(**keyword_arguments)

        monkeypatch.setattr(
            training_iteration_module, "perform_joint_model_parameter_update", failing_update
        )
        expected_random_generator = random.Random(731)
        for _ in range(2):
            training_iteration_module.sample_training_batches_for_held_models(
                ordered_model_training_samples=ordered_model_training_samples,
                held_model_ids=frozenset((4, -7)),
                batch_sample_count=3,
                python_random_generator=expected_random_generator,
            )
        with pytest.raises(ValueError, match="observed_class_labels"):
            perform_held_model_joint_training_iterations(
                requested_joint_update_iteration_count=4,
                held_model_training_bindings=held_model_training_bindings,
                ordered_model_training_samples=ordered_model_training_samples,
                batch_sample_count=3,
                python_random_generator=python_random_generator,
                local_training_settings=LocalTrainingSettings(
                    local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                    shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
                ),
                shared_feature_extractor=participating_training_batches[
                    0
                ].classifier.feature_extractor,
                shared_parameter_optimizer=shared_parameter_optimizer,
                update_shared_features=True,
            )
        assert update_calls.call_count == 2
        assert python_random_generator.getstate() == expected_random_generator.getstate()
        assert_joint_update_states_equal(
            participating_training_batches=participating_training_batches,
            shared_parameter_optimizer=shared_parameter_optimizer,
            legacy_client=legacy_client,
        )
        assert all(
            0 <= training_sample.observed_class_labels.item() <= 1
            for model_training_samples in ordered_model_training_samples
            for training_sample in model_training_samples.training_samples
        )
    finally:
        torch.set_rng_state(global_torch_random_state)


@pytest.mark.parametrize("update_shared_features", [True, False])
def test_joint_training_iterations_with_empty_shared_features(
    keyword_arguments, update_shared_features
):
    empty_shared_classifier = ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(),
        class_count=2,
    )
    expected_classifier = deepcopy(empty_shared_classifier)
    empty_shared_optimizer = torch.optim.Adam(
        tuple(empty_shared_classifier.residual_adapter.parameters())
        + tuple(empty_shared_classifier.classification_layer.parameters()),
        lr=0.01,
        amsgrad=True,
    )
    expected_parameter_optimizer = torch.optim.Adam(
        tuple(expected_classifier.parameters()), lr=0.01, amsgrad=True
    )
    ordered_model_training_samples = keyword_arguments["ordered_model_training_samples"][:1]
    expected_random_generator = random.Random(731)
    expected_losses = []
    for _ in range(4):
        sampled_training_batch = training_iteration_module.sample_training_batches_for_held_models(
            ordered_model_training_samples=ordered_model_training_samples,
            held_model_ids=frozenset((4,)),
            batch_sample_count=3,
            python_random_generator=expected_random_generator,
        )[0]
        expected_parameter_optimizer.zero_grad()
        joint_update_loss = torch.nn.BCELoss()(
            expected_classifier(sampled_training_batch.input_features),
            sampled_training_batch.observed_class_labels,
        )
        expected_losses.append(joint_update_loss.item())
        joint_update_loss.backward()
        expected_parameter_optimizer.step()
    keyword_arguments.update(
        held_model_training_bindings=(
            HeldModelTrainingBinding(
                model_id=4,
                classifier=empty_shared_classifier,
                concept_specific_parameter_optimizer=empty_shared_optimizer,
            ),
        ),
        ordered_model_training_samples=ordered_model_training_samples,
        shared_feature_extractor=empty_shared_classifier.feature_extractor,
        shared_parameter_optimizer=None,
        update_shared_features=update_shared_features,
    )
    assert tuple(empty_shared_classifier.feature_extractor.parameters()) == ()
    assert perform_held_model_joint_training_iterations(**keyword_arguments) == tuple(
        expected_losses
    )
    assert (
        keyword_arguments["python_random_generator"].getstate()
        == expected_random_generator.getstate()
    )
    for parameter, expected_parameter in zip(
        empty_shared_classifier.parameters(), expected_classifier.parameters()
    ):
        assert torch.equal(parameter, expected_parameter)
        assert torch.equal(parameter.grad, expected_parameter.grad)
    assert_nested_state_equal(
        empty_shared_optimizer.state_dict(), expected_parameter_optimizer.state_dict()
    )
