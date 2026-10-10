"""準備済みモデルの有界損失・副作用と旧実装の対応を検証する。"""

import random
import warnings
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_joint_model_parameter_update import (
    assert_joint_update_states_equal,
    assert_nested_state_equal,
    build_joint_update_oracle_pair,
    run_legacy_joint_update,
)
from test_model_computation_measurement import make_classifier as make_measured_classifier
from test_model_computation_measurement import make_inputs as make_measured_inputs
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    build_attachment_classifier,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment.clients.base import BaseClient
from federated_drift_experiment.clients.shared_backbone import (
    _SharedRepresentationFedSDAClientMixin,
)
from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import (
    initialize_model_and_class_loss_statistics_from_batch,
)
from federated_learning_experiments.learning.models.model_computation_measurement import (
    measure_model_computation,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
    evaluate_classifier_per_sample_bounded_losses_and_outputs,
    evaluate_classifiers_per_sample_bounded_losses_from_shared_features,
    validate_classifier_bounded_loss_inputs,
)
from federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration import (
    integrate_adopted_candidate_shared_features,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)


def build_bounded_loss_oracle_pair(*, class_count, sample_count, monkeypatch):
    training_batches, _, legacy_client = build_joint_update_oracle_pair(
        class_count=class_count,
        batch_sample_counts=(sample_count,),
        optimizer_variant="standard",
        monkeypatch=monkeypatch,
    )
    legacy_model = legacy_client.models[4]
    classifier = training_batches[0].classifier
    # ゼロ初期化adapterだけの対照にならないよう、実旧展開層から値を対応する。
    with torch.no_grad():
        legacy_model.adapter.up.weight.fill_(0.15)
        legacy_model.adapter.up.bias.fill_(0.05)
    classifier.residual_adapter.feature_expansion.load_state_dict(
        legacy_model.adapter.up.state_dict()
    )
    return classifier, legacy_model, training_batches[0]


@pytest.mark.parametrize("class_count", [2, 4, 10])
@pytest.mark.parametrize("sample_count", [1, 5])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_classifier_bounded_losses_match_legacy(
    class_count, sample_count, noncontiguous, training, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        classifier, legacy_model, training_batch = build_bounded_loss_oracle_pair(
            class_count=class_count, sample_count=sample_count, monkeypatch=monkeypatch
        )
        input_features = training_batch.input_features.flip(0)
        observed_class_labels = training_batch.observed_class_labels.flip(0)
        if noncontiguous:
            input_features = input_features.repeat_interleave(2, dim=1)[:, ::2]
            observed_class_labels = observed_class_labels.repeat_interleave(2, dim=1)[:, ::2]
            if sample_count > 1:
                assert not input_features.is_contiguous()
                assert not observed_class_labels.is_contiguous()
        classifier.train(training)
        legacy_model.train(training)
        # 親flagだけの検査にせず、子の異なるflagも保持する。
        classifier.residual_adapter.eval()
        for parameter in classifier.parameters():
            parameter.grad = torch.ones_like(parameter)
        state_before = snapshot_parameter_values_and_gradients(classifier.parameters())
        input_before = (input_features.clone(), observed_class_labels.clone())
        flags_before = tuple(module.training for module in classifier.modules())
        rng_before = torch.get_rng_state().clone()
        forward_calls = []
        hook = classifier.register_forward_hook(
            lambda module, args, output: forward_calls.append(torch.is_grad_enabled())
        )
        try:
            with torch.no_grad():
                expected = legacy_model.per_sample_error(input_features, observed_class_labels)
            result = evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            )
        finally:
            hook.remove()
        assert torch.equal(result, expected)
        assert result.shape == (sample_count,)
        assert result.dtype == torch.float32 and result.device.type == "cpu"
        assert not result.requires_grad and result.grad_fn is None
        assert forward_calls == [False]
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert flags_before == tuple(module.training for module in classifier.modules())
        assert_parameter_values_and_gradients_unchanged(state_before)
        result.fill_(0)
        assert torch.equal(input_features, input_before[0])
        assert torch.equal(observed_class_labels, input_before[1])
        assert_parameter_values_and_gradients_unchanged(state_before)


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("classifier", None),
        ("classifier", torch.nn.Linear(2, 1)),
        ("input_features", None),
        ("input_features", [[0.1, 0.2]]),
        ("input_features", torch.empty((2, 2), dtype=torch.float64)),
        ("input_features", torch.empty((2, 2), dtype=torch.int64)),
        ("input_features", torch.empty((2, 2), dtype=torch.bool)),
        ("input_features", torch.empty((2, 2), device="meta")),
        ("input_features", torch.ones((2, 2)).to_sparse()),
        ("input_features", torch.empty((0, 2))),
        ("input_features", torch.empty(2)),
        ("input_features", torch.empty((2, 2, 1))),
        ("input_features", torch.empty((2, 3))),
        ("input_features", torch.tensor([[float("nan"), 0.0], [0.0, 0.0]])),
        ("input_features", torch.tensor([[float("inf"), 0.0], [0.0, 0.0]])),
        ("observed_class_labels", None),
        ("observed_class_labels", [0, 1]),
        ("observed_class_labels", torch.empty((2, 1), dtype=torch.float64)),
        ("observed_class_labels", torch.zeros((2, 1), dtype=torch.int64)),
        ("observed_class_labels", torch.zeros((2, 1), dtype=torch.bool)),
        ("observed_class_labels", torch.empty((2, 1), device="meta")),
        ("observed_class_labels", torch.ones((2, 1)).to_sparse()),
        ("observed_class_labels", torch.zeros(2)),
        ("observed_class_labels", torch.zeros((2, 2))),
        ("observed_class_labels", torch.zeros((1, 1))),
        ("observed_class_labels", torch.empty((0, 1))),
        ("observed_class_labels", torch.tensor([[0.5], [1.0]])),
        ("observed_class_labels", torch.tensor([[-1.0], [1.0]])),
        ("observed_class_labels", torch.tensor([[2.0], [1.0]])),
        ("observed_class_labels", torch.tensor([[float("nan")], [1.0]])),
        ("observed_class_labels", torch.tensor([[float("inf")], [1.0]])),
    ],
)
def test_bounded_loss_evaluation_rejects_invalid_inputs_before_forward(field_name, invalid_value):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier()
        inputs = {
            "classifier": classifier,
            "input_features": torch.ones((2, 2)),
            "observed_class_labels": torch.tensor([[0.0], [1.0]]),
        }
        inputs[field_name] = invalid_value
        state_before = snapshot_parameter_values_and_gradients(classifier.parameters())
        rng_before = torch.get_rng_state().clone()
        forward_calls = []
        hook = classifier.register_forward_hook(
            lambda module, args, output: forward_calls.append(output)
        )
        try:
            with pytest.raises((TypeError, ValueError), match=field_name):
                evaluate_classifier_per_sample_bounded_losses(**inputs)
        finally:
            hook.remove()
        assert not forward_calls
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert_parameter_values_and_gradients_unchanged(state_before)


@pytest.mark.parametrize("operation", ["float64", "meta", "invalid_class_count", "nested"])
def test_bounded_loss_evaluation_rejects_invalid_classifier_parameters(operation):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier()
        input_features = torch.ones((2, 2))
        observed_class_labels = torch.zeros((2, 1))
        if operation == "invalid_class_count":
            classifier.class_count = True
        elif operation == "nested":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                input_features = torch.nested.nested_tensor([torch.ones(2), torch.ones(2)])
        else:
            classifier.to(dtype=torch.float64) if operation == "float64" else classifier.to("meta")
        forward_calls = []
        hook = classifier.register_forward_hook(
            lambda module, args, output: forward_calls.append(output)
        )
        try:
            with pytest.raises((TypeError, ValueError)):
                evaluate_classifier_per_sample_bounded_losses(
                    classifier=classifier,
                    input_features=input_features,
                    observed_class_labels=observed_class_labels,
                )
        finally:
            hook.remove()
        assert not forward_calls


@pytest.mark.parametrize(
    "invalid_value",
    [
        [[0.2], [0.3]],
        torch.zeros((2, 1), dtype=torch.float64),
        torch.zeros((2, 1), dtype=torch.int64),
        torch.empty((2, 1), device="meta"),
        torch.ones((2, 1)).to_sparse(),
        torch.zeros(2),
        torch.zeros((1, 1)),
        torch.zeros((2, 2)),
        torch.tensor([[float("nan")], [0.5]]),
        torch.tensor([[float("inf")], [0.5]]),
        torch.tensor([[-0.1], [0.5]]),
        torch.tensor([[1.1], [0.5]]),
    ],
)
def test_bounded_loss_evaluation_rejects_invalid_outputs(invalid_value):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier()
        state_before = snapshot_parameter_values_and_gradients(classifier.parameters())
        rng_before = torch.get_rng_state().clone()
        flags_before = tuple(module.training for module in classifier.modules())
        hook = classifier.register_forward_hook(lambda module, args, output: invalid_value)
        try:
            with torch.enable_grad():
                with pytest.raises((TypeError, ValueError), match="classifier_outputs"):
                    evaluate_classifier_per_sample_bounded_losses(
                        classifier=classifier,
                        input_features=torch.ones((2, 2)),
                        observed_class_labels=torch.zeros((2, 1)),
                    )
                assert torch.is_grad_enabled()
        finally:
            hook.remove()
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert tuple(module.training for module in classifier.modules()) == flags_before
        assert_parameter_values_and_gradients_unchanged(state_before)


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_bounded_loss_evaluation_preserves_environment_and_independent_results(
    class_count, grad_enabled
):
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier(class_count=class_count)
        input_features = torch.ones((2, 2), requires_grad=True)
        observed_class_labels = torch.tensor([[0.0], [1.0]], requires_grad=True)
        input_features.grad = torch.ones_like(input_features)
        observed_class_labels.grad = torch.ones_like(observed_class_labels)
        for parameter in classifier.parameters():
            parameter.grad = torch.ones_like(parameter)
        state_before = snapshot_parameter_values_and_gradients(
            tuple(classifier.parameters()) + (input_features, observed_class_labels)
        )
        rng_before = (
            torch.get_rng_state().clone(),
            random.getstate(),
            deepcopy(np.random.get_state()),
        )
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with torch.device("meta"), torch.set_grad_enabled(grad_enabled):
                result = evaluate_classifier_per_sample_bounded_losses(
                    classifier=classifier,
                    input_features=input_features,
                    observed_class_labels=observed_class_labels,
                )
                assert torch.is_grad_enabled() == grad_enabled
                assert torch.empty(0).device.type == "meta"
            assert torch.get_default_dtype() == torch.float64
        finally:
            torch.set_default_dtype(default_dtype)
        assert not result.requires_grad and result.grad_fn is None
        assert torch.equal(torch.get_rng_state(), rng_before[0])
        assert random.getstate() == rng_before[1]
        numpy_state = np.random.get_state()
        assert np.array_equal(numpy_state[1], rng_before[2][1])
        assert (numpy_state[0], *numpy_state[2:]) == (rng_before[2][0], *rng_before[2][2:])
        assert_parameter_values_and_gradients_unchanged(state_before)
        result.fill_(1)
        assert_parameter_values_and_gradients_unchanged(state_before)


def test_classifier_bounded_losses_match_legacy_multiclass_logits():
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier(class_count=4)
        classifier_outputs = torch.tensor([[100.0, -100.0, 2.0, -2.0], [-3.0, 2.0, -1.0, 3.0]])
        hook = classifier.register_forward_hook(lambda module, args, output: classifier_outputs)
        try:
            with torch.inference_mode():
                result = evaluate_classifier_per_sample_bounded_losses(
                    classifier=classifier,
                    input_features=torch.ones((2, 2)),
                    observed_class_labels=torch.tensor([[0.0], [2.0]]),
                )
                assert torch.is_inference_mode_enabled()
        finally:
            hook.remove()
        expected = 1.0 - torch.softmax(classifier_outputs, dim=1)[torch.arange(2), [0, 2]]
        assert torch.equal(result, expected)


def test_bounded_loss_evaluation_rejects_invalid_outputs_from_finite_parameters():
    with torch.random.fork_rng(devices=[]):
        classifier = build_attachment_classifier(class_count=4)
        with torch.no_grad():
            for parameter in classifier.parameters():
                parameter.fill_(torch.finfo(torch.float32).max)
        state_before = snapshot_parameter_values_and_gradients(classifier.parameters())
        with pytest.raises(ValueError, match="classifier_outputs"):
            evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier,
                input_features=torch.ones((2, 2)),
                observed_class_labels=torch.zeros((2, 1)),
            )
        assert_parameter_values_and_gradients_unchanged(state_before)


def assert_initial_loss_statistics_match_legacy(result, legacy_stats):
    assert (
        result.overall_loss_moments.observed_loss_count,
        result.overall_loss_moments.mean_loss,
        result.overall_loss_moments.sum_squared_loss_deviations,
    ) == (legacy_stats["n"], legacy_stats["mean"], legacy_stats["M2"])
    assert tuple(class_id for class_id, _ in result.class_loss_moments_by_class_id) == tuple(
        legacy_stats["class_stats"]
    )
    for class_id, moments in result.class_loss_moments_by_class_id:
        expected = legacy_stats["class_stats"][class_id]
        assert (
            moments.observed_loss_count,
            moments.mean_loss,
            moments.sum_squared_loss_deviations,
        ) == (expected["n"], expected["mean"], expected["M2"])


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [False, True])
def test_bounded_loss_evaluation_connects_prepared_model_to_initial_statistics(
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
        candidate = training_batches[1].classifier
        optimizer_settings = (
            SgdParameterOptimizerSettings(learning_rate=0.01)
            if optimizer_variant == "sgd"
            else AdamParameterOptimizerSettings(
                learning_rate=0.01, weight_decay=0.001, adam_variant=optimizer_variant
            )
        )
        candidate_owner = ParameterOptimizerState(
            parameters=tuple(candidate.residual_adapter.parameters())
            + tuple(candidate.classification_layer.parameters()),
            optimizer_settings=optimizer_settings,
        )
        training_batches = (
            training_batches[0],
            replace(
                training_batches[1],
                concept_specific_parameter_optimizer=candidate_owner.parameter_optimizer,
            ),
        )
        settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        )
        for _ in range(3):
            expected_loss = run_legacy_joint_update(
                legacy_client=legacy_client, update_shared_features=update_shared_features
            )
            actual_loss = perform_joint_model_parameter_update(
                local_training_settings=settings,
                shared_feature_extractor=candidate.feature_extractor,
                shared_parameter_optimizer=shared_parameter_optimizer,
                participating_training_batches=training_batches,
                update_shared_features=update_shared_features,
            )
            assert actual_loss == expected_loss
            assert_joint_update_states_equal(
                participating_training_batches=training_batches,
                shared_parameter_optimizer=shared_parameter_optimizer,
                legacy_client=legacy_client,
            )
        # 旧と新の準備は上位の接続。損失評価部へresetや登録を追加しない。
        prepared_legacy_candidate = (
            _SharedRepresentationFedSDAClientMixin._prepare_model_for_registration(
                legacy_client, legacy_client.models[-7]
            )
        )
        integrate_adopted_candidate_shared_features(
            adopted_candidate_classifier=candidate,
            candidate_concept_specific_parameter_optimizer_state=candidate_owner,
            active_shared_feature_extractor=training_batches[0].classifier.feature_extractor,
        )
        training_batches = (
            training_batches[0],
            replace(
                training_batches[1],
                concept_specific_parameter_optimizer=candidate_owner.parameter_optimizer,
            ),
        )
        assert_joint_update_states_equal(
            participating_training_batches=training_batches,
            shared_parameter_optimizer=shared_parameter_optimizer,
            legacy_client=legacy_client,
        )
        shared_state_before = deepcopy(shared_parameter_optimizer.state_dict())
        concept_state_before = deepcopy(candidate_owner.parameter_optimizer.state_dict())
        previous_optimizer = candidate_owner.parameter_optimizer
        model_state_before = snapshot_parameter_values_and_gradients(candidate.parameters())
        rng_before = torch.get_rng_state().clone()
        # 一標本の全体fallbackと、複数class/欠落classの全fieldを実旧登録へ対応する。
        for sample_count in (5, 1):
            input_features = training_batches[1].input_features[:sample_count]
            observed_class_labels = training_batches[1].observed_class_labels[:sample_count]
            losses = evaluate_classifier_per_sample_bounded_losses(
                classifier=candidate,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            )
            with torch.no_grad():
                expected_losses = prepared_legacy_candidate.per_sample_error(
                    input_features, observed_class_labels
                )
            assert torch.equal(losses, expected_losses)
            result = initialize_model_and_class_loss_statistics_from_batch(
                per_sample_bounded_losses=losses,
                observed_class_labels=observed_class_labels,
                class_count=class_count,
            )
            registration_client = SimpleNamespace(
                models={},
                model_stats={},
                _prepare_model_for_registration=lambda model: model,
                _record_model_compute=lambda *args: None,
            )
            # singletonの旧不偏分散warningだけをoracle内で抑制する。
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                BaseClient._register_trained_new_model(
                    registration_client,
                    temp_id=-7,
                    new_model=prepared_legacy_candidate,
                    bx=input_features,
                    by=observed_class_labels,
                    pending_ready=False,
                )
            assert_initial_loss_statistics_match_legacy(result, registration_client.model_stats[-7])
            assert torch.equal(torch.get_rng_state(), rng_before)
            assert candidate_owner.parameter_optimizer is previous_optimizer
            assert_nested_state_equal(shared_parameter_optimizer.state_dict(), shared_state_before)
            assert_nested_state_equal(
                candidate_owner.parameter_optimizer.state_dict(), concept_state_before
            )
            assert_parameter_values_and_gradients_unchanged(model_state_before)
            assert_joint_update_states_equal(
                participating_training_batches=training_batches,
                shared_parameter_optimizer=shared_parameter_optimizer,
                legacy_client=legacy_client,
            )


def build_classifiers_sharing_feature_extractor(*, class_count, classifier_count):
    """同じ共有部につながる分類器（概念固有部は、別々の値）。"""
    first_classifier = make_measured_classifier(class_count=class_count)
    classifiers = [first_classifier]
    for seed in range(1, classifier_count):
        classifier = make_measured_classifier(
            class_count=class_count,
            shared_feature_extractor=first_classifier.feature_extractor,
            seed=seed,
        )
        # 展開層は0で初期化されるので、分類器ごとに違う値を入れる。
        with torch.no_grad():
            classifier.residual_adapter.feature_expansion.weight.fill_(0.1 * seed)
        classifiers.append(classifier)
    return tuple(classifiers)


def make_bounded_loss_inputs(*, class_count, sample_count):
    input_features = make_measured_inputs(sample_count)
    observed_class_labels = torch.tensor(
        [[float(sample_position % class_count)] for sample_position in range(sample_count)]
    )
    return input_features, observed_class_labels


@pytest.mark.parametrize("class_count", [2, 3])
@pytest.mark.parametrize("sample_count", [1, 6])
def test_losses_and_outputs_come_from_one_forward_and_match_existing_evaluation(
    class_count, sample_count
):
    """損失は既存の評価と同じ値、出力は分類器の出力そのもので、順伝播は1回だけ。"""
    (classifier,) = build_classifiers_sharing_feature_extractor(
        class_count=class_count, classifier_count=1
    )
    input_features, observed_class_labels = make_bounded_loss_inputs(
        class_count=class_count, sample_count=sample_count
    )
    expected_losses = evaluate_classifier_per_sample_bounded_losses(
        classifier=classifier,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )
    with torch.no_grad():
        expected_outputs = classifier(input_features)
    with measure_model_computation() as model_computation_meter:
        per_sample_losses, classifier_outputs = (
            evaluate_classifier_per_sample_bounded_losses_and_outputs(
                classifier=classifier,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            )
        )
    assert torch.equal(per_sample_losses, expected_losses)
    assert torch.equal(classifier_outputs, expected_outputs)
    assert not per_sample_losses.requires_grad
    assert not classifier_outputs.requires_grad
    counts = model_computation_meter.get_model_computation_counts()
    assert counts.shared_part_inference_example_count == sample_count
    assert counts.concept_specific_part_inference_example_count == sample_count
    assert counts.shared_part_training_example_count == 0
    # 既存の評価も、順伝播は1回だけ。
    with measure_model_computation() as model_computation_meter:
        evaluate_classifier_per_sample_bounded_losses(
            classifier=classifier,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
        )
    assert model_computation_meter.get_model_computation_counts() == counts


@pytest.mark.parametrize("class_count", [2, 3])
@pytest.mark.parametrize("classifier_count", [1, 3])
def test_losses_from_shared_features_match_per_classifier_evaluation_with_one_shared_forward(
    class_count, classifier_count
):
    """共有部の特徴を1回だけ計算した損失が、分類器ごとに計算した損失と、完全に一致する。"""
    classifiers = build_classifiers_sharing_feature_extractor(
        class_count=class_count, classifier_count=classifier_count
    )
    input_features, observed_class_labels = make_bounded_loss_inputs(
        class_count=class_count, sample_count=5
    )
    expected_losses = [
        evaluate_classifier_per_sample_bounded_losses(
            classifier=classifier,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
        )
        for classifier in classifiers
    ]
    training_modes = [classifier.training for classifier in classifiers]
    with measure_model_computation() as model_computation_meter:
        per_sample_losses_by_classifier = (
            evaluate_classifiers_per_sample_bounded_losses_from_shared_features(
                classifiers=classifiers,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            )
        )
    assert type(per_sample_losses_by_classifier) is tuple
    assert len(per_sample_losses_by_classifier) == classifier_count
    for per_sample_losses, expected in zip(
        per_sample_losses_by_classifier, expected_losses, strict=True
    ):
        assert torch.equal(per_sample_losses, expected)
        assert not per_sample_losses.requires_grad
    if classifier_count > 1:
        # 分類器ごとに、違う損失になっている（同じ値どうしの比較ではない）。
        assert not torch.equal(
            per_sample_losses_by_classifier[0], per_sample_losses_by_classifier[1]
        )
    counts = model_computation_meter.get_model_computation_counts()
    assert counts.shared_part_inference_example_count == 5
    assert counts.concept_specific_part_inference_example_count == 5 * classifier_count
    assert [classifier.training for classifier in classifiers] == training_modes


def test_losses_from_shared_features_reject_invalid_arguments_before_any_forward():
    classifiers = build_classifiers_sharing_feature_extractor(class_count=2, classifier_count=2)
    unrelated_classifier = make_measured_classifier(class_count=2, seed=9)
    input_features, observed_class_labels = make_bounded_loss_inputs(class_count=2, sample_count=4)
    valid_arguments = dict(
        classifiers=classifiers,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )
    invalid_cases = [
        ("classifiers", list(classifiers), TypeError),
        ("classifiers", (), ValueError),
        ("classifiers", (classifiers[0], SimpleNamespace()), TypeError),
        # 別の共有部につながる分類器。
        ("classifiers", (classifiers[0], unrelated_classifier), ValueError),
        ("input_features", input_features.double(), ValueError),
        ("input_features", input_features[:, :2], ValueError),
        ("observed_class_labels", observed_class_labels + 5, ValueError),
        ("observed_class_labels", observed_class_labels[:3], ValueError),
    ]
    with measure_model_computation() as model_computation_meter:
        for argument_name, invalid_value, expected_exception_type in invalid_cases:
            with pytest.raises(expected_exception_type):
                evaluate_classifiers_per_sample_bounded_losses_from_shared_features(
                    **valid_arguments | {argument_name: invalid_value}
                )
        with pytest.raises(TypeError):
            evaluate_classifiers_per_sample_bounded_losses_from_shared_features(
                classifiers, input_features, observed_class_labels
            )
    # どの拒否も、順伝播より前。
    assert not any(vars(model_computation_meter.get_model_computation_counts()).values())


def test_input_validation_rejects_like_the_evaluation_without_any_forward():
    """公開の検査は、評価と同じ入力を、順伝播なしで拒否する。"""
    (classifier,) = build_classifiers_sharing_feature_extractor(class_count=3, classifier_count=1)
    input_features, observed_class_labels = make_bounded_loss_inputs(class_count=3, sample_count=4)
    valid_arguments = dict(
        classifier=classifier,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )
    invalid_cases = [
        ("classifier", SimpleNamespace()),
        ("input_features", input_features.double()),
        ("input_features", input_features[:, :2]),
        ("input_features", input_features[:0]),
        ("input_features", torch.full_like(input_features, float("nan"))),
        ("observed_class_labels", observed_class_labels + 5),
        ("observed_class_labels", observed_class_labels + 0.5),
        ("observed_class_labels", observed_class_labels[:3]),
        ("observed_class_labels", observed_class_labels.reshape(-1)),
    ]
    with measure_model_computation() as model_computation_meter:
        assert validate_classifier_bounded_loss_inputs(**valid_arguments) is None
        for argument_name, invalid_value in invalid_cases:
            invalid_arguments = valid_arguments | {argument_name: invalid_value}
            with pytest.raises((TypeError, ValueError)) as validation_error:
                validate_classifier_bounded_loss_inputs(**invalid_arguments)
            with pytest.raises(type(validation_error.value)) as evaluation_error:
                evaluate_classifier_per_sample_bounded_losses(**invalid_arguments)
            assert str(validation_error.value) == str(evaluation_error.value)
        with pytest.raises(TypeError):
            validate_classifier_bounded_loss_inputs(
                classifier, input_features, observed_class_labels
            )
    assert not any(vars(model_computation_meter.get_model_computation_counts()).values())
