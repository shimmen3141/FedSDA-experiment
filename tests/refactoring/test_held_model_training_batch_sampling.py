"""実旧samplerへ参加順・抽出Tensor・終端乱数状態を直接照合する。"""

import random
import warnings
from dataclasses import FrozenInstanceError, MISSING, fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np
import torch

from federated_drift_experiment.clients.shared_backbone import (
    _SharedRepresentationFedSDAClientMixin,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
    ModelTrainingSampleCollection,
    SampledModelTrainingBatch,
)
from federated_learning_experiments.learning.training.held_model_training_batch_sampling import (
    sample_training_batches_for_held_models,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.participating_model_training_batch import (
    ParticipatingModelTrainingBatch,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    TrainingDataAssignmentSettings,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from test_joint_model_parameter_update import (
    build_joint_update_oracle_pair,
    assert_joint_update_states_equal,
)


def build_sampling_oracle_inputs(*, batch_sample_count=2, input_contract_case="mixed"):
    ordered_model_training_samples = []
    legacy_training_samples = {}
    for model_id, sample_count in ((8, 5), (55, 5), (-7, 100), (2, 0), (4, batch_sample_count - 1)):
        training_samples = tuple(
            ObservedTrainingSample(
                input_features=torch.tensor(
                    [[float(training_sample_index), float(model_id)]],
                    dtype=torch.float32,
                    device="cpu",
                ),
                observed_class_labels=torch.tensor(
                    [[float(training_sample_index % 2)]], dtype=torch.float32, device="cpu"
                ),
            )
            for training_sample_index in range(sample_count)
        )
        if input_contract_case == "same_reference" and training_samples:
            training_samples = (training_samples[0],) * sample_count
        ordered_model_training_samples.append(
            ModelTrainingSampleCollection(model_id=model_id, training_samples=training_samples)
        )
        # 真の概念情報は旧oracleだけに付け、新sampleの契約へ持ち込まない。
        legacy_training_samples[model_id] = [
            (training_sample.input_features, training_sample.observed_class_labels, 0)
            for training_sample in training_samples
        ]
    held_model_ids = frozenset((8, -7, 2, 4, 999))
    if input_contract_case == "empty":
        ordered_model_training_samples = []
        legacy_training_samples = {}
    elif input_contract_case == "no_eligible":
        held_model_ids = frozenset((2, 4, 999))
    legacy_client = SimpleNamespace(
        models={model_id: None for model_id in held_model_ids},
        train_data_store=legacy_training_samples,
        batch_size=batch_sample_count,
    )
    return held_model_ids, tuple(ordered_model_training_samples), random.Random(137), legacy_client


def run_legacy_training_batch_sampling(*, legacy_client, python_random_generator):
    global_python_random_state = random.getstate()
    try:
        random.setstate(python_random_generator.getstate())
        legacy_batches = _SharedRepresentationFedSDAClientMixin._sample_training_batches(
            legacy_client
        )
        return legacy_batches, random.getstate()
    finally:
        random.setstate(global_python_random_state)


def assert_sampled_batches_equal(*, sampled_batches, legacy_batches):
    assert type(sampled_batches) is tuple
    assert len(sampled_batches) == len(legacy_batches)
    for sampled_batch, (model_id, input_features, observed_class_labels) in zip(
        sampled_batches, legacy_batches
    ):
        assert type(sampled_batch) is SampledModelTrainingBatch
        assert sampled_batch.model_id == model_id
        assert torch.equal(sampled_batch.input_features, input_features)
        assert torch.equal(sampled_batch.observed_class_labels, observed_class_labels)


@pytest.mark.parametrize(
    "input_contract_case,batch_sample_count",
    [
        ("mixed", 2),
        ("mixed", 1),
        ("mixed", 5),
        ("same_reference", 5),
        ("empty", 2),
        ("no_eligible", 2),
    ],
)
def test_training_batch_sampling_matches_legacy(input_contract_case, batch_sample_count):
    held_model_ids, ordered_model_training_samples, python_random_generator, legacy_client = (
        build_sampling_oracle_inputs(
            batch_sample_count=batch_sample_count, input_contract_case=input_contract_case
        )
    )
    global_python_random_state = random.getstate()
    for _ in range(3):
        legacy_batches, expected_random_state = run_legacy_training_batch_sampling(
            legacy_client=legacy_client, python_random_generator=python_random_generator
        )
        actual_random_state = python_random_generator.getstate()
        sampled_batches = sample_training_batches_for_held_models(
            held_model_ids=held_model_ids,
            ordered_model_training_samples=ordered_model_training_samples,
            batch_sample_count=batch_sample_count,
            python_random_generator=python_random_generator,
        )
        assert_sampled_batches_equal(sampled_batches=sampled_batches, legacy_batches=legacy_batches)
        assert python_random_generator.getstate() == expected_random_state
        assert random.getstate() == global_python_random_state
        assert tuple(sampled_batch.model_id for sampled_batch in sampled_batches) == (
            () if input_contract_case in ("empty", "no_eligible") else (8, -7)
        )
        if not sampled_batches:
            assert python_random_generator.getstate() == actual_random_state
        else:
            for sampled_batch in sampled_batches:
                assert sampled_batch.input_features.shape == (batch_sample_count, 2)
                assert sampled_batch.observed_class_labels.shape == (batch_sample_count, 1)
                if input_contract_case != "same_reference":
                    assert (
                        len(set(sampled_batch.input_features[:, 0].tolist())) == batch_sample_count
                    )
                else:
                    assert torch.equal(
                        sampled_batch.input_features,
                        sampled_batch.input_features[0:1].expand(batch_sample_count, -1),
                    )


def test_training_batch_sampling_observes_draw_order(monkeypatch):
    held_model_ids, ordered_model_training_samples, python_random_generator, legacy_client = (
        build_sampling_oracle_inputs()
    )
    monkeypatch.setattr(
        python_random_generator, "sample", Mock(wraps=python_random_generator.sample)
    )
    sampled_batches = sample_training_batches_for_held_models(
        held_model_ids=held_model_ids,
        ordered_model_training_samples=ordered_model_training_samples,
        batch_sample_count=2,
        python_random_generator=python_random_generator,
    )
    assert python_random_generator.sample.call_count == 2
    eligible_model_training_samples = (
        ordered_model_training_samples[0],
        ordered_model_training_samples[2],
    )
    for sample_call, model_training_samples in zip(
        python_random_generator.sample.call_args_list, eligible_model_training_samples
    ):
        assert sample_call.args[0] is model_training_samples.training_samples
        assert sample_call.args[1] == 2
    assert tuple(sampled_batch.model_id for sampled_batch in sampled_batches) == (8, -7)


def capture_training_sample_inputs(*, ordered_model_training_samples):
    """借用参照・Tensor値・gradを保存し、nested/metaを安全に扱う。"""
    reference_ids = [id(ordered_model_training_samples)]
    tensor_metadata = []
    tensor_values = []
    tensor_gradients = []
    for model_training_samples in ordered_model_training_samples:
        reference_ids.extend(
            (id(model_training_samples), id(model_training_samples.training_samples))
        )
        for training_sample in model_training_samples.training_samples:
            reference_ids.append(id(training_sample))
            if type(training_sample) is not ObservedTrainingSample:
                continue
            for training_tensor in (
                training_sample.input_features,
                training_sample.observed_class_labels,
            ):
                reference_ids.append(id(training_tensor))
                if not isinstance(training_tensor, torch.Tensor):
                    tensor_metadata.append((type(training_tensor).__name__, repr(training_tensor)))
                    continue
                tensor_metadata.append(
                    (
                        str(training_tensor.device),
                        str(training_tensor.dtype),
                        str(training_tensor.layout),
                        training_tensor.requires_grad,
                        None if training_tensor.is_nested else tuple(training_tensor.shape),
                    )
                )
                if training_tensor.is_nested:
                    tensor_values.extend(
                        parameter.detach().clone() for parameter in training_tensor.unbind()
                    )
                elif training_tensor.device.type != "meta":
                    tensor_values.append(training_tensor.detach().to_dense().clone())
                reference_ids.append(
                    id(training_tensor.grad) if training_tensor.grad is not None else 0
                )
                tensor_gradients.append(
                    None if training_tensor.grad is None else training_tensor.grad.detach().clone()
                )
    return (
        tuple(reference_ids),
        tuple(tensor_metadata),
        tuple(tensor_values),
        tuple(tensor_gradients),
    )


@pytest.mark.parametrize(
    "input_contract_case",
    [
        "held_type",
        "held_bool",
        "collections_list",
        "collection_type",
        "model_id_bool",
        "model_id_float",
        "duplicate_id",
        "count_bool",
        "count_zero",
        "count_negative",
        "count_float",
        "generator_none",
        "generator_system",
        "samples_list",
        "sample_type",
        "feature_type",
        "feature_dtype",
        "label_dtype",
        "feature_meta",
        "feature_sparse",
        "feature_nested",
        "label_nested",
        "feature_rank",
        "feature_zero_rows",
        "feature_two_rows",
        "feature_zero_width",
        "label_shape",
        "label_rank",
        "feature_nan",
        "label_inf",
        "feature_width_mismatch",
    ],
)
def test_training_batch_sampling_rejects_before_random_draw(input_contract_case, monkeypatch):
    held_model_ids, ordered_model_training_samples, python_random_generator, legacy_client = (
        build_sampling_oracle_inputs()
    )
    invalid_inputs = dict(
        held_model_ids=held_model_ids,
        ordered_model_training_samples=ordered_model_training_samples,
        batch_sample_count=2,
        python_random_generator=python_random_generator,
    )
    model_training_samples = ordered_model_training_samples[2]
    training_sample = model_training_samples.training_samples[-1]
    # 99番位置はこの乱数状態からの次回抽出に入らないが、事前検査の対象である。
    expected_random_generator = random.Random()
    expected_random_generator.setstate(python_random_generator.getstate())
    expected_random_generator.sample(range(5), 2)
    assert 99 not in expected_random_generator.sample(range(100), 2)
    for model_training_samples in ordered_model_training_samples:
        for training_sample in model_training_samples.training_samples:
            training_sample.input_features.grad = torch.full_like(
                training_sample.input_features, 0.25
            )
            training_sample.observed_class_labels.grad = torch.full_like(
                training_sample.observed_class_labels, -0.25
            )
    model_training_samples = ordered_model_training_samples[2]
    training_sample = model_training_samples.training_samples[-1]
    expected_error_field = r"ordered_model_training_samples\[2\]"
    if input_contract_case == "held_type":
        invalid_inputs["held_model_ids"] = set(held_model_ids)
        expected_error_field = "held_model_ids"
    elif input_contract_case == "held_bool":
        invalid_inputs["held_model_ids"] = frozenset((True,))
        expected_error_field = "held_model_ids"
    elif input_contract_case == "collections_list":
        invalid_inputs["ordered_model_training_samples"] = list(ordered_model_training_samples)
        expected_error_field = "ordered_model_training_samples"
    elif input_contract_case == "collection_type":
        invalid_inputs["ordered_model_training_samples"] = ordered_model_training_samples[:2] + (
            None,
        )
    elif input_contract_case in ("model_id_bool", "model_id_float", "duplicate_id"):
        invalid_inputs["ordered_model_training_samples"] = ordered_model_training_samples[:2] + (
            ModelTrainingSampleCollection(
                model_id={"model_id_bool": True, "model_id_float": 0.5, "duplicate_id": 8}[
                    input_contract_case
                ],
                training_samples=model_training_samples.training_samples,
            ),
        )
    elif input_contract_case.startswith("count_"):
        invalid_inputs["batch_sample_count"] = {
            "count_bool": True,
            "count_zero": 0,
            "count_negative": -1,
            "count_float": 2.0,
        }[input_contract_case]
        expected_error_field = "batch_sample_count"
    elif input_contract_case.startswith("generator_"):
        invalid_inputs["python_random_generator"] = (
            None if input_contract_case == "generator_none" else random.SystemRandom()
        )
        expected_error_field = "python_random_generator"
    elif input_contract_case == "samples_list":
        invalid_inputs["ordered_model_training_samples"] = ordered_model_training_samples[:2] + (
            ModelTrainingSampleCollection(
                model_id=-7, training_samples=list(model_training_samples.training_samples)
            ),
        )
    elif input_contract_case == "sample_type":
        object.__setattr__(
            model_training_samples,
            "training_samples",
            model_training_samples.training_samples[:-1] + (None,),
        )
    else:
        input_features = training_sample.input_features
        observed_class_labels = training_sample.observed_class_labels
        if input_contract_case == "feature_type":
            input_features = [[1.0, 2.0]]
        elif input_contract_case == "feature_dtype":
            input_features = input_features.double()
        elif input_contract_case == "label_dtype":
            observed_class_labels = observed_class_labels.long()
        elif input_contract_case == "feature_meta":
            input_features = input_features.to("meta")
        elif input_contract_case == "feature_sparse":
            input_features = input_features.to_sparse()
        elif input_contract_case in ("feature_nested", "label_nested"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                training_tensor = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)])
            if input_contract_case == "feature_nested":
                input_features = training_tensor
            else:
                observed_class_labels = training_tensor
        elif input_contract_case == "feature_rank":
            input_features = input_features.squeeze(0)
        elif input_contract_case == "feature_zero_rows":
            input_features = input_features[:0]
        elif input_contract_case == "feature_two_rows":
            input_features = input_features.repeat(2, 1)
        elif input_contract_case == "feature_zero_width":
            input_features = input_features[:, :0]
        elif input_contract_case == "label_shape":
            observed_class_labels = torch.ones(1, 2)
        elif input_contract_case == "label_rank":
            observed_class_labels = observed_class_labels.view(-1)
        elif input_contract_case == "feature_nan":
            input_features = torch.full_like(input_features, float("nan"))
        elif input_contract_case == "label_inf":
            observed_class_labels = torch.full_like(observed_class_labels, float("inf"))
        else:
            input_features = torch.ones(1, 3)
        object.__setattr__(training_sample, "input_features", input_features)
        object.__setattr__(training_sample, "observed_class_labels", observed_class_labels)
    input_snapshot_before_sampling = capture_training_sample_inputs(
        ordered_model_training_samples=ordered_model_training_samples
    )
    expected_random_state = python_random_generator.getstate()
    global_python_random_state = random.getstate()
    monkeypatch.setattr(
        python_random_generator, "sample", Mock(wraps=python_random_generator.sample)
    )
    with pytest.raises(ValueError, match=expected_error_field):
        sample_training_batches_for_held_models(**invalid_inputs)
    python_random_generator.sample.assert_not_called()
    assert python_random_generator.getstate() == expected_random_state
    assert random.getstate() == global_python_random_state
    input_snapshot_after_sampling = capture_training_sample_inputs(
        ordered_model_training_samples=ordered_model_training_samples
    )
    assert input_snapshot_after_sampling[:2] == input_snapshot_before_sampling[:2]
    torch.testing.assert_close(
        input_snapshot_after_sampling[2:],
        input_snapshot_before_sampling[2:],
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_training_batch_sampling_skips_ineligible_payloads(monkeypatch):
    held_model_ids, ordered_model_training_samples, python_random_generator, legacy_client = (
        build_sampling_oracle_inputs()
    )
    # 未保有recordにはpayload field自体がなくても、読むことなく省略できる。
    object.__delattr__(ordered_model_training_samples[1], "training_samples")
    object.__setattr__(ordered_model_training_samples[-1], "training_samples", (object(),))
    monkeypatch.setattr(
        python_random_generator, "sample", Mock(wraps=python_random_generator.sample)
    )
    sampled_batches = sample_training_batches_for_held_models(
        held_model_ids=held_model_ids,
        ordered_model_training_samples=ordered_model_training_samples,
        batch_sample_count=2,
        python_random_generator=python_random_generator,
    )
    assert tuple(sampled_batch.model_id for sampled_batch in sampled_batches) == (8, -7)
    assert python_random_generator.sample.call_count == 2
    assert "training_samples" not in vars(ordered_model_training_samples[1])
    expected_random_state = python_random_generator.getstate()
    python_random_generator.sample.reset_mock()
    object.__setattr__(ordered_model_training_samples[-1], "training_samples", [object()])
    with pytest.raises(ValueError, match=r"ordered_model_training_samples\[4\].training_samples"):
        sample_training_batches_for_held_models(
            held_model_ids=held_model_ids,
            ordered_model_training_samples=ordered_model_training_samples,
            batch_sample_count=2,
            python_random_generator=python_random_generator,
        )
    python_random_generator.sample.assert_not_called()
    assert python_random_generator.getstate() == expected_random_state


@pytest.mark.parametrize("batch_sample_count", [1, 2])
@pytest.mark.parametrize("require_grad", [True, False])
def test_training_batch_sampling_preserves_borrowed_inputs_and_environment(
    batch_sample_count, require_grad
):
    ordered_model_training_samples = tuple(
        ModelTrainingSampleCollection(
            model_id=model_id,
            training_samples=tuple(
                ObservedTrainingSample(
                    input_features=torch.nn.Parameter(
                        torch.arange(
                            input_feature_count * 2, dtype=torch.float32, device="cpu"
                        ).reshape(1, -1)[:, ::2]
                    ),
                    observed_class_labels=torch.nn.Parameter(
                        torch.tensor(
                            [[0.25 if training_sample_index == 0 else 100.25]],
                            dtype=torch.float32,
                            device="cpu",
                        )
                    ),
                )
                for training_sample_index in range(2)
            ),
        )
        for model_id, input_feature_count in ((8, 4), (-7, 3))
    )
    for model_training_samples in ordered_model_training_samples:
        for training_sample in model_training_samples.training_samples:
            assert not training_sample.input_features.is_contiguous()
            training_sample.input_features.grad = torch.full_like(
                training_sample.input_features, 0.37
            )
            training_sample.observed_class_labels.grad = torch.full_like(
                training_sample.observed_class_labels, -0.37
            )
    input_snapshot_before_sampling = capture_training_sample_inputs(
        ordered_model_training_samples=ordered_model_training_samples
    )
    python_random_generator = random.Random(137)
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    global_torch_random_state = torch.get_rng_state()
    original_default_dtype = torch.get_default_dtype()
    original_default_device = torch.get_default_device()
    original_grad_mode = torch.is_grad_enabled()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"), torch.set_grad_enabled(require_grad):
            sampled_batches = sample_training_batches_for_held_models(
                held_model_ids=frozenset((8, -7)),
                ordered_model_training_samples=ordered_model_training_samples,
                batch_sample_count=batch_sample_count,
                python_random_generator=python_random_generator,
            )
            assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_device() == torch.device("meta")
            assert torch.is_grad_enabled() is require_grad
            for sampled_batch, model_training_samples in zip(
                sampled_batches, ordered_model_training_samples
            ):
                assert sampled_batch.model_id == model_training_samples.model_id
                assert sampled_batch.input_features.shape == (
                    batch_sample_count,
                    model_training_samples.training_samples[0].input_features.shape[1],
                )
                for tensor_name in ("input_features", "observed_class_labels"):
                    training_tensor = getattr(sampled_batch, tensor_name)
                    assert training_tensor.dtype == torch.float32
                    assert training_tensor.device.type == "cpu"
                    assert training_tensor.requires_grad is require_grad
                    for training_sample in model_training_samples.training_samples:
                        assert (
                            training_tensor.data_ptr()
                            != getattr(training_sample, tensor_name).data_ptr()
                        )
                if require_grad:
                    assert sampled_batch.input_features.grad_fn is not None
                    assert sampled_batch.observed_class_labels.grad_fn is not None
            assert torch.equal(torch.get_rng_state(), global_torch_random_state)
        # 出力を更新しても、元のParameterと既存gradへ逆流しない。
        with torch.no_grad():
            sampled_batches[0].input_features.add_(10)
            sampled_batches[0].observed_class_labels.add_(10)
    finally:
        torch.set_default_dtype(original_default_dtype)
    input_snapshot_after_sampling = capture_training_sample_inputs(
        ordered_model_training_samples=ordered_model_training_samples
    )
    assert input_snapshot_after_sampling[:2] == input_snapshot_before_sampling[:2]
    torch.testing.assert_close(
        input_snapshot_after_sampling[2:], input_snapshot_before_sampling[2:], rtol=0, atol=0
    )
    assert torch.get_default_dtype() == original_default_dtype
    assert torch.get_default_device() == original_default_device
    assert torch.is_grad_enabled() == original_grad_mode
    assert random.getstate() == global_python_random_state
    actual_numpy_random_state = np.random.get_state()
    assert actual_numpy_random_state[0] == global_numpy_random_state[0]
    assert np.array_equal(actual_numpy_random_state[1], global_numpy_random_state[1])
    assert actual_numpy_random_state[2:] == global_numpy_random_state[2:]


def test_training_sample_records_are_frozen_and_explicit():
    input_features = torch.ones(1, 2)
    observed_class_labels = torch.ones(1, 1)
    training_sample = ObservedTrainingSample(
        input_features=input_features, observed_class_labels=observed_class_labels
    )
    for sampling_record in (
        training_sample,
        ModelTrainingSampleCollection(model_id=-7, training_samples=(training_sample,)),
        SampledModelTrainingBatch(
            model_id=-7,
            input_features=training_sample.input_features,
            observed_class_labels=training_sample.observed_class_labels,
        ),
    ):
        for sampling_record_field in fields(sampling_record):
            assert sampling_record_field.kw_only
            assert sampling_record_field.default is MISSING
            assert sampling_record_field.default_factory is MISSING
        with pytest.raises(FrozenInstanceError):
            setattr(sampling_record, fields(sampling_record)[0].name, None)
        with pytest.raises(TypeError):
            type(sampling_record)()
        with pytest.raises(TypeError):
            type(sampling_record)(*vars(sampling_record).values())
    assert training_sample.input_features is input_features
    assert training_sample.observed_class_labels is observed_class_labels
    # constructorは参照を束ねるだけで、解釈/検査はsamplerが担当する。
    model_training_samples = ModelTrainingSampleCollection(
        model_id=-7, training_samples=(training_sample,)
    )
    assert model_training_samples.training_samples[0] is training_sample


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_training_batch_sampling_connects_fifo_and_joint_update(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    global_python_random_state = random.getstate()
    global_torch_random_state = torch.get_rng_state()
    loss_hooks = []
    try:
        torch.manual_seed(137)
        participating_training_batches, shared_parameter_optimizer, legacy_client = (
            build_joint_update_oracle_pair(
                class_count=class_count,
                batch_sample_counts=(2, 5),
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        classifiers_by_model_id = {
            model_id: training_batch.classifier
            for model_id, training_batch in zip((4, -7), participating_training_batches)
        }
        optimizers_by_model_id = {
            model_id: training_batch.concept_specific_parameter_optimizer
            for model_id, training_batch in zip((4, -7), participating_training_batches)
        }
        observed_samples_by_index = {
            sample_index: ObservedTrainingSample(
                input_features=torch.tensor(
                    [[sample_index / 7, (sample_index % 3) / 5]], dtype=torch.float32, device="cpu"
                ),
                observed_class_labels=torch.tensor(
                    [[float(sample_index % class_count)]], dtype=torch.float32, device="cpu"
                ),
            )
            for sample_index in range(12)
        }
        pending_assignment_buffer = PendingTrainingAssignmentBuffer(
            training_data_assignment_settings=TrainingDataAssignmentSettings(
                pending_assignment_buffer_capacity_samples=3
            )
        )
        training_samples_by_model_id = {4: [], -7: []}
        released_sample_indices = []
        for sample_index in observed_samples_by_index:
            pending_assignment_buffer.append_observed_sample_index(sample_index=sample_index)
            for (
                released_sample_index
            ) in pending_assignment_buffer.release_sample_indices_exceeding_capacity():
                released_sample_indices.append(released_sample_index)
                model_id = 4 if released_sample_index % 2 == 0 else -7
                training_samples_by_model_id[model_id].append(
                    observed_samples_by_index[released_sample_index]
                )
        assert tuple(released_sample_indices) == tuple(range(9))
        assert pending_assignment_buffer.get_state_snapshot().pending_sample_indices == (9, 10, 11)
        for released_sample_index in pending_assignment_buffer.drain_pending_sample_indices():
            released_sample_indices.append(released_sample_index)
            model_id = 4 if released_sample_index % 2 == 0 else -7
            training_samples_by_model_id[model_id].append(
                observed_samples_by_index[released_sample_index]
            )
        assert tuple(released_sample_indices) == tuple(observed_samples_by_index)
        assert pending_assignment_buffer.get_state_snapshot().pending_sample_indices == ()
        ordered_model_training_samples = tuple(
            ModelTrainingSampleCollection(
                model_id=model_id, training_samples=tuple(training_samples)
            )
            for model_id, training_samples in training_samples_by_model_id.items()
        )
        assert tuple(
            model_training_samples.model_id
            for model_training_samples in ordered_model_training_samples
        ) == (4, -7)
        assert all(
            training_sample is observed_samples_by_index[sample_index]
            for model_training_samples in ordered_model_training_samples
            for training_sample, sample_index in zip(
                model_training_samples.training_samples,
                range(0 if model_training_samples.model_id == 4 else 1, 12, 2),
            )
        )
        legacy_client.train_data_store = {
            model_training_samples.model_id: [
                (training_sample.input_features, training_sample.observed_class_labels, 0)
                for training_sample in model_training_samples.training_samples
            ]
            for model_training_samples in ordered_model_training_samples
        }
        legacy_client.batch_size = 3
        legacy_batches = []
        # 旧jointの内側から実旧samplerを一回だけ呼び、その戻り値を観測する。
        legacy_client._sample_training_batches = Mock(
            side_effect=lambda: (
                legacy_batches.extend(
                    _SharedRepresentationFedSDAClientMixin._sample_training_batches(legacy_client)
                ),
                legacy_batches,
            )[1]
        )
        weighted_model_losses = []
        loss_hooks = [
            model_module.loss_fn.register_forward_hook(
                lambda model_module, training_inputs, model_loss: weighted_model_losses.append(
                    model_loss.detach().clone() * len(training_inputs[1])
                )
            )
            for model_module in legacy_client.models.values()
        ]
        python_random_generator = random.Random(731)
        local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        )
        for _ in range(3):
            legacy_batches.clear()
            weighted_model_losses.clear()
            legacy_client._sample_training_batches.reset_mock()
            random.setstate(python_random_generator.getstate())
            _SharedRepresentationFedSDAClientMixin._train_heads_together(
                legacy_client, count_multiplier=1, update_backbone=update_shared_features
            )
            expected_random_state = random.getstate()
            legacy_client._sample_training_batches.assert_called_once_with()
            assert len(weighted_model_losses) == len(legacy_batches) == 2
            expected_joint_loss = float(
                (
                    sum(weighted_model_losses)
                    / sum(len(input_features) for _, input_features, _ in legacy_batches)
                ).item()
            )
            sampled_batches = sample_training_batches_for_held_models(
                held_model_ids=frozenset(classifiers_by_model_id),
                ordered_model_training_samples=ordered_model_training_samples,
                batch_sample_count=3,
                python_random_generator=python_random_generator,
            )
            assert_sampled_batches_equal(
                sampled_batches=sampled_batches, legacy_batches=legacy_batches
            )
            assert python_random_generator.getstate() == expected_random_state
            assert random.getstate() == expected_random_state
            participating_training_batches = tuple(
                ParticipatingModelTrainingBatch(
                    classifier=classifiers_by_model_id[sampled_batch.model_id],
                    concept_specific_parameter_optimizer=optimizers_by_model_id[
                        sampled_batch.model_id
                    ],
                    input_features=sampled_batch.input_features,
                    observed_class_labels=sampled_batch.observed_class_labels,
                )
                for sampled_batch in sampled_batches
            )
            actual_joint_loss = perform_joint_model_parameter_update(
                local_training_settings=local_training_settings,
                shared_feature_extractor=classifiers_by_model_id[4].feature_extractor,
                shared_parameter_optimizer=shared_parameter_optimizer,
                participating_training_batches=participating_training_batches,
                update_shared_features=update_shared_features,
            )
            assert actual_joint_loss == expected_joint_loss
            assert python_random_generator.getstate() == expected_random_state
            assert_joint_update_states_equal(
                participating_training_batches=participating_training_batches,
                shared_parameter_optimizer=shared_parameter_optimizer,
                legacy_client=legacy_client,
            )
    finally:
        for loss_hook in loss_hooks:
            loss_hook.remove()
        random.setstate(global_python_random_state)
        torch.set_rng_state(global_torch_random_state)
