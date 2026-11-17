"""DWMのclientを、FedAvgのclient（学習器の正本）と、図1の規則とで確かめる。"""

from copy import deepcopy
from random import Random

import pytest
import torch
from numpy.random import RandomState

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.observed_sample_generation import (
    create_observed_sample_generator,
)
from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.methods.dynamic_weighted_majority.dynamic_weighted_majority_client import (
    DynamicWeightedMajorityClient,
)
from federated_learning_experiments.methods.dynamic_weighted_majority.dynamic_weighted_majority_rules import (
    vote_weighted_majority,
)
from federated_learning_experiments.methods.fedavg.fedavg_client import FedavgClient
from federated_learning_experiments.runtime.fedavg_run_settings import (
    build_default_fedavg_run_settings,
)
from federated_learning_experiments.runtime.initial_model_pretraining import pretrain_initial_model
from tests.refactoring.test_feddrift_client_round_training import assert_values_equal

BATCH = 8


def pretrained_sine2():
    """`fedavg`の既定の設定でSINE2を事前学習し、初期状態と、続きの標本の生成器を返す。"""
    conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=1,
        per_client_sample_count=200,
        server_aggregation_interval_per_client_samples=50,
    )
    settings = build_default_fedavg_run_settings(experiment_run_conditions=conditions)
    participant = settings.run_participant_settings
    generator = create_observed_sample_generator(
        dataset_name="sine2", numpy_random_generator=RandomState(0)
    )
    pretrained = pretrain_initial_model(
        initial_model_pretraining_settings=participant.initial_model_pretraining_settings,
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="independent_models_without_shared_backbone",
            residual_adapter_requested_rank=1,
        ),
        input_feature_count=2,
        hidden_layer_widths=participant.hidden_layer_widths,
        class_count=2,
        parameter_optimizer_settings=participant.parameter_optimizer_settings,
        sample_generator=generator,
        python_random_generator=Random(0),
    )
    return pretrained, generator


def stream(generator, concepts):
    return [(generator.generate_sample(concept_id=c), c) for c in concepts]


def make_client(pretrained, *, period, beta=0.5, theta=0.01, seed=17):
    return DynamicWeightedMajorityClient(
        client_id=0,
        initial_classifier=pretrained.classifier,
        initial_parameter_optimizer_state=pretrained.concept_specific_parameter_optimizer_state,
        local_training_batch_sample_count=BATCH,
        updates_per_sample=1,
        weight_decrease_factor=beta,
        removal_weight_threshold=theta,
        period_sample_count=period,
        python_random_generator=Random(seed),
    )


def feed(client, samples, concept_override=None):
    for index, (sample, concept) in enumerate(samples):
        client.process_observed_sample(
            observed_sample=sample,
            sample_index=index,
            evaluation_concept_id=concept if concept_override is None else concept_override,
        )


def expert_predictions(client, sample):
    x = torch.tensor([sample.feature_values], dtype=torch.float32)
    predictions = []
    with torch.no_grad():
        for expert in client.experts:
            scores = expert.classifier(x)
            predictions.append(int((scores > 0.5).item()))
    return tuple(predictions)


def test_single_expert_without_period_matches_the_fedavg_client():
    pretrained, generator = pretrained_sine2()
    samples = stream(generator, [0] * 40 + [1] * 40)
    dwm = make_client(pretrained, period=10_000)
    fedavg = FedavgClient(
        client_id=0,
        initial_classifier=pretrained.classifier,
        initial_parameter_optimizer_state=pretrained.concept_specific_parameter_optimizer_state,
        initial_loss_statistics=pretrained.loss_statistics,
        local_training_batch_sample_count=BATCH,
        updates_per_sample=1,
        local_update_interval_per_client_samples=1,
        python_random_generator=Random(17),
    )
    for index, (sample, concept) in enumerate(samples):
        dwm.process_observed_sample(
            observed_sample=sample, sample_index=index, evaluation_concept_id=concept
        )
        fedavg.process_observed_sample(
            observed_sample=sample, sample_index=index, evaluation_concept_id=concept
        )
    assert [p.predicted_class_id for p in dwm.sample_predictions] == [
        p.predicted_class_id for p in fedavg.sample_predictions
    ]
    (expert,) = dwm.experts
    for a, b in zip(expert.classifier.parameters(), fedavg.classifier.parameters(), strict=True):
        assert torch.equal(a, b)
    assert_values_equal(
        expert.parameter_optimizer_state.parameter_optimizer.state_dict(),
        fedavg.parameter_optimizer_state.parameter_optimizer.state_dict(),
    )
    assert dwm.python_random_generator.getstate() == fedavg.python_random_generator.getstate()
    assert expert.trained_sample_count == fedavg.trained_sample_count
    assert expert.parameter_update_step_count == fedavg.parameter_update_step_count


def test_period_updates_follow_figure_1_and_evaluation_uses_weights_before_learning():
    pretrained, generator = pretrained_sine2()
    initial_parameters = [p.detach().clone() for p in pretrained.classifier.parameters()]
    samples = stream(generator, [0] * 30 + [1] * 70)
    client = make_client(pretrained, period=5)
    for index, (sample, concept) in enumerate(samples):
        weights_before = tuple(e.weight for e in client.experts)
        ids_before = tuple(e.expert_id for e in client.experts)
        expected = vote_weighted_majority(
            expert_predicted_class_ids=expert_predictions(client, sample),
            expert_weights=weights_before,
            class_count=2,
        )
        client.process_observed_sample(
            observed_sample=sample, sample_index=index, evaluation_concept_id=concept
        )
        record = client.sample_predictions[-1]
        assert record.predicted_class_id == expected
        assert record.held_model_count_at_prediction == len(ids_before)
        if (index + 1) % 5 != 0:
            assert tuple(e.weight for e in client.experts) == weights_before
            assert tuple(e.expert_id for e in client.experts) == ids_before
        else:
            assert max(e.weight for e in client.experts) == 1.0
    # ラベルが反転するコンセプト1で専門家が足される。
    lifetimes = {life.expert_id: life for life in client.expert_lifetimes}
    added = [life for life in lifetimes.values() if life.created_sample_index >= 0]
    assert added
    assert all((life.created_sample_index + 1) % 5 == 0 for life in added)
    assert all(
        life.removed_sample_index == -1 or (life.removed_sample_index + 1) % 5 == 0
        for life in lifetimes.values()
    )
    alive = {e.expert_id for e in client.experts}
    assert alive == {i for i, life in lifetimes.items() if life.removed_sample_index == -1}
    for expert in client.experts:
        created = lifetimes[expert.expert_id].created_sample_index
        assert len(expert.training_samples) == len(samples) - max(created, 0)
    trained_new = [
        e
        for e in client.experts
        if lifetimes[e.expert_id].created_sample_index >= 0 and len(e.training_samples) >= BATCH
    ]
    assert trained_new
    for expert in trained_new:
        assert expert.parameter_update_step_count > 0
        assert any(
            not torch.equal(a, b)
            for a, b in zip(expert.classifier.parameters(), initial_parameters, strict=True)
        )
    # 元の初期状態は変わらない。
    fresh = make_client(pretrained, period=5)
    for a, b in zip(fresh.experts[0].classifier.parameters(), initial_parameters, strict=True):
        assert torch.equal(a, b)


def test_true_concepts_do_not_change_prediction_or_learning():
    pretrained, generator = pretrained_sine2()
    samples = stream(generator, [0] * 30 + [1] * 50)
    a = make_client(pretrained, period=5)
    b = make_client(pretrained, period=5)
    feed(a, samples)
    feed(b, samples, concept_override=3)
    assert [p.predicted_class_id for p in a.sample_predictions] == [
        p.predicted_class_id for p in b.sample_predictions
    ]
    assert [p.observed_concept_id for p in b.sample_predictions] == [3] * len(samples)
    assert a.python_random_generator.getstate() == b.python_random_generator.getstate()
    assert [e.expert_id for e in a.experts] == [e.expert_id for e in b.experts]
    for ea, eb in zip(a.experts, b.experts, strict=True):
        assert ea.weight == eb.weight
        for pa, pb in zip(ea.classifier.parameters(), eb.classifier.parameters(), strict=True):
            assert torch.equal(pa, pb)


@pytest.mark.parametrize(
    "change",
    ["not_sample", "skip_index", "negative_concept", "bool_index", "wrong_feature_count"],
)
def test_invalid_inputs_are_rejected_before_any_state_change(change):
    pretrained, generator = pretrained_sine2()
    samples = stream(generator, [0] * 12)
    client = make_client(pretrained, period=5)
    feed(client, samples[:10])
    before_rng = client.python_random_generator.getstate()
    before_records = len(client.sample_predictions)
    before = deepcopy(
        [
            (e.weight, [p.detach().clone() for p in e.classifier.parameters()])
            for e in client.experts
        ]
    )
    sample, concept = samples[10]
    arguments = dict(observed_sample=sample, sample_index=10, evaluation_concept_id=concept)
    if change == "not_sample":
        arguments["observed_sample"] = (sample.feature_values, sample.class_label)
    elif change == "skip_index":
        arguments["sample_index"] = 11
    elif change == "negative_concept":
        arguments["evaluation_concept_id"] = -1
    elif change == "bool_index":
        arguments["sample_index"] = True
    else:
        arguments["observed_sample"] = ObservedSample(feature_values=(0.1, 0.2, 0.3), class_label=1)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        client.process_observed_sample(**arguments)
    assert client.python_random_generator.getstate() == before_rng
    assert len(client.sample_predictions) == before_records
    after = [(e.weight, list(e.classifier.parameters())) for e in client.experts]
    assert len(after) == len(before)
    for (wa, pa), (wb, pb) in zip(after, before, strict=True):
        assert wa == wb
        assert all(torch.equal(x, y) for x, y in zip(pa, pb, strict=True))


@pytest.mark.parametrize(
    "changes",
    [dict(beta=0.0), dict(beta=1.0), dict(theta=0.0), dict(theta=1.0), dict(period=0)],
)
def test_invalid_settings_are_rejected(changes):
    pretrained, _ = pretrained_sine2()
    with pytest.raises((TypeError, ValueError)):
        make_client(pretrained, **{"period": 5, **changes})
