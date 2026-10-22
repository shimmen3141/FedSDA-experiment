"""警報の後の変化区間の長さと帰属先を、モデルを使って最小記述長の原理で選ぶ計算を検証する。"""

import hashlib
import math
import random

import numpy as np
import pytest
import torch

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
    evaluate_classifier_per_sample_log_losses,
)
from federated_learning_experiments.learning.training import (
    prequential_code_length as prequential_module,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.prequential_code_length import (
    compute_reinitialized_learner_prequential_code_length,
    compute_warm_started_learner_prequential_code_length,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    REINITIALIZED_LEARNER_NAME,
    compute_switching_code_length,
    select_assignment_by_minimum_description_length,
)
from federated_learning_experiments.runtime import (
    alarm_interval_minimum_description_length_selection as selection_module,
)
from federated_learning_experiments.runtime.alarm_interval_minimum_description_length_selection import (
    derive_reinitialized_learner_initialization_seed,
    select_alarm_interval_assignment,
)

OPTIMIZER_SETTINGS = AdamParameterOptimizerSettings(
    learning_rate=0.05, weight_decay=0.0, adam_variant="standard"
)


def _classifier(
    *, class_count, model_architecture_name="shared_backbone_residual_adapter", shared=None
):
    return ModelArchitectureClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name=model_architecture_name, residual_adapter_requested_rank=2
        ),
        input_feature_count=3,
        hidden_layer_widths=(5,),
        class_count=class_count,
        shared_feature_extractor=shared,
    )


def _samples(*, count, class_count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, 3, generator=generator)
    if class_count == 2:
        labels = torch.randint(0, 2, (count, 1), generator=generator).float()
    else:
        labels = torch.randint(0, class_count, (count, 1), generator=generator).float()
    return features, labels


@pytest.mark.parametrize("class_count", (2, 4))
def test_log_losses_are_negative_log_true_class_probabilities(class_count):
    torch.manual_seed(1)
    classifier = _classifier(class_count=class_count)
    features, labels = _samples(count=6, class_count=class_count)
    bounded = evaluate_classifier_per_sample_bounded_losses(
        classifier=classifier, input_features=features, observed_class_labels=labels
    )
    log_losses = evaluate_classifier_per_sample_log_losses(
        classifier=classifier, input_features=features, observed_class_labels=labels
    )
    assert torch.equal(log_losses, -torch.log(torch.clamp(1.0 - bounded, min=1e-12)))
    assert torch.all(log_losses >= 0)


def test_initialization_seed_uses_a_delimited_decimal_string():
    expected = int.from_bytes(hashlib.sha256(b"3:1250:15").digest()[:8], "big")
    assert (
        derive_reinitialized_learner_initialization_seed(
            client_id=3, alarm_sample_index=1250, change_interval_sample_count=15
        )
        == expected
    )
    assert derive_reinitialized_learner_initialization_seed(
        client_id=1, alarm_sample_index=23, change_interval_sample_count=4
    ) != derive_reinitialized_learner_initialization_seed(
        client_id=12, alarm_sample_index=3, change_interval_sample_count=4
    )
    for arguments in (
        dict(client_id=-1, alarm_sample_index=0, change_interval_sample_count=5),
        dict(client_id=0, alarm_sample_index=-1, change_interval_sample_count=5),
        dict(client_id=0, alarm_sample_index=0, change_interval_sample_count=0),
    ):
        with pytest.raises(ValueError):
            derive_reinitialized_learner_initialization_seed(**arguments)
    with pytest.raises(TypeError):
        derive_reinitialized_learner_initialization_seed(
            client_id=True, alarm_sample_index=0, change_interval_sample_count=5
        )


def _random_states():
    return torch.get_rng_state().clone(), random.getstate(), np.random.get_state()


def _assert_random_states_equal(before, after):
    assert torch.equal(before[0], after[0])
    assert before[1] == after[1]
    assert (
        before[2][0] == after[2][0]
        and np.array_equal(before[2][1], after[2][1])
        and before[2][2:] == after[2][2:]
    )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize(
    "optimizer_settings", (OPTIMIZER_SETTINGS, SgdParameterOptimizerSettings(learning_rate=0.1))
)
def test_prequential_code_length_does_not_change_the_template_or_run_randomness(
    class_count, optimizer_settings
):
    torch.manual_seed(2)
    template = _classifier(class_count=class_count)
    features, labels = _samples(count=10, class_count=class_count, seed=3)
    template_state = {name: value.clone() for name, value in template.state_dict().items()}
    random.seed(5)
    np.random.seed(6)
    before = _random_states()
    code_length = compute_reinitialized_learner_prequential_code_length(
        template_classifier=template,
        input_features=features,
        observed_class_labels=labels,
        parameter_optimizer_settings=optimizer_settings,
        initialization_seed=11,
    )
    _assert_random_states_equal(before, _random_states())
    assert all(
        torch.equal(template_state[name], value) for name, value in template.state_dict().items()
    )
    assert math.isfinite(code_length) and code_length > 0
    # 同じseedなら同じ値、違うseedなら（この入力では）違う値。
    again = compute_reinitialized_learner_prequential_code_length(
        template_classifier=template,
        input_features=features,
        observed_class_labels=labels,
        parameter_optimizer_settings=optimizer_settings,
        initialization_seed=11,
    )
    other = compute_reinitialized_learner_prequential_code_length(
        template_classifier=template,
        input_features=features,
        observed_class_labels=labels,
        parameter_optimizer_settings=optimizer_settings,
        initialization_seed=12,
    )
    assert again == code_length and other != code_length


def test_prequential_code_length_predicts_each_sample_before_learning_it():
    """最初の標本の符号長は、初期化し直しただけ（学習の前）の学習者の対数損失。共有部は更新しない。"""
    torch.manual_seed(4)
    template = _classifier(class_count=2)
    features, labels = _samples(count=1, class_count=2, seed=7)
    shared_before = {
        name: value.clone() for name, value in template.feature_extractor.state_dict().items()
    }
    single = compute_reinitialized_learner_prequential_code_length(
        template_classifier=template,
        input_features=features,
        observed_class_labels=labels,
        parameter_optimizer_settings=OPTIMIZER_SETTINGS,
        initialization_seed=21,
    )
    # 学習者の初期化は、写しの概念ごとの部分の初期化し直しだけで決まる（共有部は写しの値）。2件目を足しても、
    # 1件目の符号長は同じ（予測してから学ぶ）。
    features_two, labels_two = _samples(count=2, class_count=2, seed=7)
    features_two[0] = features[0]
    labels_two[0] = labels[0]
    calls = []
    original = prequential_module.evaluate_classifier_per_sample_log_losses

    def recording(**arguments):
        losses = original(**arguments)
        calls.append(float(losses.sum()))
        return losses

    prequential_module.evaluate_classifier_per_sample_log_losses = recording
    try:
        compute_reinitialized_learner_prequential_code_length(
            template_classifier=template,
            input_features=features_two,
            observed_class_labels=labels_two,
            parameter_optimizer_settings=OPTIMIZER_SETTINGS,
            initialization_seed=21,
        )
    finally:
        prequential_module.evaluate_classifier_per_sample_log_losses = original
    assert calls[0] == single
    assert all(
        torch.equal(shared_before[name], value)
        for name, value in template.feature_extractor.state_dict().items()
    )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize(
    "optimizer_settings", (OPTIMIZER_SETTINGS, SgdParameterOptimizerSettings(learning_rate=0.1))
)
def test_warm_started_learner_does_not_change_the_template_or_run_randomness(
    class_count, optimizer_settings
):
    torch.manual_seed(2)
    template = _classifier(class_count=class_count)
    features, labels = _samples(count=10, class_count=class_count, seed=3)
    template_state = {name: value.clone() for name, value in template.state_dict().items()}
    random.seed(5)
    np.random.seed(6)
    before = _random_states()
    code_length = compute_warm_started_learner_prequential_code_length(
        template_classifier=template,
        input_features=features,
        observed_class_labels=labels,
        parameter_optimizer_settings=optimizer_settings,
    )
    _assert_random_states_equal(before, _random_states())
    assert all(
        torch.equal(template_state[name], value) for name, value in template.state_dict().items()
    )
    assert math.isfinite(code_length) and code_length > 0
    assert code_length == compute_warm_started_learner_prequential_code_length(
        template_classifier=template,
        input_features=features,
        observed_class_labels=labels,
        parameter_optimizer_settings=optimizer_settings,
    )


@pytest.mark.parametrize("class_count", (2, 4))
def test_warm_started_learner_starts_from_the_template_without_reinitialization(class_count):
    """学習率0なら学ばないので、符号長は、写しのもとの分類器の、標本ごとの対数損失の和（同じ足す順）。"""
    torch.manual_seed(9)
    template = _classifier(class_count=class_count)
    features, labels = _samples(count=7, class_count=class_count, seed=4)
    expected = 0.0
    for position in range(7):
        expected += float(
            evaluate_classifier_per_sample_log_losses(
                classifier=template,
                input_features=features[position : position + 1],
                observed_class_labels=labels[position : position + 1],
            ).sum()
        )
    assert (
        compute_warm_started_learner_prequential_code_length(
            template_classifier=template,
            input_features=features,
            observed_class_labels=labels,
            parameter_optimizer_settings=SgdParameterOptimizerSettings(learning_rate=0.0),
        )
        == expected
    )
    # 学ぶと、2件目以降は1件目で学んだ後に予測するので、和は変わる（1件目は同じ）。
    calls = []
    original = prequential_module.evaluate_classifier_per_sample_log_losses

    def recording(**arguments):
        losses = original(**arguments)
        calls.append(float(losses.sum()))
        return losses

    prequential_module.evaluate_classifier_per_sample_log_losses = recording
    try:
        learned = compute_warm_started_learner_prequential_code_length(
            template_classifier=template,
            input_features=features,
            observed_class_labels=labels,
            parameter_optimizer_settings=OPTIMIZER_SETTINGS,
        )
    finally:
        prequential_module.evaluate_classifier_per_sample_log_losses = original
    assert len(calls) == 7
    assert calls[0] == float(
        evaluate_classifier_per_sample_log_losses(
            classifier=template, input_features=features[:1], observed_class_labels=labels[:1]
        ).sum()
    )
    assert learned != expected


def test_warm_started_learner_rejects_invalid_inputs():
    torch.manual_seed(1)
    template = _classifier(class_count=2)
    features, labels = _samples(count=3, class_count=2)
    for arguments in (
        dict(input_features=features[:0], observed_class_labels=labels[:0]),
        dict(input_features=features, observed_class_labels=labels[:2]),
    ):
        with pytest.raises(ValueError):
            compute_warm_started_learner_prequential_code_length(
                template_classifier=template,
                parameter_optimizer_settings=OPTIMIZER_SETTINGS,
                **arguments,
            )
    with pytest.raises(TypeError):
        compute_warm_started_learner_prequential_code_length(
            template_classifier=object(),
            input_features=features,
            observed_class_labels=labels,
            parameter_optimizer_settings=OPTIMIZER_SETTINGS,
        )


def _observations(*, count, class_count, start_index=100, seed=0):
    features, labels = _samples(count=count, class_count=class_count, seed=seed)
    return tuple(
        IndexedObservedTrainingSample(
            sample_index=start_index + position,
            training_sample=ObservedTrainingSample(
                input_features=features[position : position + 1],
                observed_class_labels=labels[position : position + 1],
            ),
            observed_concept_id=None,
        )
        for position in range(count)
    )


def _registry(*, class_count, model_ids):
    torch.manual_seed(8)
    registry = HeldModelTrainingStateRegistry()
    shared = None
    for model_id in model_ids:
        classifier = _classifier(class_count=class_count, shared=shared)
        shared = classifier.feature_extractor
        registry.register_held_model_training_state(
            model_id=model_id,
            classifier=classifier,
            concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                parameters=tuple(classifier.model_owned_feature_transform.parameters())
                + tuple(classifier.classification_layer.parameters()),
                optimizer_settings=OPTIMIZER_SETTINGS,
            ),
        )
    return registry


@pytest.mark.parametrize("sequence_length", (3, 5, 12))
def test_selection_combines_the_model_losses_and_prequential_code_lengths(sequence_length):
    class_count = 4
    registry = _registry(class_count=class_count, model_ids=(4, 9, 2))
    loss_store = ModelAndClassLossStatisticsStore()
    # 現在のモデルは9。統計を持つのは4と9（2は持たない＝行き先にしない）。
    for model_id in (4, 9):
        loss_store.record_assigned_loss(model_id=model_id, observed_loss=0.3, observed_class_id=1)
    assignment = CurrentTrainingModelAssignment(initial_model_id=9)
    observations = _observations(count=sequence_length, class_count=class_count)
    recorded_seeds = []
    original = selection_module.compute_reinitialized_learner_prequential_code_length

    def recording(**arguments):
        recorded_seeds.append(arguments["initialization_seed"])
        return original(**arguments)

    selection_module.compute_reinitialized_learner_prequential_code_length = recording
    try:
        selection = select_alarm_interval_assignment(
            client_id=3,
            alarm_sample_index=observations[-1].sample_index,
            pending_sample_observations=observations,
            held_model_training_state_registry=registry,
            loss_statistics_store=loss_store,
            current_training_model_assignment=assignment,
            parameter_optimizer_settings=OPTIMIZER_SETTINGS,
            switching_probability=1 / 120,
            change_interval_grid_sample_count=5,
            new_model_hypothesis_learners="reinitialized_learner",
        )
    finally:
        selection_module.compute_reinitialized_learner_prequential_code_length = original
    features = torch.cat([o.training_sample.input_features for o in observations])
    labels = torch.cat([o.training_sample.observed_class_labels for o in observations])

    def losses(model_id):
        return tuple(
            evaluate_classifier_per_sample_log_losses(
                classifier=registry.get_held_model_training_state(model_id=model_id).classifier,
                input_features=features,
                observed_class_labels=labels,
            ).tolist()
        )

    spans = tuple(range(5, sequence_length + 1, 5))
    new_lengths = {
        span: compute_reinitialized_learner_prequential_code_length(
            template_classifier=registry.get_held_model_training_state(model_id=9).classifier,
            input_features=features[sequence_length - span :],
            observed_class_labels=labels[sequence_length - span :],
            parameter_optimizer_settings=OPTIMIZER_SETTINGS,
            initialization_seed=derive_reinitialized_learner_initialization_seed(
                client_id=3,
                alarm_sample_index=observations[-1].sample_index,
                change_interval_sample_count=span,
            ),
        )
        for span in spans
    }
    expected = select_assignment_by_minimum_description_length(
        current_model_log_losses=losses(9),
        held_model_log_losses_by_model_id=((4, losses(4)),),
        new_model_code_lengths_by_change_interval_sample_count=new_lengths,
        new_model_learner_names_by_change_interval_sample_count={
            span: REINITIALIZED_LEARNER_NAME for span in spans
        },
        switching_code_length=compute_switching_code_length(
            switching_probability=1 / 120, destination_count=2
        ),
        change_interval_grid_sample_count=5,
    )
    assert selection == expected
    assert recorded_seeds == [
        derive_reinitialized_learner_initialization_seed(
            client_id=3,
            alarm_sample_index=observations[-1].sample_index,
            change_interval_sample_count=span,
        )
        for span in spans
    ]


def test_selection_rejects_an_alarm_index_that_is_not_the_last_pending_sample():
    registry = _registry(class_count=2, model_ids=(1,))
    observations = _observations(count=5, class_count=2)
    with pytest.raises(ValueError):
        select_alarm_interval_assignment(
            client_id=0,
            alarm_sample_index=observations[-1].sample_index + 1,
            pending_sample_observations=observations,
            held_model_training_state_registry=registry,
            loss_statistics_store=ModelAndClassLossStatisticsStore(),
            current_training_model_assignment=CurrentTrainingModelAssignment(initial_model_id=1),
            parameter_optimizer_settings=OPTIMIZER_SETTINGS,
            switching_probability=1 / 120,
            change_interval_grid_sample_count=5,
            new_model_hypothesis_learners="reinitialized_learner",
        )


@pytest.mark.parametrize("sequence_length", (5, 12))
def test_selection_uses_the_shorter_learner_with_the_learner_selection_code_length(sequence_length):
    """2つの学習者の組では、変化区間の長さごとに、初期化し直した学習者→warm startの学習者の順に計算し、
    短い方＋ln 2を「新しいモデル」の符号長にする。"""
    class_count = 4
    registry = _registry(class_count=class_count, model_ids=(4, 9))
    loss_store = ModelAndClassLossStatisticsStore()
    for model_id in (4, 9):
        loss_store.record_assigned_loss(model_id=model_id, observed_loss=0.3, observed_class_id=1)
    assignment = CurrentTrainingModelAssignment(initial_model_id=9)
    observations = _observations(count=sequence_length, class_count=class_count, seed=5)
    order = []
    original_reinitialized = selection_module.compute_reinitialized_learner_prequential_code_length
    original_warm_started = selection_module.compute_warm_started_learner_prequential_code_length

    def recording_reinitialized(**arguments):
        order.append(("reinitialized", arguments["input_features"].shape[0]))
        return original_reinitialized(**arguments)

    def recording_warm_started(**arguments):
        order.append(("warm_started", arguments["input_features"].shape[0]))
        return original_warm_started(**arguments)

    selection_module.compute_reinitialized_learner_prequential_code_length = recording_reinitialized
    selection_module.compute_warm_started_learner_prequential_code_length = recording_warm_started
    try:
        selection = select_alarm_interval_assignment(
            client_id=1,
            alarm_sample_index=observations[-1].sample_index,
            pending_sample_observations=observations,
            held_model_training_state_registry=registry,
            loss_statistics_store=loss_store,
            current_training_model_assignment=assignment,
            parameter_optimizer_settings=OPTIMIZER_SETTINGS,
            switching_probability=1 / 120,
            change_interval_grid_sample_count=5,
            new_model_hypothesis_learners="reinitialized_and_warm_started_learners",
        )
    finally:
        selection_module.compute_reinitialized_learner_prequential_code_length = (
            original_reinitialized
        )
        selection_module.compute_warm_started_learner_prequential_code_length = (
            original_warm_started
        )
    spans = tuple(range(5, sequence_length + 1, 5))
    assert order == [(kind, span) for span in spans for kind in ("reinitialized", "warm_started")]
    features = torch.cat([o.training_sample.input_features for o in observations])
    labels = torch.cat([o.training_sample.observed_class_labels for o in observations])
    template = registry.get_held_model_training_state(model_id=9).classifier
    learner_selections = {}
    for span in spans:
        reinitialized = compute_reinitialized_learner_prequential_code_length(
            template_classifier=template,
            input_features=features[sequence_length - span :],
            observed_class_labels=labels[sequence_length - span :],
            parameter_optimizer_settings=OPTIMIZER_SETTINGS,
            initialization_seed=derive_reinitialized_learner_initialization_seed(
                client_id=1,
                alarm_sample_index=observations[-1].sample_index,
                change_interval_sample_count=span,
            ),
        )
        warm_started = compute_warm_started_learner_prequential_code_length(
            template_classifier=template,
            input_features=features[sequence_length - span :],
            observed_class_labels=labels[sequence_length - span :],
            parameter_optimizer_settings=OPTIMIZER_SETTINGS,
        )
        learner_selections[span] = (
            ("warm_started_learner", warm_started + math.log(2.0))
            if warm_started < reinitialized
            else ("reinitialized_learner", reinitialized + math.log(2.0))
        )

    def losses(model_id):
        return tuple(
            evaluate_classifier_per_sample_log_losses(
                classifier=registry.get_held_model_training_state(model_id=model_id).classifier,
                input_features=features,
                observed_class_labels=labels,
            ).tolist()
        )

    expected = select_assignment_by_minimum_description_length(
        current_model_log_losses=losses(9),
        held_model_log_losses_by_model_id=((4, losses(4)),),
        new_model_code_lengths_by_change_interval_sample_count={
            span: value for span, (_, value) in learner_selections.items()
        },
        new_model_learner_names_by_change_interval_sample_count={
            span: name for span, (name, _) in learner_selections.items()
        },
        switching_code_length=compute_switching_code_length(
            switching_probability=1 / 120, destination_count=2
        ),
        change_interval_grid_sample_count=5,
    )
    assert selection == expected
