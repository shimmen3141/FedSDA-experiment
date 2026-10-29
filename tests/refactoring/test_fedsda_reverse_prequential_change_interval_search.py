"""途中までの予測的符号長と、最小変化区間からの候補の刻みを検証する。"""

import hashlib
from typing import TypedDict

import numpy as np
import pytest
import torch
from test_alarm_interval_minimum_description_length_selection import (
    OPTIMIZER_SETTINGS,
    _assert_random_states_equal,
    _classifier,
    _random_states,
    _samples,
)

from federated_learning_experiments.learning.training import prequential_code_length as training
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    REINITIALIZED_LEARNER_NAME,
    select_assignment_by_minimum_description_length,
)
from federated_learning_experiments.runtime import (
    alarm_interval_minimum_description_length_selection as runtime,
)


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("learner", ("reinitialized", "warm_started"))
@pytest.mark.parametrize(
    "optimizer_settings", (OPTIMIZER_SETTINGS, SgdParameterOptimizerSettings(learning_rate=0.1))
)
def test_prefixes_equal_every_existing_total_without_changing_state(
    class_count, learner, optimizer_settings
):
    torch.manual_seed(2)
    template = _classifier(class_count=class_count)
    features, labels = _samples(count=7, class_count=class_count, seed=3)
    # 呼出し側が決めた順序を、そのまま測る。
    features, labels = features.flip(0), labels.flip(0)
    state = {name: value.clone() for name, value in template.state_dict().items()}
    arguments: dict[str, object] = dict(
        template_classifier=template,
        input_features=features,
        observed_class_labels=labels,
        parameter_optimizer_settings=optimizer_settings,
    )
    if learner == "reinitialized":
        arguments["initialization_seed"] = 11
    compute = getattr(training, f"compute_{learner}_learner_prequential_code_length")
    prefixes_compute = getattr(
        training, f"compute_{learner}_learner_prequential_code_length_prefixes"
    )
    before = _random_states()
    prefixes = prefixes_compute(**arguments)
    _assert_random_states_equal(before, _random_states())
    assert type(prefixes) is tuple
    assert len(prefixes) == len(features)
    assert all(type(value) is float for value in prefixes)
    assert prefixes[-1] == compute(**arguments)
    for count, value in enumerate(prefixes, 1):
        assert value == compute(
            **dict(arguments, input_features=features[:count], observed_class_labels=labels[:count])
        )
    assert all(torch.equal(state[name], value) for name, value in template.state_dict().items())


class DerivedInt(int):
    pass


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize(
    "learner,damage",
    [
        (learner, damage)
        for learner in ("reinitialized", "warm_started")
        for damage in ("template", "empty", "count", "seed_type", "seed_negative")
        if learner == "reinitialized" or not damage.startswith("seed")
    ],
)
def test_prefix_input_validation_matches_existing_function(class_count, learner, damage):
    features, labels = _samples(count=3, class_count=class_count)
    arguments: dict[str, object] = dict(
        template_classifier=_classifier(class_count=class_count),
        input_features=features,
        observed_class_labels=labels,
        parameter_optimizer_settings=OPTIMIZER_SETTINGS,
    )
    if learner == "reinitialized":
        arguments["initialization_seed"] = 11
    if damage == "template":
        arguments["template_classifier"] = None
    elif damage == "empty":
        arguments.update(input_features=features[:0], observed_class_labels=labels[:0])
    elif damage == "count":
        arguments["observed_class_labels"] = labels[:2]
    elif damage == "seed_type":
        arguments["initialization_seed"] = True
    else:
        arguments["initialization_seed"] = -1
    compute = getattr(training, f"compute_{learner}_learner_prequential_code_length")
    prefixes_compute = getattr(
        training, f"compute_{learner}_learner_prequential_code_length_prefixes"
    )
    with pytest.raises((TypeError, ValueError)) as original:
        compute(**arguments)
    with pytest.raises(type(original.value), match=str(original.value)):
        prefixes_compute(**arguments)


@pytest.mark.parametrize("client_id,alarm_sample_index", ((0, 0), (3, 1250), (1, 23), (12, 3)))
def test_single_pass_seed_matches_direct_sha256(client_id, alarm_sample_index):
    expected = int.from_bytes(
        hashlib.sha256(f"{client_id}:{alarm_sample_index}".encode()).digest()[:8], "big"
    )
    assert (
        runtime.derive_reinitialized_learner_single_pass_initialization_seed(
            client_id=client_id, alarm_sample_index=alarm_sample_index
        )
        == expected
    )


@pytest.mark.parametrize("field", ("client_id", "alarm_sample_index"))
@pytest.mark.parametrize("value", (-1, True, 1.0, "1", None, np.int64(1), DerivedInt(1)))
def test_single_pass_seed_rejects_invalid_counts(field, value):
    arguments = dict(client_id=0, alarm_sample_index=0)
    arguments[field] = value
    with pytest.raises(ValueError if value == -1 else TypeError):
        runtime.derive_reinitialized_learner_single_pass_initialization_seed(**arguments)


class SelectionArguments(TypedDict):
    current_model_log_losses: tuple[float, ...]
    held_model_log_losses_by_model_id: tuple[tuple[int, tuple[float, ...]], ...]
    new_model_code_lengths_by_change_interval_sample_count: dict[int, float]
    new_model_learner_names_by_change_interval_sample_count: dict[int, str]
    switching_code_length: float
    change_interval_grid_sample_count: int
    change_interval_step_sample_count: int


def _selection_arguments(*, step=1, count=12) -> SelectionArguments:
    candidates = range(5, count + 1, step) if type(step) is int and step > 0 else ()
    return SelectionArguments(
        current_model_log_losses=(10.0,) * count,
        held_model_log_losses_by_model_id=(),
        new_model_code_lengths_by_change_interval_sample_count={k: 1.0 for k in candidates},
        new_model_learner_names_by_change_interval_sample_count={
            k: REINITIALIZED_LEARNER_NAME for k in candidates
        },
        switching_code_length=2.0,
        change_interval_grid_sample_count=5,
        change_interval_step_sample_count=step,
    )


@pytest.mark.parametrize("step", (1, 5))
def test_every_candidate_can_be_selected_with_the_requested_step(step):
    for candidate in range(5, 13, step):
        arguments = _selection_arguments(step=step)
        lengths = arguments["new_model_code_lengths_by_change_interval_sample_count"]
        for k in lengths:
            lengths[k] = 1000.0
        lengths[candidate] = 1.0
        selected = select_assignment_by_minimum_description_length(**arguments)
        assert (selected.selected_hypothesis, selected.change_interval_sample_count) == (
            "new_model",
            candidate,
        )
        assert selected.description_length == (10.0 * (12 - candidate) + 2.0) + 1.0


@pytest.mark.parametrize("step", (1, 5))
@pytest.mark.parametrize(
    "dictionary",
    (
        "new_model_code_lengths_by_change_interval_sample_count",
        "new_model_learner_names_by_change_interval_sample_count",
    ),
)
@pytest.mark.parametrize("damage", ("missing", "extra", "wrong"))
def test_candidate_dictionaries_must_have_exactly_the_requested_keys(step, dictionary, damage):
    arguments = _selection_arguments(step=step)
    values = arguments[dictionary]
    if damage != "extra":
        values.pop(5)
    if damage != "missing":
        values[4] = 1.0 if dictionary.startswith("new_model_code") else REINITIALIZED_LEARNER_NAME
    with pytest.raises(ValueError, match="cover exactly"):
        select_assignment_by_minimum_description_length(**arguments)


@pytest.mark.parametrize("step", (0, -1, True, 1.0, "1", None, np.int64(1), DerivedInt(1)))
def test_step_must_be_a_positive_builtin_int(step):
    with pytest.raises(
        ValueError if type(step) is int else TypeError, match="change_interval_step_sample_count"
    ):
        select_assignment_by_minimum_description_length(**_selection_arguments(step=step))


def test_step_is_required():
    arguments = _selection_arguments()
    # 必須引数を意図的に取り除き、呼出し時の検査を確かめる。
    del arguments["change_interval_step_sample_count"]  # pyright: ignore[reportGeneralTypeIssues]
    with pytest.raises(TypeError, match="change_interval_step_sample_count"):
        select_assignment_by_minimum_description_length(**arguments)


def test_dense_grid_keeps_tie_priority():
    arguments = _selection_arguments(count=7)
    arguments.update(
        current_model_log_losses=(1.0,) * 7,
        held_model_log_losses_by_model_id=((8, (1.0,) * 7), (3, (1.0,) * 7)),
        new_model_code_lengths_by_change_interval_sample_count={5: 5.0, 6: 6.0, 7: 7.0},
        switching_code_length=0.0,
    )
    assert (
        select_assignment_by_minimum_description_length(**arguments).selected_hypothesis
        == "current_model"
    )
    # 全ての切替えが1.0で並ぶと、短い区間、保有の順、新しいモデルの順。
    arguments.update(
        current_model_log_losses=(0.0, 0.0) + (2.0,) * 5,
        held_model_log_losses_by_model_id=((8, (0.0,) * 7), (3, (0.0,) * 7)),
        new_model_code_lengths_by_change_interval_sample_count={5: 0.0, 6: 0.0, 7: 0.0},
        switching_code_length=1.0,
    )
    selected = select_assignment_by_minimum_description_length(**arguments)
    assert (selected.change_interval_sample_count, selected.destination_model_id) == (5, 8)
    arguments["held_model_log_losses_by_model_id"] = ()
    selected = select_assignment_by_minimum_description_length(**arguments)
    assert (selected.selected_hypothesis, selected.change_interval_sample_count) == ("new_model", 5)


def test_sequence_shorter_than_minimum_has_no_candidates():
    selected = select_assignment_by_minimum_description_length(**_selection_arguments(count=4))
    assert selected.selected_hypothesis == "current_model"
    assert selected.description_length == 40.0
