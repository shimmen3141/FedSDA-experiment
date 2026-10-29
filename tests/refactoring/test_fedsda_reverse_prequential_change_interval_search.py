"""途中までの予測的符号長と、最小変化区間からの候補の刻みを検証する。"""

import hashlib
from dataclasses import replace
from typing import TypedDict

import numpy as np
import pytest
import torch
from test_alarm_interval_minimum_description_length_selection import (
    OPTIMIZER_SETTINGS,
    _assert_random_states_equal,
    _classifier,
    _observations,
    _random_states,
    _registry,
    _samples,
)
from test_fedsda_minimum_description_length_alarm_interval_assignment import build_small_settings

from federated_learning_experiments.core.settings_field_validation import RunSettingsValidationError
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.training import prequential_code_length as training
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment import (
    training_data_assignment_settings as assignment_settings,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    REINITIALIZED_LEARNER_NAME,
    select_assignment_by_minimum_description_length,
)
from federated_learning_experiments.runtime import (
    alarm_interval_minimum_description_length_selection as runtime,
)
from federated_learning_experiments.sweep import experiment_configurations as configurations

FORWARD = "minimum_interval_multiples_with_forward_prequential_code"
REVERSE = "every_sample_with_reverse_prequential_code"
CONFIGURATION = "fedsda-every-sample-change-interval-search"


@pytest.mark.parametrize(
    "name,previous_hash",
    (
        ("fedsda", "96d95864063a"),
        ("fedsda-without-prediction-weight-recalibration", "47b78748e2b9"),
        ("fedsda-without-candidate-validation", "bd8cc8da7273"),
        ("fedsda-loss-distance-clustering", "976646b4ae14"),
        ("fedsda-overall-loss-esr-detection", "b88b419e3de1"),
        ("fedsda-adahedge-prediction-weights", "1c19f963a361"),
        ("fedsda-single-model-prediction", "de1309ce291f"),
        ("fedsda-classwise-adwin-detection", "a388a2a44d05"),
        ("fedsda-overall-loss-adwin-detection", "a3c32a9b8eda"),
        ("fedsda-without-concept-specific-adapter", "8915b7c8ee96"),
        ("fedsda-independent-models", "a2f2a852f006"),
        ("feddrift", "c5772bf27573"),
        ("fedavg", "fb8821cf5897"),
        ("fedavg-local-updates-per-aggregation", "a69a48070103"),
        ("fedsda-synchronized-model-prediction", "1b4933095f3f"),
        ("fedsda-synchronized-local-mixture-prediction", "673db123b25b"),
        ("fedsda-local-parameter-prediction", "6b267382a889"),
    ),
)
def test_existing_configuration_values_and_non_fedsda_identifiers_are_unchanged(
    name, previous_hash
):
    """追加前に測った全構成のhashと、新項目だけを除いた設定を照合する。"""
    import json

    from test_experiment_sweep import SMALL_SWEEP_PLAN

    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )
    from federated_learning_experiments.sweep.experiment_sweep_plan import (
        build_planned_experiment_runs,
    )

    run = build_planned_experiment_runs(
        sweep_plan=replace(SMALL_SWEEP_PLAN, configuration_names=(name,), random_seeds=(0,))
    )[0]
    mapping = convert_settings_to_plain_mapping(run.run_settings)
    if name.startswith("fedsda"):
        assignment = mapping["run_participant_settings"]["run_client_settings"][
            "training_data_assignment_settings"
        ]
        assert assignment.pop("change_interval_search") == REVERSE
    else:
        assert run.run_identifier == f"circle2_interval10_seed0_{previous_hash}"
    canonical = json.dumps(
        {"configuration_name": name, "run_settings": mapping},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest()[:12] == previous_hash


class DerivedStr(str):
    pass


@pytest.mark.parametrize(
    "value", (FORWARD, REVERSE, "other", True, None, 1, DerivedStr(REVERSE), np.str_(REVERSE))
)
def test_change_interval_search_validation(value):
    validate = assignment_settings.validate_change_interval_search
    if type(value) is str and value in (FORWARD, REVERSE):
        validate(change_interval_search=value)
    else:
        with pytest.raises(ValueError if type(value) is str else TypeError):
            validate(change_interval_search=value)


def test_search_setting_constants_and_incompatible_policy():
    assert assignment_settings.CHANGE_INTERVAL_SEARCH_NAMES == (FORWARD, REVERSE)
    assert (
        assignment_settings.MINIMUM_INTERVAL_MULTIPLES_WITH_FORWARD_PREQUENTIAL_CODE_SEARCH_NAME
        == FORWARD
    )
    assert assignment_settings.EVERY_SAMPLE_WITH_REVERSE_PREQUENTIAL_CODE_SEARCH_NAME == REVERSE
    client = build_small_settings().run_participant_settings.run_client_settings
    assignment = client.training_data_assignment_settings
    assert assignment.change_interval_search == FORWARD
    for value in (FORWARD, REVERSE):
        assert replace(
            client,
            training_data_assignment_settings=replace(assignment, change_interval_search=value),
        )
    with pytest.raises(RunSettingsValidationError):
        replace(assignment, change_interval_search="other")
    with pytest.raises(RunSettingsValidationError, match="change_interval_search"):
        replace(
            client,
            training_data_assignment_settings=replace(
                assignment,
                change_interval_search=REVERSE,
                alarm_interval_assignment_policy="detector_change_span_and_mean_loss_increase_tolerance",
            ),
        )


def _runtime_arguments(count, learners):
    observations = _observations(count=count, class_count=2)
    return dict(
        client_id=3,
        alarm_sample_index=observations[-1].sample_index,
        pending_sample_observations=observations,
        held_model_training_state_registry=_registry(class_count=2, model_ids=(1,)),
        loss_statistics_store=ModelAndClassLossStatisticsStore(),
        current_training_model_assignment=CurrentTrainingModelAssignment(initial_model_id=1),
        parameter_optimizer_settings=OPTIMIZER_SETTINGS,
        switching_probability=1 / 120,
        change_interval_grid_sample_count=5,
        new_model_hypothesis_learners=learners,
    )


def _fail(**arguments):
    raise AssertionError("unexpected learner computation")


@pytest.mark.parametrize(
    "learners", ("reinitialized_learner", "reinitialized_and_warm_started_learners")
)
@pytest.mark.parametrize("count", (3, 5, 12))
def test_forward_search_never_calls_single_pass_functions(monkeypatch, learners, count):
    arguments = _runtime_arguments(count, learners)
    expected = runtime.select_alarm_interval_assignment(**arguments, change_interval_search=FORWARD)
    for name in (
        "compute_reinitialized_learner_prequential_code_length_prefixes",
        "compute_warm_started_learner_prequential_code_length_prefixes",
        "derive_reinitialized_learner_single_pass_initialization_seed",
    ):
        monkeypatch.setattr(runtime, name, _fail)
    assert (
        runtime.select_alarm_interval_assignment(**arguments, change_interval_search=FORWARD)
        == expected
    )


@pytest.mark.parametrize(
    "learners", ("reinitialized_learner", "reinitialized_and_warm_started_learners")
)
@pytest.mark.parametrize("count", (3, 5, 12))
def test_reverse_search_uses_one_reversed_prefix_stream_per_learner(monkeypatch, learners, count):
    arguments = _runtime_arguments(count, learners)
    observations = arguments["pending_sample_observations"]
    features = torch.cat([o.training_sample.input_features for o in observations]).flip(0)
    labels = torch.cat([o.training_sample.observed_class_labels for o in observations]).flip(0)
    calls = []
    updates = []
    selected_lengths = []
    original_step = torch.optim.Adam.step
    original_select = runtime.select_new_model_learner
    original_pure = runtime.select_assignment_by_minimum_description_length
    pure_arguments = []

    def step(self, *args, **kwargs):
        updates.append(id(self))
        return original_step(self, *args, **kwargs)

    def prefixes(name):
        original = getattr(training, f"compute_{name}_learner_prequential_code_length_prefixes")

        def recording(**kwargs):
            assert torch.equal(kwargs["input_features"], features)
            assert torch.equal(kwargs["observed_class_labels"], labels)
            if name == "reinitialized":
                assert kwargs["initialization_seed"] == int.from_bytes(
                    hashlib.sha256(f"3:{observations[-1].sample_index}".encode()).digest()[:8],
                    "big",
                )
            start = len(updates)
            result = original(**kwargs)
            assert len(updates) - start == count
            calls.append((name, result))
            return result

        return recording

    def select(**kwargs):
        selected_lengths.append(kwargs["prequential_code_lengths_by_learner_name"])
        return original_select(**kwargs)

    def pure(**kwargs):
        pure_arguments.append(kwargs)
        return original_pure(**kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    monkeypatch.setattr(
        runtime,
        "compute_reinitialized_learner_prequential_code_length_prefixes",
        prefixes("reinitialized"),
    )
    monkeypatch.setattr(
        runtime,
        "compute_warm_started_learner_prequential_code_length_prefixes",
        prefixes("warm_started"),
    )
    monkeypatch.setattr(runtime, "compute_reinitialized_learner_prequential_code_length", _fail)
    monkeypatch.setattr(runtime, "compute_warm_started_learner_prequential_code_length", _fail)
    monkeypatch.setattr(runtime, "derive_reinitialized_learner_initialization_seed", _fail)
    if count < 5:
        monkeypatch.setattr(
            runtime, "derive_reinitialized_learner_single_pass_initialization_seed", _fail
        )
    monkeypatch.setattr(runtime, "select_new_model_learner", select)
    monkeypatch.setattr(runtime, "select_assignment_by_minimum_description_length", pure)
    result = runtime.select_alarm_interval_assignment(**arguments, change_interval_search=REVERSE)
    names = ["reinitialized"] + (["warm_started"] if learners.endswith("learners") else [])
    assert [name for name, _ in calls] == (names if count >= 5 else [])
    assert pure_arguments[0]["change_interval_step_sample_count"] == 1
    assert tuple(
        pure_arguments[0]["new_model_code_lengths_by_change_interval_sample_count"]
    ) == tuple(range(5, count + 1))
    assert selected_lengths == [
        tuple((f"{name}_learner", values[k - 1]) for name, values in calls)
        for k in range(5, count + 1)
    ]
    if count < 5:
        assert result.selected_hypothesis == "current_model"
        assert selected_lengths == []
        assert updates == []


def test_every_sample_configuration_matches_default_and_comparison_keeps_forward_search():
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings

    assert configurations.EVERY_SAMPLE_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME == CONFIGURATION
    assert CONFIGURATION in configurations.EXPERIMENT_CONFIGURATION_NAMES
    conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=2,
        per_client_sample_count=200,
        server_aggregation_interval_per_client_samples=50,
    )
    default = configurations.build_experiment_run_settings(
        configuration_name="fedsda", experiment_run_conditions=conditions
    )
    new = configurations.build_experiment_run_settings(
        configuration_name=CONFIGURATION, experiment_run_conditions=conditions
    )
    client = new.run_participant_settings.run_client_settings
    assert client.training_data_assignment_settings.change_interval_search == REVERSE
    assert new == default
    for name in configurations.EXPERIMENT_CONFIGURATION_NAMES:
        settings = configurations.build_experiment_run_settings(
            configuration_name=name, experiment_run_conditions=conditions
        )
        if type(settings) is FedsdaRunSettings:
            assert (
                settings.run_participant_settings.run_client_settings.training_data_assignment_settings.change_interval_search
                == (
                    FORWARD
                    if name
                    == configurations.MINIMUM_INTERVAL_MULTIPLES_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME
                    else REVERSE
                )
            )


@pytest.mark.slow
@pytest.mark.parametrize("selected_count", (3, 6))
def test_reverse_selected_count_reaches_records_waiting_and_interval_preparation(
    monkeypatch, selected_count
):
    from federated_learning_experiments.runtime import alarm_buffer_response as response
    from federated_learning_experiments.runtime import observed_sample_processing as processing
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.runtime.fedsda_run_record_tables import (
        build_fedsda_run_record_tables,
    )

    settings = build_small_settings()
    participant = settings.run_participant_settings
    client = participant.run_client_settings
    settings = replace(
        settings,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                client,
                training_data_assignment_settings=replace(
                    client.training_data_assignment_settings, change_interval_search=REVERSE
                ),
            ),
        ),
    )
    original = processing.select_alarm_interval_assignment
    original_prepare = response.prepare_alarm_training_intervals
    selections = []
    prepared = []

    def select(**kwargs):
        assert kwargs["change_interval_search"] == REVERSE
        result = original(**kwargs)
        if len(kwargs["pending_sample_observations"]) >= selected_count:
            result = replace(
                result,
                selected_hypothesis="current_model",
                change_interval_sample_count=selected_count,
                destination_model_id=None,
                new_model_learner_name="",
            )
            selections.append((kwargs["client_id"], kwargs["alarm_sample_index"]))
        return result

    def prepare(**kwargs):
        prepared.append(kwargs["estimated_change_span_sample_count"])
        result = original_prepare(**kwargs)
        observations = kwargs["pending_sample_observations"]
        if len(observations) >= selected_count:
            assert kwargs["estimated_change_span_sample_count"] == selected_count
            assert result.change_interval_observations == observations[-selected_count:]
            assert result.earlier_observations == observations[:-selected_count]
        return result

    monkeypatch.setattr(processing, "select_alarm_interval_assignment", select)
    monkeypatch.setattr(response, "prepare_alarm_training_intervals", prepare)
    run = execute_fedsda_experiment_run(run_settings=settings)
    tables = build_fedsda_run_record_tables(experiment_run=run)
    records = dict(
        zip(
            zip(
                tables["alarm_interval_assignment_selection.client_id"].tolist(),
                tables["alarm_interval_assignment_selection.alarm_sample_index"].tolist(),
                strict=True,
            ),
            zip(
                tables["alarm_interval_assignment_selection.change_interval_sample_count"].tolist(),
                tables["alarm_interval_assignment_selection.alarm_raised"].tolist(),
                strict=True,
            ),
            strict=True,
        )
    )
    assert selections
    assert all(records[key] == (selected_count, selected_count >= 5) for key in selections)
    if selected_count < 5:
        assert prepared == []
        assert tables["loss_change_alarm.alarm_sample_index"].size == 0
    else:
        assert selected_count in prepared
        alarm_keys = list(
            zip(
                tables["loss_change_alarm.client_id"].tolist(),
                tables["loss_change_alarm.alarm_sample_index"].tolist(),
                strict=True,
            )
        )
        assert set(selections) <= set(alarm_keys)
        assert all(
            point == alarm_index - selected_count + 1
            for (client_id, alarm_index), point in zip(
                alarm_keys,
                tables["loss_change_alarm.estimated_change_point_sample_index"].tolist(),
                strict=True,
            )
            if (client_id, alarm_index) in selections
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
