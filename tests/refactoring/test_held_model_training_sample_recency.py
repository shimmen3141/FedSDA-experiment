"""保有モデルの学習batchの抜き出し方（全履歴・直近K件・指数減衰）の設定と抜き出し。"""

import csv
import json
import random
from dataclasses import dataclass, field, replace

import pytest
import torch

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)
from federated_learning_experiments.learning.training import held_model_training_batch_sampling
from federated_learning_experiments.learning.training.held_model_training_batch_sampling import (
    sample_training_batches_for_held_models,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ModelTrainingSampleCollection,
    ObservedTrainingSample,
)
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
    build_final_configuration_fedsda_run_settings,
)
from federated_learning_experiments.sweep import experiment_sweep_command
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
    describe_fedsda_training_sample_selection,
    parse_fedsda_training_sample_selection,
)
from federated_learning_experiments.sweep.experiment_sweep_records import write_run_summary_table

BASE_SETTINGS = dict(
    local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
    shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
)
UNIFORM = dict(
    training_sample_selection_policy="uniform_over_all_stored_samples",
    most_recent_stored_sample_count=None,
    stored_sample_recency_decay_factor=None,
)


def settings(**changes):
    return LocalTrainingSettings(**BASE_SETTINGS, **(UNIFORM | changes))


def recent(count):
    return settings(
        training_sample_selection_policy="uniform_over_most_recent_stored_samples",
        most_recent_stored_sample_count=count,
    )


def decayed(factor):
    return settings(
        training_sample_selection_policy="recency_decayed_stored_samples",
        stored_sample_recency_decay_factor=factor,
    )


def collection(sample_count, model_id=7):
    """位置を特徴へ書いた標本の列（0が最古）。"""
    return ModelTrainingSampleCollection(
        model_id=model_id,
        training_samples=tuple(
            ObservedTrainingSample(
                input_features=torch.tensor([[float(position)]], dtype=torch.float32),
                observed_class_labels=torch.tensor([[1.0]], dtype=torch.float32),
            )
            for position in range(sample_count)
        ),
        latest_alarm_boundary_position=None,
    )


def sampled_positions(*, population, batch, local_training_settings, seed=0, generator=None):
    generator = generator or random.Random(seed)
    (batch_result,) = sample_training_batches_for_held_models(
        held_model_ids=frozenset((population.model_id,)),
        ordered_model_training_samples=(population,),
        batch_sample_count=batch,
        python_random_generator=generator,
        local_training_settings=local_training_settings,
    )
    return [int(value) for value in batch_result.input_features[:, 0].tolist()]


# ---- 共有の検査（`int | None`・`float | None`） ----


def test_shared_validation_accepts_optional_int_and_float():
    @dataclass(frozen=True, kw_only=True)
    class OptionalSettings:
        count: int | None = field(
            metadata={"minimum_allowed_value": 1, "minimum_value_is_inclusive": True}
        )
        factor: float | None = field(
            metadata={
                "minimum_allowed_value": 0.0,
                "minimum_value_is_inclusive": False,
                "maximum_allowed_value": 1.0,
                "maximum_value_is_inclusive": False,
            }
        )

    validate_settings_field_values(OptionalSettings(count=None, factor=None))
    validate_settings_field_values(OptionalSettings(count=3, factor=0.5))
    for count, factor in ((0, None), (True, None), ("3", None), (None, 1.0), (None, float("nan"))):
        with pytest.raises(RunSettingsValidationError):
            validate_settings_field_values(OptionalSettings(count=count, factor=factor))


# ---- 設定 ----


def test_local_training_settings_accept_each_selection_policy():
    assert settings().training_sample_selection_policy == "uniform_over_all_stored_samples"
    assert recent(300).most_recent_stored_sample_count == 300
    assert decayed(0.997).stored_sample_recency_decay_factor == 0.997


@pytest.mark.parametrize(
    "changes",
    [
        dict(training_sample_selection_policy="unknown"),
        dict(most_recent_stored_sample_count=300),
        dict(stored_sample_recency_decay_factor=0.5),
        dict(training_sample_selection_policy="uniform_over_most_recent_stored_samples"),
        dict(training_sample_selection_policy="recency_decayed_stored_samples"),
        dict(
            training_sample_selection_policy="uniform_over_most_recent_stored_samples",
            most_recent_stored_sample_count=300,
            stored_sample_recency_decay_factor=0.5,
        ),
        dict(
            training_sample_selection_policy="uniform_over_most_recent_stored_samples",
            most_recent_stored_sample_count=0,
        ),
        dict(
            training_sample_selection_policy="uniform_over_most_recent_stored_samples",
            most_recent_stored_sample_count=True,
        ),
        dict(
            training_sample_selection_policy="uniform_over_most_recent_stored_samples",
            most_recent_stored_sample_count=300.0,
        ),
        *[
            dict(
                training_sample_selection_policy="recency_decayed_stored_samples",
                stored_sample_recency_decay_factor=factor,
            )
            for factor in (0.0, 1.0, 1.5, -0.1, float("nan"), float("inf"), 1, True)
        ],
    ],
)
def test_local_training_settings_reject_invalid_combinations(changes):
    with pytest.raises(RunSettingsValidationError):
        settings(**changes)


def test_most_recent_count_must_be_at_least_the_batch_sample_count():
    run_settings = build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=0,
            client_count=2,
            per_client_sample_count=50,
            server_aggregation_interval_per_client_samples=10,
        )
    )
    client_settings = run_settings.run_participant_settings.run_client_settings
    batch_sample_count = client_settings.scalar_settings.local_training_batch_sample_count
    assert replace(client_settings, local_training_settings=recent(batch_sample_count))
    with pytest.raises(RunSettingsValidationError, match="most_recent_stored_sample_count"):
        replace(client_settings, local_training_settings=recent(batch_sample_count - 1))


# ---- 抜き出し ----


def test_default_selection_draws_the_same_batch_and_random_state_as_before():
    population = collection(50)
    for local_training_settings in (None, settings()):
        generator = random.Random(3)
        positions = sampled_positions(
            population=population,
            batch=8,
            local_training_settings=local_training_settings,
            generator=generator,
        )
        reference = random.Random(3)
        assert positions == reference.sample(range(50), 8)
        assert generator.getstate() == reference.getstate()


def test_most_recent_selection_draws_only_from_the_last_k_samples():
    population = collection(200)
    seen = set()
    for seed in range(50):
        seen.update(
            sampled_positions(
                population=population, batch=8, local_training_settings=recent(20), seed=seed
            )
        )
    assert seen <= set(range(180, 200))
    assert len(seen) == 20
    # Kが列の長さ以上なら、全部の標本から（今と同じ母集団）。
    short_population = collection(30)
    generator = random.Random(1)
    positions = sampled_positions(
        population=short_population,
        batch=8,
        local_training_settings=recent(1000),
        generator=generator,
    )
    reference = random.Random(1)
    assert positions == reference.sample(range(30), 8)


def test_decayed_selection_is_reproducible_without_duplicates():
    population = collection(200)
    first = sampled_positions(
        population=population, batch=32, local_training_settings=decayed(0.97), seed=5
    )
    assert first == sampled_positions(
        population=population, batch=32, local_training_settings=decayed(0.97), seed=5
    )
    assert len(first) == 32 and len(set(first)) == 32
    assert all(0 <= position < 200 for position in first)


def test_decayed_selection_prefers_recent_samples_more_for_smaller_factors():
    population = collection(200)
    mean_ages = {}
    for factor in (0.9, 0.99, 0.999):
        counts = [0] * 200
        for seed in range(2000):
            for position in sampled_positions(
                population=population, batch=8, local_training_settings=decayed(factor), seed=seed
            ):
                counts[position] += 1
        newer, older = sum(counts[100:]), sum(counts[:100])
        assert newer > older
        if factor == 0.999:
            assert newer / older < 1.2
        mean_ages[factor] = sum(
            (199 - position) * count for position, count in enumerate(counts)
        ) / sum(counts)
    assert mean_ages[0.9] < mean_ages[0.99] < mean_ages[0.999]


def test_decayed_selection_follows_the_conditional_probabilities():
    """列の長さ3・batch 1: 位置（末尾から0・1・2件前）の頻度が、α^aの比に近い。"""
    factor = 0.5
    draws = 20000
    counts = {0: 0, 1: 0, 2: 0}
    generator = random.Random(11)
    population = collection(3)
    for _ in range(draws):
        (position,) = sampled_positions(
            population=population,
            batch=1,
            local_training_settings=decayed(factor),
            generator=generator,
        )
        counts[2 - position] += 1
    total_weight = 1 + factor + factor**2
    for age, count in counts.items():
        probability = factor**age / total_weight
        standard_error = (probability * (1 - probability) / draws) ** 0.5
        assert abs(count / draws - probability) < 4 * standard_error


@pytest.mark.parametrize("factor", [0.5, 1e-200])
def test_decayed_selection_selects_every_position_when_batch_equals_population(factor):
    population = collection(32)
    positions = sampled_positions(
        population=population, batch=32, local_training_settings=decayed(factor)
    )
    assert sorted(positions) == list(range(32))
    if factor == 1e-200:
        # 重みの差が極端なので、最新から順に選ばれる。
        assert positions == list(range(31, -1, -1))


def test_decayed_selection_uses_one_random_draw_per_sample_regardless_of_population_size():
    for population_size in (200, 100000):
        generator = random.Random(9)
        sampled_positions(
            population=collection(population_size),
            batch=32,
            local_training_settings=decayed(0.5),
            generator=generator,
        )
        reference = random.Random(9)
        for _ in range(32):
            reference.random()
        assert generator.getstate() == reference.getstate()


def test_selection_settings_are_checked_before_random_draws_even_without_eligible_models():
    generator = random.Random(4)
    state_before = generator.getstate()
    bypassed = recent(300)
    object.__setattr__(bypassed, "most_recent_stored_sample_count", 2)
    for invalid_settings in (bypassed, "uniform"):
        with pytest.raises(ValueError):
            sample_training_batches_for_held_models(
                held_model_ids=frozenset(),
                ordered_model_training_samples=(collection(10),),
                batch_sample_count=8,
                python_random_generator=generator,
                local_training_settings=invalid_settings,
            )
    bypassed_decay = decayed(0.5)
    object.__setattr__(bypassed_decay, "stored_sample_recency_decay_factor", 1.5)
    with pytest.raises(ValueError):
        sample_training_batches_for_held_models(
            held_model_ids=frozenset(),
            ordered_model_training_samples=(),
            batch_sample_count=8,
            python_random_generator=generator,
            local_training_settings=bypassed_decay,
        )
    assert generator.getstate() == state_before


def test_joint_training_iterations_pass_the_selection_to_sampling(monkeypatch):
    from federated_learning_experiments.learning.training import (
        held_model_joint_training_iterations,
    )

    passed_settings = []

    def record(**arguments):
        passed_settings.append(arguments["local_training_settings"])
        return ()

    monkeypatch.setattr(
        held_model_joint_training_iterations, "sample_training_batches_for_held_models", record
    )
    selection = recent(300)
    assert (
        held_model_joint_training_iterations.perform_held_model_joint_training_iterations(
            requested_joint_update_iteration_count=2,
            held_model_training_bindings=(),
            ordered_model_training_samples=(),
            batch_sample_count=32,
            python_random_generator=random.Random(0),
            local_training_settings=selection,
            shared_feature_extractor=None,  # pyright: ignore[reportArgumentType]
            shared_parameter_optimizer=None,
            update_shared_features=True,
        )
        == ()
    )
    assert passed_settings == [selection, selection]
    assert held_model_training_batch_sampling.sample_training_batches_for_held_models is not record


# ---- 掃引の軸 ----


def small_plan(**changes):
    return ExperimentSweepPlan(
        **(
            dict(
                configuration_names=("fedsda",),
                dataset_names=("sine2",),
                random_seeds=(0,),
                client_count=2,
                per_client_sample_count=50,
                server_aggregation_intervals_per_client_samples=(10,),
            )
            | changes
        )
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("uniform", ("uniform_over_all_stored_samples", None, None)),
        ("recent:300", ("uniform_over_most_recent_stored_samples", 300, None)),
        ("decay:0.997", ("recency_decayed_stored_samples", None, 0.997)),
        ("decay:0.5", ("recency_decayed_stored_samples", None, 0.5)),
    ],
)
def test_selection_text_is_parsed_and_described_back(text, expected):
    assert parse_fedsda_training_sample_selection(text) == expected
    policy, count, factor = expected
    assert (
        describe_fedsda_training_sample_selection(
            policy=policy, most_recent_count=count, decay_factor=factor
        )
        == text
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Uniform",
        "recent:",
        "recent:0",
        "recent:0300",
        "recent:-3",
        "recent:+3",
        "recent:3.0",
        "recent: 3",
        "decay:",
        "decay:1",
        "decay:1.0",
        "decay:0",
        "decay:0.000",
        "decay:.5",
        "decay:1e-3",
        "decay:nan",
        "decay:inf",
        "decay:-0.5",
        "decay:0.5 ",
        "window:300",
    ],
)
def test_invalid_selection_text_is_rejected(text):
    with pytest.raises(ValueError):
        parse_fedsda_training_sample_selection(text)


def local_selection(planned_run):
    local_training_settings = planned_run.run_settings.run_participant_settings.run_client_settings.local_training_settings
    return (
        local_training_settings.training_sample_selection_policy,
        local_training_settings.most_recent_stored_sample_count,
        local_training_settings.stored_sample_recency_decay_factor,
    )


def test_selection_axis_makes_one_fedsda_variant_per_selection():
    planned_runs = build_planned_experiment_runs(
        sweep_plan=small_plan(
            configuration_names=("fedsda", "fedsda-independent-models", "feddrift", "fedavg"),
            fedsda_training_sample_selections=("uniform", "recent:300", "decay:0.997"),
            feddrift_loss_increase_thresholds=(0.0, 0.2),
        )
    )
    by_configuration = {}
    for planned_run in planned_runs:
        by_configuration.setdefault(planned_run.configuration_name, []).append(planned_run)
    assert {name: len(runs) for name, runs in by_configuration.items()} == {
        "fedsda": 3,
        "fedsda-independent-models": 3,
        "feddrift": 2,
        "fedavg": 1,
    }
    assert [local_selection(run) for run in by_configuration["fedsda"]] == [
        ("uniform_over_all_stored_samples", None, None),
        ("uniform_over_most_recent_stored_samples", 300, None),
        ("recency_decayed_stored_samples", None, 0.997),
    ]
    assert len({run.run_identifier for run in planned_runs}) == len(planned_runs)
    # 軸を指定しないときの計画は、`uniform`だけを指定したときと同じ。
    assert build_planned_experiment_runs(sweep_plan=small_plan()) == build_planned_experiment_runs(
        sweep_plan=small_plan(fedsda_training_sample_selections=("uniform",))
    )


@pytest.mark.parametrize(
    "changes,error_type",
    [
        (dict(fedsda_training_sample_selections=()), ValueError),
        (dict(fedsda_training_sample_selections=["uniform"]), TypeError),
        (dict(fedsda_training_sample_selections=("recent:300", "recent:300")), ValueError),
        (dict(fedsda_training_sample_selections=("decay:0.5", "decay:0.50")), ValueError),
        (dict(fedsda_training_sample_selections=("window:300",)), ValueError),
    ],
)
def test_selection_axis_rejects_invalid_values(changes, error_type):
    with pytest.raises(error_type):
        small_plan(**changes)


def test_selection_axis_requires_a_fedsda_configuration_and_a_valid_count():
    with pytest.raises(ValueError, match="FedSDA"):
        build_planned_experiment_runs(
            sweep_plan=small_plan(
                configuration_names=("feddrift", "fedavg"),
                fedsda_training_sample_selections=("recent:300",),
            )
        )
    # Kがbatchの件数（32）より小さい。
    with pytest.raises(RunSettingsValidationError):
        build_planned_experiment_runs(
            sweep_plan=small_plan(fedsda_training_sample_selections=("recent:10",))
        )


def test_command_rejects_invalid_selections_before_running(tmp_path, capsys):
    output_directory = tmp_path / "output"
    base_arguments = [
        "--datasets",
        "sine2",
        "--seeds",
        "0",
        "--aggregation-intervals",
        "10",
        "--purpose",
        "test",
        "--output-directory",
        str(output_directory),
    ]
    for selections in (["window:300"], ["recent:10"]):
        assert (
            experiment_sweep_command.main(
                [*base_arguments, "--fedsda-training-sample-selections", *selections]
            )
            == 2
        )
        assert "invalid sweep plan" in capsys.readouterr().err
    assert (
        experiment_sweep_command.main(
            [
                *base_arguments,
                "--configurations",
                "feddrift",
                "--fedsda-training-sample-selections",
                "uniform",
            ]
        )
        == 2
    )
    assert "FedSDA" in capsys.readouterr().err
    assert not output_directory.exists()
    assert (
        experiment_sweep_command.main(
            [
                *base_arguments,
                "--fedsda-training-sample-selections",
                "uniform",
                "decay:0.997",
                "--print-plan",
            ]
        )
        == 0
    )
    assert "planned runs: 2" in capsys.readouterr().out


def write_summary(runs_directory, run_identifier, local_training_settings):
    runs_directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "format_version": 1,
        "run_settings": {
            "settings_type": "FedsdaRunSettings",
            "execution_settings": {
                "experiment_run_conditions": {
                    "dataset_name": "sine2",
                    "random_seed": len(run_identifier),
                    "client_count": 2,
                    "per_client_sample_count": 50,
                    "server_aggregation_interval_per_client_samples": 10,
                }
            },
            "run_participant_settings": {
                "run_client_settings": {"local_training_settings": local_training_settings}
            },
        },
        "run_metrics": {"prediction_accuracy": 0.5},
    }
    (runs_directory / f"{run_identifier}.json").write_text(json.dumps(summary), encoding="utf-8")


def test_run_summary_table_holds_the_selection_and_keeps_previous_rows(tmp_path):
    runs_directory = tmp_path / "fedsda" / "runs"
    write_summary(runs_directory, "a", dict(BASE_SETTINGS))
    write_summary(
        runs_directory,
        "bb",
        dict(BASE_SETTINGS)
        | dict(
            training_sample_selection_policy="recency_decayed_stored_samples",
            most_recent_stored_sample_count=None,
            stored_sample_recency_decay_factor=0.997,
        ),
    )
    write_summary(
        runs_directory,
        "ccc",
        dict(BASE_SETTINGS)
        | dict(
            training_sample_selection_policy="uniform_over_most_recent_stored_samples",
            most_recent_stored_sample_count=300,
            stored_sample_recency_decay_factor=None,
        ),
    )
    table_result = write_run_summary_table(
        configuration_directory=tmp_path / "fedsda", configuration_name="fedsda"
    )
    assert table_result.unreadable_run_summary_descriptions == ()
    with table_result.table_file_path.open(encoding="utf-8", newline="") as table_file:
        rows = {row["run_identifier"]: row for row in csv.DictReader(table_file)}
    assert {name: row["training_sample_selection"] for name, row in rows.items()} == {
        "a": "",
        "bb": "decay:0.997",
        "ccc": "recent:300",
    }
