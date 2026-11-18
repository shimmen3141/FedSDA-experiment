"""概念列: 製品のランダムな概念列（1クライアント）と同じ列・同じ乱数の消費になること（Windows）。"""

import random

import numpy as np
import pytest

# 製品のrun設定（既定値の照合）がtorchを読むので、torchの無いWSLのriverの環境ではskipする（Windowsで実行する）。
pytest.importorskip("torch")

from federated_learning_experiments.configuration.experiment_run_conditions import (  # noqa: E402
    ExperimentRunConditions,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_generation import (
    generate_random_client_concept_traces,
)
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
    build_final_configuration_fedsda_run_settings,
)
from single_stream_hard_assignment_soft_prediction.random_concept_schedule import (
    generate_random_concept_schedule,
)


def product_schedule_settings(dataset_name="sine2"):
    conditions = ExperimentRunConditions(
        dataset_name=dataset_name,
        random_seed=0,
        client_count=1,
        per_client_sample_count=5000,
        server_aggregation_interval_per_client_samples=50,
    )
    settings = build_final_configuration_fedsda_run_settings(experiment_run_conditions=conditions)
    return conditions, settings.execution_settings.concept_schedule_settings


@pytest.mark.parametrize("dataset_name, concept_count", [("sine2", 2), ("sea4", 4)])
@pytest.mark.parametrize("state_seed", [0, 1, 7, 12345])
def test_schedule_and_random_state_match_the_product(dataset_name, concept_count, state_seed):
    conditions, schedule_settings = product_schedule_settings(dataset_name)
    product_generator = random.Random(state_seed)
    (trace,) = generate_random_client_concept_traces(
        experiment_run_conditions=conditions,
        concept_schedule_settings=schedule_settings,
        python_random_generator=product_generator,
    )
    generator = random.Random(state_seed)
    schedule = generate_random_concept_schedule(
        sample_count=5000, concept_count=concept_count, python_random_generator=generator
    )
    assert schedule == trace.concept_ids_by_sample_index
    assert generator.getstate() == product_generator.getstate()
    # 比べる意味があること: 変化があり、2概念以上が出る。
    assert len(set(schedule)) >= 2 and schedule[0] == 0


def test_explicit_settings_match_the_product_with_other_values():
    conditions, _ = product_schedule_settings("sea4")
    from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
        RandomConceptScheduleSettings,
    )

    other = RandomConceptScheduleSettings(
        concept_schedule_strategy="random_changes_after_minimum_index_gap",
        minimum_sample_index_gap_before_change_trial=20,
        per_eligible_sample_concept_change_probability=0.05,
    )
    product_generator = random.Random(3)
    (trace,) = generate_random_client_concept_traces(
        experiment_run_conditions=conditions,
        concept_schedule_settings=other,
        python_random_generator=product_generator,
    )
    generator = random.Random(3)
    schedule = generate_random_concept_schedule(
        sample_count=5000,
        concept_count=4,
        python_random_generator=generator,
        minimum_sample_index_gap_before_change_trial=20,
        per_eligible_sample_concept_change_probability=0.05,
    )
    assert schedule == trace.concept_ids_by_sample_index
    assert generator.getstate() == product_generator.getstate()


def test_defaults_are_the_product_fedsda_defaults():
    _, schedule_settings = product_schedule_settings()
    import inspect

    parameters = inspect.signature(generate_random_concept_schedule).parameters
    assert (
        parameters["minimum_sample_index_gap_before_change_trial"].default
        == schedule_settings.minimum_sample_index_gap_before_change_trial
    )
    assert (
        parameters["per_eligible_sample_concept_change_probability"].default
        == schedule_settings.per_eligible_sample_concept_change_probability
    )


def test_three_concepts_use_every_concept_and_never_repeat_the_current_one():
    schedule = generate_random_concept_schedule(
        sample_count=50000, concept_count=3, python_random_generator=random.Random(0)
    )
    assert set(schedule) == {0, 1, 2}
    changes = [i for i in range(1, len(schedule)) if schedule[i] != schedule[i - 1]]
    assert all(b - a > 300 for a, b in zip([0, *changes], changes))


def test_does_not_touch_other_random_states():
    random.seed(99)
    np.random.seed(99)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    generate_random_concept_schedule(
        sample_count=3000, concept_count=3, python_random_generator=random.Random(5)
    )
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(after, numpy_state))


@pytest.mark.parametrize(
    "arguments",
    [
        dict(sample_count=-1),
        dict(sample_count=True),
        dict(sample_count=10.0),
        dict(concept_count=1),
        dict(concept_count=True),
        dict(python_random_generator=np.random.default_rng(0)),
        dict(minimum_sample_index_gap_before_change_trial=-1),
        dict(minimum_sample_index_gap_before_change_trial=False),
        dict(per_eligible_sample_concept_change_probability=1.5),
        dict(per_eligible_sample_concept_change_probability=-0.1),
        dict(per_eligible_sample_concept_change_probability=True),
    ],
)
def test_rejects_invalid_arguments_before_drawing(arguments):
    generator = random.Random(1)
    state = generator.getstate()
    full = dict(sample_count=100, concept_count=2, python_random_generator=generator)
    full.update(arguments)
    with pytest.raises((TypeError, ValueError)):
        generate_random_concept_schedule(**full)
    assert generator.getstate() == state


def test_zero_samples_give_an_empty_schedule():
    assert (
        generate_random_concept_schedule(
            sample_count=0, concept_count=2, python_random_generator=random.Random(0)
        )
        == ()
    )
