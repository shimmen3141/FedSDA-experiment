"""警報を区切りに使う学習batchの抜き出し: 区切りの記録・区切りを付ける処理・抜き出し・設定と掃引の軸。"""

import csv
import itertools
import json
import random

import pytest
import torch
from test_alarm_change_interval_resolution import (
    ALARM_INTERVAL_RESOLUTION_CASES,
    build_alarm_change_interval_resolution_oracle,
)
from test_post_alarm_candidate_validation_resolution import (
    LEGACY_RESOLUTION_CASES,
    build_resolution_oracle,
)

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
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
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.runtime.alarm_change_interval_resolution import (
    resolve_alarm_change_interval,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import (
    apply_post_alarm_candidate_validation_resolution,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    describe_fedsda_training_sample_selection,
    parse_fedsda_training_sample_selection,
)
from federated_learning_experiments.sweep.experiment_sweep_records import write_run_summary_table

BASE_SETTINGS = dict(
    local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
    shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
    most_recent_stored_sample_count=None,
)


def after_boundary():
    return LocalTrainingSettings(
        **BASE_SETTINGS,
        training_sample_selection_policy="uniform_after_latest_alarm_boundary",
        stored_sample_recency_decay_factor=None,
    )


def decayed_before_boundary(factor):
    return LocalTrainingSettings(
        **BASE_SETTINGS,
        training_sample_selection_policy="recency_decayed_before_latest_alarm_boundary",
        stored_sample_recency_decay_factor=factor,
    )


def sample(position):
    return ObservedTrainingSample(
        input_features=torch.tensor([[float(position)]], dtype=torch.float32),
        observed_class_labels=torch.tensor([[1.0]], dtype=torch.float32),
    )


def collection(sample_count, boundary=None, model_id=7):
    return ModelTrainingSampleCollection(
        model_id=model_id,
        training_samples=tuple(sample(position) for position in range(sample_count)),
        latest_alarm_boundary_position=boundary,
    )


def sampled_positions(*, population, batch, local_training_settings, generator):
    (batch_result,) = sample_training_batches_for_held_models(
        held_model_ids=frozenset((population.model_id,)),
        ordered_model_training_samples=(population,),
        batch_sample_count=batch,
        python_random_generator=generator,
        local_training_settings=local_training_settings,
    )
    return [int(value) for value in batch_result.input_features[:, 0].tolist()]


# ---- 区切りの記録 ----


def store_with(model_lengths):
    store = ModelTrainingSampleStore()
    for model_id, length in model_lengths.items():
        store.append_model_training_samples(
            model_id=model_id, training_samples=tuple(sample(i) for i in range(length))
        )
    return store


def boundaries(store):
    return {
        collection.model_id: collection.latest_alarm_boundary_position
        for collection in store.snapshot_ordered_model_training_samples()
    }


def test_store_counts_samples_and_records_boundaries():
    store = store_with({1: 5, 2: 3})
    assert store.count_model_training_samples(model_id=1) == 5
    assert store.count_model_training_samples(model_id=9) == 0
    assert boundaries(store) == {1: None, 2: None}
    store.mark_latest_alarm_boundary(model_id=1, position=4)
    store.mark_latest_alarm_boundary(model_id=2, position=3)
    assert boundaries(store) == {1: 4, 2: 3}
    store.mark_latest_alarm_boundary(model_id=1, position=0)
    assert boundaries(store) == {1: 0, 2: 3}
    for arguments, error_type in (
        (dict(model_id=9, position=0), ValueError),
        (dict(model_id=1, position=6), ValueError),
        (dict(model_id=1, position=-1), ValueError),
        (dict(model_id=1, position=True), TypeError),
        (dict(model_id=True, position=0), TypeError),
    ):
        with pytest.raises(error_type):
            store.mark_latest_alarm_boundary(**arguments)
    assert boundaries(store) == {1: 0, 2: 3}


def test_store_moves_boundaries_with_reassigned_ids():
    store = store_with({1: 5, 2: 3, 3: 2})
    store.mark_latest_alarm_boundary(model_id=1, position=2)
    store.mark_latest_alarm_boundary(model_id=3, position=1)
    store.reassign_model_training_samples_id(original_model_id=1, reassigned_model_id=10)
    assert boundaries(store) == {2: None, 3: 1, 10: 2}
    # 元に区切りが無ければ、先の区切りも消える（列の上書きと同じ）。
    store.reassign_model_training_samples_id(original_model_id=2, reassigned_model_id=3)
    assert boundaries(store) == {3: None, 10: 2}
    # 元の列が無ければ、何も変えない。
    store.reassign_model_training_samples_id(original_model_id=99, reassigned_model_id=10)
    assert boundaries(store) == {3: None, 10: 2}


def test_store_keeps_boundary_for_a_single_remapped_list_and_clears_merged_lists():
    store = store_with({1: 5, 2: 3, 3: 2, 4: 4})
    for model_id, position in ((1, 2), (2, 1), (3, 2), (4, 4)):
        store.mark_latest_alarm_boundary(model_id=model_id, position=position)
    store.remap_model_training_sample_collections(model_id_mapping={2: 1, 3: 30, 99: 4})
    assert boundaries(store) == {1: None, 30: 2, 4: 4}
    assert store.count_model_training_samples(model_id=1) == 8


# ---- 区切りを付ける処理 ----


@pytest.mark.parametrize("alarm_interval_resolution_case", sorted(ALARM_INTERVAL_RESOLUTION_CASES))
def test_change_interval_resolution_marks_boundary_only_when_current_model_continues(
    alarm_interval_resolution_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(841)
        resolution_arguments, _, legacy_client, _ = build_alarm_change_interval_resolution_oracle(
            class_count=2,
            alarm_interval_resolution_case=alarm_interval_resolution_case,
            monkeypatch=monkeypatch,
        )
        store = resolution_arguments["training_sample_store"]
        lengths_before = {
            model_id: store.count_model_training_samples(model_id=model_id)
            for model_id in boundaries(store)
        }
        resolution = resolve_alarm_change_interval(**resolution_arguments)
    marked = {
        model_id: position
        for model_id, position in boundaries(store).items()
        if position is not None
    }
    if resolution.resolution_outcome == "alarm_interval_current_model_maintained":
        assert marked == {
            resolution.assigned_model_id: lengths_before[resolution.assigned_model_id]
        }
        assert store.count_model_training_samples(model_id=resolution.assigned_model_id) == (
            lengths_before[resolution.assigned_model_id]
            + len(resolution_arguments["change_interval_training_samples"])
        )
    else:
        assert marked == {}


@pytest.mark.parametrize("pending_sample_count", [0, 4])
@pytest.mark.parametrize("legacy_resolution_case", LEGACY_RESOLUTION_CASES)
def test_validation_resolution_marks_boundary_only_when_current_model_continues(
    legacy_resolution_case, pending_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(601)
        resolution_arguments, _, _, _ = build_resolution_oracle(
            class_count=2,
            legacy_resolution_case=legacy_resolution_case,
            pending_sample_count=pending_sample_count,
            monkeypatch=monkeypatch,
        )
        store = resolution_arguments["training_sample_store"]
        current_model_id = resolution_arguments[
            "current_training_model_assignment"
        ].current_training_model_id
        length_before = store.count_model_training_samples(model_id=current_model_id)
        resolution = apply_post_alarm_candidate_validation_resolution(**resolution_arguments)
    marked = {
        model_id: position
        for model_id, position in boundaries(store).items()
        if position is not None
    }
    if resolution.resolution_outcome in (
        "candidate_rejected",
        "current_model_maintained",
    ) and store.count_model_training_samples(model_id=current_model_id):
        assert marked == {current_model_id: length_before}
    else:
        assert marked == {}


def test_failed_absorption_leaves_no_boundary(monkeypatch):
    from federated_learning_experiments.runtime import alarm_change_interval_resolution

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(841)
        resolution_arguments, _, _, _ = build_alarm_change_interval_resolution_oracle(
            class_count=2,
            alarm_interval_resolution_case="current_model_maintained",
            monkeypatch=monkeypatch,
        )

        def fail_absorption(**_):
            raise RuntimeError("simulated absorption failure")

        monkeypatch.setattr(
            alarm_change_interval_resolution,
            "absorb_assigned_training_samples_into_held_model",
            fail_absorption,
        )
        with pytest.raises(RuntimeError, match="simulated"):
            resolve_alarm_change_interval(**resolution_arguments)
    assert all(
        position is None
        for position in boundaries(resolution_arguments["training_sample_store"]).values()
    )


# ---- 抜き出し ----

# 区間の方法を下請けへ切り出す前（ea79b57）の`_sample_recency_decayed_positions`で採取した値
# （(列の長さ, batch, α, seed) → 選んだ位置の列）。切り出しの後の指数減衰が、同じ値と乱数の使い方であることを確かめる。
DECAY_POSITIONS_BEFORE_REFACTORING = {
    (50, 8, 0.9, 1): [48, 31, 36, 46, 42, 43, 38, 32],
    (40, 32, 0.5, 3): [
        39,
        37,
        38,
        35,
        36,
        34,
        33,
        30,
        32,
        31,
        22,
        29,
        26,
        28,
        27,
        25,
        23,
        21,
        24,
        19,
        18,
        20,
        15,
        16,
        17,
        14,
        11,
        13,
        10,
        9,
        12,
        5,
    ],
}

EXTRA_DECAY_POSITIONS_BEFORE_REFACTORING = {
    (200, 32, 0.997, 2): [
        11,
        14,
        191,
        186,
        43,
        66,
        81,
        149,
        94,
        95,
        100,
        174,
        128,
        135,
        67,
        1,
        15,
        108,
        126,
        158,
        195,
        196,
        120,
        147,
        136,
        28,
        107,
        99,
        162,
        194,
        145,
        177,
    ],
    (1000, 16, 0.999, 4): [
        838,
        932,
        710,
        896,
        957,
        703,
        130,
        293,
        338,
        847,
        582,
        805,
        883,
        929,
        851,
        115,
    ],
    (33, 32, 1e-200, 5): [
        32,
        31,
        30,
        29,
        28,
        27,
        26,
        25,
        24,
        23,
        22,
        21,
        20,
        19,
        18,
        17,
        16,
        15,
        14,
        13,
        12,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
    ],
}


def test_recency_decay_keeps_the_values_from_before_the_refactoring():
    expected = dict(DECAY_POSITIONS_BEFORE_REFACTORING)
    expected.update(EXTRA_DECAY_POSITIONS_BEFORE_REFACTORING)
    for (population_size, batch, factor, seed), positions in expected.items():
        generator = random.Random(seed)
        assert (
            held_model_training_batch_sampling._sample_recency_weighted_positions(
                population_size=population_size,
                sample_count=batch,
                decay_factor=factor,
                flat_age_count=0,
                python_random_generator=generator,
            )
            == positions
        )


def test_uniform_after_boundary_draws_from_the_samples_after_the_boundary():
    seen = set()
    for seed in range(60):
        seen.update(
            sampled_positions(
                population=collection(100, boundary=80),
                batch=8,
                local_training_settings=after_boundary(),
                generator=random.Random(seed),
            )
        )
    assert seen == set(range(80, 100))
    # 区切りより後がbatchより少なければ、末尾のbatchの件数から。
    seen = set()
    for seed in range(60):
        seen.update(
            sampled_positions(
                population=collection(100, boundary=97),
                batch=8,
                local_training_settings=after_boundary(),
                generator=random.Random(seed),
            )
        )
    assert seen == set(range(92, 100))
    # 区切りが無ければ、全部から（全履歴から一様と同じ乱数の使い方）。
    generator = random.Random(3)
    positions = sampled_positions(
        population=collection(50),
        batch=8,
        local_training_settings=after_boundary(),
        generator=generator,
    )
    reference = random.Random(3)
    assert positions == reference.sample(range(50), 8)


class FixedDraws:
    """`random()`が決まった値を返す代わり（内部の関数を、乱数の値の格子で確かめる）。"""

    def __init__(self, values):
        self._values = iter(values)

    def random(self):
        return next(self._values)


def exact_selection_probabilities(weights, batch):
    """重み（位置→重み）から、重複なしの逐次の抜き出しの、選んだ列の確率を全数列挙で求める。"""
    probabilities = {}
    for ordering in itertools.permutations(range(len(weights)), batch):
        probability = 1.0
        remaining = dict(enumerate(weights))
        for position in ordering:
            probability *= remaining[position] / sum(remaining.values())
            del remaining[position]
        probabilities[ordering] = probability
    return probabilities


@pytest.mark.parametrize("boundary", [None, 0, 2, 4])
@pytest.mark.parametrize("batch", [1, 2])
def test_decay_before_boundary_matches_the_exact_conditional_probabilities(boundary, batch):
    factor = 0.5
    population_size = 4
    flat_age_count = population_size if boundary is None else population_size - boundary
    # 位置（0が最古）の重み: 区切り以降は1、区切りより前は、区切りから数えて何件前かの分だけαを掛ける。
    weights = [
        1.0 if age < flat_age_count else factor ** (age - flat_age_count + 1)
        for age in (population_size - 1 - position for position in range(population_size))
    ]
    expected = exact_selection_probabilities(weights, batch)
    grid = 200
    counts = {}
    for draws in itertools.product(range(grid), repeat=batch):
        positions = tuple(
            held_model_training_batch_sampling._sample_recency_weighted_positions(
                population_size=population_size,
                sample_count=batch,
                decay_factor=factor,
                flat_age_count=flat_age_count,
                python_random_generator=FixedDraws([(draw + 0.5) / grid for draw in draws]),  # pyright: ignore[reportArgumentType]
            )
        )
        counts[positions] = counts.get(positions, 0) + 1
    for ordering, probability in expected.items():
        assert abs(counts.get(ordering, 0) / grid**batch - probability) < 0.02, (
            ordering,
            probability,
        )


@pytest.mark.parametrize("factor", [0.5, 1e-200, 5e-324])
def test_decay_before_boundary_selects_without_duplicates_for_extreme_factors(factor):
    for boundary in (None, 0, 20, 33):
        positions = sampled_positions(
            population=collection(33, boundary=boundary),
            batch=32,
            local_training_settings=decayed_before_boundary(factor),
            generator=random.Random(1),
        )
        assert len(set(positions)) == 32
        if boundary is not None and boundary > 0 and factor < 1e-100:
            # 区切り以降を全部選んでから、区切りの直前から古い順に選ぶ。
            assert set(positions[: 33 - boundary]) == set(range(boundary, 33))


def test_decay_before_boundary_uses_one_random_draw_per_sample():
    for population_size in (200, 100000):
        generator = random.Random(9)
        sampled_positions(
            population=collection(population_size, boundary=population_size // 2),
            batch=32,
            local_training_settings=decayed_before_boundary(0.5),
            generator=generator,
        )
        reference = random.Random(9)
        for _ in range(32):
            reference.random()
        assert generator.getstate() == reference.getstate()


def test_invalid_boundaries_are_rejected_before_random_draws():
    generator = random.Random(4)
    state_before = generator.getstate()
    for boundary in (-1, 11, True, 2.0):
        with pytest.raises(ValueError, match="latest_alarm_boundary_position"):
            sample_training_batches_for_held_models(
                held_model_ids=frozenset(),
                ordered_model_training_samples=(collection(10, boundary=boundary),),
                batch_sample_count=8,
                python_random_generator=generator,
                local_training_settings=after_boundary(),
            )
    assert generator.getstate() == state_before


# ---- 設定と掃引の軸 ----


def test_settings_accept_the_alarm_policies_with_matching_values():
    assert after_boundary().stored_sample_recency_decay_factor is None
    assert decayed_before_boundary(0.9).stored_sample_recency_decay_factor == 0.9
    with pytest.raises(RunSettingsValidationError):
        LocalTrainingSettings(
            **BASE_SETTINGS,
            training_sample_selection_policy="uniform_after_latest_alarm_boundary",
            stored_sample_recency_decay_factor=0.9,
        )
    with pytest.raises(RunSettingsValidationError):
        decayed_before_boundary(None)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("alarm", ("uniform_after_latest_alarm_boundary", None, None)),
        ("alarm-decay:0.99", ("recency_decayed_before_latest_alarm_boundary", None, 0.99)),
    ],
)
def test_alarm_selection_text_is_parsed_and_described_back(text, expected):
    assert parse_fedsda_training_sample_selection(text) == expected
    policy, count, factor = expected
    assert (
        describe_fedsda_training_sample_selection(
            policy=policy, most_recent_count=count, decay_factor=factor
        )
        == text
    )


@pytest.mark.parametrize(
    "text", ["alarm:", "alarm-decay:", "alarm-decay:1", "alarm-decay:.5", "Alarm", "alarm-decay"]
)
def test_invalid_alarm_selection_text_is_rejected(text):
    with pytest.raises(ValueError):
        parse_fedsda_training_sample_selection(text)


def test_run_summary_table_shows_the_alarm_selections(tmp_path):
    runs_directory = tmp_path / "fedsda" / "runs"
    runs_directory.mkdir(parents=True)
    for run_identifier, policy, factor in (
        ("a", "uniform_after_latest_alarm_boundary", None),
        ("bb", "recency_decayed_before_latest_alarm_boundary", 0.997),
    ):
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
                    "run_client_settings": {
                        "local_training_settings": dict(
                            BASE_SETTINGS,
                            training_sample_selection_policy=policy,
                            stored_sample_recency_decay_factor=factor,
                        )
                    }
                },
            },
            "run_metrics": {"prediction_accuracy": 0.5},
        }
        (runs_directory / f"{run_identifier}.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    table_result = write_run_summary_table(
        configuration_directory=tmp_path / "fedsda", configuration_name="fedsda"
    )
    with table_result.table_file_path.open(encoding="utf-8", newline="") as table_file:
        rows = {
            row["run_identifier"]: row["training_sample_selection"]
            for row in csv.DictReader(table_file)
        }
    assert rows == {"a": "alarm", "bb": "alarm-decay:0.997"}


def test_decay_before_boundary_draws_uniformly_among_the_samples_after_the_boundary():
    """区切りより後は重み1（一様）なので、最初の1件は、区切り以降のどこからも選ばれる（最新に偏らない）。"""
    first_positions = {
        sampled_positions(
            population=collection(33, boundary=20),
            batch=1,
            local_training_settings=decayed_before_boundary(1e-200),
            generator=random.Random(seed),
        )[0]
        for seed in range(200)
    }
    assert first_positions == set(range(20, 33))


def test_validation_resolution_without_samples_does_not_fail_or_mark(monkeypatch):
    """検証の標本が0件で、今のモデルの列も無いとき、区切りを付けずに確定する（例外を出さない）。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(601)
        resolution_arguments, _, _, _ = build_resolution_oracle(
            class_count=2,
            legacy_resolution_case="create_rejected",
            pending_sample_count=0,
            monkeypatch=monkeypatch,
        )
        store = ModelTrainingSampleStore()
        resolution_arguments["training_sample_store"] = store
        resolution = apply_post_alarm_candidate_validation_resolution(**resolution_arguments)
    assert resolution.resolution_outcome == "candidate_rejected"
    assert boundaries(store) == {}
