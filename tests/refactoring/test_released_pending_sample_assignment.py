"""警報のない標本での帰属確定（容量を超えた保留標本を現行モデルへ確定）を、実旧の標本処理と照合する。"""

from collections import defaultdict, deque
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from test_assigned_training_sample_absorption import (
    assert_absorption_matches_legacy,
    assert_absorption_state_unchanged,
    assert_models_optimizers_and_random_state_unchanged,
    build_absorption_oracle,
    snapshot_absorption_state,
)
from test_held_candidate_validation_progress import make_subclass_copy

import federated_learning_experiments.runtime.released_pending_sample_assignment as released_assignment_module
from federated_drift_experiment.clients.fedsda import FedSDAClient
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    TrainingDataAssignmentSettings,
)
from federated_learning_experiments.runtime.released_pending_sample_assignment import (
    assign_released_pending_samples_to_current_training_model,
)

FIRST_PENDING_SAMPLE_INDEX = 40


def build_released_assignment_oracle(*, class_count, monkeypatch, capacity, pending_sample_count):
    """吸収のoracleの標本を保留標本にし、最後の1件を観測した直後の新ownerと実旧clientを作る。"""
    (
        absorption_arguments,
        adoption_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_training_samples,
    ) = build_absorption_oracle(
        class_count=class_count,
        monkeypatch=monkeypatch,
        absorbed_sample_count=pending_sample_count,
        concept_id_case="mixed",
    )
    pending_training_assignment_buffer = PendingTrainingAssignmentBuffer(
        training_data_assignment_settings=TrainingDataAssignmentSettings(
            pending_assignment_buffer_capacity_samples=capacity
        )
    )
    pending_sample_observations = tuple(
        IndexedObservedTrainingSample(
            sample_index=FIRST_PENDING_SAMPLE_INDEX + sample_offset,
            training_sample=training_sample,
            observed_concept_id=observed_concept_id,
        )
        for sample_offset, (training_sample, observed_concept_id) in enumerate(
            zip(
                absorption_arguments["assigned_training_samples"],
                absorption_arguments["assigned_sample_concept_ids"],
            )
        )
    )
    for indexed_observation in pending_sample_observations:
        pending_training_assignment_buffer.append_observed_sample_index(
            sample_index=indexed_observation.sample_index
        )
    current_training_model_assignment = adoption_arguments["current_training_model_assignment"]
    assert (
        current_training_model_assignment.current_training_model_id
        == legacy_client.current_model_id
        == absorption_arguments["model_id"]
    )
    assignment_arguments = dict(
        pending_training_assignment_buffer=pending_training_assignment_buffer,
        pending_sample_observations=pending_sample_observations,
        current_training_model_assignment=current_training_model_assignment,
        held_model_training_state_registry=absorption_arguments[
            "held_model_training_state_registry"
        ],
        training_sample_store=absorption_arguments["training_sample_store"],
        model_training_and_assignment_counts_store=absorption_arguments[
            "model_training_and_assignment_counts_store"
        ],
        loss_statistics_store=absorption_arguments["loss_statistics_store"],
    )
    # 実旧: 最後の標本以外を保留済みにする。予測・候補検証・検出・学習は標本処理の対象外なので止め、
    # 統計・標本・概念の更新と、モデルの損失評価は実物のまま使う。
    legacy_client.buffer = deque(legacy_training_samples[:-1])
    legacy_client.fifo_size = capacity
    legacy_client.processed_samples = FIRST_PENDING_SAMPLE_INDEX + pending_sample_count - 1
    legacy_client.phase_seconds = defaultdict(float)
    legacy_client.processing_times = defaultdict(list)
    legacy_client.history_drift_type = []
    legacy_client._record_prediction = lambda *arguments: None
    legacy_client._observe_forward_validation = lambda *arguments: 0
    legacy_client._record_model_compute = lambda *arguments: None
    legacy_client._update_drift_detectors = lambda *arguments: False
    legacy_client._forced_drift_check = lambda *arguments: False
    legacy_client.train_step = lambda: None
    return (
        assignment_arguments,
        absorption_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_training_samples,
    )


def run_legacy_sample_processing_without_alarm(*, legacy_client, legacy_training_samples):
    """実旧の標本処理へ最後の標本を渡す（保留への追加と、容量を超えた分の確定が実行される）。"""
    input_features, observed_class_labels, observed_concept_id = legacy_training_samples[-1]
    FedSDAClient.process_one_step(
        legacy_client, input_features, observed_class_labels, observed_concept_id
    )


def snapshot_assignment_state(
    *, assignment_arguments, absorption_arguments, shared_optimizer_owners
):
    return (
        snapshot_absorption_state(
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        ),
        assignment_arguments["pending_training_assignment_buffer"].get_state_snapshot(),
        assignment_arguments["current_training_model_assignment"].current_training_model_id,
    )


def assert_assignment_state_unchanged(
    *, state_snapshot, assignment_arguments, absorption_arguments
):
    absorption_snapshot, pending_assignment_state, current_training_model_id = state_snapshot
    assert_absorption_state_unchanged(
        previous_snapshot=absorption_snapshot, valid_absorption_arguments=absorption_arguments
    )
    assert (
        assignment_arguments["pending_training_assignment_buffer"].get_state_snapshot()
        == pending_assignment_state
    )
    assert (
        assignment_arguments["current_training_model_assignment"].current_training_model_id
        == current_training_model_id
    )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize(
    "capacity,pending_sample_count",
    [(3, 1), (3, 3), (3, 4), (1, 2), (2, 5), (1, 6)],
)
def test_released_assignment_matches_real_legacy_sample_processing(
    class_count, capacity, pending_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(503)
        (
            assignment_arguments,
            absorption_arguments,
            shared_optimizer_owners,
            legacy_client,
            legacy_training_samples,
        ) = build_released_assignment_oracle(
            class_count=class_count,
            monkeypatch=monkeypatch,
            capacity=capacity,
            pending_sample_count=pending_sample_count,
        )
        previous_snapshot = snapshot_absorption_state(
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        pending_sample_observations = assignment_arguments["pending_sample_observations"]
        run_legacy_sample_processing_without_alarm(
            legacy_client=legacy_client, legacy_training_samples=legacy_training_samples
        )
        released_sample_observations = assign_released_pending_samples_to_current_training_model(
            **assignment_arguments
        )
        expected_released_count = max(0, pending_sample_count - capacity)
        # 解放されるのは最古の標本から。渡した標本そのものを返す。
        assert type(released_sample_observations) is tuple
        assert len(released_sample_observations) == expected_released_count
        assert all(
            released_observation is pending_observation
            for released_observation, pending_observation in zip(
                released_sample_observations, pending_sample_observations
            )
        )
        # 標本・割当概念計数・損失統計は、実旧の標本処理の後と一致する。
        assert_absorption_matches_legacy(
            absorption_arguments=absorption_arguments, legacy_client=legacy_client
        )
        # 保留に残る位置は、実旧のFIFOに残る標本と同じ（新しい側のcapacity件）。
        pending_assignment_state = assignment_arguments[
            "pending_training_assignment_buffer"
        ].get_state_snapshot()
        assert pending_assignment_state.pending_sample_indices == tuple(
            indexed_observation.sample_index
            for indexed_observation in pending_sample_observations[expected_released_count:]
        )
        assert len(pending_assignment_state.pending_sample_indices) == len(legacy_client.buffer)
        assert all(
            legacy_pending_sample[0] is indexed_observation.training_sample.input_features
            for legacy_pending_sample, indexed_observation in zip(
                legacy_client.buffer, pending_sample_observations[expected_released_count:]
            )
        )
        assert (
            pending_assignment_state.last_observed_sample_index
            == pending_sample_observations[-1].sample_index
        )
        # 学習帰属、モデル、optimizer、乱数は変わらない。
        assert (
            assignment_arguments["current_training_model_assignment"].current_training_model_id
            == legacy_client.current_model_id
        )
        assert_models_optimizers_and_random_state_unchanged(
            previous_snapshot=previous_snapshot,
            registry=assignment_arguments["held_model_training_state_registry"],
        )
        # 解放する標本がなくなった後の再実行は何も変えない。
        state_snapshot = snapshot_assignment_state(
            assignment_arguments=assignment_arguments,
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        remaining_arguments = assignment_arguments | dict(
            pending_sample_observations=pending_sample_observations[expected_released_count:]
        )
        assert (
            assign_released_pending_samples_to_current_training_model(**remaining_arguments) == ()
        )
        assert_assignment_state_unchanged(
            state_snapshot=state_snapshot,
            assignment_arguments=assignment_arguments,
            absorption_arguments=absorption_arguments,
        )


def test_released_samples_are_absorbed_before_pending_positions_are_released(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(503)
        assignment_arguments, _, _, _, _ = build_released_assignment_oracle(
            class_count=2, monkeypatch=monkeypatch, capacity=2, pending_sample_count=5
        )
        pending_training_assignment_buffer = assignment_arguments[
            "pending_training_assignment_buffer"
        ]
        pending_sample_observations = assignment_arguments["pending_sample_observations"]
        absorption_calls = []
        absorb_assigned_samples = (
            released_assignment_module.absorb_assigned_training_samples_into_held_model
        )

        def record_absorption_call(**absorption_arguments):
            absorption_calls.append(
                (
                    absorption_arguments,
                    pending_training_assignment_buffer.get_state_snapshot().pending_sample_indices,
                )
            )
            return absorb_assigned_samples(**absorption_arguments)

        monkeypatch.setattr(
            released_assignment_module,
            "absorb_assigned_training_samples_into_held_model",
            record_absorption_call,
        )
        assign_released_pending_samples_to_current_training_model(**assignment_arguments)
    # 吸収は1回。その時点で保留はまだ解放されていない。吸収へ渡るのは最古の3件と現在のモデルID。
    assert len(absorption_calls) == 1
    absorption_arguments, pending_sample_indices_at_absorption = absorption_calls[0]
    assert pending_sample_indices_at_absorption == tuple(
        indexed_observation.sample_index for indexed_observation in pending_sample_observations
    )
    assert absorption_arguments["model_id"] == (
        assignment_arguments["current_training_model_assignment"].current_training_model_id
    )
    assert len(absorption_arguments["assigned_training_samples"]) == 3
    assert all(
        training_sample is indexed_observation.training_sample
        for training_sample, indexed_observation in zip(
            absorption_arguments["assigned_training_samples"], pending_sample_observations
        )
    )
    assert absorption_arguments["assigned_sample_concept_ids"] == tuple(
        indexed_observation.observed_concept_id
        for indexed_observation in pending_sample_observations[:3]
    )
    for owner_argument_name in (
        "held_model_training_state_registry",
        "training_sample_store",
        "model_training_and_assignment_counts_store",
        "loss_statistics_store",
    ):
        assert (
            absorption_arguments[owner_argument_name] is assignment_arguments[owner_argument_name]
        )


def replace_observation(pending_sample_observations, position, replacement):
    return (
        pending_sample_observations[:position]
        + (replacement,)
        + pending_sample_observations[position + 1 :]
    )


# 条件名 -> (不正にする引数名, 正常な引数から不正な値を作る操作, 期待する例外)。保留5件・容量2（最古の3件が解放対象）。
INVALID_RELEASED_ASSIGNMENT_INPUT_CASES = {
    "buffer_other_type": (
        "pending_training_assignment_buffer",
        lambda assignment_arguments: object(),
        TypeError,
    ),
    "buffer_subclass": (
        "pending_training_assignment_buffer",
        lambda assignment_arguments: make_subclass_copy(
            assignment_arguments["pending_training_assignment_buffer"]
        ),
        TypeError,
    ),
    "assignment_other_type": (
        "current_training_model_assignment",
        lambda assignment_arguments: object(),
        TypeError,
    ),
    "assignment_subclass": (
        "current_training_model_assignment",
        lambda assignment_arguments: make_subclass_copy(
            assignment_arguments["current_training_model_assignment"]
        ),
        TypeError,
    ),
    "observations_list": (
        "pending_sample_observations",
        lambda assignment_arguments: list(assignment_arguments["pending_sample_observations"]),
        TypeError,
    ),
    "observations_tuple_subclass": (
        "pending_sample_observations",
        lambda assignment_arguments: type("TupleSubclass", (tuple,), {})(
            assignment_arguments["pending_sample_observations"]
        ),
        TypeError,
    ),
    "observation_other_type": (
        "pending_sample_observations",
        lambda assignment_arguments: replace_observation(
            assignment_arguments["pending_sample_observations"], 4, object()
        ),
        TypeError,
    ),
    "observation_subclass": (
        "pending_sample_observations",
        lambda assignment_arguments: replace_observation(
            assignment_arguments["pending_sample_observations"],
            4,
            make_subclass_copy(assignment_arguments["pending_sample_observations"][4]),
        ),
        TypeError,
    ),
    "observation_index_int_subclass": (
        "pending_sample_observations",
        lambda assignment_arguments: replace_observation(
            assignment_arguments["pending_sample_observations"],
            4,
            replace(
                assignment_arguments["pending_sample_observations"][4],
                sample_index=type("IntSubclass", (int,), {})(FIRST_PENDING_SAMPLE_INDEX + 4),
            ),
        ),
        TypeError,
    ),
    "observations_missing_newest": (
        "pending_sample_observations",
        lambda assignment_arguments: assignment_arguments["pending_sample_observations"][:-1],
        ValueError,
    ),
    "observations_missing_oldest": (
        "pending_sample_observations",
        lambda assignment_arguments: assignment_arguments["pending_sample_observations"][1:],
        ValueError,
    ),
    "observations_reordered": (
        "pending_sample_observations",
        lambda assignment_arguments: assignment_arguments["pending_sample_observations"][::-1],
        ValueError,
    ),
    "observation_index_shifted": (
        "pending_sample_observations",
        lambda assignment_arguments: replace_observation(
            assignment_arguments["pending_sample_observations"],
            0,
            replace(assignment_arguments["pending_sample_observations"][0], sample_index=7),
        ),
        ValueError,
    ),
    # 解放対象の最後（3件目）の標本だけ特徴数が違う。吸収が全標本の検査を終える前に、保留を解放してはならない。
    "released_sample_with_wrong_feature_count": (
        "pending_sample_observations",
        lambda assignment_arguments: replace_observation(
            assignment_arguments["pending_sample_observations"],
            2,
            replace(
                assignment_arguments["pending_sample_observations"][2],
                training_sample=ObservedTrainingSample(
                    input_features=torch.zeros(1, 99),
                    observed_class_labels=assignment_arguments["pending_sample_observations"][
                        2
                    ].training_sample.observed_class_labels,
                ),
            ),
        ),
        (RuntimeError, ValueError),
    ),
    "released_sample_concept_id_not_int": (
        "pending_sample_observations",
        lambda assignment_arguments: replace_observation(
            assignment_arguments["pending_sample_observations"],
            1,
            replace(
                assignment_arguments["pending_sample_observations"][1], observed_concept_id=1.0
            ),
        ),
        TypeError,
    ),
    # 現在の学習帰属のモデルが保有されていない。吸収が、更新と保留の解放より前に拒否する。
    "released_sample_for_model_that_is_not_held": (
        "current_training_model_assignment",
        lambda assignment_arguments: CurrentTrainingModelAssignment(initial_model_id=777),
        LookupError,
    ),
    "loss_statistics_store_other_type": (
        "loss_statistics_store",
        lambda assignment_arguments: object(),
        TypeError,
    ),
}


@pytest.mark.parametrize("invalid_case", INVALID_RELEASED_ASSIGNMENT_INPUT_CASES)
def test_released_assignment_rejects_invalid_input_before_any_update(invalid_case, monkeypatch):
    argument_name, make_invalid_argument, expected_exception = (
        INVALID_RELEASED_ASSIGNMENT_INPUT_CASES[invalid_case]
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(503)
        assignment_arguments, absorption_arguments, shared_optimizer_owners, _, _ = (
            build_released_assignment_oracle(
                class_count=4, monkeypatch=monkeypatch, capacity=2, pending_sample_count=5
            )
        )
        state_snapshot = snapshot_assignment_state(
            assignment_arguments=assignment_arguments,
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        with pytest.raises(expected_exception):
            assign_released_pending_samples_to_current_training_model(
                **assignment_arguments
                | {argument_name: make_invalid_argument(assignment_arguments)}
            )
        assert_assignment_state_unchanged(
            state_snapshot=state_snapshot,
            assignment_arguments=assignment_arguments,
            absorption_arguments=absorption_arguments,
        )


@pytest.mark.parametrize(
    "invalid_case",
    [
        invalid_case
        for invalid_case, (argument_name, _, _) in INVALID_RELEASED_ASSIGNMENT_INPUT_CASES.items()
        if argument_name != "loss_statistics_store"
        and not invalid_case.startswith("released_sample_")
    ],
)
def test_own_checks_reject_before_absorption_is_called(invalid_case, monkeypatch):
    argument_name, make_invalid_argument, expected_exception = (
        INVALID_RELEASED_ASSIGNMENT_INPUT_CASES[invalid_case]
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(503)
        assignment_arguments = build_released_assignment_oracle(
            class_count=2, monkeypatch=monkeypatch, capacity=2, pending_sample_count=5
        )[0]
    absorption_call = Mock(side_effect=AssertionError)
    monkeypatch.setattr(
        released_assignment_module,
        "absorb_assigned_training_samples_into_held_model",
        absorption_call,
    )
    with pytest.raises(expected_exception):
        assign_released_pending_samples_to_current_training_model(
            **assignment_arguments | {argument_name: make_invalid_argument(assignment_arguments)}
        )
    absorption_call.assert_not_called()


@pytest.mark.parametrize(
    "invalid_argument_name", ("current_training_model_assignment", "pending_sample_observations")
)
def test_type_checks_take_priority_over_index_mismatch(invalid_argument_name, monkeypatch):
    """型の不正と並びの不一致が同時にあるとき、型の検査が先に拒否する（設計4節の検査の順）。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(503)
        assignment_arguments = build_released_assignment_oracle(
            class_count=2, monkeypatch=monkeypatch, capacity=2, pending_sample_count=5
        )[0]
    mismatched_observations = assignment_arguments["pending_sample_observations"][1:]
    invalid_arguments = {
        "current_training_model_assignment": dict(
            current_training_model_assignment=object(),
            pending_sample_observations=mismatched_observations,
        ),
        "pending_sample_observations": dict(
            pending_sample_observations=list(mismatched_observations)
        ),
    }[invalid_argument_name]
    with pytest.raises(TypeError):
        assign_released_pending_samples_to_current_training_model(
            **assignment_arguments | invalid_arguments
        )


def test_absorption_is_not_called_when_no_sample_is_released(monkeypatch):
    """保留が容量以下なら吸収を呼ばない（旧も、解放する標本がなければモデルや統計に触れない）。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(503)
        assignment_arguments = build_released_assignment_oracle(
            class_count=2, monkeypatch=monkeypatch, capacity=3, pending_sample_count=3
        )[0]
    absorption_call = Mock(side_effect=AssertionError)
    monkeypatch.setattr(
        released_assignment_module,
        "absorb_assigned_training_samples_into_held_model",
        absorption_call,
    )
    assert assign_released_pending_samples_to_current_training_model(**assignment_arguments) == ()
    absorption_call.assert_not_called()


def test_invalid_sample_that_is_not_released_is_not_inspected(monkeypatch):
    """解放対象でない保留標本の中身は読まない（その標本は、後で解放されるときか警報のときに検査される）。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(503)
        assignment_arguments = build_released_assignment_oracle(
            class_count=2, monkeypatch=monkeypatch, capacity=2, pending_sample_count=5
        )[0]
        pending_sample_observations = assignment_arguments["pending_sample_observations"]
        assignment_arguments["pending_sample_observations"] = replace_observation(
            pending_sample_observations,
            4,
            replace(pending_sample_observations[4], training_sample=object()),
        )
        released_sample_observations = assign_released_pending_samples_to_current_training_model(
            **assignment_arguments
        )
    assert len(released_sample_observations) == 3
