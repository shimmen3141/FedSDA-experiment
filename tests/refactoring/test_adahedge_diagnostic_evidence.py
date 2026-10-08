"""診断証拠の旧実装一致と拒否時の状態保全を確認する。"""

import random
from collections.abc import Mapping

import numpy
import pytest
import torch

from federated_drift_experiment.expert_routing import AdaHedgeRouter
from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence import (
    AdaHedgeDiagnosticEvidence,
)


class FaultingEvidenceMapping(Mapping):
    """一つ目の値を読み取った後、二つ目の読取りを失敗させる。"""

    def __iter__(self):
        return iter((2, 3))

    def __len__(self):
        return 2

    def __getitem__(self, model_id):
        if model_id == 2:
            return 0.5
        raise RuntimeError("診断入力の読取り失敗")


def get_adahedge_evidence_snapshot(diagnostic_evidence):
    return (
        diagnostic_evidence.cumulative_losses_by_model_id,
        diagnostic_evidence.mixability_gap,
        diagnostic_evidence.model_pool_reset_count,
        diagnostic_evidence.concept_operation_restart_count,
    )


def assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router):
    assert get_adahedge_evidence_snapshot(diagnostic_evidence) == (
        legacy_router.cumulative_losses,
        legacy_router.mixability_gap,
        legacy_router.pool_reset_count,
        legacy_router.concept_restart_count,
    )


@pytest.mark.parametrize("diagnostic_operation", ["acquire", "direct"])
@pytest.mark.parametrize(
    "observed_loss_sequence",
    [
        ({2: 0, -3: 1, 0: 0.3}, {0: 1, 2: 0.6, -3: 0.4}) * 8,
        ({2: 0.5, -3: 0.5}, {2: 1e308, -3: -1e308}, {-3: 3}, {-3: -5}),
        ({0: 0, -3: 0}, {7: 0, -3: 0}, {7: 1, -3: 0}),
    ],
)
def test_diagnostic_evidence_matches_real_legacy_sequences(
    diagnostic_operation, observed_loss_sequence
):
    diagnostic_evidence = AdaHedgeDiagnosticEvidence()
    legacy_router = AdaHedgeRouter()
    assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
    for observed_losses_by_model_id in observed_loss_sequence:
        if diagnostic_operation == "acquire":
            weights = diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
                model_ids=reversed(observed_losses_by_model_id)
            )
            legacy_weights = legacy_router.probabilities(observed_losses_by_model_id)
            assert weights == legacy_weights
            assert tuple(weights) == tuple(sorted(observed_losses_by_model_id))
            assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
        else:
            weights = {
                model_id: 1 / len(observed_losses_by_model_id)
                for model_id in observed_losses_by_model_id
            }
            legacy_weights = weights.copy()
        diagnostic_evidence.update_evidence_after_loss_observation(
            observed_losses_by_model_id=observed_losses_by_model_id,
            diagnostic_weights_by_model_id=weights,
        )
        legacy_router.update(observed_losses_by_model_id, legacy_weights)
        assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
    for index in range(3):
        diagnostic_evidence.restart_evidence_after_concept_operation()
        legacy_router.restart_for_concept()
        assert diagnostic_evidence.concept_operation_restart_count == index + 1
        assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
    weights = diagnostic_evidence.get_diagnostic_weights_before_loss_observation(model_ids=[2, 1])
    assert weights == legacy_router.probabilities([1, 2])
    assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)

    for weights in ({0: 0, -1: 1}, {0: 0.8, -1: 0.2}, {0: 0.5, -1: 0.5 + 5e-13}):
        diagnostic_evidence = AdaHedgeDiagnosticEvidence()
        legacy_router = AdaHedgeRouter()
        for observed_losses_by_model_id in ({0: 0.9, -1: 0.1},) * 3:
            diagnostic_evidence.update_evidence_after_loss_observation(
                observed_losses_by_model_id=observed_losses_by_model_id,
                diagnostic_weights_by_model_id=weights,
            )
            legacy_router.update(observed_losses_by_model_id, weights)
            assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
            acquired_weights = diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
                model_ids=reversed(weights)
            )
            assert acquired_weights == legacy_router.probabilities(weights)
            assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)

    # gapがゼロでも累積損失が異なるとき、最小損失の同率IDだけへ配る。
    diagnostic_evidence = AdaHedgeDiagnosticEvidence()
    legacy_router = AdaHedgeRouter()
    diagnostic_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id={2: 1, 0: 0, -1: 0},
        diagnostic_weights_by_model_id={2: 0, 0: 0.5, -1: 0.5},
    )
    legacy_router.update({2: 1, 0: 0, -1: 0}, {2: 0, 0: 0.5, -1: 0.5})
    acquired_weights = diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
        model_ids=[2, 0, -1]
    )
    assert acquired_weights == legacy_router.probabilities([2, 0, -1]) == {-1: 0.5, 0: 0.5, 2: 0}
    assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)


@pytest.mark.parametrize(
    "invalid_model_ids,exception_type",
    [
        ([], ValueError),
        ([1, 1], ValueError),
        ([True], TypeError),
        ([1.0], TypeError),
        ([numpy.int64(1)], TypeError),
        (None, TypeError),
    ],
)
def test_diagnostic_evidence_rejects_invalid_model_ids_before_synchronization(
    invalid_model_ids, exception_type
):
    diagnostic_evidence = AdaHedgeDiagnosticEvidence()
    diagnostic_evidence.get_diagnostic_weights_before_loss_observation(model_ids=[-1, 0])
    evidence_snapshot = get_adahedge_evidence_snapshot(diagnostic_evidence)
    with pytest.raises(exception_type):
        diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
            model_ids=invalid_model_ids
        )
    assert get_adahedge_evidence_snapshot(diagnostic_evidence) == evidence_snapshot
    with pytest.raises(ZeroDivisionError):
        diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
            model_ids=(model_id // (model_id - 1) for model_id in [0, 1])
        )
    assert get_adahedge_evidence_snapshot(diagnostic_evidence) == evidence_snapshot


@pytest.mark.parametrize(
    "invalid_update_case,exception_type",
    [
        ((None, {2: 1}), TypeError),
        (({2: 0}, None), TypeError),
        (({}, {}), ValueError),
        (({True: 0}, {2: 1}), TypeError),
        (({2: 0}, {True: 1}), TypeError),
        (({2: 0}, {3: 1}), ValueError),
        (({2: 0, 3: 1}, {2: 0.5, 3: 0.5 + 2e-12}), ValueError),
        ((FaultingEvidenceMapping(), {2: 0.5, 3: 0.5}), RuntimeError),
        (({2: 0, 3: 1}, FaultingEvidenceMapping()), RuntimeError),
        *[(({2: value}, {2: 1}), TypeError) for value in (True, "1", numpy.float64(1))],
        *[(({2: 0}, {2: value}), TypeError) for value in (True, "1", numpy.float64(1))],
        *[(({2: value}, {2: 1}), ValueError) for value in (float("nan"), float("inf"), 10**1000)],
        *[
            (({2: 0}, {2: value}), ValueError)
            for value in (float("nan"), float("inf"), 10**1000, -0.1, 1.1, 0, 0.5)
        ],
    ],
)
def test_diagnostic_evidence_rejects_invalid_update_before_synchronization(
    invalid_update_case, exception_type
):
    diagnostic_evidence = AdaHedgeDiagnosticEvidence()
    diagnostic_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id={-1: 0, 0: 1}, diagnostic_weights_by_model_id={-1: 0.5, 0: 0.5}
    )
    evidence_snapshot = get_adahedge_evidence_snapshot(diagnostic_evidence)
    with pytest.raises(exception_type):
        diagnostic_evidence.update_evidence_after_loss_observation(
            observed_losses_by_model_id=invalid_update_case[0],
            diagnostic_weights_by_model_id=invalid_update_case[1],
        )
    assert get_adahedge_evidence_snapshot(diagnostic_evidence) == evidence_snapshot


def test_diagnostic_evidence_isolates_copies_and_independent_owners():
    diagnostic_evidence = AdaHedgeDiagnosticEvidence()
    independent_evidence = AdaHedgeDiagnosticEvidence()
    losses, weights = {2: 0, -1: 1}, {2: 0.5, -1: 0.5}
    diagnostic_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id=losses, diagnostic_weights_by_model_id=weights
    )
    assert losses == {2: 0, -1: 1} and weights == {2: 0.5, -1: 0.5}
    evidence_snapshot = get_adahedge_evidence_snapshot(diagnostic_evidence)
    acquired_losses = diagnostic_evidence.cumulative_losses_by_model_id
    acquired_weights = diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
        model_ids=[2, -1]
    )
    acquired_losses.clear()
    acquired_weights.clear()
    losses.clear()
    weights.clear()
    assert get_adahedge_evidence_snapshot(diagnostic_evidence) == evidence_snapshot
    assert get_adahedge_evidence_snapshot(independent_evidence) == ({}, 0.0, 0, 0)
    # 観測なしでも非空集合の変更を一回として数える。
    independent_evidence.get_diagnostic_weights_before_loss_observation(model_ids=[3])
    independent_evidence.get_diagnostic_weights_before_loss_observation(model_ids=[4])
    assert independent_evidence.model_pool_reset_count == 1
    for parameter_name in (
        "mixability_gap",
        "model_pool_reset_count",
        "concept_operation_restart_count",
    ):
        with pytest.raises(AttributeError):
            setattr(diagnostic_evidence, parameter_name, 100)


def test_diagnostic_evidence_preserves_all_random_states():
    random_states_before = (
        random.getstate(),
        numpy.random.get_state(),
        torch.get_rng_state().clone(),
    )
    diagnostic_evidence = AdaHedgeDiagnosticEvidence()
    weights = diagnostic_evidence.get_diagnostic_weights_before_loss_observation(model_ids=[0, -1])
    diagnostic_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id={0: 0, -1: 1}, diagnostic_weights_by_model_id=weights
    )
    diagnostic_evidence.restart_evidence_after_concept_operation()
    random_states_after = (random.getstate(), numpy.random.get_state(), torch.get_rng_state())
    assert random_states_before[0] == random_states_after[0]
    assert random_states_before[1][0] == random_states_after[1][0]
    assert numpy.array_equal(random_states_before[1][1], random_states_after[1][1])
    assert random_states_before[1][2:] == random_states_after[1][2:]
    assert torch.equal(random_states_before[2], random_states_after[2])
