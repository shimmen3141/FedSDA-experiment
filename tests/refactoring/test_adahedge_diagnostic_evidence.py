"""診断証拠の旧実装一致と拒否時の状態保全を確認する。"""

import math
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
        diagnostic_evidence.aggregation_restart_count,
        diagnostic_evidence.aggregation_recalibration_count,
        diagnostic_evidence.aggregation_recalibration_sample_count,
    )


def assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router):
    assert get_adahedge_evidence_snapshot(diagnostic_evidence) == (
        legacy_router.cumulative_losses,
        legacy_router.mixability_gap,
        legacy_router.pool_reset_count,
        legacy_router.concept_restart_count,
        legacy_router.aggregation_restart_count,
        legacy_router.aggregation_recalibration_count,
        legacy_router.aggregation_recalibration_sample_count,
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
    invalid_model_ids, exception_type, monkeypatch
):
    diagnostic_evidence = AdaHedgeDiagnosticEvidence()
    diagnostic_evidence.get_diagnostic_weights_before_loss_observation(model_ids=[-1, 0])
    evidence_snapshot = get_adahedge_evidence_snapshot(diagnostic_evidence)
    with pytest.raises(exception_type):
        diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
            model_ids=invalid_model_ids
        )
    assert get_adahedge_evidence_snapshot(diagnostic_evidence) == evidence_snapshot
    # 集合変更の候補を作った後でも、演算例外なら元の四状態を保持する。
    with monkeypatch.context() as monkeypatch:
        monkeypatch.setattr(math, "isinf", lambda value: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            diagnostic_evidence.get_diagnostic_weights_before_loss_observation(model_ids=[2, 3])
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
        (([(2, 0)], {2: 1}), TypeError),
        (({2: 0}, [(2, 1)]), TypeError),
        (({}, {}), ValueError),
        (({True: 0}, {2: 1}), TypeError),
        (({2: 0}, {True: 1}), TypeError),
        (({2: 0}, {3: 1}), ValueError),
        (({2: 0, 3: 1}, {2: 0.5, 3: 0.5 + 2e-12}), ValueError),
        (({2: 0.5, 3: 0.5}, {2: -0.1, 3: 1.1}), ValueError),
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
    invalid_update_case, exception_type, monkeypatch
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
    # 入力が有効でも、候補の数値演算が失敗した時点ではcommitしない。
    with monkeypatch.context() as monkeypatch:
        monkeypatch.setattr(math, "isinf", lambda value: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            diagnostic_evidence.update_evidence_after_loss_observation(
                observed_losses_by_model_id={2: 0, 3: 1},
                diagnostic_weights_by_model_id={2: 0.5, 3: 0.5},
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
    assert get_adahedge_evidence_snapshot(independent_evidence) == ({}, 0.0, 0, 0, 0, 0, 0)
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


def observe_losses_in_both(diagnostic_evidence, legacy_router, observed_losses_by_model_id):
    """観測前の重みの取得と、損失の観測による更新を、両実装で1回行う。"""
    weights = diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
        model_ids=observed_losses_by_model_id
    )
    legacy_weights = legacy_router.probabilities(observed_losses_by_model_id)
    diagnostic_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id=observed_losses_by_model_id,
        diagnostic_weights_by_model_id=weights,
    )
    legacy_router.update(observed_losses_by_model_id, legacy_weights)


# 集約後に再生する損失の列。空、モデルが1つ、途中でモデル集合が変わる列、範囲外の損失を含む。
AGGREGATION_REPLAY_LOSS_SEQUENCES = [
    (),
    ({2: 0.25, -3: 0.75, 0: 0.5},),
    ({2: 0, -3: 1, 0: 0.3}, {0: 1, 2: 0.6, -3: 0.4}) * 5,
    ({-3: 0.5}, {-3: 0.25}),
    ({0: 0.1, -3: 0.9}, {0: 0.2, -3: 0.8}, {7: 0.5, -3: 0.5}, {7: 1, -3: 0}),
    ({2: 3, -3: -5}, {2: 1e308, -3: -1e308}, {2: 1, -3: 0}),
]


@pytest.mark.parametrize("replayed_loss_sequence", AGGREGATION_REPLAY_LOSS_SEQUENCES)
@pytest.mark.parametrize(
    "preceding_loss_sequence",
    [(), ({2: 0, -3: 1, 0: 0.3}, {0: 1, 2: 0.6, -3: 0.4}) * 4, ({5: 0.5, 6: 0.25},)],
)
def test_diagnostic_evidence_replay_and_restart_after_aggregation_match_real_legacy(
    preceding_loss_sequence, replayed_loss_sequence
):
    """通常の観測の後で、集約後の再生と再始動を、実旧のAdaHedgeと同じ順に行い、各操作の後に全状態を照合する。"""
    diagnostic_evidence = AdaHedgeDiagnosticEvidence()
    legacy_router = AdaHedgeRouter()
    for observed_losses_by_model_id in preceding_loss_sequence:
        observe_losses_in_both(diagnostic_evidence, legacy_router, observed_losses_by_model_id)
    assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
    concept_operation_restart_count = diagnostic_evidence.concept_operation_restart_count
    # 集約後の再生（1回の反復で読み切る入力も受け取る）。
    diagnostic_evidence.replay_observed_losses_after_aggregation(
        observed_loss_sequence=iter(replayed_loss_sequence)
    )
    legacy_router.replay_after_aggregation(replayed_loss_sequence)
    assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
    assert diagnostic_evidence.aggregation_recalibration_count == int(bool(replayed_loss_sequence))
    assert diagnostic_evidence.aggregation_recalibration_sample_count == len(replayed_loss_sequence)
    assert diagnostic_evidence.aggregation_restart_count == 0
    # 再生の後も、通常の観測を続けられる。
    for observed_losses_by_model_id in ({2: 0.5, -3: 0.25, 0: 1}, {2: 0, -3: 1, 0: 0.5}):
        observe_losses_in_both(diagnostic_evidence, legacy_router, observed_losses_by_model_id)
        assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
    # 集約後の再始動は、証拠が空でも数える。
    for restart_index in range(2):
        diagnostic_evidence.restart_evidence_after_aggregation()
        legacy_router.restart_after_aggregation()
        assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
        assert diagnostic_evidence.aggregation_restart_count == restart_index + 1
        assert diagnostic_evidence.cumulative_losses_by_model_id == {}
        assert diagnostic_evidence.mixability_gap == 0.0
    # 再始動の後の再生と観測。
    diagnostic_evidence.replay_observed_losses_after_aggregation(
        observed_loss_sequence=replayed_loss_sequence
    )
    legacy_router.replay_after_aggregation(replayed_loss_sequence)
    assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
    observe_losses_in_both(diagnostic_evidence, legacy_router, {2: 0.5, -3: 0.25})
    assert_adahedge_matches_legacy(diagnostic_evidence, legacy_router)
    # 集約後の操作は、概念操作による再始動の回数を変えない。
    assert diagnostic_evidence.concept_operation_restart_count == concept_operation_restart_count


def raise_while_enumerating_loss_sequence():
    yield {2: 0.5, 3: 0.25}
    raise RuntimeError("損失の列の列挙の失敗")


@pytest.mark.parametrize(
    "make_invalid_loss_sequence,expected_exception",
    [
        (lambda: ({2: 0.5, 3: 0.25}, [(2, 0.5)]), TypeError),
        (lambda: ({2: 0.5, 3: 0.25}, None), TypeError),
        (lambda: ({2: 0.5, 3: 0.25}, {True: 0.5, 3: 0.25}), TypeError),
        (lambda: ({2: 0.5, 3: 0.25}, {"2": 0.5}), TypeError),
        (lambda: ({2: 0.5, 3: 0.25}, {}), ValueError),
        (lambda: ({2: 0.5, 3: 0.25}, {2: math.nan, 3: 0.25}), ValueError),
        (lambda: ({2: 0.5, 3: 0.25}, {2: math.inf, 3: 0.25}), ValueError),
        (lambda: ({2: 0.5, 3: 0.25}, {2: "0.5", 3: 0.25}), TypeError),
        (lambda: ({2: 0.5, 3: 0.25}, {2: torch.tensor(0.5), 3: 0.25}), TypeError),
        (lambda: ({2: 0.5, 3: 0.25}, FaultingEvidenceMapping()), RuntimeError),
        (raise_while_enumerating_loss_sequence, RuntimeError),
        (lambda: None, TypeError),
    ],
)
def test_diagnostic_evidence_rejects_invalid_replay_after_aggregation_without_change(
    make_invalid_loss_sequence, expected_exception
):
    """不正な行が後ろにあっても、証拠も計数も変える前に拒否する。"""
    diagnostic_evidence = AdaHedgeDiagnosticEvidence()
    for observed_losses_by_model_id in ({2: 0, 3: 1}, {2: 0.4, 3: 0.6}):
        weights = diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
            model_ids=observed_losses_by_model_id
        )
        diagnostic_evidence.update_evidence_after_loss_observation(
            observed_losses_by_model_id=observed_losses_by_model_id,
            diagnostic_weights_by_model_id=weights,
        )
    evidence_snapshot = get_adahedge_evidence_snapshot(diagnostic_evidence)
    assert evidence_snapshot[0] != {}
    with pytest.raises(expected_exception):
        diagnostic_evidence.replay_observed_losses_after_aggregation(
            observed_loss_sequence=make_invalid_loss_sequence()
        )
    assert get_adahedge_evidence_snapshot(diagnostic_evidence) == evidence_snapshot
