"""保持した診断証拠と帰属変更通知を実旧クライアントへ照合する。"""

import random
from collections import defaultdict

import numpy
import pytest
import torch
from test_adahedge_diagnostic_evidence import (
    assert_adahedge_matches_legacy,
    get_adahedge_evidence_snapshot,
)

from federated_drift_experiment.clients.fedsda import (
    RestartingSoftRoutingClassConditionalESRFedSDAClient,
)
from federated_drift_experiment.expert_routing import AdaHedgeRouter
from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import (
    AdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import (
    notify_diagnostics_of_training_assignment_change,
)


def build_legacy_diagnostic_client():
    legacy_client = RestartingSoftRoutingClassConditionalESRFedSDAClient.__new__(
        RestartingSoftRoutingClassConditionalESRFedSDAClient
    )
    legacy_client.current_model_id = -1
    legacy_client.expert_router = AdaHedgeRouter()
    legacy_client.context_expert_routers = {}
    legacy_client.shadow_meta_routers = {}
    legacy_client.routing_active_set = None
    legacy_client.oracle_concept_expert_routers = defaultdict(AdaHedgeRouter)
    return legacy_client


def get_diagnostic_collection_snapshot(diagnostic_collection):
    return (
        get_adahedge_evidence_snapshot(diagnostic_collection.global_diagnostic_evidence),
        diagnostic_collection.created_true_concept_ids,
        tuple(
            get_adahedge_evidence_snapshot(
                diagnostic_collection.get_true_concept_diagnostic_evidence(
                    true_concept_id=concept_id
                )
            )
            for concept_id in diagnostic_collection.created_true_concept_ids
        ),
    )


def assert_diagnostic_collection_matches_legacy(diagnostic_collection, legacy_client):
    assert_adahedge_matches_legacy(
        diagnostic_collection.global_diagnostic_evidence, legacy_client.expert_router
    )
    assert diagnostic_collection.created_true_concept_ids == tuple(
        legacy_client.oracle_concept_expert_routers
    )
    for concept_id in diagnostic_collection.created_true_concept_ids:
        assert_adahedge_matches_legacy(
            diagnostic_collection.get_true_concept_diagnostic_evidence(true_concept_id=concept_id),
            legacy_client.oracle_concept_expert_routers[concept_id],
        )


def test_held_diagnostic_evidence_matches_real_legacy_notifications():
    diagnostic_collection = AdaHedgeDiagnosticEvidenceCollection()
    legacy_client = build_legacy_diagnostic_client()
    assert_diagnostic_collection_matches_legacy(diagnostic_collection, legacy_client)
    assignment_changes = (None, (-1, -1), (-1, 2), (2, -3), (-3, -3), (-3, 2), (2, -3))
    for index, case in enumerate(assignment_changes):
        for concept_id in (None, 7, -4, 7):
            if concept_id is None:
                diagnostic_evidence = diagnostic_collection.global_diagnostic_evidence
                legacy_router = legacy_client.expert_router
            else:
                diagnostic_evidence = diagnostic_collection.get_true_concept_diagnostic_evidence(
                    true_concept_id=concept_id
                )
                legacy_router = legacy_client.oracle_concept_expert_routers[concept_id]
            observed_losses = {2: 0.1, -1: 0.8} if index % 2 else {5: 0.7, -1: 0.2, 2: 0.4}
            diagnostic_weights = diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
                model_ids=reversed(observed_losses)
            )
            legacy_weights = legacy_router.probabilities(observed_losses)
            assert diagnostic_weights == legacy_weights
            assert_diagnostic_collection_matches_legacy(diagnostic_collection, legacy_client)
            diagnostic_evidence.update_evidence_after_loss_observation(
                observed_losses_by_model_id=observed_losses,
                diagnostic_weights_by_model_id=diagnostic_weights,
            )
            legacy_router.update(observed_losses, legacy_weights)
            assert_diagnostic_collection_matches_legacy(diagnostic_collection, legacy_client)
        before = get_diagnostic_collection_snapshot(diagnostic_collection)
        notify_diagnostics_of_training_assignment_change(
            assignment_change=(
                None
                if case is None
                else TrainingModelAssignmentChange(
                    previous_model_id=case[0], current_model_id=case[1]
                )
            ),
            diagnostic_evidence_collection=diagnostic_collection,
        )
        if case is not None:
            assert legacy_client.current_model_id == case[0]
            legacy_client._set_local_current_model(case[1])
        after = get_diagnostic_collection_snapshot(diagnostic_collection)
        assert after[1:] == before[1:]
        assert after[0][2] == before[0][2]
        if case is None or case[0] == case[1]:
            assert after == before
        else:
            assert after[0] == ({}, 0.0, before[0][2], before[0][3] + 1, *before[0][4:])
        assert_diagnostic_collection_matches_legacy(diagnostic_collection, legacy_client)
        for concept_id in (None, -4, 7):
            diagnostic_evidence = (
                diagnostic_collection.global_diagnostic_evidence
                if concept_id is None
                else diagnostic_collection.get_true_concept_diagnostic_evidence(
                    true_concept_id=concept_id
                )
            )
            legacy_router = (
                legacy_client.expert_router
                if concept_id is None
                else legacy_client.oracle_concept_expert_routers[concept_id]
            )
            assert diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
                model_ids=[2, -1]
            ) == legacy_router.probabilities([2, -1])
            assert_diagnostic_collection_matches_legacy(diagnostic_collection, legacy_client)
    # 同じ変更recordの再通知も毎回数える。履歴・重複防止は保持しない。
    before = get_diagnostic_collection_snapshot(diagnostic_collection)
    for index in range(2):
        notify_diagnostics_of_training_assignment_change(
            assignment_change=TrainingModelAssignmentChange(
                previous_model_id=2, current_model_id=-3
            ),
            diagnostic_evidence_collection=diagnostic_collection,
        )
        assert diagnostic_collection.global_diagnostic_evidence.concept_operation_restart_count == (
            before[0][3] + index + 1
        )
        assert get_diagnostic_collection_snapshot(diagnostic_collection)[1:] == before[1:]


def test_diagnostic_collection_keeps_distinct_live_owners():
    first_collection = AdaHedgeDiagnosticEvidenceCollection()
    second_collection = AdaHedgeDiagnosticEvidenceCollection()
    assert first_collection.created_true_concept_ids == ()
    assert get_diagnostic_collection_snapshot(first_collection) == (
        ({}, 0.0, 0, 0, 0, 0, 0),
        (),
        (),
    )
    diagnostic_evidence = first_collection.global_diagnostic_evidence
    assert diagnostic_evidence is first_collection.global_diagnostic_evidence
    assert diagnostic_evidence is not second_collection.global_diagnostic_evidence
    concept_evidence = first_collection.get_true_concept_diagnostic_evidence(true_concept_id=-7)
    assert concept_evidence is first_collection.get_true_concept_diagnostic_evidence(
        true_concept_id=-7
    )
    assert concept_evidence is not diagnostic_evidence
    concept_ids_before = first_collection.created_true_concept_ids
    assert concept_ids_before == (-7,)
    assert first_collection.get_true_concept_diagnostic_evidence(true_concept_id=2) is not (
        concept_evidence
    )
    assert concept_evidence is not second_collection.get_true_concept_diagnostic_evidence(
        true_concept_id=-7
    )
    assert first_collection.created_true_concept_ids == (-7, 2)
    assert concept_ids_before == (-7,)
    before = get_diagnostic_collection_snapshot(second_collection)
    concept_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id={2: 0, -1: 1},
        diagnostic_weights_by_model_id={2: 0.5, -1: 0.5},
    )
    assert first_collection.get_true_concept_diagnostic_evidence(
        true_concept_id=-7
    ).cumulative_losses_by_model_id == {-1: 1.0, 2: 0.0}
    assert get_adahedge_evidence_snapshot(diagnostic_evidence) == ({}, 0.0, 0, 0, 0, 0, 0)
    assert get_adahedge_evidence_snapshot(
        first_collection.get_true_concept_diagnostic_evidence(true_concept_id=2)
    ) == ({}, 0.0, 0, 0, 0, 0, 0)
    assert get_diagnostic_collection_snapshot(second_collection) == before
    diagnostic_evidence.restart_evidence_after_concept_operation()
    assert first_collection.global_diagnostic_evidence.concept_operation_restart_count == 1


@pytest.mark.parametrize(
    "invalid_notification_case",
    [
        "collection_none",
        "collection_object",
        "collection_subclass",
        "record_object",
        "record_subclass",
        "previous_bool",
        "current_bool",
        "previous_none",
        "current_none",
        "previous_float",
        "current_numpy_int",
        "previous_int_subclass",
        "current_int_subclass",
        "equal_bool",
    ],
)
def test_diagnostic_notification_rejects_invalid_input_before_restart(invalid_notification_case):
    diagnostic_collection = AdaHedgeDiagnosticEvidenceCollection()
    for diagnostic_evidence in (
        diagnostic_collection.global_diagnostic_evidence,
        diagnostic_collection.get_true_concept_diagnostic_evidence(true_concept_id=-3),
    ):
        diagnostic_evidence.update_evidence_after_loss_observation(
            observed_losses_by_model_id={-1: 0, 2: 1},
            diagnostic_weights_by_model_id={-1: 0.5, 2: 0.5},
        )
        diagnostic_evidence.get_diagnostic_weights_before_loss_observation(model_ids=[3, 2])
    notification_arguments = {
        "assignment_change": TrainingModelAssignmentChange(
            previous_model_id=-1, current_model_id=2
        ),
        "diagnostic_evidence_collection": diagnostic_collection,
    }
    if invalid_notification_case == "collection_none":
        notification_arguments["diagnostic_evidence_collection"] = None
        notification_arguments["assignment_change"] = None
    elif invalid_notification_case == "collection_object":
        notification_arguments["diagnostic_evidence_collection"] = object()
    elif invalid_notification_case == "collection_subclass":
        notification_arguments["diagnostic_evidence_collection"] = type(
            "RejectedCollection", (AdaHedgeDiagnosticEvidenceCollection,), {}
        )()
    elif invalid_notification_case == "record_object":
        notification_arguments["assignment_change"] = object()
    elif invalid_notification_case == "record_subclass":
        notification_arguments["assignment_change"] = type(
            "RejectedAssignmentChange", (TrainingModelAssignmentChange,), {}
        )(previous_model_id=-1, current_model_id=2)
    else:
        notification_arguments["assignment_change"] = TrainingModelAssignmentChange(
            previous_model_id={
                "previous_bool": False,
                "previous_none": None,
                "previous_float": -1.0,
                "previous_int_subclass": type("RejectedModelId", (int,), {})(-1),
                "equal_bool": True,
            }.get(invalid_notification_case, -1),
            current_model_id={
                "current_bool": True,
                "current_none": None,
                "current_numpy_int": numpy.int64(2),
                "current_int_subclass": type("RejectedModelId", (int,), {})(2),
                "equal_bool": True,
            }.get(invalid_notification_case, 2),
        )
    diagnostic_states_before = get_diagnostic_collection_snapshot(diagnostic_collection)
    with pytest.raises(TypeError):
        notify_diagnostics_of_training_assignment_change(**notification_arguments)
    diagnostic_states_after = get_diagnostic_collection_snapshot(diagnostic_collection)
    assert diagnostic_states_after == diagnostic_states_before


@pytest.mark.parametrize(
    "invalid_concept_id",
    [True, False, None, 2.0, numpy.int64(2), "2", type("RejectedConceptId", (int,), {})(2)],
)
def test_diagnostic_collection_rejects_invalid_concept_id_without_creation(invalid_concept_id):
    diagnostic_collection = AdaHedgeDiagnosticEvidenceCollection()
    diagnostic_collection.get_true_concept_diagnostic_evidence(true_concept_id=3)
    before = get_diagnostic_collection_snapshot(diagnostic_collection)
    with pytest.raises(TypeError):
        diagnostic_collection.get_true_concept_diagnostic_evidence(
            true_concept_id=invalid_concept_id
        )
    assert get_diagnostic_collection_snapshot(diagnostic_collection) == before


def test_diagnostic_notification_preserves_random_states():
    random_states_before = (
        random.getstate(),
        numpy.random.get_state(),
        torch.get_rng_state().clone(),
    )
    diagnostic_collection = AdaHedgeDiagnosticEvidenceCollection()
    diagnostic_evidence = diagnostic_collection.global_diagnostic_evidence
    diagnostic_weights = diagnostic_evidence.get_diagnostic_weights_before_loss_observation(
        model_ids=[2, -1]
    )
    diagnostic_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id={2: 0, -1: 1},
        diagnostic_weights_by_model_id=diagnostic_weights,
    )
    diagnostic_collection.get_true_concept_diagnostic_evidence(true_concept_id=-7)
    for case in (None, (-1, -1), (-1, 2), (2, -3)):
        notify_diagnostics_of_training_assignment_change(
            assignment_change=(
                None
                if case is None
                else TrainingModelAssignmentChange(
                    previous_model_id=case[0], current_model_id=case[1]
                )
            ),
            diagnostic_evidence_collection=diagnostic_collection,
        )
    random_states_after = (random.getstate(), numpy.random.get_state(), torch.get_rng_state())
    assert random_states_before[0] == random_states_after[0]
    assert random_states_before[1][0] == random_states_after[1][0]
    assert numpy.array_equal(random_states_before[1][1], random_states_after[1][1])
    assert random_states_before[1][2:] == random_states_after[1][2:]
    assert torch.equal(random_states_before[2], random_states_after[2])
