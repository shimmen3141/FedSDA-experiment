"""全標本の証拠と、真の概念別の診断証拠の、2つのownerと、帰属変更の通知を、実旧クライアントへ照合する。"""

import random
from collections import defaultdict

import numpy
import pytest
import torch
from test_adahedge_weight_evidence import (
    assert_adahedge_matches_legacy,
    get_adahedge_evidence_snapshot,
)

from federated_drift_experiment.clients.fedsda import (
    RestartingSoftRoutingClassConditionalESRFedSDAClient,
)
from federated_drift_experiment.expert_routing import AdaHedgeRouter
from federated_learning_experiments.evaluation.true_concept_adahedge_diagnostic_evidence_collection import (
    TrueConceptAdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.all_sample_adahedge_weight_evidence_holder import (
    AllSampleAdaHedgeWeightEvidenceHolder,
)
from federated_learning_experiments.runtime.training_assignment_all_sample_adahedge_weight_evidence_restart import (
    restart_all_sample_adahedge_weight_evidence_after_training_assignment_change,
)


class ClientAdaHedgeEvidenceOwners:
    """1つのclientが持つ、2つのowner（全標本の側、真の概念別の側）を、並べて読むためのtestのhelper。"""

    def __init__(
        self,
        *,
        all_sample_adahedge_weight_evidence_holder=None,
        true_concept_adahedge_diagnostic_evidence_collection=None,
    ):
        self.all_sample_adahedge_weight_evidence_holder = (
            AllSampleAdaHedgeWeightEvidenceHolder()
            if all_sample_adahedge_weight_evidence_holder is None
            else all_sample_adahedge_weight_evidence_holder
        )
        self.true_concept_adahedge_diagnostic_evidence_collection = (
            TrueConceptAdaHedgeDiagnosticEvidenceCollection()
            if true_concept_adahedge_diagnostic_evidence_collection is None
            else true_concept_adahedge_diagnostic_evidence_collection
        )

    @property
    def all_sample_weight_evidence(self):
        return self.all_sample_adahedge_weight_evidence_holder.all_sample_weight_evidence

    @property
    def created_true_concept_ids(self):
        return self.true_concept_adahedge_diagnostic_evidence_collection.created_true_concept_ids

    def get_true_concept_diagnostic_evidence(self, *, true_concept_id):
        return self.true_concept_adahedge_diagnostic_evidence_collection.get_true_concept_diagnostic_evidence(
            true_concept_id=true_concept_id
        )


def collect_client_adahedge_evidence_owners(arguments):
    """引数の辞書（予測・標本1件の処理の引数）から、2つのownerを取り出して、並べる。"""
    return ClientAdaHedgeEvidenceOwners(
        all_sample_adahedge_weight_evidence_holder=arguments[
            "all_sample_adahedge_weight_evidence_holder"
        ],
        true_concept_adahedge_diagnostic_evidence_collection=arguments[
            "true_concept_adahedge_diagnostic_evidence_collection"
        ],
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
        get_adahedge_evidence_snapshot(diagnostic_collection.all_sample_weight_evidence),
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
        diagnostic_collection.all_sample_weight_evidence, legacy_client.expert_router
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
    diagnostic_collection = ClientAdaHedgeEvidenceOwners()
    legacy_client = build_legacy_diagnostic_client()
    assert_diagnostic_collection_matches_legacy(diagnostic_collection, legacy_client)
    assignment_changes = (None, (-1, -1), (-1, 2), (2, -3), (-3, -3), (-3, 2), (2, -3))
    for index, case in enumerate(assignment_changes):
        for concept_id in (None, 7, -4, 7):
            if concept_id is None:
                adahedge_weight_evidence = diagnostic_collection.all_sample_weight_evidence
                legacy_router = legacy_client.expert_router
            else:
                adahedge_weight_evidence = (
                    diagnostic_collection.get_true_concept_diagnostic_evidence(
                        true_concept_id=concept_id
                    )
                )
                legacy_router = legacy_client.oracle_concept_expert_routers[concept_id]
            observed_losses = {2: 0.1, -1: 0.8} if index % 2 else {5: 0.7, -1: 0.2, 2: 0.4}
            diagnostic_weights = adahedge_weight_evidence.get_weights_before_loss_observation(
                model_ids=reversed(observed_losses)
            )
            legacy_weights = legacy_router.probabilities(observed_losses)
            assert diagnostic_weights == legacy_weights
            assert_diagnostic_collection_matches_legacy(diagnostic_collection, legacy_client)
            adahedge_weight_evidence.update_evidence_after_loss_observation(
                observed_losses_by_model_id=observed_losses,
                weights_by_model_id=diagnostic_weights,
            )
            legacy_router.update(observed_losses, legacy_weights)
            assert_diagnostic_collection_matches_legacy(diagnostic_collection, legacy_client)
        before = get_diagnostic_collection_snapshot(diagnostic_collection)
        restart_all_sample_adahedge_weight_evidence_after_training_assignment_change(
            assignment_change=(
                None
                if case is None
                else TrainingModelAssignmentChange(
                    previous_model_id=case[0], current_model_id=case[1]
                )
            ),
            all_sample_adahedge_weight_evidence_holder=diagnostic_collection.all_sample_adahedge_weight_evidence_holder,
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
            adahedge_weight_evidence = (
                diagnostic_collection.all_sample_weight_evidence
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
            assert adahedge_weight_evidence.get_weights_before_loss_observation(
                model_ids=[2, -1]
            ) == legacy_router.probabilities([2, -1])
            assert_diagnostic_collection_matches_legacy(diagnostic_collection, legacy_client)
    # 同じ変更recordの再通知も毎回数える。履歴・重複防止は保持しない。
    before = get_diagnostic_collection_snapshot(diagnostic_collection)
    for index in range(2):
        restart_all_sample_adahedge_weight_evidence_after_training_assignment_change(
            assignment_change=TrainingModelAssignmentChange(
                previous_model_id=2, current_model_id=-3
            ),
            all_sample_adahedge_weight_evidence_holder=diagnostic_collection.all_sample_adahedge_weight_evidence_holder,
        )
        assert diagnostic_collection.all_sample_weight_evidence.concept_operation_restart_count == (
            before[0][3] + index + 1
        )
        assert get_diagnostic_collection_snapshot(diagnostic_collection)[1:] == before[1:]


def test_diagnostic_collection_keeps_distinct_live_owners():
    first_collection = ClientAdaHedgeEvidenceOwners()
    second_collection = ClientAdaHedgeEvidenceOwners()
    assert first_collection.created_true_concept_ids == ()
    assert get_diagnostic_collection_snapshot(first_collection) == (
        ({}, 0.0, 0, 0, 0, 0, 0),
        (),
        (),
    )
    adahedge_weight_evidence = first_collection.all_sample_weight_evidence
    assert adahedge_weight_evidence is first_collection.all_sample_weight_evidence
    assert adahedge_weight_evidence is not second_collection.all_sample_weight_evidence
    concept_evidence = first_collection.get_true_concept_diagnostic_evidence(true_concept_id=-7)
    assert concept_evidence is first_collection.get_true_concept_diagnostic_evidence(
        true_concept_id=-7
    )
    assert concept_evidence is not adahedge_weight_evidence
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
        weights_by_model_id={2: 0.5, -1: 0.5},
    )
    assert first_collection.get_true_concept_diagnostic_evidence(
        true_concept_id=-7
    ).cumulative_losses_by_model_id == {-1: 1.0, 2: 0.0}
    assert get_adahedge_evidence_snapshot(adahedge_weight_evidence) == ({}, 0.0, 0, 0, 0, 0, 0)
    assert get_adahedge_evidence_snapshot(
        first_collection.get_true_concept_diagnostic_evidence(true_concept_id=2)
    ) == ({}, 0.0, 0, 0, 0, 0, 0)
    assert get_diagnostic_collection_snapshot(second_collection) == before
    adahedge_weight_evidence.restart_evidence_after_concept_operation()
    assert first_collection.all_sample_weight_evidence.concept_operation_restart_count == 1


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
    diagnostic_collection = ClientAdaHedgeEvidenceOwners()
    for adahedge_weight_evidence in (
        diagnostic_collection.all_sample_weight_evidence,
        diagnostic_collection.get_true_concept_diagnostic_evidence(true_concept_id=-3),
    ):
        adahedge_weight_evidence.update_evidence_after_loss_observation(
            observed_losses_by_model_id={-1: 0, 2: 1},
            weights_by_model_id={-1: 0.5, 2: 0.5},
        )
        adahedge_weight_evidence.get_weights_before_loss_observation(model_ids=[3, 2])
    notification_arguments = {
        "assignment_change": TrainingModelAssignmentChange(
            previous_model_id=-1, current_model_id=2
        ),
        "all_sample_adahedge_weight_evidence_holder": diagnostic_collection.all_sample_adahedge_weight_evidence_holder,
    }
    if invalid_notification_case == "collection_none":
        notification_arguments["all_sample_adahedge_weight_evidence_holder"] = None
        notification_arguments["assignment_change"] = None
    elif invalid_notification_case == "collection_object":
        notification_arguments["all_sample_adahedge_weight_evidence_holder"] = object()
    elif invalid_notification_case == "collection_subclass":
        notification_arguments["all_sample_adahedge_weight_evidence_holder"] = type(
            "RejectedCollection", (AllSampleAdaHedgeWeightEvidenceHolder,), {}
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
        restart_all_sample_adahedge_weight_evidence_after_training_assignment_change(
            **notification_arguments
        )
    diagnostic_states_after = get_diagnostic_collection_snapshot(diagnostic_collection)
    assert diagnostic_states_after == diagnostic_states_before


@pytest.mark.parametrize(
    "invalid_concept_id",
    [True, False, None, 2.0, numpy.int64(2), "2", type("RejectedConceptId", (int,), {})(2)],
)
def test_diagnostic_collection_rejects_invalid_concept_id_without_creation(invalid_concept_id):
    diagnostic_collection = ClientAdaHedgeEvidenceOwners()
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
    diagnostic_collection = ClientAdaHedgeEvidenceOwners()
    adahedge_weight_evidence = diagnostic_collection.all_sample_weight_evidence
    diagnostic_weights = adahedge_weight_evidence.get_weights_before_loss_observation(
        model_ids=[2, -1]
    )
    adahedge_weight_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id={2: 0, -1: 1},
        weights_by_model_id=diagnostic_weights,
    )
    diagnostic_collection.get_true_concept_diagnostic_evidence(true_concept_id=-7)
    for case in (None, (-1, -1), (-1, 2), (2, -3)):
        restart_all_sample_adahedge_weight_evidence_after_training_assignment_change(
            assignment_change=(
                None
                if case is None
                else TrainingModelAssignmentChange(
                    previous_model_id=case[0], current_model_id=case[1]
                )
            ),
            all_sample_adahedge_weight_evidence_holder=diagnostic_collection.all_sample_adahedge_weight_evidence_holder,
        )
    random_states_after = (random.getstate(), numpy.random.get_state(), torch.get_rng_state())
    assert random_states_before[0] == random_states_after[0]
    assert random_states_before[1][0] == random_states_after[1][0]
    assert numpy.array_equal(random_states_before[1][1], random_states_after[1][1])
    assert random_states_before[1][2:] == random_states_after[1][2:]
    assert torch.equal(random_states_before[2], random_states_after[2])


def test_evidence_not_updated_by_prediction_is_not_restarted_by_assignment_change():
    """予測の処理が更新しない構成の証拠は、学習帰属の変更でも、始め直さない（入力の検査は、同じに行う）。"""
    maintained_collection = AllSampleAdaHedgeWeightEvidenceHolder()
    assert maintained_collection.evidence_is_updated_by_prediction is True
    unused_collection = AllSampleAdaHedgeWeightEvidenceHolder(
        evidence_is_updated_by_prediction=False
    )
    assert unused_collection.evidence_is_updated_by_prediction is False
    assignment_change = TrainingModelAssignmentChange(previous_model_id=1, current_model_id=-1)
    for evidence_collection, expected_restart_count in (
        (maintained_collection, 1),
        (unused_collection, 0),
    ):
        restart_all_sample_adahedge_weight_evidence_after_training_assignment_change(
            assignment_change=assignment_change,
            all_sample_adahedge_weight_evidence_holder=evidence_collection,
        )
        assert (
            evidence_collection.all_sample_weight_evidence.concept_operation_restart_count
            == expected_restart_count
        )
    with pytest.raises(TypeError):
        restart_all_sample_adahedge_weight_evidence_after_training_assignment_change(
            assignment_change="change", all_sample_adahedge_weight_evidence_holder=unused_collection
        )
    for invalid_flag in (None, 1, "True"):
        with pytest.raises(TypeError):
            AllSampleAdaHedgeWeightEvidenceHolder(evidence_is_updated_by_prediction=invalid_flag)
