"""AdaHedgeの証拠の2つのowner（全標本の側、真の概念別の側）が、証拠の部品をそのまま保持し、互いの証拠を持たないこと。"""

import inspect

import pytest

from federated_learning_experiments.evaluation.true_concept_adahedge_diagnostic_evidence_collection import (
    TrueConceptAdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.adahedge_weight_evidence import (
    AdaHedgeWeightEvidence,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.all_sample_adahedge_weight_evidence_holder import (
    AllSampleAdaHedgeWeightEvidenceHolder,
)
from federated_learning_experiments.runtime.alarm_occurrence_handling import handle_alarm_occurrence
from federated_learning_experiments.runtime.held_candidate_validation_progress import (
    advance_held_candidate_validation,
    apply_alarm_response_to_validation_session_holder,
    finalize_held_incomplete_candidate_validation,
)
from federated_learning_experiments.runtime.observed_sample_prediction import (
    predict_observed_sample_and_update_prediction_weights,
)
from federated_learning_experiments.runtime.observed_sample_processing import (
    process_observed_sample,
)
from federated_learning_experiments.runtime.post_aggregation_prediction_recalibration import (
    recalibrate_prediction_state_after_aggregation,
)
from federated_learning_experiments.runtime.training_assignment_all_sample_adahedge_weight_evidence_restart import (
    restart_all_sample_adahedge_weight_evidence_after_training_assignment_change,
)

# 標本ごとの、真の概念ID（無ければNone）と、保有モデルごとの損失。途中で、保有モデルが増える。
OBSERVED_SAMPLES = (
    (3, {0: 0.2, 1: 0.7}),
    (None, {0: 0.1, 1: 0.9}),
    (1, {0: 0.8, 1: 0.3}),
    (3, {0: 0.4, 1: 0.5, 2: 0.6}),
    (1, {0: 0.9, 1: 0.2, 2: 0.1}),
    (7, {0: 0.3, 1: 0.3, 2: 0.3}),
    (3, {0: 0.05, 1: 0.95, 2: 0.5}),
)


class DirectlyHeldEvidenceReference:
    """ownerを介さずに、証拠の部品を直接持つ、期待の側（全標本の証拠1つと、真の概念IDごとの証拠）。"""

    def __init__(self):
        self.all_sample_weight_evidence = AdaHedgeWeightEvidence()
        self._true_concept_diagnostic_evidence_by_id = {}

    def get_true_concept_diagnostic_evidence(self, *, true_concept_id):
        return self._true_concept_diagnostic_evidence_by_id.setdefault(
            true_concept_id, AdaHedgeWeightEvidence()
        )

    @property
    def created_true_concept_ids(self):
        return tuple(self._true_concept_diagnostic_evidence_by_id)


def describe_evidence(weight_evidence):
    return (
        weight_evidence.cumulative_losses_by_model_id,
        weight_evidence.mixability_gap,
        weight_evidence.model_pool_reset_count,
        weight_evidence.concept_operation_restart_count,
        weight_evidence.aggregation_restart_count,
        weight_evidence.aggregation_recalibration_count,
        weight_evidence.aggregation_recalibration_sample_count,
    )


def observe_sample(weight_evidence, observed_losses_by_model_id):
    """予測の処理と同じ順（観測前の重み→観測後の更新）で、1標本ぶん、証拠を進める。"""
    weights_by_model_id = weight_evidence.get_weights_before_loss_observation(
        model_ids=tuple(observed_losses_by_model_id)
    )
    weight_evidence.update_evidence_after_loss_observation(
        weights_by_model_id=weights_by_model_id,
        observed_losses_by_model_id=observed_losses_by_model_id,
    )
    return weights_by_model_id


@pytest.mark.parametrize("evidence_is_updated_by_prediction", [True, False])
def test_two_owners_hold_evidence_that_behaves_like_directly_held_evidence(
    evidence_is_updated_by_prediction,
):
    """同じ操作の列で、2つのownerの証拠の値・作成順が、証拠の部品を直接持った場合と同じ（ownerは、計算を足さない）。"""
    directly_held_reference = DirectlyHeldEvidenceReference()
    all_sample_holder = AllSampleAdaHedgeWeightEvidenceHolder(
        evidence_is_updated_by_prediction=evidence_is_updated_by_prediction
    )
    true_concept_collection = TrueConceptAdaHedgeDiagnosticEvidenceCollection()
    assert all_sample_holder.evidence_is_updated_by_prediction is evidence_is_updated_by_prediction
    for sample_position, (true_concept_id, observed_losses_by_model_id) in enumerate(
        OBSERVED_SAMPLES
    ):
        assert observe_sample(
            all_sample_holder.all_sample_weight_evidence, observed_losses_by_model_id
        ) == observe_sample(
            directly_held_reference.all_sample_weight_evidence, observed_losses_by_model_id
        )
        if true_concept_id is not None:
            assert observe_sample(
                true_concept_collection.get_true_concept_diagnostic_evidence(
                    true_concept_id=true_concept_id
                ),
                observed_losses_by_model_id,
            ) == observe_sample(
                directly_held_reference.get_true_concept_diagnostic_evidence(
                    true_concept_id=true_concept_id
                ),
                observed_losses_by_model_id,
            )
        if sample_position == 4:
            # 学習帰属の変更（全標本の証拠だけ）と、集約後の再較正（全標本は再生、真の概念別は再始動）。
            for weight_evidence in (
                all_sample_holder.all_sample_weight_evidence,
                directly_held_reference.all_sample_weight_evidence,
            ):
                weight_evidence.restart_evidence_after_concept_operation()
                weight_evidence.replay_observed_losses_after_aggregation(
                    observed_loss_sequence=({0: 0.25, 1: 0.5, 2: 0.75},)
                )
            assert (
                true_concept_collection.created_true_concept_ids
                == directly_held_reference.created_true_concept_ids
            )
            for created_true_concept_id in true_concept_collection.created_true_concept_ids:
                for owner in (true_concept_collection, directly_held_reference):
                    owner.get_true_concept_diagnostic_evidence(
                        true_concept_id=created_true_concept_id
                    ).restart_evidence_after_aggregation()
        assert describe_evidence(all_sample_holder.all_sample_weight_evidence) == describe_evidence(
            directly_held_reference.all_sample_weight_evidence
        )
        assert (
            true_concept_collection.created_true_concept_ids
            == directly_held_reference.created_true_concept_ids
        )
        for created_true_concept_id in true_concept_collection.created_true_concept_ids:
            assert describe_evidence(
                true_concept_collection.get_true_concept_diagnostic_evidence(
                    true_concept_id=created_true_concept_id
                )
            ) == describe_evidence(
                directly_held_reference.get_true_concept_diagnostic_evidence(
                    true_concept_id=created_true_concept_id
                )
            )
    # 作成順（最初に観測した順）。全標本の証拠と、真の概念別の証拠は、別の値になっている。
    assert true_concept_collection.created_true_concept_ids == (3, 1, 7)
    assert describe_evidence(all_sample_holder.all_sample_weight_evidence) != describe_evidence(
        true_concept_collection.get_true_concept_diagnostic_evidence(true_concept_id=3)
    )


def test_all_sample_holder_returns_one_live_evidence_and_has_no_true_concept_access():
    all_sample_holder = AllSampleAdaHedgeWeightEvidenceHolder()
    assert all_sample_holder.evidence_is_updated_by_prediction is True
    weight_evidence = all_sample_holder.all_sample_weight_evidence
    assert type(weight_evidence) is AdaHedgeWeightEvidence
    assert all_sample_holder.all_sample_weight_evidence is weight_evidence
    # 別のholderの証拠とは、独立。
    assert AllSampleAdaHedgeWeightEvidenceHolder().all_sample_weight_evidence is not weight_evidence
    # 真の概念別の証拠を得る口・作成済みのIDの口を、持たない。公開の口は、2つだけ。
    assert not hasattr(all_sample_holder, "get_true_concept_diagnostic_evidence")
    assert not hasattr(all_sample_holder, "created_true_concept_ids")
    assert {name for name in dir(all_sample_holder) if not name.startswith("_")} == {
        "all_sample_weight_evidence",
        "evidence_is_updated_by_prediction",
    }


@pytest.mark.parametrize("invalid_value", [1, 0, None, "True"])
def test_all_sample_holder_rejects_non_bool_argument(invalid_value):
    with pytest.raises(TypeError):
        AllSampleAdaHedgeWeightEvidenceHolder(evidence_is_updated_by_prediction=invalid_value)


def test_true_concept_collection_creates_independent_evidence_once_and_has_no_all_sample_evidence():
    true_concept_collection = TrueConceptAdaHedgeDiagnosticEvidenceCollection()
    assert true_concept_collection.created_true_concept_ids == ()
    first_evidence = true_concept_collection.get_true_concept_diagnostic_evidence(true_concept_id=5)
    assert type(first_evidence) is AdaHedgeWeightEvidence
    # 2回目は、同じ証拠を返す（作り直さない）。別のIDは、独立した証拠。
    assert (
        true_concept_collection.get_true_concept_diagnostic_evidence(true_concept_id=5)
        is first_evidence
    )
    second_evidence = true_concept_collection.get_true_concept_diagnostic_evidence(
        true_concept_id=2
    )
    assert second_evidence is not first_evidence
    observe_sample(first_evidence, {0: 0.9, 1: 0.1})
    assert describe_evidence(second_evidence) == describe_evidence(AdaHedgeWeightEvidence())
    assert true_concept_collection.created_true_concept_ids == (5, 2)
    # 作成済みのIDの並びは、snapshot（後の作成で変わらない）。
    created_true_concept_ids = true_concept_collection.created_true_concept_ids
    true_concept_collection.get_true_concept_diagnostic_evidence(true_concept_id=9)
    assert created_true_concept_ids == (5, 2)
    assert true_concept_collection.created_true_concept_ids == (5, 2, 9)
    # 全標本の証拠と、構成の印を、持たない。公開の口は、2つだけ。
    assert not hasattr(true_concept_collection, "all_sample_weight_evidence")
    assert not hasattr(true_concept_collection, "evidence_is_updated_by_prediction")
    assert {name for name in dir(true_concept_collection) if not name.startswith("_")} == {
        "created_true_concept_ids",
        "get_true_concept_diagnostic_evidence",
    }


@pytest.mark.parametrize("invalid_true_concept_id", [True, 1.0, "1", None])
def test_true_concept_collection_rejects_invalid_id_without_creating(invalid_true_concept_id):
    true_concept_collection = TrueConceptAdaHedgeDiagnosticEvidenceCollection()
    with pytest.raises(TypeError):
        true_concept_collection.get_true_concept_diagnostic_evidence(
            true_concept_id=invalid_true_concept_id
        )
    assert true_concept_collection.created_true_concept_ids == ()


@pytest.mark.parametrize(
    ("entry_function", "receives_true_concept_collection"),
    [
        (predict_observed_sample_and_update_prediction_weights, True),
        (recalibrate_prediction_state_after_aggregation, True),
        (process_observed_sample, True),
        (handle_alarm_occurrence, False),
        (advance_held_candidate_validation, False),
        (apply_alarm_response_to_validation_session_holder, False),
        (finalize_held_incomplete_candidate_validation, False),
        (restart_all_sample_adahedge_weight_evidence_after_training_assignment_change, False),
    ],
)
def test_true_concept_collection_is_received_only_by_prediction_and_recalibration_paths(
    entry_function, receives_true_concept_collection
):
    """真の概念別の証拠のownerを受け取るのは、更新・再始動する処理と、その通り道だけ（判断の処理は、受け取らない）。"""
    parameter_names = set(inspect.signature(entry_function).parameters)
    assert (
        "true_concept_adahedge_diagnostic_evidence_collection" in parameter_names
    ) is receives_true_concept_collection
    # 終端回収と、警報の応答の、保持への反映は、学習帰属を変えないので、全標本の側のownerも受け取らない。
    assert ("all_sample_adahedge_weight_evidence_holder" in parameter_names) is (
        entry_function
        not in (
            finalize_held_incomplete_candidate_validation,
            apply_alarm_response_to_validation_session_holder,
        )
    )
