"""警報区間での保有モデル再利用評価を、純粋判定と実旧の警報処理の両方で検証する。"""

import math
import random
from collections import Counter, deque
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_joint_model_parameter_update import assert_nested_state_equal
from test_post_alarm_candidate_validation_session_start import build_session_start_oracle
from test_post_alarm_reference_model_fixation import (
    set_overall_loss_statistics_in_both_implementations,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment import config
from federated_drift_experiment.clients.fedsda import FedSDAClient
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection import (
    alarm_interval_model_reuse_assessment as assessment_module,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
    AlarmIntervalModelReuseAssessment,
    assess_alarm_interval_model_reuse,
)
from federated_learning_experiments.runtime import (
    alarm_interval_model_reuse_assessment as reuse_evaluation_module,
)
from federated_learning_experiments.runtime.alarm_interval_model_reuse_assessment import (
    evaluate_held_models_for_alarm_interval_reuse,
)


@pytest.mark.parametrize(
    "baseline_supported_interval_mean_losses_by_model_id,reuse_baseline_mean_losses_by_model_id,"
    "maximum_alarm_interval_mean_loss_increase,reusable_mean_losses_by_model_id,"
    "selected_reuse_model_id",
    [
        # 閾値の直前/等値/直後。差は0.5−0.25=0.25。
        (((4, 0.5),), {4: 0.25}, math.nextafter(0.25, 0.0), (), None),
        (((4, 0.5),), {4: 0.25}, 0.25, ((4, 0.5),), 4),
        (((4, 0.5),), {4: 0.25}, math.nextafter(0.25, 1.0), ((4, 0.5),), 4),
        # 区間平均が履歴より低い場合も同じ式。閾値0でも適合する。
        (((4, 0.125),), {4: 0.5}, 0.0, ((4, 0.125),), 4),
        (((4, 0.5),), {4: 0.5}, 0, ((4, 0.5),), 4),
        # 適合なしと空の評価列。
        (((4, 0.75), (9, 1.0)), {4: 0.25, 9: 0.5}, 0.125, (), None),
        ((), {}, 0.25, (), None),
        # 適合列は入力順の部分列で、選択は平均最小。
        (
            ((4, 0.5), (9, 0.25), (2, 0.75)),
            {4: 0.5, 9: 0.25, 2: 0.25},
            0.125,
            ((4, 0.5), (9, 0.25)),
            9,
        ),
        # 同率は先着。IDの大小、負の一時ID、基準dictの順序に依らない。
        (((9, 0.25), (4, 0.25)), {4: 0.25, 9: 0.25}, 0.0, ((9, 0.25), (4, 0.25)), 9),
        (((4, 0.25), (9, 0.25)), {9: 0.25, 4: 0.25}, 0.0, ((4, 0.25), (9, 0.25)), 4),
        (((7, 0.25), (-103, 0.25)), {7: 0.25, -103: 0.25}, 0.0, ((7, 0.25), (-103, 0.25)), 7),
        (((-103, 0.25), (7, 0.25)), {7: 0.25, -103: 0.25}, 0.0, ((-103, 0.25), (7, 0.25)), -103),
        # 平均最小のモデルが不適合なら、適合した中の最小を選ぶ。
        (
            ((4, 0.125), (-103, 0.5), (9, 0.75)),
            {4: 0.0625, -103: 0.5, 9: 0.75},
            0.0,
            ((-103, 0.5), (9, 0.75)),
            -103,
        ),
        # 整数の平均・基準・閾値も数値として扱う。
        (((4, 1), (9, 0)), {4: 1, 9: 1}, 0, ((4, 1), (9, 0)), 9),
    ],
)
def test_alarm_interval_reuse_assessment_selects_ordered_minimum(
    baseline_supported_interval_mean_losses_by_model_id,
    reuse_baseline_mean_losses_by_model_id,
    maximum_alarm_interval_mean_loss_increase,
    reusable_mean_losses_by_model_id,
    selected_reuse_model_id,
):
    alarm_interval_reuse_assessment = assess_alarm_interval_model_reuse(
        baseline_supported_interval_mean_losses_by_model_id=baseline_supported_interval_mean_losses_by_model_id,
        reuse_baseline_mean_losses_by_model_id=reuse_baseline_mean_losses_by_model_id,
        maximum_alarm_interval_mean_loss_increase=maximum_alarm_interval_mean_loss_increase,
    )
    assert type(alarm_interval_reuse_assessment) is AlarmIntervalModelReuseAssessment
    assert (
        alarm_interval_reuse_assessment.baseline_supported_interval_mean_losses_by_model_id
        == baseline_supported_interval_mean_losses_by_model_id
    )
    assert type(alarm_interval_reuse_assessment.reusable_mean_losses_by_model_id) is tuple
    assert (
        alarm_interval_reuse_assessment.reusable_mean_losses_by_model_id
        == reusable_mean_losses_by_model_id
    )
    assert alarm_interval_reuse_assessment.selected_reuse_model_id == selected_reuse_model_id
    # 実旧の式と選択関数（selfを使わない）で同じ結果になる。
    legacy_valid_candidates = [
        (model_id, mean_loss)
        for model_id, mean_loss in baseline_supported_interval_mean_losses_by_model_id
        if mean_loss - reuse_baseline_mean_losses_by_model_id[model_id]
        <= maximum_alarm_interval_mean_loss_increase
    ]
    assert list(alarm_interval_reuse_assessment.reusable_mean_losses_by_model_id) == (
        legacy_valid_candidates
    )
    if legacy_valid_candidates:
        assert (
            alarm_interval_reuse_assessment.selected_reuse_model_id
            == FedSDAClient._select_reuse_candidate(None, legacy_valid_candidates)[0]
        )


def test_alarm_interval_reuse_assessment_is_immutable():
    alarm_interval_reuse_assessment = assess_alarm_interval_model_reuse(
        baseline_supported_interval_mean_losses_by_model_id=((4, 0.5), (9, 0.25)),
        reuse_baseline_mean_losses_by_model_id={4: 0.5, 9: 0.25},
        maximum_alarm_interval_mean_loss_increase=0.0,
    )
    assert tuple(field.name for field in fields(AlarmIntervalModelReuseAssessment)) == (
        "baseline_supported_interval_mean_losses_by_model_id",
        "reusable_mean_losses_by_model_id",
    )
    assert all(field.kw_only for field in fields(AlarmIntervalModelReuseAssessment))
    for field_name, field_value in (
        ("baseline_supported_interval_mean_losses_by_model_id", ()),
        ("reusable_mean_losses_by_model_id", ()),
        ("selected_reuse_model_id", 4),
    ):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(alarm_interval_reuse_assessment, field_name, field_value)
    assert alarm_interval_reuse_assessment.selected_reuse_model_id == 9
    with pytest.raises(TypeError):
        AlarmIntervalModelReuseAssessment(((4, 0.5),), ((4, 0.5),))
    with pytest.raises(TypeError):
        AlarmIntervalModelReuseAssessment(
            baseline_supported_interval_mean_losses_by_model_id=((4, 0.5),)
        )
    with pytest.raises(TypeError):
        assess_alarm_interval_model_reuse(((4, 0.5),), {4: 0.5}, 0.0)
    assert not hasattr(assessment_module, "__all__")


class TupleSubclass(tuple):
    pass


class DictSubclass(dict):
    pass


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


@pytest.mark.parametrize(
    "field_name,field_value,expected_exception",
    [
        # 評価列の構造。
        ("baseline_supported_interval_mean_losses_by_model_id", [(4, 0.5), (9, 0.25)], TypeError),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            TupleSubclass(((4, 0.5), (9, 0.25))),
            TypeError,
        ),
        ("baseline_supported_interval_mean_losses_by_model_id", {4: 0.5, 9: 0.25}, TypeError),
        ("baseline_supported_interval_mean_losses_by_model_id", ([4, 0.5], (9, 0.25)), TypeError),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            (TupleSubclass((4, 0.5)), (9, 0.25)),
            TypeError,
        ),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            ((4, 0.5, 0.5), (9, 0.25)),
            ValueError,
        ),
        ("baseline_supported_interval_mean_losses_by_model_id", ((4,), (9, 0.25)), ValueError),
        # ID。
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            ((True, 0.5), (9, 0.25)),
            TypeError,
        ),
        ("baseline_supported_interval_mean_losses_by_model_id", ((4.0, 0.5), (9, 0.25)), TypeError),
        ("baseline_supported_interval_mean_losses_by_model_id", (("4", 0.5), (9, 0.25)), TypeError),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            ((IntSubclass(4), 0.5), (9, 0.25)),
            TypeError,
        ),
        ("baseline_supported_interval_mean_losses_by_model_id", ((4, 0.5), (4, 0.25)), ValueError),
        # 評価列と基準のkey不一致。
        ("baseline_supported_interval_mean_losses_by_model_id", ((4, 0.5),), ValueError),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            ((4, 0.5), (9, 0.25), (2, 0.5)),
            ValueError,
        ),
        ("baseline_supported_interval_mean_losses_by_model_id", ((4, 0.5), (8, 0.25)), ValueError),
        ("baseline_supported_interval_mean_losses_by_model_id", (), ValueError),
        # 区間平均。
        ("baseline_supported_interval_mean_losses_by_model_id", ((4, True), (9, 0.25)), TypeError),
        ("baseline_supported_interval_mean_losses_by_model_id", ((4, "0.5"), (9, 0.25)), TypeError),
        ("baseline_supported_interval_mean_losses_by_model_id", ((4, None), (9, 0.25)), TypeError),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            ((4, FloatSubclass(0.5)), (9, 0.25)),
            TypeError,
        ),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            ((4, math.nan), (9, 0.25)),
            ValueError,
        ),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            ((4, math.inf), (9, 0.25)),
            ValueError,
        ),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            ((4, -0.125), (9, 0.25)),
            ValueError,
        ),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            ((4, math.nextafter(1.0, 2.0)), (9, 0.25)),
            ValueError,
        ),
        ("baseline_supported_interval_mean_losses_by_model_id", ((4, 2), (9, 0.25)), ValueError),
        (
            "baseline_supported_interval_mean_losses_by_model_id",
            ((4, 10**400), (9, 0.25)),
            ValueError,
        ),
        # 履歴基準。
        ("reuse_baseline_mean_losses_by_model_id", ((4, 0.5), (9, 0.25)), TypeError),
        ("reuse_baseline_mean_losses_by_model_id", DictSubclass({4: 0.5, 9: 0.25}), TypeError),
        ("reuse_baseline_mean_losses_by_model_id", None, TypeError),
        ("reuse_baseline_mean_losses_by_model_id", {4: 0.5, 9.0: 0.25}, TypeError),
        ("reuse_baseline_mean_losses_by_model_id", {4: 0.5, "9": 0.25}, TypeError),
        ("reuse_baseline_mean_losses_by_model_id", {4: 0.5, IntSubclass(9): 0.25}, TypeError),
        ("reuse_baseline_mean_losses_by_model_id", {4: 0.5}, ValueError),
        ("reuse_baseline_mean_losses_by_model_id", {4: 0.5, 9: 0.25, 2: 0.5}, ValueError),
        ("reuse_baseline_mean_losses_by_model_id", {4: 0.5, 8: 0.25}, ValueError),
        ("reuse_baseline_mean_losses_by_model_id", {}, ValueError),
        ("reuse_baseline_mean_losses_by_model_id", {4: True, 9: 0.25}, TypeError),
        ("reuse_baseline_mean_losses_by_model_id", {4: "0.5", 9: 0.25}, TypeError),
        ("reuse_baseline_mean_losses_by_model_id", {4: None, 9: 0.25}, TypeError),
        ("reuse_baseline_mean_losses_by_model_id", {4: FloatSubclass(0.5), 9: 0.25}, TypeError),
        ("reuse_baseline_mean_losses_by_model_id", {4: math.nan, 9: 0.25}, ValueError),
        ("reuse_baseline_mean_losses_by_model_id", {4: math.inf, 9: 0.25}, ValueError),
        ("reuse_baseline_mean_losses_by_model_id", {4: -0.125, 9: 0.25}, ValueError),
        ("reuse_baseline_mean_losses_by_model_id", {4: 1.5, 9: 0.25}, ValueError),
        ("reuse_baseline_mean_losses_by_model_id", {4: 0.0, 9: 0.25}, ValueError),
        ("reuse_baseline_mean_losses_by_model_id", {4: -0.0, 9: 0.25}, ValueError),
        ("reuse_baseline_mean_losses_by_model_id", {4: 0, 9: 0.25}, ValueError),
        # 許容損失増加量。
        ("maximum_alarm_interval_mean_loss_increase", True, TypeError),
        ("maximum_alarm_interval_mean_loss_increase", "0.125", TypeError),
        ("maximum_alarm_interval_mean_loss_increase", None, TypeError),
        ("maximum_alarm_interval_mean_loss_increase", FloatSubclass(0.125), TypeError),
        ("maximum_alarm_interval_mean_loss_increase", IntSubclass(1), TypeError),
        ("maximum_alarm_interval_mean_loss_increase", math.nan, ValueError),
        ("maximum_alarm_interval_mean_loss_increase", math.inf, ValueError),
        ("maximum_alarm_interval_mean_loss_increase", -math.inf, ValueError),
        ("maximum_alarm_interval_mean_loss_increase", -0.125, ValueError),
        ("maximum_alarm_interval_mean_loss_increase", -1, ValueError),
        ("maximum_alarm_interval_mean_loss_increase", 10**400, ValueError),
    ],
)
def test_alarm_interval_reuse_assessment_rejects_invalid_inputs(
    field_name, field_value, expected_exception
):
    valid_assessment_arguments = dict(
        baseline_supported_interval_mean_losses_by_model_id=((4, 0.5), (9, 0.25)),
        reuse_baseline_mean_losses_by_model_id={4: 0.5, 9: 0.25},
        maximum_alarm_interval_mean_loss_increase=0.125,
    )
    # 正常入力は受理される（拒否がfixtureの不備でないこと）。
    assert (
        assess_alarm_interval_model_reuse(**valid_assessment_arguments).selected_reuse_model_id == 9
    )
    with pytest.raises((TypeError, ValueError)) as exception_info:
        assess_alarm_interval_model_reuse(**{**valid_assessment_arguments, field_name: field_value})
    assert type(exception_info.value) is expected_exception


# 保有順のモデルごとの役割。履歴の与え方と、許容増加量に対する適合/不適合を表す。
HISTORY_AND_FIT_CASES = {
    "single_model_fits": ((4,), ("fits_at_equal_mean",)),
    "single_model_exceeds": ((4,), ("exceeds_history",)),
    "single_model_without_statistics": ((4,), ("unregistered_statistics",)),
    "only_last_model_has_usable_history": (
        (4, 9, -3, 7, 2),
        (
            "no_observation",
            "single_observation",
            "zero_mean",
            "unregistered_statistics",
            "fits_below_history",
        ),
    ),
    "all_fit": ((9, -3, 4), ("fits_at_equal_mean",) * 3),
    "none_fit": ((9, -3, 4), ("exceeds_history",) * 3),
    "only_negative_id_fits": (
        (4, -3, 9),
        ("exceeds_history", "fits_below_history", "exceeds_history"),
    ),
    "mixed": (
        (9, -3, 4, 7),
        ("zero_mean", "exceeds_history", "fits_at_equal_mean", "fits_below_history"),
    ),
}


def build_alarm_interval_reuse_oracle(*, class_count, held_model_ids, monkeypatch):
    """同じ実NN・統計・区間を持つ新owner群と実旧clientを準備する。"""
    session_start_arguments, shared_optimizer_owners, legacy_client = build_session_start_oracle(
        class_count=class_count,
        optimizer_variant="standard",
        monkeypatch=monkeypatch,
        held_model_ids=held_model_ids,
    )
    reuse_evaluation_arguments = dict(
        input_features=session_start_arguments["input_features"],
        observed_class_labels=session_start_arguments["observed_class_labels"],
        held_model_training_state_registry=session_start_arguments[
            "held_model_training_state_registry"
        ],
        loss_statistics_store=session_start_arguments["loss_statistics_store"],
        maximum_alarm_interval_mean_loss_increase=0.0,
    )
    with torch.no_grad():
        legacy_interval_mean_losses_by_model_id = {
            model_id: float(
                torch.mean(
                    legacy_model.per_sample_error(
                        reuse_evaluation_arguments["input_features"],
                        reuse_evaluation_arguments["observed_class_labels"],
                    )
                ).item()
            )
            for model_id, legacy_model in legacy_client.models.items()
        }
    assert all(0 < mean_loss < 1 for mean_loss in legacy_interval_mean_losses_by_model_id.values())
    return (
        reuse_evaluation_arguments,
        session_start_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_interval_mean_losses_by_model_id,
    )


def resolve_alarm_interval_in_legacy_client(
    *,
    legacy_client,
    reuse_evaluation_arguments,
    monkeypatch,
    start_candidate_validation_session=False,
):
    """実旧_resolve_driftを実行し、評価済み候補列・適合列・選択IDを観測する。

    評価より後の副作用（吸収・イベント記録・検出器reset・帰属切替）は差し替える。
    """
    input_features = reuse_evaluation_arguments["input_features"]
    observed_class_labels = reuse_evaluation_arguments["observed_class_labels"]
    monkeypatch.setattr(config, "MIN_DRIFT_DATA", 1)
    legacy_client.verbose = False
    legacy_client._forward_validation = None
    legacy_client.reuse_selection_counts = Counter()
    legacy_client._estimated_new_concept_span = lambda sample_index: len(input_features)
    legacy_client._absorb_into_store = lambda model_id, data_list: None
    legacy_client._record_adaptation_event = lambda **event_fields: None
    legacy_client._reset_drift_detectors = lambda: None
    legacy_client._set_local_current_model = lambda model_id: None
    if not start_candidate_validation_session:
        legacy_client._begin_forward_validation = lambda *session_arguments: None
    legacy_initialization_parameters = []
    legacy_evaluated_candidates = []
    legacy_valid_candidates = []
    legacy_selected_reuse_model_id = None

    def select_initialization_parameters(evaluated_candidates):
        legacy_evaluated_candidates[:] = evaluated_candidates
        legacy_initialization_parameters.append(
            FedSDAClient._select_initialization_params(legacy_client, evaluated_candidates)
        )
        return legacy_initialization_parameters[-1]

    def select_reuse_candidate(valid_candidates):
        legacy_valid_candidates[:] = valid_candidates
        return FedSDAClient._select_reuse_candidate(legacy_client, valid_candidates)

    legacy_client._select_initialization_params = select_initialization_parameters
    legacy_client._select_reuse_candidate = select_reuse_candidate
    distance_thresholds = [reuse_evaluation_arguments["maximum_alarm_interval_mean_loss_increase"]]
    if not start_candidate_validation_session:
        # 適合があると旧は評価済み候補列を外へ渡さない。どのモデルも適合しない閾値で先に観測する。
        distance_thresholds.insert(0, -math.inf)
    for distance_threshold in distance_thresholds:
        legacy_valid_candidates.clear()
        legacy_client.buffer = deque(
            (
                input_features[sample_index : sample_index + 1],
                observed_class_labels[sample_index : sample_index + 1],
                0,
            )
            for sample_index in range(len(input_features))
        )
        legacy_client.distance_threshold = distance_threshold
        forward_call_count = legacy_client.compute_counters["detection_forward_calls"]
        legacy_client._resolve_drift(40, 35, 7)
        # 履歴基準の有無にかかわらず、旧は全保有モデルをforwardする。
        assert legacy_client.compute_counters[
            "detection_forward_calls"
        ] - forward_call_count == len(legacy_client.models)
    if legacy_valid_candidates:
        legacy_selected_reuse_model_id = FedSDAClient._select_reuse_candidate(
            legacy_client, legacy_valid_candidates
        )[0]
    return (
        legacy_evaluated_candidates,
        legacy_valid_candidates,
        legacy_selected_reuse_model_id,
        legacy_initialization_parameters,
    )


def snapshot_reuse_evaluation_state(*, reuse_evaluation_arguments, shared_optimizer_owners):
    registry = reuse_evaluation_arguments["held_model_training_state_registry"]
    held_model_training_states = (
        registry.snapshot_ordered_held_model_training_states()
        if type(registry) is HeldModelTrainingStateRegistry
        else ()
    )
    loss_statistics_store = reuse_evaluation_arguments["loss_statistics_store"]
    return dict(
        registry=registry,
        held_model_training_states=held_model_training_states,
        parameter_snapshot=snapshot_parameter_values_and_gradients(
            tuple(
                parameter
                for state in held_model_training_states
                for parameter in state.classifier.parameters()
            )
            + tuple(
                tensor
                for tensor in (
                    reuse_evaluation_arguments["input_features"],
                    reuse_evaluation_arguments["observed_class_labels"],
                )
                # NaNは自身と等しくならないので、値の不変比較から外す。
                if isinstance(tensor, torch.Tensor) and not tensor.isnan().any()
            )
        ),
        training_modes=tuple(state.classifier.training for state in held_model_training_states),
        optimizer_state_snapshots=tuple(
            (owner, owner.parameter_optimizer, deepcopy(owner.parameter_optimizer.state_dict()))
            for owner in tuple(
                state.concept_specific_parameter_optimizer_state
                for state in held_model_training_states
            )
            + tuple(shared_optimizer_owners)
        ),
        loss_statistics_store=loss_statistics_store,
        loss_statistics_snapshot=(
            loss_statistics_store.get_state_snapshot()
            if type(loss_statistics_store) is ModelAndClassLossStatisticsStore
            else None
        ),
        random_states=(torch.get_rng_state().clone(), random.getstate(), np.random.get_state()),
    )


def assert_reuse_evaluation_state_unchanged(state_snapshot):
    registry = state_snapshot["registry"]
    if type(registry) is HeldModelTrainingStateRegistry:
        held_model_training_states = registry.snapshot_ordered_held_model_training_states()
        assert len(held_model_training_states) == len(state_snapshot["held_model_training_states"])
        for state, previous_state in zip(
            held_model_training_states, state_snapshot["held_model_training_states"]
        ):
            assert state is previous_state
            assert state.classifier is previous_state.classifier
    assert_parameter_values_and_gradients_unchanged(state_snapshot["parameter_snapshot"])
    assert (
        tuple(state.classifier.training for state in state_snapshot["held_model_training_states"])
        == state_snapshot["training_modes"]
    )
    for owner, previous_optimizer, previous_optimizer_state in state_snapshot[
        "optimizer_state_snapshots"
    ]:
        assert owner.parameter_optimizer is previous_optimizer
        assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
    if state_snapshot["loss_statistics_snapshot"] is not None:
        assert (
            state_snapshot["loss_statistics_store"].get_state_snapshot()
            == state_snapshot["loss_statistics_snapshot"]
        )
    torch_random_state, python_random_state, numpy_random_state = state_snapshot["random_states"]
    assert torch.equal(torch.get_rng_state(), torch_random_state)
    assert random.getstate() == python_random_state
    numpy_state = np.random.get_state()
    assert numpy_state[0] == numpy_random_state[0]
    assert np.array_equal(numpy_state[1], numpy_random_state[1])
    assert numpy_state[2:] == numpy_random_state[2:]


def assert_alarm_interval_reuse_assessment_matches_legacy(
    *,
    reuse_evaluation_arguments,
    shared_optimizer_owners,
    legacy_client,
    monkeypatch,
):
    """新runtimeの評価列・適合列・選択を実旧の警報処理と照合し、新側の全状態の不変を確かめる。"""
    (
        legacy_evaluated_candidates,
        legacy_valid_candidates,
        legacy_selected_reuse_model_id,
        _,
    ) = resolve_alarm_interval_in_legacy_client(
        legacy_client=legacy_client,
        reuse_evaluation_arguments=reuse_evaluation_arguments,
        monkeypatch=monkeypatch,
    )
    state_snapshot = snapshot_reuse_evaluation_state(
        reuse_evaluation_arguments=reuse_evaluation_arguments,
        shared_optimizer_owners=shared_optimizer_owners,
    )
    with patch.object(
        reuse_evaluation_module,
        "evaluate_classifier_per_sample_bounded_losses",
        wraps=reuse_evaluation_module.evaluate_classifier_per_sample_bounded_losses,
    ) as loss_evaluation_calls:
        alarm_interval_reuse_assessment = evaluate_held_models_for_alarm_interval_reuse(
            **reuse_evaluation_arguments
        )
    assert_reuse_evaluation_state_unchanged(state_snapshot)
    # 基準を使えないモデルも含め、保有順に1回ずつ区間全体をforwardする。
    held_model_training_states = reuse_evaluation_arguments[
        "held_model_training_state_registry"
    ].snapshot_ordered_held_model_training_states()
    assert len(loss_evaluation_calls.call_args_list) == len(held_model_training_states)
    for loss_evaluation_call, state in zip(
        loss_evaluation_calls.call_args_list, held_model_training_states
    ):
        assert loss_evaluation_call.args == ()
        assert loss_evaluation_call.kwargs["classifier"] is state.classifier
        assert (
            loss_evaluation_call.kwargs["input_features"]
            is reuse_evaluation_arguments["input_features"]
        )
        assert (
            loss_evaluation_call.kwargs["observed_class_labels"]
            is reuse_evaluation_arguments["observed_class_labels"]
        )
    assert type(alarm_interval_reuse_assessment) is AlarmIntervalModelReuseAssessment
    assert (
        list(alarm_interval_reuse_assessment.baseline_supported_interval_mean_losses_by_model_id)
        == legacy_evaluated_candidates
    )
    assert (
        list(alarm_interval_reuse_assessment.reusable_mean_losses_by_model_id)
        == legacy_valid_candidates
    )
    for (
        evaluated_model_loss
    ) in alarm_interval_reuse_assessment.baseline_supported_interval_mean_losses_by_model_id:
        assert type(evaluated_model_loss) is tuple
        assert type(evaluated_model_loss[0]) is int
        assert type(evaluated_model_loss[1]) is float
    assert alarm_interval_reuse_assessment.selected_reuse_model_id == legacy_selected_reuse_model_id
    return alarm_interval_reuse_assessment


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("history_and_fit_case", list(HISTORY_AND_FIT_CASES))
def test_alarm_interval_reuse_evaluation_matches_legacy(
    class_count, history_and_fit_case, monkeypatch
):
    held_model_ids, history_and_fit_roles = HISTORY_AND_FIT_CASES[history_and_fit_case]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(831)
        (
            reuse_evaluation_arguments,
            _,
            shared_optimizer_owners,
            legacy_client,
            legacy_interval_mean_losses_by_model_id,
        ) = build_alarm_interval_reuse_oracle(
            class_count=class_count, held_model_ids=held_model_ids, monkeypatch=monkeypatch
        )
        # 不適合の差（平均の3/4）より小さく、0より大きい許容増加量。
        reuse_evaluation_arguments["maximum_alarm_interval_mean_loss_increase"] = (
            min(legacy_interval_mean_losses_by_model_id.values()) / 8
        )
        set_overall_loss_statistics_in_both_implementations(
            loss_statistics_store=reuse_evaluation_arguments["loss_statistics_store"],
            legacy_client=legacy_client,
            statistics_by_model_id={
                model_id: {
                    "unregistered_statistics": None,
                    "no_observation": (0, 0.0),
                    "single_observation": (1, legacy_interval_mean_losses_by_model_id[model_id]),
                    "zero_mean": (3, 0.0),
                    "fits_at_equal_mean": (2, legacy_interval_mean_losses_by_model_id[model_id]),
                    "fits_below_history": (
                        3,
                        (legacy_interval_mean_losses_by_model_id[model_id] + 1) / 2,
                    ),
                    "exceeds_history": (3, legacy_interval_mean_losses_by_model_id[model_id] / 4),
                }[history_and_fit_role]
                for model_id, history_and_fit_role in zip(held_model_ids, history_and_fit_roles)
            },
        )
        alarm_interval_reuse_assessment = assert_alarm_interval_reuse_assessment_matches_legacy(
            reuse_evaluation_arguments=reuse_evaluation_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        # 旧との一致に加えて、役割から決まる期待値を直接確かめる。
        assert (
            alarm_interval_reuse_assessment.baseline_supported_interval_mean_losses_by_model_id
            == tuple(
                (model_id, legacy_interval_mean_losses_by_model_id[model_id])
                for model_id, history_and_fit_role in zip(held_model_ids, history_and_fit_roles)
                if history_and_fit_role
                in ("fits_at_equal_mean", "fits_below_history", "exceeds_history")
            )
        )
        assert alarm_interval_reuse_assessment.reusable_mean_losses_by_model_id == tuple(
            (model_id, legacy_interval_mean_losses_by_model_id[model_id])
            for model_id, history_and_fit_role in zip(held_model_ids, history_and_fit_roles)
            if history_and_fit_role in ("fits_at_equal_mean", "fits_below_history")
        )
        if history_and_fit_case == "only_negative_id_fits":
            assert alarm_interval_reuse_assessment.selected_reuse_model_id == -3
            assert legacy_client.current_model_id != -3
        if history_and_fit_case == "none_fit":
            assert alarm_interval_reuse_assessment.selected_reuse_model_id is None


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("threshold_boundary_case", ["just_below", "equal", "just_above"])
def test_alarm_interval_reuse_evaluation_threshold_boundary_matches_legacy(
    class_count, threshold_boundary_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(832)
        (
            reuse_evaluation_arguments,
            _,
            shared_optimizer_owners,
            legacy_client,
            legacy_interval_mean_losses_by_model_id,
        ) = build_alarm_interval_reuse_oracle(
            class_count=class_count, held_model_ids=(4, 9), monkeypatch=monkeypatch
        )
        reuse_baseline_mean_loss = legacy_interval_mean_losses_by_model_id[4] / 2
        # 旧と同じPython floatの差。
        mean_loss_increase = legacy_interval_mean_losses_by_model_id[4] - reuse_baseline_mean_loss
        reuse_evaluation_arguments["maximum_alarm_interval_mean_loss_increase"] = {
            "just_below": math.nextafter(mean_loss_increase, 0.0),
            "equal": mean_loss_increase,
            "just_above": math.nextafter(mean_loss_increase, 1.0),
        }[threshold_boundary_case]
        set_overall_loss_statistics_in_both_implementations(
            loss_statistics_store=reuse_evaluation_arguments["loss_statistics_store"],
            legacy_client=legacy_client,
            # 現行でない先頭のモデルだけが履歴基準を持つ。
            statistics_by_model_id={4: (2, reuse_baseline_mean_loss), 9: (3, 0.0)},
        )
        alarm_interval_reuse_assessment = assert_alarm_interval_reuse_assessment_matches_legacy(
            reuse_evaluation_arguments=reuse_evaluation_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        assert (
            alarm_interval_reuse_assessment.baseline_supported_interval_mean_losses_by_model_id
            == ((4, legacy_interval_mean_losses_by_model_id[4]),)
        )
        assert alarm_interval_reuse_assessment.selected_reuse_model_id == (
            None if threshold_boundary_case == "just_below" else 4
        )
        assert legacy_client.current_model_id != 4


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("held_model_ids", [(9, -3, 4), (4, 9, -3), (-3, 4, 9)])
def test_alarm_interval_reuse_evaluation_tie_follows_held_order(
    class_count, held_model_ids, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(833)
        (
            reuse_evaluation_arguments,
            _,
            shared_optimizer_owners,
            legacy_client,
            _,
        ) = build_alarm_interval_reuse_oracle(
            class_count=class_count, held_model_ids=held_model_ids, monkeypatch=monkeypatch
        )
        registry = reuse_evaluation_arguments["held_model_training_state_registry"]
        held_model_training_states = registry.snapshot_ordered_held_model_training_states()
        # 後ろ2件を同じ値の実モデルにし、先頭は不適合にする。現行は最後のモデル。
        held_model_training_states[2].classifier.load_state_dict(
            held_model_training_states[1].classifier.state_dict()
        )
        legacy_client.models[held_model_ids[2]].load_state_dict(
            legacy_client.models[held_model_ids[1]].state_dict()
        )
        assert legacy_client.current_model_id == held_model_ids[2]
        with torch.no_grad():
            legacy_interval_mean_losses_by_model_id = {
                model_id: float(
                    torch.mean(
                        legacy_model.per_sample_error(
                            reuse_evaluation_arguments["input_features"],
                            reuse_evaluation_arguments["observed_class_labels"],
                        )
                    ).item()
                )
                for model_id, legacy_model in legacy_client.models.items()
            }
        assert (
            legacy_interval_mean_losses_by_model_id[held_model_ids[1]]
            == legacy_interval_mean_losses_by_model_id[held_model_ids[2]]
        )
        set_overall_loss_statistics_in_both_implementations(
            loss_statistics_store=reuse_evaluation_arguments["loss_statistics_store"],
            legacy_client=legacy_client,
            statistics_by_model_id={
                held_model_ids[0]: (
                    3,
                    legacy_interval_mean_losses_by_model_id[held_model_ids[0]] / 4,
                ),
                held_model_ids[1]: (3, legacy_interval_mean_losses_by_model_id[held_model_ids[1]]),
                held_model_ids[2]: (3, legacy_interval_mean_losses_by_model_id[held_model_ids[2]]),
            },
        )
        alarm_interval_reuse_assessment = assert_alarm_interval_reuse_assessment_matches_legacy(
            reuse_evaluation_arguments=reuse_evaluation_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        assert tuple(
            model_id
            for model_id, _ in alarm_interval_reuse_assessment.reusable_mean_losses_by_model_id
        ) == (held_model_ids[1], held_model_ids[2])
        # 現行（最後）でもIDの最小でもなく、保有順で先のモデルを選ぶ。
        assert alarm_interval_reuse_assessment.selected_reuse_model_id == held_model_ids[1]


@pytest.mark.parametrize(
    "invalid_case,field_value,expected_exception",
    [
        ("held_model_training_state_registry", None, TypeError),
        ("held_model_training_state_registry", {}, TypeError),
        ("subclass:held_model_training_state_registry", None, TypeError),
        ("empty:held_model_training_state_registry", None, ValueError),
        ("loss_statistics_store", None, TypeError),
        ("loss_statistics_store", {}, TypeError),
        ("subclass:loss_statistics_store", None, TypeError),
        ("maximum_alarm_interval_mean_loss_increase", True, TypeError),
        ("maximum_alarm_interval_mean_loss_increase", "0.125", TypeError),
        ("maximum_alarm_interval_mean_loss_increase", None, TypeError),
        ("maximum_alarm_interval_mean_loss_increase", torch.tensor(0.125), TypeError),
        ("maximum_alarm_interval_mean_loss_increase", np.float64(0.125), TypeError),
        ("maximum_alarm_interval_mean_loss_increase", math.nan, ValueError),
        ("maximum_alarm_interval_mean_loss_increase", math.inf, ValueError),
        ("maximum_alarm_interval_mean_loss_increase", -0.125, ValueError),
        ("maximum_alarm_interval_mean_loss_increase", -1, ValueError),
        ("maximum_alarm_interval_mean_loss_increase", 10**400, ValueError),
        ("input_features", None, TypeError),
        ("input_features", [[0.0, 1.0]] * 11, TypeError),
        ("input_features", torch.zeros(11, 2, dtype=torch.float64), ValueError),
        ("input_features", torch.zeros(11, 3), ValueError),
        ("input_features", torch.zeros(0, 2), ValueError),
        ("input_features", torch.zeros(11), ValueError),
        ("input_features", torch.full((11, 2), math.nan), ValueError),
        ("input_features", torch.full((11, 2), math.inf), ValueError),
        ("observed_class_labels", None, TypeError),
        ("observed_class_labels", [[0.0]] * 11, TypeError),
        ("observed_class_labels", torch.zeros(11, 1, dtype=torch.int64), ValueError),
        ("observed_class_labels", torch.zeros(10, 1), ValueError),
        ("observed_class_labels", torch.zeros(11), ValueError),
        ("observed_class_labels", torch.full((11, 1), 4.0), ValueError),
        ("observed_class_labels", torch.full((11, 1), -1.0), ValueError),
        ("observed_class_labels", torch.full((11, 1), 0.5), ValueError),
        ("observed_class_labels", torch.full((11, 1), math.nan), ValueError),
        ("later_model_feature_count_mismatch", None, ValueError),
    ],
)
def test_alarm_interval_reuse_evaluation_rejects_without_mutation(
    invalid_case, field_value, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(834)
        (
            reuse_evaluation_arguments,
            _,
            shared_optimizer_owners,
            _,
            _,
        ) = build_alarm_interval_reuse_oracle(
            class_count=4, held_model_ids=(4, 9), monkeypatch=monkeypatch
        )
        # 正常入力は受理される（拒否がfixtureの不備でないこと）。
        assert (
            type(evaluate_held_models_for_alarm_interval_reuse(**reuse_evaluation_arguments))
            is AlarmIntervalModelReuseAssessment
        )
        valid_state_snapshot = snapshot_reuse_evaluation_state(
            reuse_evaluation_arguments=reuse_evaluation_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        if invalid_case in reuse_evaluation_arguments:
            reuse_evaluation_arguments[invalid_case] = field_value
        elif invalid_case.startswith("subclass:"):
            field_name = invalid_case.removeprefix("subclass:")
            field_value = type("InputSubclass", (type(reuse_evaluation_arguments[field_name]),), {})
            field_value = object.__new__(field_value)
            field_value.__dict__.update(reuse_evaluation_arguments[field_name].__dict__)
            reuse_evaluation_arguments[field_name] = field_value
        elif invalid_case == "empty:held_model_training_state_registry":
            reuse_evaluation_arguments["held_model_training_state_registry"] = (
                HeldModelTrainingStateRegistry()
            )
        else:
            assert invalid_case == "later_model_feature_count_mismatch"
            mismatched_feature_count_classifier = ResidualAdapterClassifier(
                model_architecture_settings=ModelArchitectureSettings(
                    model_architecture_name="shared_backbone_residual_adapter",
                    residual_adapter_requested_rank=2,
                ),
                input_feature_count=3,
                hidden_layer_widths=(5, 4),
                class_count=4,
            )
            reuse_evaluation_arguments[
                "held_model_training_state_registry"
            ].register_held_model_training_state(
                model_id=11,
                classifier=mismatched_feature_count_classifier,
                concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                    parameters=tuple(
                        mismatched_feature_count_classifier.residual_adapter.parameters()
                    )
                    + tuple(mismatched_feature_count_classifier.classification_layer.parameters()),
                    optimizer_settings=SgdParameterOptimizerSettings(learning_rate=0.01),
                ),
            )
        state_snapshot = snapshot_reuse_evaluation_state(
            reuse_evaluation_arguments=reuse_evaluation_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        with patch.object(
            reuse_evaluation_module,
            "evaluate_classifier_per_sample_bounded_losses",
            wraps=reuse_evaluation_module.evaluate_classifier_per_sample_bounded_losses,
        ) as loss_evaluation_calls:
            with pytest.raises((TypeError, ValueError)) as exception_info:
                evaluate_held_models_for_alarm_interval_reuse(**reuse_evaluation_arguments)
        assert type(exception_info.value) is expected_exception
        assert_reuse_evaluation_state_unchanged(state_snapshot)
        if invalid_case in (
            "held_model_training_state_registry",
            "subclass:held_model_training_state_registry",
            "empty:held_model_training_state_registry",
            "loss_statistics_store",
            "subclass:loss_statistics_store",
            "maximum_alarm_interval_mean_loss_increase",
        ):
            # 状態所有者と閾値の不正は、分類器の評価より前に拒否する。
            assert loss_evaluation_calls.call_count == 0
        if invalid_case == "later_model_feature_count_mismatch":
            # 先行する2モデルのforward後に、3件目の形状検査で拒否される。
            assert loss_evaluation_calls.call_count == 3
        # 差し替えていない元のownerも変わらない。
        assert_parameter_values_and_gradients_unchanged(valid_state_snapshot["parameter_snapshot"])
        assert (
            valid_state_snapshot["loss_statistics_store"].get_state_snapshot()
            == valid_state_snapshot["loss_statistics_snapshot"]
        )
