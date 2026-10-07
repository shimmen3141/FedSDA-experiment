"""警報区間での保有モデル再利用評価を、純粋判定と実旧の警報処理の両方で検証する。"""

import math
from dataclasses import FrozenInstanceError, fields

import pytest

from federated_drift_experiment.clients.fedsda import FedSDAClient
from federated_learning_experiments.methods.fedsda.candidate_model_selection import (
    alarm_interval_model_reuse_assessment as assessment_module,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
    AlarmIntervalModelReuseAssessment,
    assess_alarm_interval_model_reuse,
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
