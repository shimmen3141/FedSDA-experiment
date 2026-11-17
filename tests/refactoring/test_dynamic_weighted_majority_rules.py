"""DWM（Kolter & Maloof 2007の図1）の投票と、周期の標本での重みの更新の規則を、手で計算した例で確かめる。"""

import math

import pytest

from federated_learning_experiments.methods.dynamic_weighted_majority.dynamic_weighted_majority_rules import (
    DynamicWeightedMajorityPeriodUpdate,
    update_weights_at_period,
    vote_weighted_majority,
)

BETA = 0.5
THETA = 0.01


def update(predictions, weights, observed, *, class_count=2, beta=BETA, theta=THETA):
    return update_weights_at_period(
        expert_predicted_class_ids=tuple(predictions),
        expert_weights=tuple(weights),
        observed_class_id=observed,
        class_count=class_count,
        weight_decrease_factor=beta,
        removal_weight_threshold=theta,
    )


def test_vote_adds_each_weight_to_the_predicted_class():
    assert (
        vote_weighted_majority(
            expert_predicted_class_ids=(2, 0, 2), expert_weights=(0.3, 0.5, 0.3), class_count=3
        )
        == 2
    )
    assert (
        vote_weighted_majority(
            expert_predicted_class_ids=(2, 0, 2), expert_weights=(0.2, 0.5, 0.2), class_count=3
        )
        == 0
    )


def test_vote_breaks_ties_by_the_smallest_class_id():
    assert (
        vote_weighted_majority(
            expert_predicted_class_ids=(1, 0), expert_weights=(1.0, 1.0), class_count=2
        )
        == 0
    )
    # 票の無いクラス（重み0）も同点の対象になる。全員が重み0なら、最小のクラス。
    assert (
        vote_weighted_majority(
            expert_predicted_class_ids=(2,), expert_weights=(0.0,), class_count=3
        )
        == 0
    )


def test_period_update_keeps_weights_when_every_expert_is_correct():
    result = update((1, 1), (1.0, 0.5), 1)
    assert result == DynamicWeightedMajorityPeriodUpdate(
        updated_weights=(1.0, 0.5),
        kept_expert_positions=(0, 1),
        reweighted_global_prediction_class_id=1,
        new_expert_is_added=False,
    )


def test_period_update_normalizes_after_every_expert_is_wrong_and_adds_an_expert():
    # 0.5, 0.25に減衰→投票し直すとクラス0（誤り）→最大0.5で割って1, 0.5→追加する。
    result = update((0, 0), (1.0, 0.5), 1)
    assert result.updated_weights == (1.0, 0.5)
    assert result.kept_expert_positions == (0, 1)
    assert result.reweighted_global_prediction_class_id == 0
    assert result.new_expert_is_added is True


def test_period_update_removes_experts_below_the_threshold_after_normalization():
    # 2つ目だけ誤り: 0.015→0.0075。最大は1のまま。0.0075 < θ=0.01なので消す。
    result = update((1, 0), (1.0, 0.015), 1)
    assert result.kept_expert_positions == (0,)
    assert result.updated_weights == (1.0,)
    assert result.reweighted_global_prediction_class_id == 1
    assert result.new_expert_is_added is False


def test_addition_uses_the_vote_before_normalization_and_removal():
    # y=0。専門家0はクラス1（誤り）で1→0.5。専門家1〜3はクラス0（正しい）で0.2, 0.2, 0.15。
    # 減衰の直後の投票: クラス0が0.55、クラス1が0.5 → クラス0（正しい）なので追加しない。
    # 正規化（最大0.5で割る）: 1, 0.4, 0.4, 0.3。θ=0.5なら専門家1〜3を消す。
    # 消した後に投票し直すとクラス1（誤り）になり追加してしまうので、順序を区別できる。
    result = update((1, 0, 0, 0), (1.0, 0.2, 0.2, 0.15), 0, theta=0.5)
    assert result.reweighted_global_prediction_class_id == 0
    assert result.new_expert_is_added is False
    assert result.kept_expert_positions == (0,)
    assert result.updated_weights == (1.0,)


def test_maximum_weight_survives_so_an_expert_always_remains():
    result = update((0, 0, 0), (1.0, 1.0, 1.0), 1, theta=0.99)
    assert result.updated_weights == (1.0, 1.0, 1.0)
    assert result.kept_expert_positions == (0, 1, 2)
    result = update((0, 1), (1.0, 0.003), 1, theta=0.99)
    # 0→0.5、1は正しく0.003のまま。最大0.5で割ると1と0.006。θ=0.99で1つ目だけ残る。
    assert result.kept_expert_positions == (0,)
    assert math.isclose(result.updated_weights[0], 1.0)


@pytest.mark.parametrize(
    "arguments",
    [
        dict(expert_predicted_class_ids=(), expert_weights=()),
        dict(expert_predicted_class_ids=(0, 1), expert_weights=(1.0,)),
        dict(expert_predicted_class_ids=(2,), expert_weights=(1.0,)),
        dict(expert_predicted_class_ids=(-1,), expert_weights=(1.0,)),
        dict(expert_predicted_class_ids=(True,), expert_weights=(1.0,)),
        dict(expert_predicted_class_ids=(0,), expert_weights=(math.nan,)),
        dict(expert_predicted_class_ids=(0,), expert_weights=(-0.1,)),
        dict(expert_predicted_class_ids=[0], expert_weights=(1.0,)),
    ],
)
def test_vote_rejects_invalid_inputs(arguments):
    with pytest.raises((TypeError, ValueError)):
        vote_weighted_majority(class_count=2, **arguments)


@pytest.mark.parametrize(
    "changes",
    [
        dict(observed=2),
        dict(observed=True),
        dict(beta=0.0),
        dict(beta=1.0),
        dict(theta=0.0),
        dict(theta=1.0),
        dict(class_count=1),
        dict(weights=(0.0, 0.0)),
    ],
)
def test_period_update_rejects_invalid_inputs(changes):
    arguments = dict(predictions=(0, 1), weights=(1.0, 0.5), observed=1)
    arguments.update({k: v for k, v in changes.items() if k in ("weights",)})
    observed = changes.get("observed", arguments.pop("observed"))
    keyword = {k: v for k, v in changes.items() if k in ("beta", "theta", "class_count")}
    with pytest.raises((TypeError, ValueError)):
        update(arguments["predictions"], arguments["weights"], observed, **keyword)
