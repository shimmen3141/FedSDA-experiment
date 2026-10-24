"""同時交差・概念による打切り・保存値の破損を検証する。"""

from copy import deepcopy

import pytest

from tools.research_shadow_optimizer_analysis import summarize_episode, validate_plan


def example(*, d, keep, reset, concepts=None):
    concepts = concepts or [0] * len(d)
    episode = dict(
        episode_id=0,
        client_id=2,
        alarm_sample_index=100,
        alarm_concept_id=0,
        initial_score=-2.0,
        observed_sample_count=len(d),
        learner_name="reinitialized_learner",
        pending_is_single_concept=True,
        end_reason="horizon",
        last_sample_index=100 + len(d),
        end_sample_index=100 + len(d),
        last_score=d[-1],
        control_score=keep[-1],
        reset_control_score=reset[-1],
    )
    trace, previous, same = [], [-2.0] * 3, True
    for i, values in enumerate(zip(d, keep, reset)):
        same = same and concepts[i] == 0
        trace.append(
            dict(
                episode_id=0,
                client_id=2,
                sample_index=101 + i,
                age=i + 1,
                observed_concept_id=concepts[i],
                same_alarm_concept_so_far=same,
                shadow_log_loss=10.0,
                current_log_loss=10.0 + values[0] - previous[0],
                online_current_log_loss=10.0 + values[1] - previous[1],
                reset_current_log_loss=10.0 + values[2] - previous[2],
                score=values[0],
                control_score=values[1],
                reset_control_score=values[2],
            )
        )
        previous = list(values)
    for values, prefix in (
        (d, "first_crossing"),
        (keep, "control_first_crossing"),
        (reset, "reset_control_first_crossing"),
    ):
        first = next((i for i, v in enumerate(values) if v > 0), None)
        episode[prefix + "_age"] = first + 1 if first is not None else None
        episode[prefix + "_same_alarm_concept"] = (
            trace[first]["same_alarm_concept_so_far"] if first is not None else None
        )
    return episode, trace


def test_different_crossing_times_are_not_joint_and_reversal_is_retained():
    e, t = example(d=[1, -1, 1], keep=[-1, 1, -1], reset=[1, 1, 1])
    r = summarize_episode(e, t)
    assert r["d_age"] == r["reset_age"] == 1
    assert r["keep_age"] == 2
    assert r["joint_keep_age"] is None
    assert r["joint_reset_age"] == 1
    assert r["joint_reset_reversed"] is True
    assert r["joint_reset_observed_after_crossing"] == 2


def test_return_of_same_concept_does_not_remove_censoring():
    e, t = example(d=[-1, 1, 1], keep=[-1, 1, 1], reset=[-1, 1, 1], concepts=[0, 1, 0])
    r = summarize_episode(e, t)
    assert r["same_concept_observations"] == 1
    assert r["d_age"] is r["joint_keep_age"] is None
    assert r["d_reversed"] is None
    corrupted = deepcopy(t)
    corrupted[-1]["same_alarm_concept_so_far"] = True
    with pytest.raises(ValueError, match="概念維持"):
        summarize_episode(e, corrupted)


def test_corrupt_loss_or_order_is_rejected():
    e, t = example(d=[1, 1], keep=[1, 1], reset=[1, 1])
    corrupted = deepcopy(t)
    corrupted[-1]["reset_current_log_loss"] += 0.5
    with pytest.raises(ValueError, match="累積差"):
        summarize_episode(e, corrupted)
    with pytest.raises(ValueError, match="順序"):
        summarize_episode(e, t[::-1])


def test_last_observation_crossing_has_no_followup_and_no_claim_of_stability():
    e, t = example(d=[-1, 1], keep=[-1, 1], reset=[-1, 1])
    r = summarize_episode(e, t)
    assert r["joint_keep_age"] == 2
    assert r["joint_keep_reversed"] is False
    assert r["joint_keep_observed_after_crossing"] == 0


def test_inconsistent_episode_end_is_rejected():
    e, t = example(d=[1], keep=[1], reset=[1])
    e["end_sample_index"] += 1
    with pytest.raises(ValueError, match="終端"):
        summarize_episode(e, t)


def test_plan_rejects_duplicate_seed_unbalanced_groups_and_mixed_scales():
    base = dict(
        dataset="sea4",
        stream="drift",
        interval=100,
        seed=0,
        sample_count=5000,
        horizon_samples=240,
        task="optimizer",
        disabled=False,
    )
    jobs = [
        base,
        dict(base, seed=1),
        dict(base, stream="fixed2"),
        dict(base, stream="fixed2", seed=1),
    ]
    validate_plan(jobs)
    with pytest.raises(ValueError, match="二重計上"):
        validate_plan([base, dict(base, output="別の出力")])
    with pytest.raises(ValueError, match="seed集合"):
        validate_plan(jobs[:-1])
    with pytest.raises(ValueError, match="規模"):
        validate_plan([base, dict(base, seed=1, sample_count=1500)])
