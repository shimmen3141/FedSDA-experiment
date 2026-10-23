"""概念変化と逆転後の再反転・追跡無しを区別する境界検証。"""

import pytest

from tools.research_shadow_learner_diagnostic.analyze_stability import (
    summarize_episode,
    aggregate,
)


def build(scores, concepts, first, same):
    episode = dict(
        episode_id=0,
        client_id=1,
        alarm_sample_index=100,
        alarm_concept_id=2,
        initial_score=-2.0,
        first_crossing_age=first,
        first_crossing_same_alarm_concept=same,
        end_reason="alarm_current_model",
    )
    trace = [
        dict(
            age=i + 1,
            sample_index=101 + i,
            client_id=1,
            score=s,
            observed_concept_id=c,
            same_alarm_concept_so_far=all(v == 2 for v in concepts[: i + 1]),
        )
        for i, (s, c) in enumerate(zip(scores, concepts, strict=True))
    ]
    return episode, trace


def test_reversal_including_zero_and_recovery_is_counted():
    e, t = build([-1, 1, 0, 2], [2, 2, 2, 2], 2, True)
    r = summarize_episode(e, t)
    assert r["first_crossing_age"] == 2
    assert r["first_reversal_after_crossing"] == 1
    assert r["positive_fraction_of_followup"] == 0.5
    assert aggregate([r])["crossed_last_nonpositive"] == 0


def test_later_concept_change_and_recurrence_are_excluded():
    e, t = build([-1, 1, -3, -4], [2, 2, 0, 2], 2, True)
    r = summarize_episode(e, t)
    assert r["followup_count"] == 0
    assert r["first_reversal_after_crossing"] is None
    assert r["same_concept_end_reason"] == "concept_change"
    assert aggregate([r])["crossed_without_followup"] == 1


def test_crossing_only_after_concept_change_does_not_count():
    e, t = build([-1, 2], [2, 0], 2, False)
    assert aggregate([summarize_episode(e, t)])["crossed"] == 0


def test_empty_trace_and_zero_episode_group_are_explicit():
    e, t = build([], [], None, None)
    assert summarize_episode(e, t)["last_same_concept_score"] == -2.0
    assert aggregate([])["episodes"] == 0


def test_control_score_metadata_and_malformed_trace_are_checked():
    e, t = build([-1, 1], [2, 2], 2, True)
    e.update(
        control_first_crossing_age=None, control_first_crossing_same_alarm_concept=None
    )
    for r in t:
        r["control_score"] = -1
    assert summarize_episode(e, t, "control_score")["first_crossing_age"] is None
    t[1]["same_alarm_concept_so_far"] = False
    with pytest.raises(ValueError, match="継続フラグ"):
        summarize_episode(e, t)


def test_nonfinite_score_is_rejected():
    e, t = build([float("nan")], [2], None, None)
    with pytest.raises(ValueError, match="非有限"):
        summarize_episode(e, t)
