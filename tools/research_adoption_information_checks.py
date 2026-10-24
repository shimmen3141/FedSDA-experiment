"""採用時までの特徴、seed分離、学習だけでの閾値選択を検査する。"""

from copy import deepcopy
import pytest

from tools.research_adoption_information import (
    FEATURES,
    adoption_features,
    evaluate,
    fit,
    predict,
    label,
)


def candidate(seed, x, y):
    return dict(
        seed=seed,
        dataset="sea4",
        stream="drift",
        client=0,
        episode=0,
        features={name: x for name in FEATURES},
        labels={"full_log": y},
    )


def test_features_ignore_future_trace_and_oracle_fields():
    episode = dict(
        initial_score=-1.0,
        last_score=0.5,
        observed_sample_count=2,
        adoption_sample_index=12,
        current_code_length=4.0,
        new_code_length=5.0,
        change_interval_sample_count=3,
        learner_name="warm_started_learner",
        adoption_pending_sample_count=2,
        alarm_concept_id=100,
    )
    trace = [
        dict(
            age=1,
            sample_index=11,
            current_log_loss=1.0,
            shadow_log_loss=0.5,
            score=-0.5,
            observed_concept_id=100,
        ),
        dict(
            age=2,
            sample_index=12,
            current_log_loss=1.5,
            shadow_log_loss=0.5,
            score=0.5,
            observed_concept_id=100,
        ),
    ]
    f = adoption_features(episode, trace)
    altered = deepcopy(trace)
    altered[0]["observed_concept_id"] = 999
    altered.append(
        dict(
            sample_index=13, age=3, current_log_loss=1000, shadow_log_loss=0, score=1000
        )
    )
    assert adoption_features(episode, altered) == f
    assert f["recent30_gap"] == 0.75 and set(f) == set(FEATURES)
    with pytest.raises(ValueError):
        adoption_features(episode, trace[:1])


def test_threshold_uses_training_only_and_neutrals_are_not_harmful():
    train = [
        candidate(0, 0, -1),
        candidate(0, 1, -1),
        candidate(1, 3, 1),
        candidate(1, 4, 1),
    ]
    model = fit(train, "full_log")
    assert model["feature"] == FEATURES[0] and model["threshold"] == 2
    before = deepcopy(model)
    test = [
        candidate(2, 0, 1),
        candidate(3, 4, -1),
        candidate(2, 5, 0),
        candidate(3, 6, None),
    ]
    result = evaluate(model, test, "full_log")
    assert (
        result["balanced_accuracy"] == 0
        and result["neutral"] == 1
        and result["missing"] == 1
    )
    assert result["evaluated"] == 2 and model == before
    assert predict(model, train[0]["features"]) == -1


def test_seed_weighting_and_single_class_test():
    model = dict(feature=None, constant=1)
    rows = [candidate(0, 0, -1)] * 9 + [candidate(1, 1, 1)]
    result = evaluate(model, rows, "full_log")
    assert result["seed_weighted_accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    single = evaluate(model, [candidate(2, 0, 1)], "full_log")
    assert single["balanced_accuracy"] is None and single["seed_weighted_accuracy"] == 1
    assert label(1e-12, True) == 0 and label(None, False) is None
