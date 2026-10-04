"""有界損失の不変集計値を旧逐次計算へ直接照合する。"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
    LossMeanAndSampleVariance,
    accumulate_bounded_loss_observation,
    estimate_loss_mean_and_sample_variance,
)


@pytest.mark.parametrize("loss_sequence", [(0.0,) * 5, (1.0,) * 5,
    (0.0, 1.0) * 5, (0.2, 0.7, 0.3, 0.9), (1e-15, 2e-15, 3e-15)])
@pytest.mark.parametrize("legacy_stats", [
    {"n": 0, "mean": 0.0, "M2": 0.0},
    {"n": 1, "mean": 0.25, "M2": 0.1},
    {"n": 10, "mean": 0.3, "M2": 0.0},
])
def test_loss_moments_match_legacy_updates(loss_sequence, legacy_stats):
    legacy_stats = legacy_stats.copy()
    loss_moments = BoundedLossMoments(observed_loss_count=legacy_stats["n"],
        mean_loss=legacy_stats["mean"], sum_squared_loss_deviations=legacy_stats["M2"])
    legacy_client = object.__new__(BaseClient)
    legacy_client.model_stats = {0: legacy_stats}
    for observed_loss in loss_sequence:
        state_before_call = vars(loss_moments).copy()
        result = accumulate_bounded_loss_observation(
            loss_moments=loss_moments, observed_loss=observed_loss)
        assert vars(loss_moments) == state_before_call
        assert result is not loss_moments
        BaseClient._update_running_stats(legacy_stats, observed_loss)
        assert (result.observed_loss_count, result.mean_loss,
                result.sum_squared_loss_deviations) == (
                    legacy_stats["n"], legacy_stats["mean"], legacy_stats["M2"])
        estimate = estimate_loss_mean_and_sample_variance(loss_moments=result)
        if legacy_stats["n"] < 2:
            assert estimate is None
            assert BaseClient._get_model_stats(legacy_client, 0) == (0.0, 0.0)
        else:
            assert type(estimate) is LossMeanAndSampleVariance
            assert estimate.observed_loss_count == legacy_stats["n"]
            assert (estimate.mean_loss, estimate.sample_variance) == (
                BaseClient._get_model_stats(legacy_client, 0))
        loss_moments = result


@pytest.mark.parametrize("field_name, invalid_value", [
    ("observed_loss_count", True), ("observed_loss_count", -1),
    ("observed_loss_count", 1.0), ("observed_loss_count", np.int64(1)),
    ("observed_loss_count", 10 ** 400),
    ("mean_loss", True), ("mean_loss", -0.1), ("mean_loss", 1.1),
    ("mean_loss", float("nan")), ("mean_loss", float("inf")),
    ("mean_loss", np.float64(0.2)), ("mean_loss", [0.2]),
    ("sum_squared_loss_deviations", True),
    ("sum_squared_loss_deviations", -0.1),
    ("sum_squared_loss_deviations", float("nan")),
    ("sum_squared_loss_deviations", float("inf")),
    ("sum_squared_loss_deviations", 10 ** 400),
    ("sum_squared_loss_deviations", np.float64(0.2)),
])
def test_loss_moments_reject_invalid_fields(field_name, invalid_value):
    state_before_call = dict(observed_loss_count=1, mean_loss=0.2,
                             sum_squared_loss_deviations=0.1)
    state_before_call[field_name] = invalid_value
    with pytest.raises((TypeError, ValueError), match=field_name):
        BoundedLossMoments(**state_before_call)
    loss_moments = BoundedLossMoments(observed_loss_count=1, mean_loss=0.2,
                                    sum_squared_loss_deviations=0.1)
    object.__setattr__(loss_moments, field_name, invalid_value)
    state_before_call = vars(loss_moments).copy()
    for operation in (accumulate_bounded_loss_observation,
                      estimate_loss_mean_and_sample_variance):
        with pytest.raises((TypeError, ValueError), match=field_name):
            if operation is accumulate_bounded_loss_observation:
                operation(loss_moments=loss_moments, observed_loss=0.5)
            else:
                operation(loss_moments=loss_moments)
        assert all(vars(loss_moments)[field_name] is invalid_value
                   for field_name, invalid_value in state_before_call.items())


@pytest.mark.parametrize("invalid_value", [True, -0.1, 1.1, float("nan"),
    float("inf"), 10 ** 400, np.float64(0.2), [0.2], "0.2", None])
def test_loss_moments_reject_invalid_observations(invalid_value):
    loss_moments = BoundedLossMoments(observed_loss_count=1, mean_loss=0.2,
                                    sum_squared_loss_deviations=0.1)
    state_before_call = vars(loss_moments).copy()
    with pytest.raises((TypeError, ValueError), match="observed_loss"):
        accumulate_bounded_loss_observation(loss_moments=loss_moments,
                                            observed_loss=invalid_value)
    assert vars(loss_moments) == state_before_call


def test_loss_moments_distinguish_unsupported_from_zero():
    for sample_index in (0, 1):
        loss_moments = BoundedLossMoments(observed_loss_count=sample_index,
            mean_loss=0, sum_squared_loss_deviations=0)
        assert type(loss_moments.mean_loss) is float
        assert type(loss_moments.sum_squared_loss_deviations) is float
        assert estimate_loss_mean_and_sample_variance(loss_moments=loss_moments) is None
    loss_moments = BoundedLossMoments(observed_loss_count=2, mean_loss=0,
                                    sum_squared_loss_deviations=0)
    estimate = estimate_loss_mean_and_sample_variance(loss_moments=loss_moments)
    assert estimate == LossMeanAndSampleVariance(observed_loss_count=2,
                                               mean_loss=0, sample_variance=0)
    assert type(estimate.mean_loss) is float
    assert type(estimate.sample_variance) is float
    for result in (loss_moments, estimate):
        for field_name in vars(result):
            with pytest.raises(FrozenInstanceError):
                setattr(result, field_name, 3)
    for field_name in ("mean_loss", "sum_squared_loss_deviations"):
        with pytest.raises(ValueError, match=field_name):
            BoundedLossMoments(**(dict(observed_loss_count=0, mean_loss=0,
                sum_squared_loss_deviations=0) | {field_name: 0.1}))
    # 再検査が既存instanceをfloatへ書き換えないことを確認する。
    object.__setattr__(loss_moments, "mean_loss", 0)
    object.__setattr__(loss_moments, "sum_squared_loss_deviations", 0)
    estimate_loss_mean_and_sample_variance(loss_moments=loss_moments)
    accumulate_bounded_loss_observation(loss_moments=loss_moments, observed_loss=1)
    assert type(loss_moments.mean_loss) is int
    assert type(loss_moments.sum_squared_loss_deviations) is int
    for operation in (accumulate_bounded_loss_observation,
                      estimate_loss_mean_and_sample_variance):
        with pytest.raises(TypeError, match="loss_moments"):
            if operation is accumulate_bounded_loss_observation:
                operation(loss_moments=None, observed_loss=0)
            else:
                operation(loss_moments=None)
    # 有限floatへ変換可能な最大境界のcountは、次の追加で契約外になる。
    loss_moments = BoundedLossMoments(
        observed_loss_count=int(float.fromhex("0x1.fffffffffffffp+1023")) + 2 ** 970 - 1,
        mean_loss=0.5, sum_squared_loss_deviations=0.1)
    state_before_call = vars(loss_moments).copy()
    with pytest.raises(ValueError, match="observed_loss_count"):
        accumulate_bounded_loss_observation(loss_moments=loss_moments, observed_loss=0.5)
    assert vars(loss_moments) == state_before_call
