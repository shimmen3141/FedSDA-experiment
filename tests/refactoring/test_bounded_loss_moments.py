"""有界損失の不変集計値を旧逐次計算へ直接照合する。"""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
    LossMeanAndSampleVariance,
    accumulate_bounded_loss_observation,
    estimate_loss_mean_and_sample_variance,
)


@pytest.mark.parametrize(
    "loss_sequence",
    [(0.0,) * 5, (1.0,) * 5, (0.0, 1.0) * 5, (0.2, 0.7, 0.3, 0.9), (1e-15, 2e-15, 3e-15)],
)
@pytest.mark.parametrize(
    "legacy_stats",
    [
        {"n": 0, "mean": 0.0, "M2": 0.0},
        {"n": 1, "mean": 0.25, "M2": 0.1},
        {"n": 10, "mean": 0.3, "M2": 0.0},
    ],
)
def test_loss_moments_match_legacy_updates(loss_sequence, legacy_stats):
    legacy_stats = legacy_stats.copy()
    loss_moments = BoundedLossMoments(
        observed_loss_count=legacy_stats["n"],
        mean_loss=legacy_stats["mean"],
        sum_squared_loss_deviations=legacy_stats["M2"],
    )
    legacy_client = object.__new__(BaseClient)
    legacy_client.model_stats = {0: legacy_stats}
    for observed_loss in loss_sequence:
        state_before_call = vars(loss_moments).copy()
        result = accumulate_bounded_loss_observation(
            loss_moments=loss_moments, observed_loss=observed_loss
        )
        assert vars(loss_moments) == state_before_call
        assert result is not loss_moments
        BaseClient._update_running_stats(legacy_stats, observed_loss)
        assert (
            result.observed_loss_count,
            result.mean_loss,
            result.sum_squared_loss_deviations,
        ) == (legacy_stats["n"], legacy_stats["mean"], legacy_stats["M2"])
        estimate = estimate_loss_mean_and_sample_variance(loss_moments=result)
        if legacy_stats["n"] < 2:
            assert estimate is None
            assert BaseClient._get_model_stats(legacy_client, 0) == (0.0, 0.0)
        else:
            assert type(estimate) is LossMeanAndSampleVariance
            assert estimate.observed_loss_count == legacy_stats["n"]
            assert (estimate.mean_loss, estimate.sample_variance) == (
                BaseClient._get_model_stats(legacy_client, 0)
            )
        loss_moments = result


@pytest.mark.parametrize(
    "field_name, invalid_value",
    [
        ("observed_loss_count", True),
        ("observed_loss_count", -1),
        ("observed_loss_count", 1.0),
        ("observed_loss_count", np.int64(1)),
        ("observed_loss_count", 10**400),
        ("mean_loss", True),
        ("mean_loss", -0.1),
        ("mean_loss", 1.1),
        ("mean_loss", float("nan")),
        ("mean_loss", float("inf")),
        ("mean_loss", np.float64(0.2)),
        ("mean_loss", [0.2]),
        ("sum_squared_loss_deviations", True),
        ("sum_squared_loss_deviations", -0.1),
        ("sum_squared_loss_deviations", float("nan")),
        ("sum_squared_loss_deviations", float("inf")),
        ("sum_squared_loss_deviations", 10**400),
        ("sum_squared_loss_deviations", np.float64(0.2)),
    ],
)
def test_loss_moments_reject_invalid_fields(field_name, invalid_value):
    state_before_call = dict(observed_loss_count=1, mean_loss=0.2, sum_squared_loss_deviations=0.1)
    state_before_call[field_name] = invalid_value
    with pytest.raises((TypeError, ValueError), match=field_name):
        BoundedLossMoments(**state_before_call)
    loss_moments = BoundedLossMoments(
        observed_loss_count=1, mean_loss=0.2, sum_squared_loss_deviations=0.1
    )
    object.__setattr__(loss_moments, field_name, invalid_value)
    state_before_call = vars(loss_moments).copy()
    for operation in (accumulate_bounded_loss_observation, estimate_loss_mean_and_sample_variance):
        with pytest.raises((TypeError, ValueError), match=field_name):
            if operation is accumulate_bounded_loss_observation:
                operation(loss_moments=loss_moments, observed_loss=0.5)
            else:
                operation(loss_moments=loss_moments)
        assert all(
            vars(loss_moments)[field_name] is invalid_value
            for field_name, invalid_value in state_before_call.items()
        )


@pytest.mark.parametrize(
    "invalid_value",
    [True, -0.1, 1.1, float("nan"), float("inf"), 10**400, np.float64(0.2), [0.2], "0.2", None],
)
def test_loss_moments_reject_invalid_observations(invalid_value):
    loss_moments = BoundedLossMoments(
        observed_loss_count=1, mean_loss=0.2, sum_squared_loss_deviations=0.1
    )
    state_before_call = vars(loss_moments).copy()
    with pytest.raises((TypeError, ValueError), match="observed_loss"):
        accumulate_bounded_loss_observation(loss_moments=loss_moments, observed_loss=invalid_value)
    assert vars(loss_moments) == state_before_call


def test_loss_moments_distinguish_unsupported_from_zero():
    for sample_index in (0, 1):
        loss_moments = BoundedLossMoments(
            observed_loss_count=sample_index, mean_loss=0, sum_squared_loss_deviations=0
        )
        assert type(loss_moments.mean_loss) is float
        assert type(loss_moments.sum_squared_loss_deviations) is float
        assert estimate_loss_mean_and_sample_variance(loss_moments=loss_moments) is None
    loss_moments = BoundedLossMoments(
        observed_loss_count=2, mean_loss=0, sum_squared_loss_deviations=0
    )
    estimate = estimate_loss_mean_and_sample_variance(loss_moments=loss_moments)
    assert estimate == LossMeanAndSampleVariance(
        observed_loss_count=2, mean_loss=0, sample_variance=0
    )
    assert type(estimate.mean_loss) is float
    assert type(estimate.sample_variance) is float
    for result in (loss_moments, estimate):
        for field_name in vars(result):
            with pytest.raises(FrozenInstanceError):
                setattr(result, field_name, 3)
    for field_name in ("mean_loss", "sum_squared_loss_deviations"):
        with pytest.raises(ValueError, match=field_name):
            BoundedLossMoments(
                **(
                    dict(observed_loss_count=0, mean_loss=0, sum_squared_loss_deviations=0)
                    | {field_name: 0.1}
                )
            )
    # 再検査が既存instanceをfloatへ書き換えないことを確認する。
    object.__setattr__(loss_moments, "mean_loss", 0)
    object.__setattr__(loss_moments, "sum_squared_loss_deviations", 0)
    estimate_loss_mean_and_sample_variance(loss_moments=loss_moments)
    accumulate_bounded_loss_observation(loss_moments=loss_moments, observed_loss=1)
    assert type(loss_moments.mean_loss) is int
    assert type(loss_moments.sum_squared_loss_deviations) is int
    for operation in (accumulate_bounded_loss_observation, estimate_loss_mean_and_sample_variance):
        with pytest.raises(TypeError, match="loss_moments"):
            if operation is accumulate_bounded_loss_observation:
                operation(loss_moments=None, observed_loss=0)
            else:
                operation(loss_moments=None)
    # 有限floatへ変換可能な最大境界のcountは、次の追加で契約外になる。
    loss_moments = BoundedLossMoments(
        observed_loss_count=int(float.fromhex("0x1.fffffffffffffp+1023")) + 2**970 - 1,
        mean_loss=0.5,
        sum_squared_loss_deviations=0.1,
    )
    state_before_call = vars(loss_moments).copy()
    with pytest.raises(ValueError, match="observed_loss_count"):
        accumulate_bounded_loss_observation(loss_moments=loss_moments, observed_loss=0.5)
    assert vars(loss_moments) == state_before_call


@pytest.mark.parametrize(
    "loss_sequence",
    [
        ((7, 0, 0.1), (7, 1, 0.9), (7, 0, 0.3), (-2, 1, 0.0), (-2, 1, 1.0)),
        ((7, 0, 0.0), (7, 0, 0.0), (7, 1, 1.0), (-2, 0, 0.5)),
    ],
)
def test_loss_moments_match_legacy_class_updates(loss_sequence):
    """帰属順で全体と正解classの独立系列を旧辞書へ照合する。"""
    legacy_client = SimpleNamespace(
        model_stats={}, _update_running_stats=BaseClient._update_running_stats
    )
    overall_loss_moments = {}
    class_loss_moments = {}
    for model_id, class_id, observed_loss in loss_sequence:
        BaseClient._update_model_stats(legacy_client, model_id, observed_loss, class_id)
        overall_loss_moments[model_id] = accumulate_bounded_loss_observation(
            loss_moments=overall_loss_moments.get(
                model_id,
                BoundedLossMoments(
                    observed_loss_count=0, mean_loss=0, sum_squared_loss_deviations=0
                ),
            ),
            observed_loss=observed_loss,
        )
        class_loss_moments[model_id, class_id] = accumulate_bounded_loss_observation(
            loss_moments=class_loss_moments.get(
                (model_id, class_id),
                BoundedLossMoments(
                    observed_loss_count=0, mean_loss=0, sum_squared_loss_deviations=0
                ),
            ),
            observed_loss=observed_loss,
        )
        for model_id, result in overall_loss_moments.items():
            legacy_stats = legacy_client.model_stats[model_id]
            assert (
                result.observed_loss_count,
                result.mean_loss,
                result.sum_squared_loss_deviations,
            ) == (legacy_stats["n"], legacy_stats["mean"], legacy_stats["M2"])
        for (model_id, class_id), result in class_loss_moments.items():
            legacy_class_stats = legacy_client.model_stats[model_id]["class_stats"][class_id]
            assert (
                result.observed_loss_count,
                result.mean_loss,
                result.sum_squared_loss_deviations,
            ) == (legacy_class_stats["n"], legacy_class_stats["mean"], legacy_class_stats["M2"])


@pytest.mark.parametrize("loss_sequence", [(), (0.2,), (0.2, 0.4), (0.0, 0.0), (1.0, 1.0)])
def test_loss_moments_connect_to_monitor_and_reference_evaluation(loss_sequence):
    """1件の保存平均は監視に使え、候補履歴には2件以上だけを渡す。"""
    from federated_drift_experiment.clients.fedsda import ESRFedSDAClient
    from federated_drift_experiment.provisional_model import select_forward_fitting_reference
    from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
        select_available_reference_within_historical_loss_tolerance,
    )
    from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
        LossChangeDetectionSettings,
    )
    from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
        OverallAndTrueClassLossMonitor,
    )

    legacy_stats = dict(n=0, mean=0.0, M2=0.0)
    overall_loss_moments = BoundedLossMoments(
        observed_loss_count=0, mean_loss=0, sum_squared_loss_deviations=0
    )
    for observed_loss in loss_sequence:
        BaseClient._update_running_stats(legacy_stats, observed_loss)
        overall_loss_moments = accumulate_bounded_loss_observation(
            loss_moments=overall_loss_moments, observed_loss=observed_loss
        )
    legacy_client = SimpleNamespace(model_stats={7: legacy_stats}, current_model_id=7)
    # clipと件数条件は接続側の旧方針。数値部品には持ち込まない。
    result = (
        0.01
        if overall_loss_moments.observed_loss_count < 1
        else min(1.0 - 1e-6, max(0.01, overall_loss_moments.mean_loss))
    )
    assert result == ESRFedSDAClient._e_detector_baseline(legacy_client)
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=LossChangeDetectionSettings(
            drift_detector_name="e_sr",
            loss_monitoring_scope="overall_and_true_class_losses",
            e_sr_false_alarm_control_alpha=0.01,
        ),
        class_count=2,
        initial_baseline_loss_mean=result,
        maximum_retained_candidate_count=5,
        betting_fractions=(0.1, 0.4),
    )
    class_loss_moments = BoundedLossMoments(
        observed_loss_count=2, mean_loss=0.9, sum_squared_loss_deviations=0.0
    )
    observation = monitor.observe_loss_after_label_observation(
        observed_loss=0.5,
        observed_class_id=0,
        sample_index=0,
        current_model_baseline_loss_mean=result,
    )
    assert observation.component_update_count == 2
    assert monitor.get_state_snapshot().overall_esr_state.baseline_loss_mean == result
    assert (
        monitor.get_state_snapshot().class_esr_states_by_class_id[0][1].baseline_loss_mean == result
    )
    assert class_loss_moments.mean_loss == 0.9  # 独立クラス統計の平均へ置換しない。
    estimate = estimate_loss_mean_and_sample_variance(loss_moments=overall_loss_moments)
    reference_historical_mean_losses_by_model_id = (
        {} if estimate is None else {7: estimate.mean_loss}
    )
    assert reference_historical_mean_losses_by_model_id == (
        {7: legacy_stats["mean"]} if legacy_stats["n"] >= 2 else {}
    )
    reference_losses_by_model_id = {7: (0.1, 0.2)}
    available_reference_model_ids = (7,)
    current_training_model_id = 7
    assert select_available_reference_within_historical_loss_tolerance(
        reference_losses_by_model_id=reference_losses_by_model_id,
        reference_historical_mean_losses_by_model_id=reference_historical_mean_losses_by_model_id,
        available_reference_model_ids=available_reference_model_ids,
        current_training_model_id=current_training_model_id,
        maximum_reference_mean_loss_increase=0.2,
    ) == select_forward_fitting_reference(
        reference_losses_by_model_id,
        reference_historical_mean_losses_by_model_id,
        0.2,
        preferred_model_id=7,
    )


def test_loss_moments_preserve_shared_numeric_state():
    """純粋集計は共有RNG・既定型・deviceに影響しない。"""
    import random

    import numpy as np
    import torch

    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    global_torch_random_state = torch.get_rng_state().clone()
    global_default_dtype = torch.get_default_dtype()
    global_default_device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            overall_loss_moments = BoundedLossMoments(
                observed_loss_count=1, mean_loss=0.25, sum_squared_loss_deviations=0.1
            )
            result = accumulate_bounded_loss_observation(
                loss_moments=overall_loss_moments, observed_loss=0.75
            )
            assert (
                estimate_loss_mean_and_sample_variance(loss_moments=result).sample_variance == 0.225
            )
            assert torch.get_default_device() == torch.device("meta")
            assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(global_default_dtype)
    assert random.getstate() == global_python_random_state
    assert np.random.get_state()[0] == global_numpy_random_state[0]
    assert np.array_equal(np.random.get_state()[1], global_numpy_random_state[1])
    assert np.random.get_state()[2:] == global_numpy_random_state[2:]
    assert torch.equal(torch.get_rng_state(), global_torch_random_state)
    assert torch.get_default_dtype() == global_default_dtype
    assert torch.get_default_device() == global_default_device


def test_loss_moments_use_keyword_arguments():
    """意味を取り違えないkeyword-only呼出を固定する。"""
    overall_loss_moments = BoundedLossMoments(
        observed_loss_count=2, mean_loss=0, sum_squared_loss_deviations=0
    )
    with pytest.raises(TypeError):
        BoundedLossMoments(2, 0, 0)
    with pytest.raises(TypeError):
        accumulate_bounded_loss_observation(overall_loss_moments, 0.2)
    with pytest.raises(TypeError):
        estimate_loss_mean_and_sample_variance(overall_loss_moments)
