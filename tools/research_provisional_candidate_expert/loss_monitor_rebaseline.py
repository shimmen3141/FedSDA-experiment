"""IMPROVE-031の研究介入。監視の再設定だけを製品の外から行う。"""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch
from weakref import WeakKeyDictionary

LOSS_MONITOR_REBASELINE_VARIANTS = {
    "product-default-loss-monitor-decrease-rebaseline": (False, None),
    "product-default-loss-monitor-decrease-rebaseline-observe-only": (True, None),
    "product-default-loss-monitor-ratio-rebaseline-1p5": (False, 1.5),
    "product-default-loss-monitor-ratio-rebaseline-2": (False, 2.0),
}


def _validate_baseline(value):
    if type(value) not in (int, float) or not 0.01 <= value <= 1.0 - 1e-6:
        raise ValueError("監視の基準は有限な[0.01, 1 - 1e-6]が必要です")


@contextmanager
def loss_monitor_rebaseline_policy(*, variant, diagnostics):
    """下がる向きの検定、または比の閾値で基準を下げる。ADWINは拒否する。"""
    if variant == "product-default":
        yield WeakKeyDictionary()
        return
    observe_only, ratio = LOSS_MONITOR_REBASELINE_VARIANTS[variant]

    from federated_learning_experiments.methods.fedsda.loss_change_detection.bounded_loss_e_sr_detection import (
        BoundedLossESRDetector,
    )
    from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
        OverallAndTrueClassLossMonitor,
    )

    states = WeakKeyDictionary()
    monitor_type = OverallAndTrueClassLossMonitor
    init_original = monitor_type.__init__
    reset_original = monitor_type.reset
    observe_original = monitor_type.observe_loss_after_label_observation
    diagnostics.update(
        monitor_count=0,
        rebaseline_count=0,
        decrease_alarm_count=0,
        upward_alarm_count=0,
        rebaselines=[],
        monitors=[],
    )

    def make_decrease_detector(self, frozen):
        # 全体の検出器と同じ設定。frozenだけは制限後の内部基準から取る。
        return BoundedLossESRDetector(
            baseline_loss_mean=1.0 - frozen,
            false_alarm_control_alpha=float(
                self._loss_change_detection_settings.e_sr_false_alarm_control_alpha
            ),
            maximum_retained_candidate_count=self._maximum_retained_candidate_count,
            betting_fractions=self._betting_fractions,
        )

    def restart(self, state):
        state["frozen"] = self._overall_detector._baseline_loss_mean
        _validate_baseline(state["frozen"])
        state["decrease_detector"] = (
            make_decrease_detector(self, state["frozen"]) if ratio is None else None
        )

    def require_state(self):
        if self not in states:
            raise ValueError("変種を有効にする前に作られた監視器は使えません")
        return states[self]

    def initialize(self, **arguments):
        _validate_baseline(arguments["initial_baseline_loss_mean"])
        init_original(self, **arguments)
        if self._uses_adwin:
            raise ValueError("基準の再設定の変種はe-SR専用です（ADWINは対象外）")
        monitor_diagnostics = dict(
            monitor_index=diagnostics["monitor_count"],
            rebaseline_count=0,
            decrease_alarm_count=0,
            upward_alarm_count=0,
        )
        state = dict(
            monitor_index=diagnostics["monitor_count"],
            pending_rebaselines=[],
            diagnostics=monitor_diagnostics,
        )
        restart(self, state)
        states[self] = state
        diagnostics["monitors"].append(monitor_diagnostics)
        diagnostics["monitor_count"] += 1

    def reset(self, *, baseline_loss_mean):
        state = require_state(self)
        _validate_baseline(baseline_loss_mean)
        result = reset_original(self, baseline_loss_mean=baseline_loss_mean)
        restart(self, state)
        return result

    def observe(self, **arguments):
        state = require_state(self)
        current = arguments["current_model_baseline_loss_mean"]
        _validate_baseline(current)
        observation = observe_original(self, **arguments)
        decreased = False
        if ratio is None:
            decreased = (
                state["decrease_detector"]
                .observe_loss(observed_loss=1.0 - arguments["observed_loss"])
                .drift_detected
            )
            diagnostics["decrease_alarm_count"] += int(decreased)
            state["diagnostics"]["decrease_alarm_count"] += int(decreased)
        if observation.drift_detected:
            # 監視器の上がる向きの判定の回数。製品の警報（短い変化区間の方針で取り消されうる`alarm_raised`）や、
            # 警報の応答の回数とは一致しないことがある（実装レビューのMinor）。
            diagnostics["upward_alarm_count"] += 1
            state["diagnostics"]["upward_alarm_count"] += 1
            for event in state["pending_rebaselines"]:
                event["samples_until_next_upward_alarm"] = (
                    arguments["sample_index"] - event["sample_index"]
                )
            state["pending_rebaselines"].clear()
            return observation
        frozen = state["frozen"]
        should_rebaseline = (
            decreased and current < frozen
            if ratio is None
            else current < frozen / ratio
        )
        if should_rebaseline and not observe_only:
            self.reset(baseline_loss_mean=current)
            # 全件を保存する。未到来の次の警報はNoneで、打切りと0標本を区別する。
            event = dict(
                monitor_index=state["monitor_index"],
                sample_index=arguments["sample_index"],
                frozen_baseline_loss_mean=frozen,
                new_baseline_loss_mean=state["frozen"],
                samples_until_next_upward_alarm=None,
            )
            diagnostics["rebaseline_count"] += 1
            state["diagnostics"]["rebaseline_count"] += 1
            diagnostics["rebaselines"].append(event)
            state["pending_rebaselines"].append(event)
        elif decreased:
            # observe-onlyも補助検出器だけを再開し、製品の状態には触れない。
            state["decrease_detector"] = make_decrease_detector(self, frozen)
        return observation

    with ExitStack() as stack:
        stack.enter_context(patch.object(monitor_type, "__init__", initialize))
        stack.enter_context(patch.object(monitor_type, "reset", reset))
        stack.enter_context(
            patch.object(monitor_type, "observe_loss_after_label_observation", observe)
        )
        yield states
