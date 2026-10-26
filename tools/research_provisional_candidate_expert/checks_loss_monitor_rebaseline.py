"""IMPROVE-031専用の検証。通常のtestsへ組み込まず、pytestで明示実行する。"""

import gc
import json
from dataclasses import replace
from math import exp
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    LossChangeDetectionSettings,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)
from tools.research_provisional_candidate_expert.loss_monitor_rebaseline import (
    LOSS_MONITOR_REBASELINE_VARIANTS,
    loss_monitor_rebaseline_policy,
)

_DECREASE = "product-default-loss-monitor-decrease-rebaseline"
_OBSERVE_ONLY = _DECREASE + "-observe-only"


def monitor(baseline=0.6, *, detector="e_sr", scope="overall_loss"):
    return OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=LossChangeDetectionSettings(
            drift_detector_name=detector,
            loss_monitoring_scope=scope,
            e_sr_false_alarm_control_alpha=0.001,
            adwin_confidence_delta=0.002,
            adwin_minimum_test_window_sample_count=5,
            short_change_interval_alarm_policy="raise_alarm_immediately",
        ),
        class_count=2,
        initial_baseline_loss_mean=baseline,
        maximum_retained_candidate_count=1000,
        betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
    )


def observe(m, index, *, loss=0.1, current=0.1):
    return m.observe_loss_after_label_observation(
        observed_loss=loss,
        observed_class_id=0,
        sample_index=index,
        current_model_baseline_loss_mean=current,
    )


@pytest.mark.parametrize("baseline", [0.01, 0.6, 1.0 - 1e-6])
@pytest.mark.parametrize("loss", [0.0, 0.2, 1.0])
def test_decrease_increment_matches_hand_calculation(baseline, loss):
    with loss_monitor_rebaseline_policy(
        variant=_OBSERVE_ONLY, diagnostics={}
    ) as states:
        m = monitor(baseline)
        assert (
            states[m]["frozen"]
            == m._overall_detector.get_state_snapshot().baseline_loss_mean
        )
        detector = states[m]["decrease_detector"]
        detector.observe_loss(observed_loss=1.0 - loss)
        capitals = detector.get_state_snapshot().candidate_log_capitals[0]
        for fraction, capital in zip(m._betting_fractions, capitals):
            assert exp(capital) == pytest.approx(
                1.0 + fraction * (baseline - loss) / (1.0 - baseline)
            )
        assert (
            detector._maximum_retained_candidate_count
            == m._maximum_retained_candidate_count
        )
        assert detector._log_alarm_threshold == m._overall_detector._log_alarm_threshold


def test_low_loss_rebaselines_once_and_returns_original_observation():
    diagnostics = {}
    original = OverallAndTrueClassLossMonitor.reset
    resets = []

    def reset(self, **kw):
        resets.append(kw["baseline_loss_mean"])
        return original(self, **kw)

    with patch.object(OverallAndTrueClassLossMonitor, "reset", reset):
        with loss_monitor_rebaseline_policy(
            variant=_DECREASE, diagnostics=diagnostics
        ) as states:
            m = monitor(scope="overall_and_true_class_losses")
            for index in range(80):
                observation = observe(m, index)
                assert observation.sample_index == index
                assert not observation.drift_detected
                if resets:
                    break
            assert resets == [0.1]
            assert index < 80
            assert states[m]["frozen"] == 0.1
            assert m.get_state_snapshot().class_esr_states_by_class_id == ()
            assert m.last_observation is None
            first_index_after_reset = index + 1
            for later_index in range(first_index_after_reset, first_index_after_reset + 80):
                observe(m, later_index)
            assert resets == [0.1]
    assert diagnostics["rebaseline_count"] == 1
    assert diagnostics["decrease_alarm_count"] >= 1
    assert diagnostics["rebaselines"][0]["new_baseline_loss_mean"] == 0.1


def test_loss_at_baseline_does_not_rebaseline():
    diagnostics = {}
    with loss_monitor_rebaseline_policy(
        variant=_DECREASE, diagnostics=diagnostics
    ) as states:
        m = monitor()
        for index in range(200):
            observe(m, index, loss=0.6)
        assert states[m]["frozen"] == 0.6
    assert diagnostics["rebaseline_count"] == diagnostics["decrease_alarm_count"] == 0


@pytest.mark.parametrize("current", [0.6, 0.8])
def test_lower_detector_restarts_when_current_is_not_lower(current):
    diagnostics = {}
    with loss_monitor_rebaseline_policy(
        variant=_DECREASE, diagnostics=diagnostics
    ) as states:
        m = monitor()
        detector = states[m]["decrease_detector"]
        for index in range(80):
            observe(m, index, current=current)
        assert states[m]["frozen"] == 0.6
        assert states[m]["decrease_detector"] is not detector
    assert diagnostics["decrease_alarm_count"] > 0
    assert diagnostics["rebaseline_count"] == 0


@pytest.mark.parametrize("variant", LOSS_MONITOR_REBASELINE_VARIANTS)
def test_upward_alarm_prevents_rebaseline_and_observation_is_preserved(variant):
    original = OverallAndTrueClassLossMonitor.observe_loss_after_label_observation
    observations = []

    def alarm(self, **arguments):
        result = replace(original(self, **arguments), drift_detected=True)
        observations.append(result)
        return result

    diagnostics = {}
    with patch.object(
        OverallAndTrueClassLossMonitor, "observe_loss_after_label_observation", alarm
    ):
        with loss_monitor_rebaseline_policy(
            variant=variant, diagnostics=diagnostics
        ) as states:
            m = monitor()
            for index in range(40):
                assert observe(m, index) is observations[-1]
            assert states[m]["frozen"] == 0.6
    assert diagnostics["rebaseline_count"] == 0
    assert diagnostics["upward_alarm_count"] == 40
    if "decrease" in variant:
        assert diagnostics["decrease_alarm_count"] > 0


@pytest.mark.parametrize("ratio,suffix", [(1.5, "1p5"), (2, "2")])
@pytest.mark.parametrize("offset,expected", [(0.0, 0), (0.001, 1), (-0.001, 0)])
def test_ratio_threshold_is_strict(ratio, suffix, offset, expected):
    diagnostics = {}
    variant = f"product-default-loss-monitor-ratio-rebaseline-{suffix}"
    with loss_monitor_rebaseline_policy(
        variant=variant, diagnostics=diagnostics
    ) as states:
        m = monitor()
        current = 0.6 / ratio - offset
        observe(m, 0, current=current)
        assert states[m]["decrease_detector"] is None
        assert states[m]["frozen"] == (current if expected else 0.6)
    assert diagnostics["rebaseline_count"] == expected
    assert diagnostics["decrease_alarm_count"] == 0


def test_external_reset_updates_both_baselines_and_states_are_weak():
    with loss_monitor_rebaseline_policy(variant=_DECREASE, diagnostics={}) as states:
        a, b = monitor(), monitor(0.4)
        a.reset(baseline_loss_mean=0.2)
        assert states[a]["frozen"] == 0.2
        assert (
            states[a]["decrease_detector"].get_state_snapshot().baseline_loss_mean
            == 0.8
        )
        assert states[b]["frozen"] == 0.4
        del a
        gc.collect()
        assert len(states) == 1


def test_delay_is_measured_per_monitor_until_next_alarm():
    original = OverallAndTrueClassLossMonitor.observe_loss_after_label_observation

    def alarm(self, **arguments):
        result = original(self, **arguments)
        return replace(result, drift_detected=arguments["sample_index"] == 5)

    diagnostics = {}
    variant = "product-default-loss-monitor-ratio-rebaseline-2"
    with patch.object(
        OverallAndTrueClassLossMonitor, "observe_loss_after_label_observation", alarm
    ):
        with loss_monitor_rebaseline_policy(variant=variant, diagnostics=diagnostics):
            a, b = monitor(), monitor()
            observe(a, 0)
            observe(b, 0)
            for index in range(1, 6):
                observe(a, index)
    assert [
        e["samples_until_next_upward_alarm"] for e in diagnostics["rebaselines"]
    ] == [5, None]


def test_observe_only_runs_detector_without_changing_product_snapshot():
    expected = monitor()
    expected_observations = [observe(expected, i) for i in range(80)]
    diagnostics = {}
    original = OverallAndTrueClassLossMonitor.reset
    with loss_monitor_rebaseline_policy(variant=_OBSERVE_ONLY, diagnostics=diagnostics):
        actual = monitor()
        actual_observations = [observe(actual, i) for i in range(80)]
    assert expected_observations == actual_observations
    assert actual.get_state_snapshot() == expected.get_state_snapshot()
    assert diagnostics["decrease_alarm_count"] > 0
    assert diagnostics["rebaseline_count"] == 0
    assert OverallAndTrueClassLossMonitor.reset is original


@pytest.mark.parametrize("bad", [0.0, 0.009, 1.0, float("nan"), float("inf"), True])
def test_out_of_range_baselines_are_rejected_before_product_mutation(bad):
    with loss_monitor_rebaseline_policy(variant=_DECREASE, diagnostics={}):
        with pytest.raises(ValueError):
            monitor(bad)
        m = monitor()
        before = m.get_state_snapshot()
        with pytest.raises(ValueError):
            m.reset(baseline_loss_mean=bad)
        with pytest.raises(ValueError):
            observe(m, 0, current=bad)
        assert m.get_state_snapshot() == before


def test_adwin_and_preexisting_monitor_are_rejected():
    old = monitor()
    with loss_monitor_rebaseline_policy(variant=_DECREASE, diagnostics={}):
        with pytest.raises(ValueError, match="ADWIN"):
            monitor(detector="adwin")
        with pytest.raises(ValueError, match="前に作られた"):
            observe(old, 0)


def sweep_arguments(output):
    return [
        "--configurations",
        "fedsda",
        "--datasets",
        "sine2",
        "--seeds",
        "0",
        "--aggregation-intervals",
        "100",
        "--per-client-sample-count",
        "600",
        "--client-count",
        "10",
        "--workers",
        "1",
        "--output-directory",
        str(output),
        "--purpose",
        "IMPROVE-031の専用検証",
    ]


@pytest.mark.parametrize(
    "status,existing,code",
    [
        ("completed", False, 0),
        ("skipped_existing", True, 0),
        ("skipped_existing", False, 2),
        ("failed", False, 1),
    ],
)
def test_run_one_preserves_diagnostics_on_skip_and_failure(
    tmp_path, status, existing, code
):
    import federated_learning_experiments.sweep.experiment_sweep_command as command
    from tools import research_loss_monitor_rebaseline as entry

    path = tmp_path / entry._DIAGNOSTICS_NAME
    saved = dict(
        variant=_OBSERVE_ONLY, diagnostics={"rebaseline_count": 0, "sentinel": 42}
    )
    if existing:
        entry._write_json(path, saved)

    def sweep(arguments):
        entry._write_json(
            tmp_path / "sweep_record_fake.json",
            dict(
                run_outcomes=[
                    dict(
                        configuration_name="fedsda",
                        run_status=status,
                    )
                ]
            ),
        )
        return 1 if status == "failed" else 0

    with patch.object(command, "main", sweep):
        assert (
            entry.main(
                [
                    "run-one",
                    "--variant",
                    _OBSERVE_ONLY,
                    "--",
                    *sweep_arguments(tmp_path),
                ]
            )
            == code
        )
    if existing:
        assert entry._read_json(path) == saved
    else:
        assert path.exists() == (status == "completed")


@pytest.mark.parametrize(
    "replacement",
    [
        ["--workers", "2"],
        ["--datasets", "sine2", "circle2"],
        ["--seeds", "0-1"],
        ["--configurations", "fedavg"],
        ["--aggregation-intervals", "50", "100"],
        ["--work", "2"],
        ["--fedsda-fixed-share-time-scales", "50", "100"],
    ],
)
def test_run_one_rejects_multiple_runs_and_workers(replacement):
    from tools import research_loss_monitor_rebaseline as entry

    output = Path("_loss_monitor_rebaseline_invalid_no_output")
    assert not output.exists()
    arguments = sweep_arguments(output) + replacement
    try:
        code = entry.main(["run-one", "--variant", _DECREASE, "--", *arguments])
    except SystemExit as error:
        code = error.code
    assert code == 2
    assert not output.exists()


def test_single_run_plan_and_child_command_are_read_only():
    from tools import research_loss_monitor_rebaseline as entry

    output = Path("_loss_monitor_rebaseline_plan_no_output")
    arguments = SimpleNamespace(
        sample_count=600,
        client_count=10,
        purpose="専用検証",
    )
    command = entry._child_command(arguments, _DECREASE, "sine2", 100, 0, output)
    sweep = command[command.index("--") + 1 :]
    assert entry._validate_sweep_arguments(sweep) == output
    assert command[:4] == [
        entry.sys.executable,
        "-m",
        "tools.research_loss_monitor_rebaseline",
        "run-one",
    ]
    assert sweep[sweep.index("--workers") + 1] == "1"
    assert not output.exists()


def test_launch_uses_child_processes_continues_after_failure_and_records_status(
    tmp_path,
):
    from tools import research_loss_monitor_rebaseline as entry

    commands = []

    def child(command, **kwargs):
        commands.append(command)
        args = command[command.index("--") + 1 :]
        output = entry._validate_sweep_arguments(args)
        output.mkdir()
        status = "completed" if "s0" in output.name else "failed"
        entry._write_json(
            output / "sweep_record_fake.json",
            dict(
                run_outcomes=[
                    dict(
                        configuration_name="fedsda",
                        run_status=status,
                    )
                ]
            ),
        )
        return SimpleNamespace(returncode=0 if status == "completed" else 1)

    with (
        patch.object(entry.subprocess, "run", child),
        patch.object(entry, "_provenance", lambda: {}),
    ):
        assert (
            entry.main(
                [
                    "--output",
                    str(tmp_path),
                    "--variants",
                    _DECREASE,
                    "--datasets",
                    "sine2",
                    "--intervals",
                    "100",
                    "--seeds",
                    "0",
                    "1",
                    "--workers",
                    "2",
                    "--purpose",
                    "専用検証",
                ]
            )
            == 1
        )
    assert len(commands) == 2
    assert all(
        "--study" not in c and "--label" not in c and "--new-output" not in c
        for c in commands
    )
    record = entry._read_json(tmp_path / "launch.json")["launches"][-1]
    assert sorted(r["exit_code"] for r in record["run_outcomes"]) == [0, 1]
    assert (tmp_path / _DECREASE / "sine2-i100-s0.log").exists()


def test_launch_rejects_different_variant_before_spawning_any_child(tmp_path):
    from tools import research_loss_monitor_rebaseline as entry

    output = tmp_path / _DECREASE / "sine2-i100-s1"
    output.mkdir(parents=True)
    entry._write_json(output / entry._DIAGNOSTICS_NAME, dict(variant="product-default"))
    with patch.object(entry.subprocess, "run") as child:
        assert (
            entry.main(
                [
                    "--output",
                    str(tmp_path),
                    "--variants",
                    _DECREASE,
                    "--datasets",
                    "sine2",
                    "--intervals",
                    "100",
                    "--seeds",
                    "0",
                    "1",
                    "--workers",
                    "1",
                ]
            )
            == 2
        )
        child.assert_not_called()
    assert not (tmp_path / "launch.json").exists()


def test_artifact_comparison_includes_all_arrays_and_non_time_summary_items(tmp_path):
    from tools import research_loss_monitor_rebaseline as entry

    roots = [tmp_path / "expected", tmp_path / "actual"]
    for root in roots:
        directory = root / "fedsda/runs"
        directory.mkdir(parents=True)
        np.savez(
            directory / "run.npz",
            loss=np.array([float("nan"), 0.1]),
            label=np.array([1, 0]),
        )
        entry._write_json(
            directory / "run.json", dict(accuracy=0.5, duration_seconds=1, digest="abc")
        )
    assert entry.compare_run_artifacts(*roots)["array_count"] == 2
    summary = roots[1] / "fedsda/runs/run.json"
    entry._write_json(summary, dict(accuracy=0.5, duration_seconds=99, digest="abc"))
    entry.compare_run_artifacts(*roots)
    entry._write_json(
        summary, dict(accuracy=0.5, duration_seconds=99, digest="changed")
    )
    with pytest.raises(ValueError, match="要約"):
        entry.compare_run_artifacts(*roots)
    entry._write_json(summary, dict(accuracy=0.5, duration_seconds=99, digest="abc"))
    np.savez(
        summary.with_suffix(".npz"),
        loss=np.array([float("nan"), 0.2]),
        label=np.array([1, 0]),
    )
    with pytest.raises(ValueError, match="raw"):
        entry.compare_run_artifacts(*roots)


def test_artifact_comparison_finds_expected_runs_by_name_under_saved_results(tmp_path):
    from tools import research_loss_monitor_rebaseline as entry

    def write_run(directory, name, value):
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / f"{name}.npz", loss=np.array([value]))
        entry._write_json(directory / f"{name}.json", dict(accuracy=value))

    # 保存済みの結果: 別の実行のディレクトリの下にあり、照合に使わないrunも含む。
    saved = tmp_path / "saved"
    write_run(saved / "20261023_sweep/fedsda/runs", "mnist4_interval500_seed7_x", 0.5)
    write_run(saved / "20261023_sweep/fedsda/runs", "mnist4_interval500_seed8_y", 0.6)
    actual = tmp_path / "output"
    write_run(actual / _OBSERVE_ONLY / "mnist4-i500-s7/fedsda/runs", "mnist4_interval500_seed7_x", 0.5)
    assert entry.compare_run_artifacts(saved, actual) == dict(run_count=1, array_count=1)
    # 相手が見つからないrunは拒否する。
    write_run(actual / _OBSERVE_ONLY / "mnist4-i500-s9/fedsda/runs", "mnist4_interval500_seed9_z", 0.7)
    with pytest.raises(ValueError, match="相手"):
        entry.compare_run_artifacts(saved, actual)
    (actual / _OBSERVE_ONLY / "mnist4-i500-s9").rename(tmp_path / "unused")
    # 同じ名前で中身も同じrawの写し（例: 実行ごとの出力先と、まとめた置き場所）は、1つとして扱う。
    write_run(saved / "copy/fedsda/runs", "mnist4_interval500_seed7_x", 0.5)
    assert entry.compare_run_artifacts(saved, actual) == dict(run_count=1, array_count=1)
    # 同じ名前で中身の違うrawがあると、どれと比べるか決まらないので拒否する。
    write_run(saved / "other/fedsda/runs", "mnist4_interval500_seed7_x", 0.9)
    with pytest.raises(ValueError, match="中身の違う"):
        entry.compare_run_artifacts(saved, actual)


def test_artifact_comparison_ignores_code_commit_but_not_baseline_version(tmp_path):
    from tools import research_loss_monitor_rebaseline as entry

    def write_run(root, commit, baseline_version):
        directory = root / "fedsda/runs"
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "run.npz", loss=np.array([0.1]))
        entry._write_json(
            directory / "run.json",
            dict(
                accuracy=0.5,
                run_provenance=dict(
                    code_commit=commit,
                    has_uncommitted_changes=commit == "b",
                    code_revision_unavailable_reason=None,
                    behavior_baseline_version=baseline_version,
                ),
            ),
        )

    write_run(tmp_path / "saved", "a", 12)
    write_run(tmp_path / "actual", "b", 12)
    assert entry.compare_run_artifacts(tmp_path / "saved", tmp_path / "actual")["run_count"] == 1
    write_run(tmp_path / "actual", "b", 13)
    with pytest.raises(ValueError, match="要約"):
        entry.compare_run_artifacts(tmp_path / "saved", tmp_path / "actual")


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["sine2", "circle2", "mnist4"])
def test_observe_only_product_sweep_matches_all_saved_items(tmp_path, dataset):
    from tools import research_loss_monitor_rebaseline as entry

    expected, actual = tmp_path / "product", tmp_path / "observe-only"
    for variant, output in (("product-default", expected), (_OBSERVE_ONLY, actual)):
        args = sweep_arguments(output)
        args[args.index("--datasets") + 1] = dataset
        assert entry.main(["run-one", "--variant", variant, "--", *args]) == 0
    assert entry.compare_run_artifacts(expected, actual)["run_count"] == 1
    diagnostics = json.loads(
        (actual / entry._DIAGNOSTICS_NAME).read_text(encoding="utf-8")
    )
    assert diagnostics["diagnostics"]["monitor_count"] == 10
    assert diagnostics["diagnostics"]["rebaseline_count"] == 0


@pytest.mark.slow
def test_observe_only_research_run_matches_product_including_parameter_digest():
    from tools.research_provisional_candidate_expert.run import _execute_job

    job = dict(
        dataset="sine2",
        interval=100,
        seed=0,
        sample_count=600,
        check=False,
        record_prediction_period_metrics=True,
    )
    expected = _execute_job({**job, "variant": "product-default"})
    actual = _execute_job({**job, "variant": _OBSERVE_ONLY})
    assert actual.pop("loss_monitor_rebaseline")["rebaseline_count"] == 0
    for record in (actual, expected):
        for key in ("variant", "seconds"):
            record.pop(key)
    assert actual == expected
