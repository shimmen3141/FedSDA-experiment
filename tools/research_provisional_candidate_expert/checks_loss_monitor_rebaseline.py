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


def run_arguments(output, *, variants, seeds=("0",), workers="1", dataset="sine2"):
    return [
        "--output",
        str(output),
        "--datasets",
        dataset,
        "--intervals",
        "100",
        "--seeds",
        *seeds,
        "--sample-count",
        "600",
        "--variants",
        *variants,
        "--workers",
        workers,
        "--purpose",
        "IMPROVE-031の専用検証",
    ]


@pytest.mark.parametrize("status,written", [("completed", True), ("failed", False)])
def test_execute_run_with_variant_enables_policy_and_writes_diagnostics_only_when_completed(
    tmp_path, monkeypatch, status, written
):
    from tools import research_loss_monitor_rebaseline as entry

    original_init = OverallAndTrueClassLossMonitor.__init__
    seen = {}

    def product_run(*, planned_run, output_directory, run_provenance):
        # 製品のrunの実行の中では、変種の包みが有効になっている。
        seen["patched"] = OverallAndTrueClassLossMonitor.__init__ is not original_init
        return SimpleNamespace(run_status=status)

    monkeypatch.setenv(entry._VARIANT_ENVIRONMENT_VARIABLE, _DECREASE)
    with patch.object(entry, "_execute_planned_experiment_run", product_run):
        outcome = entry._execute_run_with_variant(
            planned_run=SimpleNamespace(run_identifier="sine2_interval100_seed0_x"),
            output_directory=tmp_path,
            run_provenance=None,
        )
    assert outcome.run_status == status
    assert seen["patched"]
    assert OverallAndTrueClassLossMonitor.__init__ is original_init
    path = tmp_path / entry._DIAGNOSTICS_DIRECTORY_NAME / "sine2_interval100_seed0_x.json"
    assert path.exists() is written
    if written:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["variant"] == _DECREASE
        assert record["run_identifier"] == "sine2_interval100_seed0_x"


def test_execute_run_with_variant_rejects_unknown_variant(monkeypatch):
    from tools import research_loss_monitor_rebaseline as entry

    monkeypatch.setenv(entry._VARIANT_ENVIRONMENT_VARIABLE, "unknown")
    with pytest.raises(ValueError, match="変種"):
        entry._execute_run_with_variant(planned_run=None, output_directory=None, run_provenance=None)


def test_run_calls_product_sweep_per_variant_with_patch_and_restores(tmp_path, monkeypatch):
    import federated_learning_experiments.sweep.experiment_sweep_command as command
    from tools import research_loss_monitor_rebaseline as entry

    monkeypatch.delenv(entry._VARIANT_ENVIRONMENT_VARIABLE, raising=False)
    calls = []
    original_execute = entry.experiment_sweep_execution.execute_planned_experiment_run

    def product_main(arguments):
        output = Path(arguments[arguments.index("--output-directory") + 1])
        calls.append(
            dict(
                arguments=list(arguments),
                patched=entry.experiment_sweep_execution.execute_planned_experiment_run
                is entry._execute_run_with_variant,
                variant=entry.os.environ[entry._VARIANT_ENVIRONMENT_VARIABLE],
                output=output,
            )
        )
        output.mkdir(parents=True)
        return 0 if len(calls) == 1 else 1

    with (
        patch.object(command, "main", product_main),
        patch.object(entry, "_provenance", lambda: {}),
    ):
        code = entry.main(run_arguments(tmp_path, variants=[_DECREASE, _OBSERVE_ONLY], seeds=("0", "1"), workers="4"))
    # 終了コードは変種ごとの製品のコマンドの終了コードの最大。
    assert code == 1
    assert [c["variant"] for c in calls] == [_DECREASE, _OBSERVE_ONLY]
    assert all(c["patched"] for c in calls)
    assert entry.experiment_sweep_execution.execute_planned_experiment_run is original_execute
    assert entry._VARIANT_ENVIRONMENT_VARIABLE not in entry.os.environ
    first = calls[0]["arguments"]
    assert first[first.index("--seeds") + 1 : first.index("--aggregation-intervals")] == ["0", "1"]
    assert first[first.index("--workers") + 1] == "4"
    assert first[first.index("--per-client-sample-count") + 1] == "600"
    assert calls[0]["output"] == tmp_path.resolve() / _DECREASE
    assert "--study" not in first and "--label" not in first
    for variant in (_DECREASE, _OBSERVE_ONLY):
        record = json.loads((tmp_path / variant / entry._VARIANT_RECORD_NAME).read_text(encoding="utf-8"))
        assert record["variant"] == variant


def test_run_rejects_other_variant_record_before_calling_product(tmp_path):
    import federated_learning_experiments.sweep.experiment_sweep_command as command
    from tools import research_loss_monitor_rebaseline as entry

    output = tmp_path / _DECREASE
    output.mkdir()
    entry._write_json(output / entry._VARIANT_RECORD_NAME, dict(variant="product-default"))
    with patch.object(command, "main") as product_main:
        assert entry.main(run_arguments(tmp_path, variants=[_DECREASE])) == 2
        product_main.assert_not_called()


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

    arguments = run_arguments(tmp_path, variants=["product-default", _OBSERVE_ONLY], dataset=dataset)
    assert entry.main(arguments) == 0
    assert entry.compare_run_artifacts(tmp_path / "product-default", tmp_path / _OBSERVE_ONLY)["run_count"] == 1
    (record,) = (tmp_path / _OBSERVE_ONLY / entry._DIAGNOSTICS_DIRECTORY_NAME).glob("*.json")
    diagnostics = json.loads(record.read_text(encoding="utf-8"))["diagnostics"]
    assert diagnostics["monitor_count"] == 10
    assert diagnostics["rebaseline_count"] == 0


@pytest.mark.slow
def test_parallel_workers_apply_variant_and_write_diagnostics(tmp_path):
    from tools import research_loss_monitor_rebaseline as entry

    # 2 seed・2 workerで、製品の掃引の並列の経路（spawnのworker）を通す。
    assert entry.main(run_arguments(tmp_path, variants=[_DECREASE], seeds=("0", "1"), workers="2")) == 0
    runs = sorted((tmp_path / _DECREASE / "fedsda/runs").glob("*.npz"))
    records = sorted((tmp_path / _DECREASE / entry._DIAGNOSTICS_DIRECTORY_NAME).glob("*.json"))
    assert [r.stem for r in records] == [r.stem for r in runs] and len(runs) == 2
    diagnostics = [json.loads(r.read_text(encoding="utf-8"))["diagnostics"] for r in records]
    assert all(d["monitor_count"] == 10 for d in diagnostics)
    # workerの中でも変種が働き、基準の再設定が起きている。
    assert sum(d["rebaseline_count"] for d in diagnostics) > 0


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
