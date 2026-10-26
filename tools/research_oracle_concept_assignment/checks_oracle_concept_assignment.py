"""IMPROVE-030の専用検証。pytestでこのファイルを明示する。"""

import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from tools.research_oracle_concept_assignment import run as entry
from tools.research_oracle_concept_assignment.policy import (
    ORACLE_VARIANT,
    _ConceptChangeAlarm,
    _oracle_pair_scores,
    _unique_majority,
    _select_destination_model_id,
)
from tools.research_loss_monitor_rebaseline_metrics import _table, run_metrics


class _WorkspaceTemporaryPaths:
    def mktemp(self, name):
        # Windowsのsandboxではpytestのmode=0o700の一時領域へ再アクセスできない。
        # workspace内の通常のディレクトリを使い、検証の成果物を残す。
        path = (
            Path(__file__).resolve().parents[2]
            / "results/oracle-concept-assignment/validation-artifacts"
            / f"{name}-{uuid4().hex}"
        )
        path.mkdir(parents=True)
        return path


@pytest.fixture(scope="module")
def tmp_path_factory():
    return _WorkspaceTemporaryPaths()


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("check")


def _arguments(root, variants, *, seeds=(0,), workers=1):
    return [
        "--output",
        str(root),
        "--datasets",
        "sea2",
        "--intervals",
        "100",
        "--seeds",
        *map(str, seeds),
        "--sample-count",
        "600",
        "--client-count",
        "10",
        "--variants",
        *variants,
        "--workers",
        str(workers),
    ]


@pytest.mark.parametrize("gap", [1, 2, 3, 4])
def test_close_changes_fail(gap):
    alarm = _ConceptChangeAlarm()
    alarm.observe(sample_index=0, concept_id=0)
    alarm.observe(sample_index=10, concept_id=1)
    with pytest.raises(ValueError, match="5標本以内"):
        alarm.observe(sample_index=10 + gap, concept_id=0)


def test_alarm_is_on_fifth_sample_and_only_once():
    alarm = _ConceptChangeAlarm()
    raised = [i for i in range(25) if alarm.observe(sample_index=i, concept_id=int(i >= 10))[1]]
    assert raised == [14]


@pytest.mark.parametrize("counts, expected", [({}, None), ({0: 4, 1: 4}, None), ({0: 5, 1: 4}, 0)])
def test_unique_majority(counts, expected):
    assert _unique_majority(counts) == expected


def test_pair_scores_replace_same_different_missing_and_tie():
    scores = _oracle_pair_scores(
        model_ids=(0, 1, 2, 3), majority_concept_ids={0: 0, 1: 0, 2: 1}, threshold=0.2
    )
    assert scores[(0, 1)] == 0.0
    assert all(score > 0.2 for pair, score in scores.items() if pair != (0, 1))


def test_destination_uses_current_held_ids_majority_and_sample_count():
    counts = {0: {0: 80}, 7: {1: 10}, 8: {1: 20}, 9: {0: 30, 1: 30}, 10: {1: 100}}
    store = SimpleNamespace(
        get_model_assigned_sample_concept_counts=lambda model_id: counts.get(model_id, {})
    )
    assert (
        _select_destination_model_id(
            held_model_ids=(0, 7, 8, 9, 11), assignment_counts_store=store, concept_id=1
        )
        == 8
    )
    # 非保有の10と、多数派の同数の9を使わない。同期後の保有順を毎回読む。
    assert (
        _select_destination_model_id(
            held_model_ids=(0, 9, 11), assignment_counts_store=store, concept_id=1
        )
        is None
    )
    assert (
        _select_destination_model_id(
            held_model_ids=(7, 0), assignment_counts_store=store, concept_id=0
        )
        == 0
    )


def test_print_plan_creates_no_output(tmp_path):
    output = tmp_path / "plan"
    assert entry.main([*_arguments(output, [ORACLE_VARIANT]), "--print-plan"]) == 0
    assert not output.exists()


@pytest.fixture(scope="module")
def oracle_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("oracle")
    assert entry.main(_arguments(root, [ORACLE_VARIANT])) == 0
    (raw,) = (root / ORACLE_VARIANT / "fedsda/runs").glob("*.npz")
    (record,) = (root / ORACLE_VARIANT / entry._DIAGNOSTICS_DIRECTORY_NAME).glob("*.json")
    return raw, json.loads(record.read_text(encoding="utf-8"))["diagnostics"]


def test_small_run_alarm_assignment_and_consolidation(oracle_run):
    raw, diagnostics = oracle_run
    with np.load(raw, allow_pickle=False) as z:
        concepts = z["sample_prediction.observed_concept_id"]
        expected = sorted(
            (c, int(i + 4))
            for c, row in enumerate(concepts)
            for i in np.flatnonzero(row[1:] != row[:-1]) + 1
            if i + 4 < len(row)
        )
        actual = sorted(
            (r["client_id"], r["alarm_sample_index"]) for r in _table(z, "loss_change_alarm")
        )
        assert actual == expected and actual
        selections = _table(z, "alarm_interval_assignment_selection")
        assert all(r["change_interval_sample_count"] == 5 for r in selections)
        outcomes = Counter(r["adaptation_outcome"] for r in _table(z, "adaptation"))
        assert outcomes["alarm_change_interval_too_short"] == 0
    assert len(diagnostics["alarms"]) == len(expected)
    assert diagnostics["suppressed_monitor_signals"] > 0
    assert diagnostics["mixed_assignment_count"] == 0
    assert (
        diagnostics["assigned_sample_count"] + diagnostics["pending_sample_count"]
        == diagnostics["processed_sample_count"]
    )
    assert diagnostics["processed_sample_count"] == concepts.size
    assert all(
        len(counts) == 1
        for store in diagnostics["final_assignment_concept_counts"]
        for counts in store.values()
    )
    assert run_metrics(raw)["persistent"] == 0
    assert diagnostics["consolidations"]
    for event in diagnostics["consolidations"]:
        majority = event["majority_concept_ids"]
        for cluster in event["model_clusters"]:
            if len(cluster) > 1:
                assert len({majority[str(model_id)] for model_id in cluster}) == 1


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    from federated_learning_experiments.sweep.experiment_sweep_command import (
        main as product_main,
    )

    tmp_path = tmp_path_factory.mktemp("default")
    assert entry.main(_arguments(tmp_path / "entry", ["product-default"])) == 0
    arguments = entry._run_parser().parse_args(_arguments(tmp_path / "unused", ["product-default"]))
    assert (
        product_main(entry._sweep_arguments(arguments, "product-default", tmp_path / "direct")) == 0
    )
    return tmp_path


def test_default_entry_matches_direct_product_all_records(default_run):
    tmp_path = default_run
    result = entry.compare_run_artifacts(tmp_path / "direct", tmp_path / "entry")
    assert result["run_count"] == 1 and result["array_count"] > 100


def test_analysis_checks_summary_and_writes_three_accuracies(default_run, oracle_run, tmp_path):
    from tools.research_oracle_concept_assignment.analyze import analyze

    raw, _ = oracle_run
    shutil.copytree(default_run / "entry/product-default", tmp_path / "product-default")
    shutil.copytree(raw.parents[2], tmp_path / ORACLE_VARIANT)
    report = analyze(tmp_path, expected_pairs=1)
    assert "oracle a" in report
    assert (tmp_path / "analysis_accuracy_periods.csv").exists()
    result = json.loads(
        (tmp_path / "analysis_oracle_concept_assignment.json").read_text(encoding="utf-8")
    )
    assert result["rows"][0]["oracle"]["persistent"] == 0
    assert result["decisions"]["sea2"]["seed_standard_deviation_pp"] is None


def test_spawn_workers_apply_oracle(tmp_path, capsys):
    exit_code = entry.main(_arguments(tmp_path, [ORACLE_VARIANT], seeds=(0, 1), workers=2))
    output = capsys.readouterr()
    if exit_code == 2 and "[WinError 5]" in output.err:
        pytest.skip("WindowsのsandboxがspawnのIPCを拒否したため未検証")
    assert exit_code == 0, output.err
    records = list((tmp_path / ORACLE_VARIANT / entry._DIAGNOSTICS_DIRECTORY_NAME).glob("*.json"))
    assert len(records) == 2
    for record in records:
        diagnostics = json.loads(record.read_text(encoding="utf-8"))["diagnostics"]
        assert diagnostics["alarms"]
        assert diagnostics["mixed_assignment_count"] == 0
