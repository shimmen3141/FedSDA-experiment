"""FedHASP基準版16のOracle診断の専用検証。"""

import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from tools.research_loss_monitor_rebaseline_metrics import _table, run_metrics
from tools.research_oracle_concept_assignment import run as entry
from tools.research_oracle_concept_assignment.policy import (
    ORACLE_CHANGE_ALARM_VARIANT,
    ORACLE_VARIANT,
    _ConceptChangeAlarm,
    _oracle_pair_scores,
    _select_destination_model_id,
    _unique_majority,
)


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


def _assert_small_run_alarm_assignment_and_consolidation(oracle_run):
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


def test_small_run_alarm_assignment_and_consolidation(oracle_run):
    _assert_small_run_alarm_assignment_and_consolidation(oracle_run)
    assert oracle_run[1]["suppressed_monitor_signals"] > 0


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


def test_change_alarm_variant_replaces_only_alarm(tmp_path):
    from tools.research_oracle_concept_assignment.analyze import analyze

    root = tmp_path / "alarm"
    assert entry.main(_arguments(root, [ORACLE_CHANGE_ALARM_VARIANT])) == 0
    (raw,) = (root / ORACLE_CHANGE_ALARM_VARIANT / "fedsda/runs").glob("*.npz")
    (record,) = (root / ORACLE_CHANGE_ALARM_VARIANT / entry._DIAGNOSTICS_DIRECTORY_NAME).glob(
        "*.json"
    )
    diagnostics = json.loads(record.read_text(encoding="utf-8"))["diagnostics"]
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
        # 帰属の選択は製品のMDLのまま（今のモデルの継続は保留の全長、それ以外は5の倍数の区間）。
        selections = _table(z, "alarm_interval_assignment_selection")
        assert selections and all(
            r["selected_hypothesis"] == "current_model" or r["change_interval_sample_count"] % 5 == 0
            for r in selections
        )
    assert diagnostics["selections"] == [] and diagnostics["consolidations"] == []
    assert (
        diagnostics["assigned_sample_count"] + diagnostics["pending_sample_count"]
        == diagnostics["processed_sample_count"]
    )
    # 集計は、全体の正解のoracleと別の名前へ保存する。
    analysis_root = tmp_path / "analysis"
    shutil.copytree(raw.parents[2], analysis_root / ORACLE_CHANGE_ALARM_VARIANT)
    default_root = tmp_path / "default"
    assert entry.main(_arguments(default_root, ["product-default"])) == 0
    shutil.copytree(default_root / "product-default", analysis_root / "product-default")
    analyze(analysis_root, expected_pairs=1, variant=ORACLE_CHANGE_ALARM_VARIANT)
    assert (analysis_root / "analysis_oracle_change_alarm.md").exists()
    assert (analysis_root / "analysis_accuracy_periods_oracle_change_alarm.csv").exists()
    assert not (analysis_root / "analysis_oracle_concept_assignment.md").exists()


def test_delayed_alarm_is_on_fiftieth_sample_and_rejects_close_changes():
    alarm = _ConceptChangeAlarm(alarm_sample_count=50)
    raised = [i for i in range(100) if alarm.observe(sample_index=i, concept_id=int(i >= 10))[1]]
    assert raised == [59]
    alarm = _ConceptChangeAlarm(alarm_sample_count=50)
    alarm.observe(sample_index=0, concept_id=0)
    alarm.observe(sample_index=10, concept_id=1)
    with pytest.raises(ValueError, match="50標本以内"):
        alarm.observe(sample_index=59, concept_id=0)


@pytest.mark.parametrize("fault", ["short", "shift", "keys", "missing", "nan", "inf", "name"])
def test_interval_validation_rejects_broken_inputs(fault):
    from tools.research_oracle_concept_assignment.policy import (
        _validate_new_model_interval_dictionaries,
        _validate_pending_change_interval,
    )

    pending = tuple(SimpleNamespace(sample_index=i) for i in range(10, 15))
    lengths = {5: 1.0}
    names = {5: "reinitialized_learner"}
    if fault == "short":
        pending = pending[1:]
    elif fault == "shift":
        pending = tuple(SimpleNamespace(sample_index=i) for i in range(11, 16))
    elif fault == "keys":
        names[6] = "warm_started_learner"
    elif fault == "missing":
        lengths, names = {6: 1.0}, {6: "warm_started_learner"}
    elif fault in ("nan", "inf"):
        lengths[5] = float(fault)
    else:
        names[5] = float("nan")
    with pytest.raises(ValueError):
        _validate_pending_change_interval(
            pending_sample_observations=pending, change_sample_index=10
        )
        _validate_new_model_interval_dictionaries(lengths=lengths, names=names)


@pytest.mark.parametrize(
    "dataset", ["sea2", "mnist4", "fmnist2", "fmnist4", "rotmnist2", "rotmnist4"]
)
def test_v16_small_full_run(dataset, tmp_path):
    arguments = _arguments(tmp_path / dataset, [ORACLE_VARIANT])
    arguments[arguments.index("sea2")] = dataset
    arguments[arguments.index("100")] = "50"
    # 計画確認は読取り専用。全体runの直前に同じ条件を確認する。
    assert entry.main([*arguments, "--print-plan"]) == 0
    assert not (tmp_path / dataset).exists()
    assert entry.main(arguments) == 0
    directory = tmp_path / dataset / ORACLE_VARIANT
    (raw,) = (directory / "fedsda/runs").glob("*.npz")
    (record,) = (directory / entry._DIAGNOSTICS_DIRECTORY_NAME).glob("*.json")
    diagnostics = json.loads(record.read_text(encoding="utf-8"))["diagnostics"]
    assert diagnostics["changes"] and diagnostics["alarms"]
    assert diagnostics["interval_validation_count"] == len(diagnostics["alarms"])
    assert len(diagnostics["selections"]) == len(diagnostics["alarms"])
    events = {(row["client_id"], row["sample_index"]) for row in diagnostics["alarms"]}
    computed = set()
    for evaluation in diagnostics["learner_code_length_evaluations"]:
        assert set(evaluation["code_lengths"]) == {"reinitialized_learner", "warm_started_learner"}
        assert all(np.isfinite(value) for value in evaluation["code_lengths"].values())
        computed.add((evaluation["client_id"], evaluation["sample_index"]))
    assert computed == events
    # 既存の全体runと同じ帰属・統合の不変条件を適用する。
    _assert_small_run_alarm_assignment_and_consolidation((raw, diagnostics))


def _known_three_variant_rows():
    from tools.research_oracle_concept_assignment.analyze import _THREE_VARIANTS

    rows = []
    for dataset, base in (("sea2", 0.5), ("sea4", 0.6)):
        for seed, difference in enumerate((-0.01, 0.0, 0.02)):
            row = dict(dataset=dataset, interval=50, seed=seed)
            for variant, accuracy in zip(
                _THREE_VARIANTS, (base, base + difference / 2, base + difference), strict=True
            ):
                row[variant] = dict(
                    accuracy_a=accuracy,
                    post_change_accuracy=0.25,
                    other_accuracy=0.75,
                    new_model_adoptions=seed,
                    uploaded_bytes=10,
                    downloaded_bytes=20,
                    client_forward_mac=30,
                    client_backward_mac=40,
                    server_mac=50,
                    elapsed_seconds=1,
                )
            rows.append(row)
    return rows


def test_three_variant_statistics_match_hand_calculation():
    from tools.research_oracle_concept_assignment.analyze import (
        _summarize_three_variant_rows,
    )

    summaries = _summarize_three_variant_rows(_known_three_variant_rows())
    cell = [row for row in summaries if row["dataset"] == "sea2"]
    difference = next(row for row in cell if row["name"] == f"{ORACLE_VARIANT} - product-default")
    # 差[-1,0,2]pp: 平均1/3、標本分散7/3、t(.975,2)=4.3026527299。
    assert difference["mean"] == pytest.approx(1 / 3)
    assert difference["sd"] == pytest.approx(np.sqrt(7 / 3))
    half = 4.3026527299 * np.sqrt(7) / 3
    assert difference["ci_low"] == pytest.approx(1 / 3 - half)
    assert difference["ci_high"] == pytest.approx(1 / 3 + half)
    assert (difference["wins"], difference["losses"], difference["ties"]) == (1, 1, 1)
    for row in cell:
        if row["kind"] == "paired_difference" and row is not difference:
            assert row["mean"] == pytest.approx(1 / 6)
            assert row["ci_low"] == pytest.approx((1 / 3 - half) / 2)
            assert row["ci_high"] == pytest.approx((1 / 3 + half) / 2)
            assert (row["wins"], row["losses"], row["ties"]) == (1, 1, 1)
    oracle = next(
        row for row in cell if row["name"] == ORACLE_VARIANT and row["metric"] == "accuracy_a"
    )
    assert oracle["mean"] == pytest.approx(50 + 1 / 3)
    assert oracle["sd"] == pytest.approx(np.sqrt(7 / 3))
    for variant, mean, sd in (
        ("product-default", 50, 0),
        (ORACLE_CHANGE_ALARM_VARIANT, 50 + 1 / 6, np.sqrt(7 / 3) / 2),
    ):
        value = next(
            row for row in cell if row["name"] == variant and row["metric"] == "accuracy_a"
        )
        assert value["mean"] == pytest.approx(mean)
        assert value["sd"] == pytest.approx(sd)
    for row in cell:
        if row["kind"] == "variant" and row["metric"] == "new_model_adoptions":
            assert row["mean"] == 1 and row["sd"] == 1
    all_oracle = next(
        row
        for row in summaries
        if row["dataset"] == "all"
        and row["name"] == ORACLE_VARIANT
        and row["metric"] == "accuracy_a"
    )
    assert all_oracle["mean"] == pytest.approx(55 + 1 / 3)
    assert all_oracle["sd"] == pytest.approx(np.sqrt(7 / 3))
    assert all_oracle["seed_count"] == 3


def test_post_change_accuracy_uses_union_and_excludes_initial_concept():
    from tools.research_oracle_concept_assignment.analyze import _post_change_accuracies

    values = _post_change_accuracies(
        np.array([[0, 0, 1, 1, 2, 2, 2, 2]]), np.array([[1, 1, 0, 0, 1, 1, 0, 1]]), 3
    )
    assert values == {"post_change_accuracy": 2 / 5, "other_accuracy": 1.0}
    assert _post_change_accuracies(np.zeros((1, 3)), np.ones((1, 3)), 3) == {
        "post_change_accuracy": None,
        "other_accuracy": 1.0,
    }


@pytest.mark.parametrize(
    "fault", [None, "missing", "identity", "concept", "label", "extra", "seeds"]
)
def test_three_variant_loading_and_outputs(tmp_path, monkeypatch, fault):
    from tools.research_oracle_concept_assignment import analyze as analysis

    rows = _known_three_variant_rows()
    paths = {
        variant: {
            (row["dataset"], row["interval"], row["seed"]): tmp_path
            / variant
            / f"{row['dataset']}-{row['seed']}.npz"
            for row in rows
        }
        for variant in analysis._THREE_VARIANTS
    }
    if fault == "missing":
        paths[ORACLE_VARIANT].pop(("sea2", 50, 0))
    if fault == "seeds":
        for values in paths.values():
            values[("sea2", 50, 4)] = values.pop(("sea2", 50, 0))
    if fault == "extra":
        directory = tmp_path / "unexpected-variant"
        directory.mkdir()
        (directory / "raw.npz").touch()
    monkeypatch.setattr(analysis, "_collect", lambda directories: paths[directories[0].name])

    def read(path, directory, variant, key):
        row = next(row for row in rows if (row["dataset"], row["interval"], row["seed"]) == key)
        broken = variant == ORACLE_VARIANT and key == ("sea2", 50, 0)
        identity = (*key, 11 if fault == "identity" and broken else 10, 600)
        concept = np.array([[int(fault == "concept" and broken)]])
        label = np.array([[int(fault == "label" and broken)]])
        return identity, concept, label, row[variant]

    monkeypatch.setattr(analysis, "_read_three_variant_run", read)
    if fault:
        with pytest.raises(ValueError):
            analysis.analyze_three_variants(tmp_path, expected_cells=2, expected_seeds=3)
        assert not (tmp_path / "analysis_three_variants.csv").exists()
    else:
        report = analysis.analyze_three_variants(tmp_path, expected_cells=2, expected_seeds=3)
        assert "95% CI" in report
        assert (tmp_path / "analysis_three_variants.csv").exists()


def test_three_variant_analysis_with_real_raw(default_run, oracle_run, tmp_path):
    from tools.research_oracle_concept_assignment.analyze import analyze_three_variants

    raw, _ = oracle_run
    shutil.copytree(default_run / "entry/product-default", tmp_path / "product-default")
    shutil.copytree(raw.parents[2], tmp_path / ORACLE_VARIANT)
    assert entry.main(_arguments(tmp_path, [ORACLE_CHANGE_ALARM_VARIANT])) == 0
    assert "基準版16" in analyze_three_variants(tmp_path, expected_cells=1, expected_seeds=1)
    # ファイル名が同じでも入力のclient数が違えば拒否する。
    summary_path = next((tmp_path / ORACLE_VARIANT / "fedsda/runs").glob("*.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_settings"]["execution_settings"]["experiment_run_conditions"]["client_count"] = 11
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="client数"):
        analyze_three_variants(tmp_path, expected_cells=1, expected_seeds=1)


@pytest.mark.parametrize(
    "field",
    [
        "dataset_name",
        "random_seed",
        "server_aggregation_interval_per_client_samples",
        "client_count",
        "per_client_sample_count",
        "variant",
    ],
)
def test_three_variant_reader_rejects_actual_metadata_mismatch(tmp_path, monkeypatch, field):
    from tools.research_oracle_concept_assignment import analyze as analysis

    path = tmp_path / "run.npz"
    conditions = dict(
        dataset_name="sea2",
        random_seed=0,
        server_aggregation_interval_per_client_samples=50,
        client_count=2,
        per_client_sample_count=6,
    )
    if field != "variant":
        conditions[field] = "sea4" if field == "dataset_name" else conditions[field] + 1
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(run_settings=dict(execution_settings=dict(experiment_run_conditions=conditions)))
        ),
        encoding="utf-8",
    )
    diagnostics = tmp_path / "oracle_concept_assignment_diagnostics"
    diagnostics.mkdir()
    (diagnostics / "run.json").write_text(
        json.dumps(dict(variant=ORACLE_VARIANT if field == "variant" else "product-default")),
        encoding="utf-8",
    )
    np.savez(
        path,
        **{
            "sample_prediction.observed_concept_id": np.zeros((2, 6)),
            "sample_prediction.observed_class_id": np.zeros((2, 6)),
            "sample_prediction.combined_prediction_is_correct": np.ones((2, 6)),
        },
    )
    monkeypatch.setattr(analysis, "_read_run", lambda *a, **kw: {})
    with pytest.raises(ValueError):
        analysis._read_three_variant_run(path, tmp_path, "product-default", ("sea2", 50, 0))
