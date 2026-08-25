import csv
import os
import sys
from types import SimpleNamespace

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import run_pareto_sweep as sweep
from federated_drift_experiment import config


def test_cli_help_groups_related_sweep_options():
    help_text = sweep.build_parser().format_help()

    assert "FedSDAの手法・掃引" in help_text
    assert "--fixed-adwin-delta" in help_text
    assert "--fixed-aggregation-interval" in help_text
    assert "--aggregation-intervals" in help_text
    assert "--model-reuse-policy" not in help_text
    assert "FedDriftの手法・掃引" in help_text
    assert "--fixed-feddrift-distance-threshold" in help_text
    assert "--feddrift-detection-batch-sizes" in help_text
    assert "既存CSVの再描画" in help_text
    assert "他の実験設定は無視" in help_text
    assert "--plot-x-metric" in help_text
    assert "--plot-sweep-kind" in help_text


def test_large_or_deprecated_settings_are_opt_in_for_default_sweep():
    parser = sweep.build_parser()
    defaults = parser.parse_args([])
    assert defaults.datasets == ["sea4", "circle2", "sine2"]
    assert defaults.agg_sweep == [50, 100, 200, 500]
    assert defaults.batches == [50, 100, 200, 500]
    assert defaults.concept_schedule == "random"
    assert defaults.shared_backbone_training == "sequential"
    assert defaults.shared_backbone_gradient_strategy == "mean"
    assert defaults.shared_backbone_routing_recalibration == "none"
    assert defaults.soft_routing_context == "global"
    assert defaults.soft_routing_top_combination == "leader"
    assert defaults.soft_routing_meta_loss == "zero_one"
    assert not defaults.routing_archive_shadow_diagnostics
    assert defaults.shared_adapter_rank == config.SHARED_ADAPTER_RANK
    assert defaults.duplicate_policy == "error"
    selected = parser.parse_args([
        "--datasets", "sea2", "mnist2", "mnist4",
        "--concept-schedule", "feddrift_fixed",
    ])
    assert selected.datasets == ["sea2", "mnist2", "mnist4"]
    assert selected.concept_schedule == "feddrift_fixed"
    assert parser.parse_args([
        "--routing-archive-shadow-diagnostics"
    ]).routing_archive_shadow_diagnostics


def test_long_experiment_slug_is_shortened_with_stable_hash():
    first = sweep._experiment_slug(
        ["sea2", "sea4", "circle2", "sine2", "mnist2", "mnist4"],
        list(range(10)), 5000, tag="very-long-experiment-name-" * 8,
    )
    second = sweep._experiment_slug(
        ["sea2", "sea4", "circle2", "sine2", "mnist2", "mnist4"],
        list(range(10)), 5000, tag="very-long-experiment-name-" * 7 + "other",
    )

    assert len(first) <= 72
    assert first != second


def test_parallel_workers_are_explicit_and_positive():
    parser = sweep.build_parser()
    assert parser.parse_args([]).workers == 1
    with pytest.raises(SystemExit):
        sweep.main(["--workers", "0", "--print-plan"])


def test_parallel_sweep_returns_rows_in_plan_order(monkeypatch):
    experiments = [
        SimpleNamespace(
            dataset="sea4", mode="Oblivious", seed=seed,
            sweep_parameter=None, sweep_value=None,
        )
        for seed in range(3)
    ]
    plan = SimpleNamespace(iter_experiments=lambda: iter(experiments))

    class FakeFuture:
        def __init__(self, row):
            self.row = row

        def result(self):
            return self.row

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, experiment, raw_dir):
            return FakeFuture({
                "seed": experiment.seed,
                "stable_accuracy": 0.5,
                "comm_models_total": 0,
                "final_model_count": 1,
            })

    monkeypatch.setattr(sweep, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        sweep, "as_completed", lambda futures: reversed(list(futures)),
    )
    monkeypatch.setattr(
        sweep.multiprocessing, "get_context", lambda method: method,
    )

    rows = sweep.run_sweep_plan(
        plan, workers=2, runtime_config={},
    )

    assert [row["seed"] for row in rows] == [0, 1, 2]


def test_parallel_sweep_submits_expensive_runs_first_but_keeps_plan_order(
    monkeypatch,
):
    experiments = [
        SimpleNamespace(
            dataset=dataset, mode="FedSDA_NoCached_ClassESR", seed=seed,
            sweep_parameter=sweep.AGGREGATION_INTERVAL, sweep_value=50,
        )
        for seed, dataset in enumerate(("sea4", "mnist2", "mnist4", "circle2"))
    ]
    plan = SimpleNamespace(iter_experiments=lambda: iter(experiments))
    submitted = []

    class FakeFuture:
        def __init__(self, row):
            self.row = row

        def result(self):
            return self.row

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, experiment, raw_dir):
            submitted.append(experiment.dataset)
            return FakeFuture({
                "seed": experiment.seed,
                "stable_accuracy": 0.5,
                "comm_models_total": 0,
                "final_model_count": 1,
            })

    monkeypatch.setattr(sweep, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(sweep, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        sweep.multiprocessing, "get_context", lambda method: method,
    )

    rows = sweep.run_sweep_plan(plan, workers=2, runtime_config={})

    assert submitted == ["mnist4", "mnist2", "sea4", "circle2"]
    assert [row["seed"] for row in rows] == [0, 1, 2, 3]


def test_explicit_disable_flags_resolve_to_empty_plan_collections():
    argv = [
        "--no-fedsda", "--no-feddrift", "--no-baselines",
        "--no-adwin-sweep", "--no-aggregation-sweep",
        "--no-feddrift-batch-sweep", "--no-feddrift-distance-sweep",
    ]
    parser = sweep.build_parser()
    args = sweep.apply_collection_disables(parser, parser.parse_args(argv), argv)
    assert args.fedsda_modes == []
    assert args.feddrift_modes == []
    assert args.baseline_modes == []
    assert args.adwin_deltas == []
    assert args.agg_sweep == []
    assert args.batches == []
    assert args.deltas == []


def test_disable_flag_rejects_non_empty_values():
    argv = ["--no-adwin-sweep", "--adwin-deltas", "0.05"]
    parser = sweep.build_parser()
    with pytest.raises(SystemExit):
        sweep.apply_collection_disables(parser, parser.parse_args(argv), argv)


@pytest.mark.parametrize("option", [
    "--fedsda-modes", "--feddrift-modes", "--baseline-modes",
    "--adwin-deltas", "--aggregation-intervals",
    "--feddrift-detection-batch-sizes", "--feddrift-distance-thresholds",
])
def test_legacy_empty_collection_syntax_is_rejected(option):
    with pytest.raises(SystemExit):
        sweep.build_parser().parse_args([option])


def _fake_row(**kwargs):
    row = dict(kwargs)
    row.update({key: 0.0 for key in sweep.METRIC_KEYS})
    return row


def test_new_rows_use_method_specific_parameter_schema(monkeypatch):
    monkeypatch.setattr(
        sweep,
        "run_random_drift_experiment",
        lambda **kwargs: {key: 0.0 for key in sweep.METRIC_KEYS},
    )

    row = sweep._run(
        mode="FedDrift",
        dataset="sea4",
        seed=0,
        series="FedDrift B_detect sweep (δ_FedDrift=0.1)",
        sweep_parameter=sweep.FEDDRIFT_DETECTION_BATCH_SIZE,
        sweep_value=50,
        feddrift_batch=50,
        distance_threshold=0.1,
    )

    assert row["parameter_schema_version"] == 1
    assert row["sweep_parameter"] == sweep.FEDDRIFT_DETECTION_BATCH_SIZE
    assert row[sweep.FEDDRIFT_DETECTION_BATCH_SIZE] == 50
    assert row[sweep.FEDDRIFT_DISTANCE_THRESHOLD] == 0.1
    assert row[sweep.AGGREGATION_INTERVAL] is None
    assert row[sweep.FEDSDA_DISTANCE_THRESHOLD] is None
    assert row[sweep.ADWIN_DELTA] is None


def test_run_sweep_schedules_selected_versions(monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(dict(kwargs))
        return _fake_row(**kwargs)

    monkeypatch.setattr(sweep, "_run", fake_run)
    rows = sweep.run_sweep(
        datasets=["sea4"], seeds=[0], batches=[25], deltas=[0.1, 0.2],
        adwin_deltas=[0.05, 0.3], fixed_delta=0.1, fixed_batch=50,
        fixed_gamma=0.1, agg_sweep=[100], fixed_adwin=0.1,
        fedsda_modes=["FedSDA_NoCached_ADWIN", "FedSDA_Cached_ADWIN"],
        feddrift_modes=["FedDrift"],
        baseline_modes=["FedSDA_without_server", "Oblivious"],
    )

    assert len(rows) == 11
    assert {call["mode"] for call in calls} == {
        "FedSDA_NoCached_ADWIN", "FedSDA_Cached_ADWIN", "FedDrift",
        "FedSDA_without_server", "Oblivious",
    }
    for mode in ("FedSDA_NoCached_ADWIN", "FedSDA_Cached_ADWIN"):
        mode_calls = [call for call in calls if call["mode"] == mode]
        assert [call["agg_interval"] for call in mode_calls] == [
            sweep.config.AGGREGATION_INTERVAL, sweep.config.AGGREGATION_INTERVAL, 100,
        ]


def test_adwin_sweep_uses_fixed_aggregation_interval(monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(dict(kwargs))
        return _fake_row(**kwargs)

    monkeypatch.setattr(sweep, "_run", fake_run)
    sweep.run_sweep(
        datasets=["sea4"], seeds=[0], batches=[], deltas=[],
        adwin_deltas=[0.05], fixed_delta=0.1, fixed_batch=50,
        fixed_gamma=0.1, agg_sweep=[], fixed_adwin=0.1, fixed_agg=500,
        fedsda_modes=["FedSDA_NoCached_ADWIN"], feddrift_modes=[], baseline_modes=[],
    )

    assert len(calls) == 1
    assert calls[0]["agg_interval"] == 500


def test_adwin_delta_sweep_skips_non_adwin_detectors(monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(dict(kwargs))
        return _fake_row(**kwargs)

    monkeypatch.setattr(sweep, "_run", fake_run)
    sweep.run_sweep(
        datasets=["sea4"], seeds=[0], batches=[], deltas=[],
        adwin_deltas=[0.05, 0.1], fixed_delta=0.1, fixed_batch=50,
        fixed_gamma=0.1, agg_sweep=[100], fixed_adwin=0.1,
        fedsda_modes=["FedSDA_NoCached_ESR", "FedSDA_NoCached_HDDMA"],
        feddrift_modes=[], baseline_modes=[],
    )

    assert len(calls) == 2
    assert {call["mode"] for call in calls} == {
        "FedSDA_NoCached_ESR", "FedSDA_NoCached_HDDMA",
    }
    assert all("A sweep" in call["series"] for call in calls)


def test_load_csv_rejects_legacy_parameter_columns(tmp_path):
    old_keys = [
        key for key in sweep.ROW_KEYS
        if key not in (
            "parameter_schema_version", "sweep_parameter",
            "concept_schedule", sweep.AGGREGATION_INTERVAL,
            sweep.FEDDRIFT_DETECTION_BATCH_SIZE,
            sweep.FEDSDA_DISTANCE_THRESHOLD,
            sweep.FEDDRIFT_DISTANCE_THRESHOLD,
            "clustering_policy",
            "detection_episodes",
        )
    ] + ["feddrift_batch", "distance_threshold"]
    path = tmp_path / "old.csv"
    row = {key: "0" for key in old_keys}
    row.update({
        "mode": "FedSDA", "dataset": "sea", "series": "FedSDA sweep",
        "sweep_value": "0.1", "feddrift_batch": "50",
    })
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=old_keys)
        writer.writeheader()
        writer.writerow(row)

    try:
        sweep._load_csv(path)
    except ValueError as error:
        assert "Legacy parameter columns" in str(error)
    else:
        raise AssertionError("旧パラメータ列を受理してはいけない")


def test_load_csv_accepts_canonical_feddrift_baseline_names(tmp_path):
    path = tmp_path / "feddrift.csv"
    row = {key: "0" for key in sweep.METRIC_KEYS}
    row.update({
        "mode": "FedDrift",
        "parameter_schema_version": "1",
        "dataset": "circle2",
        "concept_schedule": "random",
        "seed": "0",
        "series": "FedDrift B_detect sweep (δ_FedDrift=0.1)",
        "sweep_parameter": sweep.FEDDRIFT_DETECTION_BATCH_SIZE,
        "sweep_value": "50",
        sweep.FEDDRIFT_DETECTION_BATCH_SIZE: "50",
        sweep.FEDDRIFT_DISTANCE_THRESHOLD: "0.1",
    })
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    loaded = sweep._load_csv(path)

    assert loaded[0]["mode"] == "FedDrift"
    assert loaded[0][sweep.FEDDRIFT_DETECTION_BATCH_SIZE] == 50
    assert loaded[0][sweep.FEDDRIFT_DISTANCE_THRESHOLD] == 0.1
    assert loaded[0]["sweep_parameter"] == sweep.FEDDRIFT_DETECTION_BATCH_SIZE


def test_series_style_distinguishes_method_and_sweep_type():
    fedsda_delta = sweep._series_style("FedSDA_NoCached_ADWIN δ_ADWIN sweep (A=50, γ=0.1)")
    feddrift_delta = sweep._series_style("FedDrift δ_FedDrift sweep (B_detect=50)")
    fedsda_agg = sweep._series_style("FedSDA_NoCached_ADWIN A sweep (δ_ADWIN=0.05, γ=0.1)")

    assert fedsda_delta != feddrift_delta
    assert fedsda_delta[0] == fedsda_agg[0]
    assert fedsda_delta[1:] != fedsda_agg[1:]


def test_plot_pareto_draws_baseline_standard_deviation_band(tmp_path, monkeypatch):
    spans = []
    line_labels = []
    original = sweep.plt.Axes.axhspan
    original_line = sweep.plt.Axes.axhline

    def record_span(self, ymin, ymax, *args, **kwargs):
        spans.append((ymin, ymax))
        return original(self, ymin, ymax, *args, **kwargs)

    def record_line(self, y, *args, **kwargs):
        line_labels.append(kwargs.get("label"))
        return original_line(self, y, *args, **kwargs)

    monkeypatch.setattr(sweep.plt.Axes, "axhspan", record_span)
    monkeypatch.setattr(sweep.plt.Axes, "axhline", record_line)
    rows = []
    for mode, accuracies in {
        "FedSDA_without_server": (0.7, 0.9),
        "Oblivious": (0.6, 0.8),
    }.items():
        for seed, accuracy in enumerate(accuracies):
            rows.append({
                "mode": mode, "dataset": "sea4", "seed": seed,
                "series": f"{mode} [feddrift_fixed]",
                "sweep_value": None, "comm_models_total": 0.0,
                "stable_accuracy": accuracy,
                sweep.AGGREGATION_INTERVAL: 50,
                sweep.ADWIN_DELTA: 0.1,
            })

    path = tmp_path / "pareto.png"
    sweep.plot_pareto(rows, ["sea4"], path)

    assert path.exists()
    assert len(spans) == 2
    assert "FedSDA_without_server (δ_ADWIN=0.1, mean±std)" in line_labels
    assert "Oblivious (A=50, mean±std)" in line_labels


def test_plot_pareto_can_use_overall_accuracy(tmp_path):
    rows = [{
        "mode": "FedSDA_NoCached_ADWIN", "dataset": "sea4", "seed": 0,
        "series": "FedSDA_NoCached_ADWIN δ_ADWIN sweep (A=50, γ=0.1)", "sweep_value": 0.1,
        "comm_models_total": 100.0, "stable_accuracy": 0.9, "accuracy": 0.8,
    }]

    path = tmp_path / "overall.png"
    sweep.plot_pareto(rows, ["sea4"], path, y_key="accuracy")

    assert path.exists()


def test_replot_filter_selects_interval_sweeps_and_plot_accepts_compute_x(tmp_path):
    rows = [
        {
            "mode": "FedSDA_NoCached_ClassADWIN", "dataset": "sea4", "seed": 0,
            "series": "FedSDA_NoCached_ClassADWIN A sweep", "sweep_value": 50.0,
            "compute_model_examples_total": 1000.0,
            "stable_accuracy": 0.9, "accuracy": 0.8,
        },
        {
            "mode": "FedSDA_NoCached_ClassADWIN", "dataset": "sea4", "seed": 0,
            "series": "FedSDA_NoCached_ClassADWIN δ_ADWIN sweep", "sweep_value": 0.1,
            "compute_model_examples_total": 1100.0,
            "stable_accuracy": 0.91, "accuracy": 0.81,
        },
        {
            "mode": "FedDrift", "dataset": "sea4", "seed": 0,
            "series": "FedDrift B_detect sweep", "sweep_value": 50.0,
            "compute_model_examples_total": 900.0,
            "stable_accuracy": 0.88, "accuracy": 0.79,
        },
    ]
    filtered = sweep._filter_replot_rows(
        rows, modes=["FedSDA_NoCached_ClassADWIN", "FedDrift"], sweep_kind="interval"
    )
    assert [row["series"] for row in filtered] == [
        "FedSDA_NoCached_ClassADWIN A sweep", "FedDrift B_detect sweep"
    ]

    path = tmp_path / "compute.png"
    sweep.plot_pareto(
        filtered, ["sea4"], path, y_key="accuracy",
        x_key="compute_model_examples_total",
    )
    assert path.exists()
