import csv
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import run_pareto_sweep as sweep


def test_cli_help_groups_related_sweep_options():
    help_text = sweep.build_parser().format_help()

    assert "FedSDAの手法・掃引" in help_text
    assert "--fixed-adwin" in help_text
    assert "--agg-sweepが空なら未使用" in help_text
    assert "FedDriftの手法・掃引" in help_text
    assert "--fixed-delta" in help_text
    assert "--batchesが空なら未使用" in help_text
    assert "既存CSVの再描画" in help_text
    assert "他の実験設定は無視" in help_text


def test_new_paper_datasets_are_opt_in_for_default_sweep():
    parser = sweep.build_parser()
    defaults = parser.parse_args([])
    assert defaults.datasets == ["blobs", "sea", "circle", "sine"]
    assert defaults.concept_schedule == "random"
    selected = parser.parse_args([
        "--datasets", "sea2", "mnist2", "mnist4",
        "--concept-schedule", "feddrift_fixed",
    ])
    assert selected.datasets == ["sea2", "mnist2", "mnist4"]
    assert selected.concept_schedule == "feddrift_fixed"


def _fake_row(**kwargs):
    row = dict(kwargs)
    row.update({key: 0.0 for key in sweep.METRIC_KEYS})
    return row


def test_run_sweep_schedules_selected_versions(monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(dict(kwargs))
        return _fake_row(**kwargs)

    monkeypatch.setattr(sweep, "_run", fake_run)
    rows = sweep.run_sweep(
        datasets=["sea"], seeds=[0], batches=[25], deltas=[0.1, 0.2],
        adwin_deltas=[0.05, 0.3], fixed_delta=0.1, fixed_batch=50,
        fixed_gamma=0.1, agg_sweep=[100], fixed_adwin=0.1,
        fedsda_modes=["FedSDA_v2", "FedSDA_v3"],
        feddrift_modes=["FedDrift_v2"],
        baseline_modes=["FedSDA_without_server", "Oblivious"],
    )

    assert len(rows) == 11
    assert {call["mode"] for call in calls} == {
        "FedSDA_v2", "FedSDA_v3", "FedDrift_v2",
        "FedSDA_without_server", "Oblivious",
    }
    for mode in ("FedSDA_v2", "FedSDA_v3"):
        mode_calls = [call for call in calls if call["mode"] == mode]
        assert [call["agg_interval"] for call in mode_calls] == [
            sweep.config.AGG_INTERVAL, sweep.config.AGG_INTERVAL, 100,
        ]


def test_load_csv_accepts_previous_format_without_agg_interval(tmp_path):
    old_keys = [
        key for key in sweep.ROW_KEYS
        if key not in (
            "concept_schedule", "agg_interval", "e_detector_baseline_strategy",
            "e_detector_baseline_beta",
        )
    ]
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

    loaded = sweep._load_csv(path)

    assert loaded[0]["agg_interval"] == ""
    assert loaded[0]["concept_schedule"] == "random"
    assert "e_detector_baseline_strategy" not in loaded[0]
    assert math.isnan(loaded[0]["e_detector_baseline_beta"])
    assert loaded[0]["sweep_value"] == 0.1


def test_series_style_distinguishes_method_and_sweep_type():
    fedsda_delta = sweep._series_style("FedSDA_v2 δ_adwin sweep (γ=0.1)")
    feddrift_delta = sweep._series_style("FedDrift_v2 δ sweep (batch=50)")
    fedsda_agg = sweep._series_style("FedSDA_v2 AGG_INTERVAL sweep (δ_adwin=0.05)")

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
                "mode": mode, "dataset": "sea", "seed": seed, "series": mode,
                "sweep_value": None, "comm_models_total": 0.0,
                "stable_accuracy": accuracy, "agg_interval": 50,
                "adwin_delta": 0.1,
            })

    path = tmp_path / "pareto.png"
    sweep.plot_pareto(rows, ["sea"], path)

    assert path.exists()
    assert len(spans) == 2
    assert "FedSDA_without_server (δ_adwin=0.1, mean±std)" in line_labels
    assert "Oblivious (AGG_INTERVAL=50, mean±std)" in line_labels


def test_plot_pareto_can_use_overall_accuracy(tmp_path):
    rows = [{
        "mode": "FedSDA_v2", "dataset": "sea", "seed": 0,
        "series": "FedSDA_v2 δ_adwin sweep (γ=0.1)", "sweep_value": 0.1,
        "comm_models_total": 100.0, "stable_accuracy": 0.9, "accuracy": 0.8,
    }]

    path = tmp_path / "overall.png"
    sweep.plot_pareto(rows, ["sea"], path, y_key="accuracy")

    assert path.exists()
