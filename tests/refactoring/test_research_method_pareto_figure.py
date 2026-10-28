"""論文用Pareto図を小さなrunの要約から検算する。"""

import csv
import json

import pytest

from tools import research_method_pareto_figure as figure
from tools.research_feddrift_delta_comparison import _read_runs


@pytest.mark.parametrize(
    "freedom, expected",
    [
        (1, 12.7062047364),
        (2, 4.3026527297),
        (9, 2.2621571629),
        (29, 2.0452296421),
        (99, 1.9842169515),
    ],
)
def test_student_t_critical_values(freedom, expected):
    assert figure._t_critical_95(freedom) == pytest.approx(expected, rel=1e-9)


def _run(
    root,
    *,
    seed=0,
    delta=None,
    accuracy=0.8,
    upload=100,
    download=200,
    configuration="feddrift",
    study="study",
    dataset="sea2",
    interval=50,
):
    path = (
        root / study / configuration / "runs" / f"{dataset}_interval{interval}_seed{seed}_test.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_settings": {
                    "run_participant_settings": {
                        "run_client_settings": {"loss_increase_threshold": delta}
                    }
                },
                "run_metrics": {
                    "prediction_accuracy": accuracy,
                    "communication_volume": {
                        "uploaded_byte_count": upload,
                        "downloaded_byte_count": download,
                    },
                    "computation_cost_summary": {
                        "client_forward_multiply_accumulate_count": 11,
                        "client_estimated_backward_multiply_accumulate_count": 22,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _arguments(root, output):
    return [
        "--series",
        f"Method={root}:feddrift",
        "--datasets",
        "sea2",
        "--intervals",
        "50",
        "--output",
        str(output),
    ]


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_statistics_and_outputs(tmp_path):
    for seed, accuracy, upload in [(0, 0.7, 100), (1, 0.8, 200), (2, 0.9, 300)]:
        _run(tmp_path, seed=seed, accuracy=accuracy, upload=upload, delta=0.1)
    output = tmp_path / "figure"
    assert figure.main(_arguments(tmp_path, output)) == 0
    row = _rows(output.with_suffix(".csv"))[0]
    assert float(row["mean_accuracy_percent"]) == pytest.approx(80)
    # 自由度2のtの97.5百分位=4.3026527297、標本標準偏差10pp。
    assert float(row["accuracy_ci95_half_width_pp"]) == pytest.approx(24.8413771172)
    assert float(row["mean_communication_byte"]) == 400
    assert row["seed_count"] == "3"
    assert row["seeds"] == "0 1 2"
    assert float(row["delta"]) == 0.1
    for extension in ("pdf", "png"):
        assert output.with_suffix(f".{extension}").stat().st_size > 100
    run = next(iter(_read_runs(tmp_path).values()))
    assert run["client_mac"] == 33


def test_minimal_summary_and_configuration_filter(tmp_path):
    path = _run(tmp_path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    del summary["run_metrics"]["computation_cost_summary"]
    path.write_text(json.dumps(summary), encoding="utf-8")
    # 選ばない構成は集計の前に除外し、その指標や重複を要求しない。
    other = _run(tmp_path, configuration="unselected")
    other.write_text("{}", encoding="utf-8")
    assert figure.main(_arguments(tmp_path, tmp_path / "figure")) == 0


def test_frontier_dominance_and_single_seed(tmp_path):
    main_root, tuning = tmp_path / "main", tmp_path / "tuning"
    _run(main_root, delta=0.1)
    # 同じ通信で低精度、同じ精度で多通信、完全同値、トレードオフを含む。
    points = [(100, 0.8), (100, 0.7), (200, 0.8), (100, 0.8), (300, 0.9)]
    for index, (communication, accuracy) in enumerate(points):
        _run(
            tuning,
            seed=100,
            delta=index / 10,
            accuracy=accuracy,
            upload=communication,
            download=0,
            study=str(index),
        )
    output = tmp_path / "figure"
    assert (
        figure.main(_arguments(main_root, output) + ["--frontier", f"Tuning={tuning}:feddrift"])
        == 0
    )
    rows = _rows(tmp_path / "figure-frontier.csv")
    assert rows[0]["accuracy_ci95_half_width_pp"] == ""
    background = [row for row in rows if row["kind"] == "frontier"]
    assert [row["is_nondominated"] for row in background] == [
        "True",
        "False",
        "False",
        "True",
        "True",
    ]
    assert all(row["seed_count"] == "1" and row["seeds"] == "100" for row in background)
    assert (tmp_path / "figure-frontier.pdf").stat().st_size > 100
    assert (tmp_path / "figure-frontier.png").stat().st_size > 100


@pytest.mark.parametrize("second_delta", [0.1, 0.2])
def test_duplicate_runs_and_mixed_deltas_fail(tmp_path, second_delta):
    _run(tmp_path, delta=0.1)
    _run(tmp_path, delta=second_delta, study="other")
    assert figure.main(_arguments(tmp_path, tmp_path / "figure")) == 2
    assert not (tmp_path / "figure.pdf").exists()


def test_different_seeds_fail_or_warn(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    _run(left, seed=0)
    _run(right, seed=1)
    arguments = _arguments(left, tmp_path / "figure") + ["--series", f"Other={right}:feddrift"]
    assert figure.main(arguments) == 2
    with pytest.warns(UserWarning, match="seed"):
        assert figure.main(arguments + ["--allow-different-seeds"]) == 0


@pytest.mark.parametrize(
    "suffix", [".pdf", ".png", ".csv", "-frontier.pdf", "-frontier.png", "-frontier.csv"]
)
def test_existing_outputs_are_never_overwritten(tmp_path, suffix):
    root = tmp_path / "input"
    _run(root)
    existing = tmp_path / f"figure{suffix}"
    existing.write_bytes(b"keep")
    arguments = _arguments(root, tmp_path / "figure") + ["--frontier", f"Tuning={root}:feddrift"]
    assert figure.main(arguments) == 2
    assert existing.read_bytes() == b"keep"
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == [existing.name]


def test_six_series_fail(tmp_path):
    arguments = ["--output", str(tmp_path / "figure")]
    for index in range(6):
        arguments += ["--series", f"{index}={tmp_path}:feddrift"]
    assert figure.main(arguments) == 2
    assert not (tmp_path / "figure.pdf").exists()


def test_nested_studies_and_requested_order(tmp_path):
    for dataset in ("sea2", "sea4"):
        for interval in (50, 100):
            _run(
                tmp_path,
                dataset=dataset,
                interval=interval,
                delta=interval / 1000,
                study=f"study{interval}",
            )
    output = tmp_path / "figure"
    arguments = [
        "--series",
        f"Method={tmp_path}:feddrift",
        "--datasets",
        "sea4",
        "sea2",
        "--intervals",
        "100",
        "50",
        "--output",
        str(output),
    ]
    assert figure.main(arguments) == 0
    assert [(r["dataset"], r["interval"]) for r in _rows(output.with_suffix(".csv"))] == [
        ("sea4", "100"),
        ("sea4", "50"),
        ("sea2", "100"),
        ("sea2", "50"),
    ]


@pytest.mark.parametrize("case", ["missing", "zero", "nan", "accuracy", "empty", "intervals"])
def test_invalid_or_missing_values_fail(tmp_path, case):
    _run(
        tmp_path,
        upload=0 if case == "zero" else 100,
        download=0 if case == "zero" else 200,
        accuracy=float("nan") if case == "nan" else (1.1 if case == "accuracy" else 0.8),
    )
    arguments = _arguments(tmp_path, tmp_path / "figure")
    if case == "missing":
        arguments += ["--datasets", "sea4"]
    elif case == "empty":
        arguments += ["--series", f"={tmp_path}:feddrift"]
    elif case == "intervals":
        arguments += ["--intervals", "50", "50"]
    assert figure.main(arguments) == 2
