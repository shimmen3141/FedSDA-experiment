"""手法比較の累積精度と上下通信量を、seed間のばらつき（既定は±1 SD、縦と横）付きで描く。

例: python -m tools.research_method_pareto_figure --series "FedSDA=results/study:fedsda"
    --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 --output results/figures/methods
`--error-bars ci95`で、平均の95%信頼区間（seed間のt分布）に替える。
補足図は調整用seedの取得済みrunを全て使う。未完了のδもseed数と共にCSVへ残す。
"""

import argparse
import csv
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from tools.research_feddrift_delta_comparison import _read_runs, _t_critical_95

_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
_MARKERS = ("o", "s", "^", "D", "v")
_ERROR_BAR_DESCRIPTIONS = {
    "sd": "±1 SD over seeds",
    "ci95": "95% confidence interval of the mean over seeds",
}


def _spread(values):
    """seed間のSDと、平均の95%信頼区間の半幅（1 seedならNone）。"""
    count = len(values)
    if count < 2:
        return None, None
    standard_deviation = float(np.std(values, ddof=1))
    return standard_deviation, float(_t_critical_95(count - 1) * standard_deviation / np.sqrt(count))


def _parse_series(item):
    try:
        label, source = item.split("=", 1)
        root, configuration = source.rsplit(":", 1)
    except ValueError as error:
        raise ValueError(f"表示名=ディレクトリ:構成で指定してください: {item}") from error
    if not label.strip() or not root.strip() or not configuration.strip():
        raise ValueError(f"表示名・ディレクトリ・構成は空にできません: {item}")
    return label, root, configuration


def _collect_cells(items, datasets, intervals, *, frontier=False):
    cells = {}
    for item in items:
        label, root, configuration = _parse_series(item)
        if label in cells:
            raise ValueError(f"表示名が重複しています: {label}")
        groups = defaultdict(dict)
        runs = _read_runs(root, configuration_filter=configuration, include_computation=False)
        for (_, dataset, interval, seed, delta), run in runs.items():
            if (datasets is not None and dataset not in datasets) or interval not in intervals:
                continue
            key = (dataset, interval, delta if frontier else None)
            if seed in groups[key]:
                raise ValueError(
                    f"同じデータ・間隔・seedのrunが2つあります（δの混在を含む）: {label} {key} {seed}"
                )
            accuracy = run["accuracy"]
            upload, download = float(run["upload"]), float(run["download"])
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
                raise ValueError(f"精度が不正です: {label} {key} {seed}")
            if (
                not all(math.isfinite(v) and v >= 0 for v in (upload, download))
                or upload + download <= 0
            ):
                raise ValueError(f"対数軸に描ける正の通信量が必要です: {label} {key} {seed}")
            if delta is not None and not math.isfinite(float(delta)):
                raise ValueError(f"δが不正です: {label} {key} {seed}")
            groups[key][seed] = dict(run, delta=delta)
        if not groups:
            raise ValueError(f"対象のrunがありません: {label} {root}:{configuration}")
        cells[label] = (configuration, groups)
    return cells


def _summarize(cells, datasets, intervals, *, frontier=False):
    rows = []
    for label, (configuration, groups) in cells.items():
        for dataset in datasets:
            for interval in intervals:
                keys = [key for key in groups if key[:2] == (dataset, interval)]
                if not keys:
                    raise ValueError(f"対象のrunがありません: {label} {dataset} 間隔{interval}")
                for key in sorted(keys, key=lambda key: (key[2] is not None, key[2])):
                    seeds = sorted(groups[key])
                    runs = [groups[key][seed] for seed in seeds]
                    accuracies = [run["accuracy"] for run in runs]
                    communications = [run["upload"] + run["download"] for run in runs]
                    count = len(seeds)
                    accuracy_sd, half_width = _spread(accuracies)
                    communication_sd, communication_half_width = _spread(communications)
                    deltas = {run["delta"] for run in runs}
                    delta = (
                        next(iter(deltas))
                        if configuration == "feddrift" and len(deltas) == 1
                        else None
                    )
                    rows.append(
                        dict(
                            kind="frontier" if frontier else "series",
                            series=label,
                            configuration=configuration,
                            dataset=dataset,
                            interval=interval,
                            seed_count=count,
                            seeds=" ".join(map(str, seeds)),
                            mean_accuracy_percent=float(np.mean(accuracies)),
                            accuracy_sd_pp=accuracy_sd,
                            accuracy_ci95_half_width_pp=half_width,
                            mean_communication_byte=float(np.mean(communications)),
                            communication_sd_byte=communication_sd,
                            communication_ci95_half_width_byte=communication_half_width,
                            delta=delta,
                            is_nondominated=None,
                        )
                    )
    return rows


def _check_seeds(cells, datasets, intervals, allow_different):
    for dataset in datasets:
        for interval in intervals:
            seed_sets = {
                label: sorted(groups.get((dataset, interval, None), {}))
                for label, (_, groups) in cells.items()
            }
            if len({tuple(seeds) for seeds in seed_sets.values()}) > 1:
                message = f"系列間のseedがそろっていません: {dataset} 間隔{interval} {seed_sets}"
                if not allow_different:
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)


def _mark_frontier(rows):
    # 同値の点は両方残す。一方の軸で厳密に良い場合だけ支配とする。
    for row in rows:
        row["is_nondominated"] = not any(
            other["dataset"] == row["dataset"]
            and other["mean_communication_byte"] <= row["mean_communication_byte"]
            and other["mean_accuracy_percent"] >= row["mean_accuracy_percent"]
            and (
                other["mean_communication_byte"] < row["mean_communication_byte"]
                or other["mean_accuracy_percent"] > row["mean_accuracy_percent"]
            )
            for other in rows
        )


def _draw(rows, datasets, labels, prefix, frontier_rows=(), error_bars="sd"):
    accuracy_key, communication_key = {
        "sd": ("accuracy_sd_pp", "communication_sd_byte"),
        "ci95": ("accuracy_ci95_half_width_pp", "communication_ci95_half_width_byte"),
    }[error_bars]

    def error(row, key, scale=1.0):
        return row[key] / scale if row[key] is not None else float("nan")

    columns = min(3, len(datasets))
    row_count = math.ceil(len(datasets) / columns)
    with plt.rc_context(
        {
            "font.size": 10,
            "text.color": "black",
            "axes.labelcolor": "black",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(
            row_count, columns, figsize=(4.1 * columns, 3.2 * row_count), squeeze=False
        )
        handles = []
        try:
            for index, (axis, dataset) in enumerate(zip(axes.flat, datasets)):
                background = [row for row in frontier_rows if row["dataset"] == dataset]
                if background:
                    axis.scatter(
                        [r["mean_communication_byte"] / 1e6 for r in background],
                        [r["mean_accuracy_percent"] for r in background],
                        color="0.75",
                        s=13,
                        alpha=0.55,
                        zorder=1,
                    )
                    frontier = sorted(
                        (r for r in background if r["is_nondominated"]),
                        key=lambda r: r["mean_communication_byte"],
                    )
                    axis.plot(
                        [r["mean_communication_byte"] / 1e6 for r in frontier],
                        [r["mean_accuracy_percent"] for r in frontier],
                        color="0.4",
                        linewidth=1.2,
                        zorder=2,
                    )
                for series_index, label in enumerate(labels):
                    points = [r for r in rows if r["dataset"] == dataset and r["series"] == label]
                    x = [r["mean_communication_byte"] / 1e6 for r in points]
                    y = [r["mean_accuracy_percent"] for r in points]
                    handle = axis.errorbar(
                        x,
                        y,
                        yerr=[error(r, accuracy_key) for r in points],
                        xerr=[error(r, communication_key, 1e6) for r in points],
                        label=label,
                        color=_COLORS[series_index],
                        marker=_MARKERS[series_index],
                        linewidth=1.5,
                        markersize=6.5,
                        capsize=3,
                        zorder=3,
                    )
                    if index == 0:
                        handles.append(handle)
                    if series_index == 0:
                        for px, py, point in zip(x, y, points):
                            axis.annotate(
                                str(point["interval"]),
                                (px, py),
                                xytext=(6, 5),
                                textcoords="offset points",
                                fontsize=8,
                                color="0.25",
                            )
                axis.set_title(dataset.upper())
                axis.set_xscale("log")
                axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                axis.xaxis.set_minor_formatter(NullFormatter())
                axis.set_xlabel("Communication (upload + download, MB)")
                axis.set_ylabel("Cumulative accuracy (%)")
                axis.grid(True, color="0.9", linewidth=0.5)
                axis.margins(x=0.18, y=0.22)
            for axis in list(axes.flat)[len(datasets) :]:
                axis.set_visible(False)
            legend_labels = list(labels)
            if frontier_rows:
                from matplotlib.lines import Line2D

                handles.append(Line2D([], [], marker="o", color="0.4", markersize=3, linewidth=1.2))
                legend_labels.append(" / ".join(dict.fromkeys(r["series"] for r in frontier_rows)))
            legend_columns = min(3, len(handles))
            legend_height = 0.32 * math.ceil(len(handles) / legend_columns)
            figure.legend(
                handles, legend_labels, loc="lower center", ncol=legend_columns, frameon=False
            )
            figure.tight_layout(rect=(0, legend_height / figure.get_figheight(), 1, 1))
            for extension in ("pdf", "png"):
                # 排他的に作り、事前確認後に作られた既存ファイルも上書きしない。
                with Path(f"{prefix}.{extension}").open("xb") as stream:
                    figure.savefig(stream, format=extension, dpi=300, bbox_inches="tight")
        finally:
            plt.close(figure)


def _write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", action="append", required=True)
    parser.add_argument("--frontier", action="append", default=[])
    parser.add_argument("--datasets", nargs="+", help="対象と並び（省略時は主図のデータを名前順）")
    parser.add_argument("--intervals", nargs="+", type=int, default=[50, 100, 200, 500])
    parser.add_argument("--allow-different-seeds", action="store_true")
    parser.add_argument(
        "--error-bars",
        choices=sorted(_ERROR_BAR_DESCRIPTIONS),
        default="sd",
        help="誤差棒（縦は精度、横は通信）: sd=±1 SD（既定）、ci95=平均の95%%信頼区間",
    )
    parser.add_argument("--output", required=True, help="拡張子なしの出力prefix")
    arguments = parser.parse_args(command_arguments)
    try:
        if len(arguments.series) > len(_COLORS):
            raise ValueError("主図は5系列までです")
        if len(set(arguments.intervals)) != len(arguments.intervals) or any(
            i <= 0 for i in arguments.intervals
        ):
            raise ValueError("間隔は重複のない正の整数にしてください")
        cells = _collect_cells(arguments.series, arguments.datasets, arguments.intervals)
        datasets = arguments.datasets or sorted(
            {key[0] for _, groups in cells.values() for key in groups}
        )
        if len(set(datasets)) != len(datasets):
            raise ValueError("データの指定が重複しています")
        rows = _summarize(cells, datasets, arguments.intervals)
        _check_seeds(cells, datasets, arguments.intervals, arguments.allow_different_seeds)
        frontier_rows = []
        if arguments.frontier:
            frontier_cells = _collect_cells(
                arguments.frontier, datasets, arguments.intervals, frontier=True
            )
            frontier_rows = _summarize(frontier_cells, datasets, arguments.intervals, frontier=True)
            _mark_frontier(frontier_rows)
        prefix = Path(arguments.output)
        prefixes = [prefix] + ([Path(f"{prefix}-frontier")] if frontier_rows else [])
        for destination in prefixes:
            for extension in ("pdf", "png", "csv"):
                path = Path(f"{destination}.{extension}")
                if path.exists():
                    raise ValueError(f"既存のファイルは上書きしません: {path}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        _draw(rows, datasets, list(cells), prefix, error_bars=arguments.error_bars)
        _write_csv(Path(f"{prefix}.csv"), rows)
        if frontier_rows:
            _draw(
                rows,
                datasets,
                list(cells),
                prefixes[1],
                frontier_rows,
                error_bars=arguments.error_bars,
            )
            _write_csv(Path(f"{prefixes[1]}.csv"), rows + frontier_rows)
        print(f"出力: {', '.join(map(str, prefixes))} (.pdf, .png, .csv)")
        print(f"誤差棒（図の説明に書く）: {_ERROR_BAR_DESCRIPTIONS[arguments.error_bars]}")
        return 0
    except (ValueError, KeyError, OSError, IndexError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
