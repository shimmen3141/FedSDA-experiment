"""論文の表1の手法の精度を「変化の直後」と「それ以外」に分けて集計する研究ツール（追加の実験なし）。

問い: 連合学習でのFedHASPの優位は、評価の条件（変化の直後の標本を含む全標本の精度）によるものか。
FedDriftの論文の評価は変化が起きた時刻を平均から除くので、「それ以外」の精度がそれに近い。

入力: 論文のリポジトリの`figures/paper_results.py`の`load_methods()`が読む結果（表1と同じrun）。各runの
`run_summary.csv`と同じフォルダの`runs/<run_identifier>.npz`から、標本ごとの正誤（FedHASPは
`sample_prediction.combined_prediction_is_correct`、ほかは`sample_prediction.prediction_is_correct`）と真の概念
（`sample_prediction.observed_concept_id`）を読む。各runで、正誤の平均が要約の`prediction_accuracy`と一致しなければ止める。
表1から外した手法（FedAvg+Window、Adaptive-FedAvg）は除く。

「変化の直後」: 各clientで真の概念が変わった標本から、集約の間隔Iの標本（重なりはまとめる）。最初の概念（標本0）は
変化に数えない。「それ以外」: 残りの標本。精度は各runで（全clientの標本をまとめて）求め、seedで平均する。
差（FedHASP−手法）の95%区間はseedで対にしたt区間（多重比較の補正なし）。

出力（--output）: `analysis_post_change_accuracy_breakdown.md`と`post_change_accuracy_cells.csv`。

使い方（プロジェクトのvenv、worktreeのルートで）:
    python -m tools.research_post_change_accuracy_breakdown.analyze \\
        --paper-repository C:/Users/yshin/Research/FedHASP-paper --output <出力先> --workers 8
"""

import argparse
import csv
import importlib.util
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

EXCLUDED_METHODS = ("FedAvg+Window", "Adaptive-FedAvg")
REFERENCE_METHOD = "FedHASP"


def load_paper_results_module(paper_repository):
    path = Path(paper_repository) / "figures" / "paper_results.py"
    spec = importlib.util.spec_from_file_location("paper_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_run_paths(paper_results):
    """`load_methods()`が返す各runに、読んだ`run_summary.csv`の場所を付けて返す。"""
    original_load = paper_results._load

    def load_with_source(path, with_delta=False):
        rows = original_load(path, with_delta=with_delta)
        for row in rows.values():
            row["__source_directory"] = str(Path(path).parent)
        return rows

    paper_results._load = load_with_source
    try:
        methods = paper_results.load_methods()
    finally:
        paper_results._load = original_load
    tasks = []
    for method, rows in methods.items():
        if method in EXCLUDED_METHODS:
            continue
        for (dataset, interval, seed), row in rows.items():
            if dataset not in paper_results.DATASETS or interval not in paper_results.INTERVALS:
                continue
            if seed not in paper_results.SEEDS:
                continue
            npz_path = Path(row["__source_directory"]) / "runs" / f"{row['run_identifier']}.npz"
            tasks.append(
                (method, dataset, interval, seed, str(npz_path), float(row["prediction_accuracy"]))
            )
    return tasks


def post_change_mask(concept_ids, interval):
    """各clientで真の概念が変わった標本からinterval件をTrueにした配列（client×標本）。"""
    mask = np.zeros(concept_ids.shape, dtype=bool)
    for client in range(concept_ids.shape[0]):
        changes = np.flatnonzero(np.diff(concept_ids[client]) != 0) + 1
        for start in changes:
            mask[client, start : start + interval] = True
    return mask


def summarize_run(task):
    method, dataset, interval, seed, npz_path, summary_accuracy = task
    with np.load(npz_path, allow_pickle=False) as loaded:
        name = (
            "sample_prediction.combined_prediction_is_correct"
            if method == REFERENCE_METHOD
            else "sample_prediction.prediction_is_correct"
        )
        correct = loaded[name].astype(bool)
        concept_ids = loaded["sample_prediction.observed_concept_id"]
    if correct.shape != concept_ids.shape:
        raise ValueError(f"形が合いません: {npz_path}")
    if abs(correct.mean() - summary_accuracy) > 1e-9:
        raise ValueError(f"正誤の平均が要約の精度と違います: {npz_path}")
    mask = post_change_mask(concept_ids, interval)
    return dict(
        method=method,
        dataset=dataset,
        interval=interval,
        seed=seed,
        overall=100 * correct.mean(),
        post_change=100 * correct[mask].mean() if mask.any() else float("nan"),
        rest=100 * correct[~mask].mean(),
        post_change_fraction=100 * mask.mean(),
    )


def paired(values_reference, values_method):
    differences = np.asarray(values_reference) - np.asarray(values_method)
    mean = float(differences.mean())
    half = float(stats.t.ppf(0.975, len(differences) - 1) * differences.std(ddof=1) / np.sqrt(len(differences)))
    return mean, mean - half, mean + half, int((differences > 0).sum())


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--paper-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(command_arguments)
    paper_results = load_paper_results_module(arguments.paper_repository)
    tasks = collect_run_paths(paper_results)
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        records = list(executor.map(summarize_run, tasks, chunksize=8))
    by_key = {(r["method"], r["dataset"], r["interval"], r["seed"]): r for r in records}
    methods = sorted({r["method"] for r in records}, key=lambda m: (m != REFERENCE_METHOD, m))
    datasets, intervals, seeds = paper_results.DATASETS, paper_results.INTERVALS, list(paper_results.SEEDS)
    for method in methods:
        for dataset in datasets:
            for interval in intervals:
                for seed in seeds:
                    if (method, dataset, interval, seed) not in by_key:
                        raise ValueError(f"runが欠けています: {method} {dataset} {interval} {seed}")
    arguments.output.mkdir(parents=True, exist_ok=True)
    cell_rows = []
    for method in methods:
        for dataset in datasets:
            for interval in intervals:
                runs = [by_key[(method, dataset, interval, s)] for s in seeds]
                cell_rows.append(
                    dict(
                        method=method,
                        dataset=dataset,
                        interval=interval,
                        **{
                            name: st.mean(r[name] for r in runs)
                            for name in ("overall", "post_change", "rest", "post_change_fraction")
                        },
                    )
                )
    with (arguments.output / "post_change_accuracy_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    cell = {(c["method"], c["dataset"], c["interval"]): c for c in cell_rows}

    def seed_mean(method, name, selected_datasets, selected_intervals):
        return [
            st.mean(
                by_key[(method, d, i, s)][name] for d in selected_datasets for i in selected_intervals
            )
            for s in seeds
        ]

    lines = [
        "# 表1の手法の精度: 変化の直後とそれ以外",
        "",
        f"- 手法{len(methods)}（表1と同じrun。FedAvg+Window・Adaptive-FedAvgは除く）、データ{len(datasets)}、間隔{intervals}、seed {seeds[0]}〜{seeds[-1]}。"
        "各runで、標本ごとの正誤の平均が要約の精度と一致することを確かめた。",
        "- 「直後」は各clientの真の概念の変化から集約の間隔I件。「それ以外」はFedDriftの論文の評価（変化の時刻を除く）に近い。",
        "- 差はFedHASP−手法（pp）、95%区間はseedで対にしたt区間（多重比較の補正なし）。",
        "",
        "## 全体（10データの等重み）、間隔ごと",
        "",
        "| 手法 | 間隔 | 直後の割合 | 全体 | 直後 | それ以外 | 差（全体） | 差（それ以外） | 95%区間（それ以外） | FedHASPが上のseed（それ以外） |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method in methods:
        for interval in [*intervals, None]:
            selected = intervals if interval is None else [interval]
            values = {
                name: st.mean(seed_mean(method, name, datasets, selected))
                for name in ("post_change_fraction", "overall", "post_change", "rest")
            }
            if method == REFERENCE_METHOD:
                differences = "| — | — | — | — |"
            else:
                overall_mean = paired(
                    seed_mean(REFERENCE_METHOD, "overall", datasets, selected),
                    seed_mean(method, "overall", datasets, selected),
                )[0]
                mean, low, high, wins = paired(
                    seed_mean(REFERENCE_METHOD, "rest", datasets, selected),
                    seed_mean(method, "rest", datasets, selected),
                )
                differences = f"| {overall_mean:+.2f} | {mean:+.2f} | [{low:+.2f}, {high:+.2f}] | {wins}/{len(seeds)} |"
            label = "全間隔" if interval is None else str(interval)
            lines.append(
                f"| {method} | {label} | {values['post_change_fraction']:.1f}% | {values['overall']:.2f} | "
                f"{values['post_change']:.2f} | {values['rest']:.2f} " + differences
            )
    lines += [
        "",
        "## セルごとの「それ以外」の差（FedHASP−手法、pp）の符号の数（40セル＝10データ×4間隔）",
        "",
        "| 手法 | 全体の差が正のセル | それ以外の差が正のセル | それ以外の差の区間が0より上 | それ以外の差の区間が0より下 | それ以外の差の最小（セル） |",
        "|---|---|---|---|---|---|",
    ]
    for method in methods[1:]:
        overall_positive = rest_positive = rest_above = rest_below = 0
        worst = None
        for dataset in datasets:
            for interval in intervals:
                reference = [by_key[(REFERENCE_METHOD, dataset, interval, s)] for s in seeds]
                other = [by_key[(method, dataset, interval, s)] for s in seeds]
                overall_positive += paired([r["overall"] for r in reference], [r["overall"] for r in other])[0] > 0
                mean, low, high, _ = paired([r["rest"] for r in reference], [r["rest"] for r in other])
                rest_positive += mean > 0
                rest_above += low > 0
                rest_below += high < 0
                if worst is None or mean < worst[0]:
                    worst = (mean, f"{dataset}・{interval}")
        lines.append(
            f"| {method} | {overall_positive}/40 | {rest_positive}/40 | {rest_above} | {rest_below} | {worst[0]:+.2f}（{worst[1]}） |"
        )
    lines += [
        "",
        "## データ×間隔ごとの「それ以外」の精度（%、seedの平均）",
        "",
        "| データ | 間隔 | " + " | ".join(methods) + " |",
        "|---|---|" + "---|" * len(methods),
    ]
    for dataset in datasets:
        for interval in intervals:
            lines.append(
                f"| {dataset} | {interval} | "
                + " | ".join(f"{cell[(m, dataset, interval)]['rest']:.2f}" for m in methods)
                + " |"
            )
    output = arguments.output / "analysis_post_change_accuracy_breakdown.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
