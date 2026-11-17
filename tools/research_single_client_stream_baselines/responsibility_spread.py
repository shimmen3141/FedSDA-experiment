"""柔らかい学習の責任度が、どれだけのモデルに分かれているかを、連合学習と1クライアントで比べる（追加の実験なし）。

仮説: 連合学習では、他のclientが作ったモデルが配られて手元にあるので、柔らかい学習の責任度が多くのモデルへ分かれ、
固い学習との精度の差が大きくなる。1クライアントでは、手元のモデルは自分が作ったものだけで、分かれ方が小さい。

各設定の柔らかい学習（`fedsda-soft-training`）のrunから、次を求める。
- 予測の時点の保有モデル数（`sample_prediction.held_model_count_at_prediction`。標本ごとで、標本の番号と対応する）。
  区間の種類（`recurrence_breakdown.segment_categories`と同じ）ごとの平均も出す。
- 責任度の実効モデル数（`training_responsibility.effective_model_count`。FIFOから放した標本ごとの、責任度の二乗和の逆数）
  の平均と、1.05を超える標本の割合（責任度が実際に2つ以上のモデルへ分かれた標本）。放した件数がclientごとに少し違い、
  標本の番号との対応を確かめていないので、区間の種類別には分けない。
- 柔らかい学習の1回の更新に参加したモデル数（`soft_training_update.participating_model_count`）の平均。
同じ組の固い学習（`fedsda-training-mass-diagnostics`）との精度の差（固い−柔らかい）を、全体と区間の種類ごとに併記する。
値は、セル（データ×間隔）ごとにseedの平均をとり、セルを等しい重みで平均。セル単位の、分かれ方と精度の差の相関（記述値）も出す。

使い方（プロジェクトのvenv、worktreeから）:
    python -m tools.research_single_client_stream_baselines.responsibility_spread --results-root ../../results --output <出力先>
"""

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.research_single_client_stream_baselines.recurrence_breakdown import (
    CATEGORIES,
    CATEGORY_LABELS,
    segment_categories,
)

csv.field_size_limit(10**9)

HARD = "fedsda-training-mass-diagnostics"
SOFT = "fedsda-soft-training"
SPREAD_THRESHOLD = 1.05
SETTINGS = {
    "連合学習（10 client×5,000、seed 0〜9）": "fedsda-soft-training-v16/20261105_182656_soft-vs-hard-i50-i500-seeds0-9",
    "1クライアント×5,000（seed 0〜29）": "fedhasp-single-client-pilot-v16/20261117_101402_single-client-seeds0-29",
    "1クライアント×50,000（seed 0〜9）": "fedhasp-single-client-pilot-v16/20261117_195811_single-client-long-n50000-seeds0-9",
}


def load(directory, configuration):
    runs = {}
    with (directory / configuration / "run_summary.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["dataset_name"],
                int(row["server_aggregation_interval_per_client_samples"]),
                int(row["random_seed"]),
            )
            if key in runs:
                raise ValueError(f"同じ組のrunが2つあります: {configuration} {key}")
            runs[key] = (
                directory / configuration / "runs" / f"{row['run_identifier']}.npz",
                float(row["prediction_accuracy"]),
            )
    return runs


def category_accuracy(correct, category):
    return {
        kind: (
            float(correct.mean())
            if kind == "all"
            else (float(correct[category == kind].mean()) if (category == kind).any() else None)
        )
        for kind in CATEGORIES
    }


def read_pair(hard_run, soft_run):
    with np.load(soft_run[0]) as npz:
        correct = npz["sample_prediction.combined_prediction_is_correct"].astype(bool)
        concepts = npz["sample_prediction.observed_concept_id"]
        held = npz["sample_prediction.held_model_count_at_prediction"]
        effective = npz["training_responsibility.effective_model_count"]
        participating = npz["soft_training_update.participating_model_count"]
    with np.load(hard_run[0]) as npz:
        hard_correct = npz["sample_prediction.combined_prediction_is_correct"].astype(bool)
        hard_concepts = npz["sample_prediction.observed_concept_id"]
    if not np.array_equal(concepts, hard_concepts):
        raise ValueError(f"固い学習と柔らかい学習の真のコンセプトの系列が違います: {soft_run[0]}")
    for run, values in ((soft_run, correct), (hard_run, hard_correct)):
        if abs(values.mean() - run[1]) > 1e-9:
            raise ValueError(f"標本ごとの正誤の平均が精度と違います: {run[0]}")
    category, _ = segment_categories(concepts)
    held_by_category = {
        kind: (
            float(held.mean())
            if kind == "all"
            else (float(held[category == kind].mean()) if (category == kind).any() else None)
        )
        for kind in CATEGORIES
    }
    soft_accuracy = category_accuracy(correct, category)
    hard_accuracy = category_accuracy(hard_correct, category)
    return dict(
        held=held_by_category,
        difference={
            kind: None
            if soft_accuracy[kind] is None or hard_accuracy[kind] is None
            else 100 * (hard_accuracy[kind] - soft_accuracy[kind])
            for kind in CATEGORIES
        },
        effective_mean=float(effective.mean()) if effective.size else None,
        spread_fraction=float((effective > SPREAD_THRESHOLD).mean()) if effective.size else None,
        participating_mean=float(participating.mean()) if participating.size else None,
    )


def cell_average(per_key, keys, interval, value):
    cells = defaultdict(list)
    for key in keys:
        if key[1] == interval:
            v = value(per_key[key])
            if v is not None:
                cells[key[0]].append(v)
    if not cells:
        return None, {}
    means = {dataset: st.mean(values) for dataset, values in cells.items()}
    return st.mean(means.values()), means


def fmt(value, digits=2, sign=False):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(command_arguments)
    lines = ["# 柔らかい学習の責任度の分かれ方（連合学習と1クライアント）", ""]
    lines.append(
        "- 値はセル（データ×間隔）ごとにseedの平均をとり、セルを等しい重みで平均。精度の差は固い−柔らかい（pp）。"
        f"「分かれた標本の割合」は、責任度の実効モデル数が{SPREAD_THRESHOLD}を超える、FIFOから放した標本の割合。"
        "保有モデル数・精度の差は柔らかい学習のrunの標本ごとの記録から、区間の種類ごとにも求めた（固い学習と真のコンセプトの系列が一致することを確かめた）。"
    )
    cell_rows = []
    for setting, relative in SETTINGS.items():
        directory = arguments.results_root / relative
        hard_runs, soft_runs = load(directory, HARD), load(directory, SOFT)
        if set(hard_runs) != set(soft_runs):
            raise ValueError(f"{setting}: 固い学習と柔らかい学習のrunの組がそろっていません")
        keys = sorted(soft_runs)
        per_key = {key: read_pair(hard_runs[key], soft_runs[key]) for key in keys}
        lines += ["", f"## {setting}", ""]
        lines.append(
            "| 間隔 | 精度の差 | 責任度の実効モデル数 | 分かれた標本の割合 | 1回の更新に参加したモデル数 | 予測の時点の保有モデル数 |"
        )
        lines.append("|---|---|---|---|---|---|")
        intervals = sorted({k[1] for k in keys})
        for interval in intervals:
            difference, difference_cells = cell_average(per_key, keys, interval, lambda r: r["difference"]["all"])
            effective, _ = cell_average(per_key, keys, interval, lambda r: r["effective_mean"])
            spread, spread_cells = cell_average(per_key, keys, interval, lambda r: r["spread_fraction"])
            participating, _ = cell_average(per_key, keys, interval, lambda r: r["participating_mean"])
            held, _ = cell_average(per_key, keys, interval, lambda r: r["held"]["all"])
            lines.append(
                f"| {interval} | {fmt(difference, sign=True)} | {fmt(effective, 3)} | {fmt(100 * spread, 1)}% | "
                f"{fmt(participating)} | {fmt(held)} |"
            )
            for dataset in sorted(difference_cells):
                cell_rows.append(
                    dict(
                        setting=setting,
                        dataset=dataset,
                        interval=interval,
                        difference=difference_cells[dataset],
                        spread_fraction=spread_cells.get(dataset),
                    )
                )
        lines += ["", "区間の種類ごと（保有モデル数／精度の差）:", ""]
        lines.append("| 間隔 | " + " | ".join(CATEGORY_LABELS[k] for k in CATEGORIES[1:]) + " |")
        lines.append("|---|" + "---|" * (len(CATEGORIES) - 1))
        for interval in intervals:
            cells = []
            for kind in CATEGORIES[1:]:
                held, _ = cell_average(per_key, keys, interval, lambda r, k=kind: r["held"][k])
                difference, _ = cell_average(per_key, keys, interval, lambda r, k=kind: r["difference"][k])
                cells.append(f"{fmt(held)}／{fmt(difference, sign=True)}")
            lines.append(f"| {interval} | " + " | ".join(cells) + " |")
    lines += ["", "## セル単位の、分かれた標本の割合と精度の差の相関（記述値）", ""]
    lines.append("| 範囲 | セルの数 | Pearsonの相関 |")
    lines.append("|---|---|---|")
    for label, selected in [("全設定", cell_rows)] + [
        (setting, [r for r in cell_rows if r["setting"] == setting]) for setting in SETTINGS
    ]:
        pairs = [(r["spread_fraction"], r["difference"]) for r in selected if r["spread_fraction"] is not None]
        if len(pairs) >= 3:
            x, y = np.array(pairs).T
            correlation = float(np.corrcoef(x, y)[0, 1])
            lines.append(f"| {label} | {len(pairs)} | {correlation:+.2f} |")
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "analysis_responsibility_spread.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (arguments.output / "responsibility_spread_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    print(f"wrote {arguments.output / 'analysis_responsibility_spread.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
