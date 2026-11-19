"""変化からの経過標本数ごとの精度を、ソフト予測・ハード予測・Oracleで並べる研究ツール（追加の実験なし）。

入力の4変種と照合は`hard_soft_prediction.py`と同じ。各clientで、直前の真の概念の変化からの経過標本数
（変化の標本が0）で標本を帯に分ける。最初の変化より前の標本は「最初の変化の前」とする。
精度は各runで帯ごとに（全clientの標本をまとめて）求め、データを等しい重みで平均し、seedで平均 ± 標準偏差。
出力: `analysis_post_change_profile.md`。

使い方（プロジェクトのvenv、worktreeのルートで）:
    python -m tools.research_post_change_accuracy_breakdown.post_change_profile \\
        --results ../../results --output ../../results/hard-soft-prediction-v16 --workers 12
"""

import argparse
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tools.research_post_change_accuracy_breakdown.hard_soft_prediction import (
    DATASETS,
    INTERVALS,
    LABELS,
    SEEDS,
    VARIANTS,
    collect_tasks,
)

BINS = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, None)]
BEFORE = "最初の変化の前"


def bin_label(bin_range):
    start, end = bin_range
    return f"{start}〜" if end is None else f"{start}〜{end - 1}"


def summarize(task):
    variant, dataset, interval, seed, npz, summary_accuracy = task
    name = (
        "sample_prediction.prediction_is_correct"
        if variant == "hard-oracle-feddrift"
        else "sample_prediction.combined_prediction_is_correct"
    )
    with np.load(npz, allow_pickle=False) as loaded:
        correct = loaded[name].astype(bool)
        concepts = loaded["sample_prediction.observed_concept_id"]
    if abs(correct.mean() - summary_accuracy) > 1e-9:
        raise ValueError(f"正誤の平均が要約の精度と違います: {npz}")
    since = np.full(concepts.shape, -1, dtype=np.int64)
    for client in range(concepts.shape[0]):
        last = -1
        for index in range(concepts.shape[1]):
            if index > 0 and concepts[client, index] != concepts[client, index - 1]:
                last = index
            since[client, index] = -1 if last < 0 else index - last
    result = {BEFORE: 100 * correct[since < 0].mean()}
    for bin_range in BINS:
        start, end = bin_range
        mask = (since >= start) & ((since < end) if end is not None else True)
        result[bin_label(bin_range)] = 100 * correct[mask].mean() if mask.any() else float("nan")
    return (variant, dataset, interval, seed), result


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(command_arguments)
    tasks = collect_tasks(arguments.results)
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        values = dict(executor.map(summarize, tasks, chunksize=8))
    columns = [BEFORE, *(bin_label(b) for b in BINS)]
    lines = [
        "# 変化からの経過標本数ごとの精度（連合学習、基準版16）",
        "",
        "- 10 client×5000標本、10データ×seed 0〜9。帯は各clientの直前の真の概念の変化からの経過標本数（変化の標本が0）。",
        "- 値は10データの等重み平均の精度（%）のseedの平均 ± 標準偏差。各runで、正誤の平均が要約の精度と一致することを確かめた。",
        "",
    ]
    for interval in INTERVALS:
        lines += [f"## 間隔{interval}", "", "| 変種 | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        for variant in VARIANTS:
            cells = []
            for column in columns:
                per_seed = [
                    st.mean(values[(variant, d, interval, s)][column] for d in DATASETS) for s in SEEDS
                ]
                cells.append(f"{st.mean(per_seed):.2f} ± {st.stdev(per_seed):.2f}")
            lines.append(f"| {LABELS[variant]} | " + " | ".join(cells) + " |")
        lines.append("")
    arguments.output.mkdir(parents=True, exist_ok=True)
    output = arguments.output / "analysis_post_change_profile.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
