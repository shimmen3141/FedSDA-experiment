"""予測を混ぜる効果と学習を固くする効果を、直前の集約からの標本の数（集約の周期の中の位置）で分けて集計する（仮説3の切り分け）。

仮説3: 連合学習では、集約のたびに全モデルのパラメータが他のclientの学習を受けて一斉に変わる。混合の予測は、
その変化への保険になっていて、効果は集約の直後に集中する。

各標本の位置を、標本の番号を集約の間隔で割った余り（0が集約の直後の1標本目）とし、間隔を10の帯に分ける。
最初の周期（標本の番号が間隔より小さい。まだ集約が無い）は除く。コンセプトの変化の時刻は、集約の時刻と無関係に
乱数で決まる。入力のrunと、対にする方法は`recurrence_breakdown.py`と同じ（真のコンセプトの系列の一致も確かめる）。

使い方（プロジェクトのvenv、リポジトリのルートで）:
    python -m tools.research_single_client_stream_baselines.sync_phase_breakdown --output <出力先>
"""

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.research_single_client_stream_baselines.recurrence_breakdown import (
    COMPARISONS,
    HARD_MIX,
    SETTINGS,
    load_runs,
    paired_interval,
)

BIN_COUNT = 10


def run_phase_accuracy(npz_path, accuracy, interval):
    with np.load(npz_path) as npz:
        correct = npz["sample_prediction.combined_prediction_is_correct"].astype(bool)
        concepts = npz["sample_prediction.observed_concept_id"]
    if abs(correct.mean() - accuracy) > 1e-9:
        raise ValueError(f"標本ごとの正誤の平均が精度と違います: {npz_path}")
    sample_index = np.arange(correct.shape[1])
    after_first_sync = sample_index >= interval
    phase_bin = (sample_index % interval) * BIN_COUNT // interval
    result = {}
    for b in range(BIN_COUNT):
        mask = after_first_sync & (phase_bin == b)
        result[b] = float(correct[:, mask].mean()) if mask.any() else None
    return result, concepts


def fmt(value):
    return "—" if value is None else f"{value:+.2f}"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(command_arguments)
    lines = ["# 集約の周期の中の位置ごとの効果（仮説3の切り分け）", ""]
    lines.append(
        f"- 差はpp。集約の間隔を{BIN_COUNT}の帯に分ける（帯1が集約の直後）。最初の周期（まだ集約が無い）は除く。"
        "セル（データ）ごとにseedの平均をとり、データを等しい重みで平均。95%区間はseedごとのデータの平均の差のt区間。"
    )
    lines.append("- 対にした2つのrunの真のコンセプトの系列が全標本で一致することを確かめた。")
    csv_rows = []
    for setting, spec in SETTINGS.items():
        runs = {name: load_runs(roots, spec["intervals"]) for name, roots in spec["roots"].items()}
        keys = set(runs[HARD_MIX])
        for name, configuration_runs in runs.items():
            if set(configuration_runs) != keys:
                raise ValueError(f"{setting} {name}: runの組がそろっていません")
        per_run = {}
        for name, configuration_runs in runs.items():
            for key, (path, accuracy) in configuration_runs.items():
                per_run[(name, key)] = run_phase_accuracy(path, accuracy, key[1])
        for key in keys:
            reference = per_run[(HARD_MIX, key)][1]
            for name in runs:
                if not np.array_equal(per_run[(name, key)][1], reference):
                    raise ValueError(f"{setting} {key}: {name}の真のコンセプトの系列が違います")
        lines += ["", f"## {setting}", ""]
        for interval in spec["intervals"]:
            interval_keys = [k for k in keys if k[1] == interval]
            if setting.startswith("1クライアント") and interval == 5000:
                # 間隔5000（＝系列の長さ）では、系列の途中に集約が無い。
                continue
            seeds = sorted({k[2] for k in interval_keys})
            header = "| 比較 | " + " | ".join(f"帯{b + 1}" for b in range(BIN_COUNT)) + " | 帯1−帯2〜10の平均 |"
            lines += [f"### 間隔{interval}（帯の幅{interval // BIN_COUNT}標本）", "", header,
                      "|---|" + "---|" * (BIN_COUNT + 1)]
            for label, first, second in COMPARISONS:
                by_seed = defaultdict(lambda: defaultdict(list))
                for key in interval_keys:
                    a, b = per_run[(first, key)][0], per_run[(second, key)][0]
                    for bin_index in range(BIN_COUNT):
                        by_seed[key[2]][bin_index].append(100 * (a[bin_index] - b[bin_index]))
                cells = []
                for bin_index in range(BIN_COUNT):
                    mean, low, high = paired_interval([st.mean(by_seed[s][bin_index]) for s in seeds])
                    cells.append(f"{fmt(mean)} [{fmt(low)}, {fmt(high)}]")
                    csv_rows.append(dict(setting=setting, interval=interval, comparison=label, bin=bin_index + 1,
                                         mean=mean, low=low, high=high))
                contrast = [
                    st.mean(by_seed[s][0]) - st.mean(st.mean(by_seed[s][b]) for b in range(1, BIN_COUNT))
                    for s in seeds
                ]
                mean, low, high = paired_interval(contrast)
                lines.append(f"| {label} | " + " | ".join(cells) + f" | {fmt(mean)} [{fmt(low)}, {fmt(high)}] |")
            lines.append("")
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "analysis_sync_phase_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (arguments.output / "sync_phase_breakdown.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {arguments.output / 'analysis_sync_phase_breakdown.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
