"""学習を固くする効果と予測を混ぜる効果を、コンセプトの区間の種類ごとに分けて集計する（仮説1の切り分け）。

仮説1: 混合の予測と固い割り当てが効くのは、新しいコンセプトに合う既存のモデルがあるとき（再来、または
連合学習で他のclientがそのコンセプトを先に経験しているとき）に限られる。1クライアントでは、初めて出たコンセプトに
合うモデルが無く、効果が小さくなる。

各clientの評価の系列を、真のコンセプトが変わらない区間に分け、区間を次の4種類に分ける（標本の時刻はclientの間で
そろっている。連合学習ではclientは同じ標本の番号を同時に処理する）。
- start: 系列の最初の区間（概念0。初期モデルは概念0で事前学習してある）。
- recurrence: そのclientが前に経験したコンセプト（概念0は事前学習で経験済みとみなす）。
- seen_by_others: そのclientは初めてだが、別のclientが区間の開始より前に経験したコンセプト（連合学習だけ）。
- new: どのclientもまだ経験していないコンセプト。

各runで、種類ごとに正解の割合を出す（区間の全標本と、区間の最初のW標本の2通り）。構成の組の差は、データ・間隔・seedで
対にして、種類ごとに、セル（データ×間隔）の平均を等しい重みで平均する。95%区間は、seedごとのセルの平均の差のt区間。
対にする2つのrunの真のコンセプトの系列が一致しなければ止める。

使い方（プロジェクトのvenv、リポジトリのルートで）:
    python -m tools.research_single_client_stream_baselines.recurrence_breakdown --output <出力先>
入力は下の`SETTINGS`の結果のディレクトリ（results/の下）。
"""

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

csv.field_size_limit(10**9)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS = _REPOSITORY_ROOT / "results"
CATEGORIES = ("start", "recurrence", "seen_by_others", "new")
CATEGORY_LABELS = {
    "start": "最初の区間（概念0）",
    "recurrence": "再来（自分が経験済み）",
    "seen_by_others": "自分は初めて・他clientが経験済み",
    "new": "全体で初めて",
}
HARD_MIX, SOFT_MIX, HARD_SINGLE, SOFT_SINGLE = "hard_mix", "soft_mix", "hard_single", "soft_single"
COMPARISONS = (
    ("学習を固くする効果（混合の予測）", HARD_MIX, SOFT_MIX),
    ("学習を固くする効果（現在のモデルだけの予測）", HARD_SINGLE, SOFT_SINGLE),
    ("予測を混ぜる効果（固い学習）", HARD_MIX, HARD_SINGLE),
    ("予測を混ぜる効果（柔らかい学習）", SOFT_MIX, SOFT_SINGLE),
)
_PILOT = "fedhasp-single-client-pilot-v16/20261117_101402_single-client-seeds0-29/"
_SOFT = "fedsda-soft-training-v16/20261105_182656_soft-vs-hard-i50-i500-seeds0-9/"
SETTINGS = {
    "連合学習（10 client、間隔50・500、seed 0〜9）": dict(
        intervals=(50, 500),
        roots={
            HARD_MIX: [_SOFT + "fedsda-training-mass-diagnostics"],
            SOFT_MIX: [_SOFT + "fedsda-soft-training"],
            HARD_SINGLE: [
                "fedsda-component-ablation-v16/20261029_231413_core-components-v16-seeds0-9/fedsda-single-model-prediction",
                "fedsda-component-ablation-new-datasets-v16/20261031_062435_warm-start-and-single-model-seeds0-9/fedsda-single-model-prediction",
            ],
            SOFT_SINGLE: [
                "fedsda-soft-training-single-model-prediction-v16/20261107_134630_soft-single-i50-i500-seeds0-9/fedsda-soft-training-single-model-prediction"
            ],
        },
    ),
    "1クライアント（間隔50・5000、seed 0〜29）": dict(
        intervals=(50, 5000),
        roots={
            HARD_MIX: [_PILOT + "fedsda-training-mass-diagnostics"],
            SOFT_MIX: [_PILOT + "fedsda-soft-training"],
            HARD_SINGLE: [_PILOT + "fedsda-single-model-prediction"],
            SOFT_SINGLE: [_PILOT + "fedsda-soft-training-single-model-prediction"],
        },
    ),
}


def segment_categories(concepts):
    """concepts: [client, sample]の真のコンセプト。標本ごとの区間の種類と、区間の中の位置を返す。"""
    client_count, sample_count = concepts.shape
    category = np.empty(concepts.shape, dtype=object)
    position = np.zeros(concepts.shape, dtype=np.int64)
    # 各コンセプトを、各clientが最初に経験した標本の番号（概念0は事前学習で経験済みとみなし-1）。
    first_seen = defaultdict(lambda: np.full(client_count, sample_count + 1, dtype=np.int64))
    for client in range(client_count):
        for index in range(sample_count):
            concept = int(concepts[client, index])
            if first_seen[concept][client] > index:
                first_seen[concept][client] = index
    for client in range(client_count):
        start = 0
        while start < sample_count:
            concept = int(concepts[client, start])
            end = start
            while end < sample_count and concepts[client, end] == concept:
                end += 1
            if start == 0:
                kind = "start"
            elif concept == 0 or first_seen[concept][client] < start:
                kind = "recurrence"
            elif np.any(np.delete(first_seen[concept], client) < start):
                kind = "seen_by_others"
            else:
                kind = "new"
            category[client, start:end] = kind
            position[client, start:end] = np.arange(end - start)
            start = end
    return category, position


def load_runs(roots, intervals):
    runs = {}
    for root in roots:
        directory = RESULTS / root
        with (directory / "run_summary.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                interval = int(row["server_aggregation_interval_per_client_samples"])
                if interval not in intervals:
                    continue
                key = (row["dataset_name"], interval, int(row["random_seed"]))
                if key in runs:
                    raise ValueError(f"同じ組のrunが2つあります: {root} {key}")
                runs[key] = (directory / "runs" / f"{row['run_identifier']}.npz", float(row["prediction_accuracy"]))
    return runs


def run_category_accuracy(npz_path, accuracy, window):
    with np.load(npz_path) as npz:
        correct = npz["sample_prediction.combined_prediction_is_correct"].astype(bool)
        concepts = npz["sample_prediction.observed_concept_id"]
    if abs(correct.mean() - accuracy) > 1e-9:
        raise ValueError(f"標本ごとの正誤の平均が精度と違います: {npz_path}")
    category, position = segment_categories(concepts)
    result = {}
    for kind in CATEGORIES:
        for scope, mask in (("all", category == kind), (f"first{window}", (category == kind) & (position < window))):
            count = int(mask.sum())
            result[(kind, scope)] = (float(correct[mask].mean()) if count else None, count)
    return result, concepts


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def fmt(value):
    return "—" if value is None else f"{value:+.2f}"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=200)
    arguments = parser.parse_args(command_arguments)
    window = arguments.window
    scopes = ("all", f"first{window}")
    lines = ["# 区間の種類ごとの効果（仮説1の切り分け）", ""]
    lines.append(
        f"- 差はpp。セル（データ×間隔）ごとにseedの平均をとり、セルを等しい重みで平均。95%区間はseedごとのセルの平均の差のt区間"
        f"（多重比較の補正なし）。「最初の{window}標本」は区間の開始から{window}標本。"
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
                per_run[(name, key)] = run_category_accuracy(path, accuracy, window)
        for key in keys:
            reference = per_run[(HARD_MIX, key)][1]
            for name in runs:
                if not np.array_equal(per_run[(name, key)][1], reference):
                    raise ValueError(f"{setting} {key}: {name}の真のコンセプトの系列が違います")
        seeds = sorted({s for _, _, s in keys})
        lines += ["", f"## {setting}", ""]
        total = defaultdict(int)
        for key in keys:
            for kind in CATEGORIES:
                total[kind] += per_run[(HARD_MIX, key)][0][(kind, "all")][1]
        grand = sum(total.values())
        lines.append(
            "- 標本の割合: "
            + "、".join(f"{CATEGORY_LABELS[k]} {100 * total[k] / grand:.1f}%" for k in CATEGORIES)
        )
        lines.append("")
        lines.append("| 比較 | 区間の種類 | 範囲 | 間隔 | 平均 | 95%区間 | 上のseed | セルの数 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        interval_sets = [("両方", spec["intervals"])] + [(str(i), (i,)) for i in spec["intervals"]]
        for label, first, second in COMPARISONS:
            for kind in CATEGORIES:
                for scope, (interval_label, selected_intervals) in [
                    (scope, interval_set) for scope in scopes for interval_set in interval_sets
                ]:
                    cell_values = defaultdict(list)
                    for key in keys:
                        if key[1] not in selected_intervals:
                            continue
                        a = per_run[(first, key)][0][(kind, scope)][0]
                        b = per_run[(second, key)][0][(kind, scope)][0]
                        if a is not None:
                            cell_values[key[:2]].append((key[2], 100 * (a - b)))
                    if not cell_values:
                        continue
                    cell_means = [st.mean(v for _, v in values) for values in cell_values.values()]
                    by_seed = defaultdict(list)
                    for values in cell_values.values():
                        for seed, value in values:
                            by_seed[seed].append(value)
                    seed_means = [st.mean(by_seed[s]) for s in seeds if by_seed[s]]
                    _, low, high = paired_interval(seed_means)
                    mean = st.mean(cell_means)
                    scope_label = "区間の全標本" if scope == "all" else f"最初の{window}標本"
                    lines.append(
                        f"| {label} | {CATEGORY_LABELS[kind]} | {scope_label} | {interval_label} | {fmt(mean)} | "
                        f"[{fmt(low)}, {fmt(high)}] | {sum(m > 0 for m in seed_means)}/{len(seed_means)} | {len(cell_means)} |"
                    )
                    csv_rows.append(
                        dict(setting=setting, comparison=label, category=kind, scope=scope, intervals=interval_label,
                             mean=mean, low=low, high=high,
                             cells=len(cell_means), seeds=len(seed_means))
                    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "analysis_recurrence_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (arguments.output / "recurrence_breakdown.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {arguments.output / 'analysis_recurrence_breakdown.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
