"""1クライアントで柔らかい学習が固い学習を上回る場面の診断（追加の実験なし）。

1クライアント（50,000標本・間隔50、5,000標本・間隔50と5000）の固い学習（`fedsda-training-mass-diagnostics`）と
柔らかい学習（`fedsda-soft-training`）を、データ・間隔・seedで対にして、次を求める。
1. 時期: 真のコンセプトの切り替わりからの経過標本数で標本を分け、区間ごとの精度の差（柔らかい−固い）。
   最初の区間（系列の先頭のコンセプト）は除き、切り替わりの後の標本だけを使う。
2. 割り当ての誤り（固い学習の記録）:
   - 学習量の不純度: 最後の学習量の記録`training_mass_by_concept`で、各モデルの学習量のうち、そのモデルの最多のコンセプト
     以外の割合（学習量で重み付け。負のmodel IDも含める）。固い学習ではモデルへ入った標本の、別のコンセプトの割合。
   - 予測のモデルの取り違え: 標本ごとの最大の重みのモデル（`sample_prediction.maximum_weight_model_id`）の最多のコンセプト
     （上の記録から決める）が、真のコンセプトと違う標本の割合。最後の記録に無いモデル（統合などで消えた）の標本は数えず、
     数えた割合（被覆）を併記する。
   - 警報の回数と、学習のモデルの切り替えの回数。
3. データ別の対応: セル（データ×間隔）ごとの、精度の差と上の量のPearsonの相関（記述値）。
柔らかい学習の側の同じ量も併記する（柔らかい学習では学習量の不純度は責任度で分けた量の不純度）。

使い方（プロジェクトのvenv、worktreeから）:
    python -m tools.research_single_client_stream_baselines.soft_training_advantage --results-root ../../results --output <出力先>
"""

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

csv.field_size_limit(10**9)

HARD = "fedsda-training-mass-diagnostics"
SOFT = "fedsda-soft-training"
SETTINGS = {
    "1クライアント×50,000（間隔50、seed 0〜9）": (
        "fedhasp-single-client-pilot-v16/20261117_195811_single-client-long-n50000-seeds0-9",
        (50,),
    ),
    "1クライアント×5,000（間隔50・5000、seed 0〜29）": (
        "fedhasp-single-client-pilot-v16/20261117_101402_single-client-seeds0-29",
        (50, 5000),
    ),
}
BINS = ((0, 30), (30, 100), (100, 300), (300, 1000), (1000, 3000), (3000, None))


def bin_label(low, high):
    return f"{low}〜" if high is None else f"{low}〜{high - 1}"


def load(directory, configuration, intervals):
    runs = {}
    with (directory / configuration / "run_summary.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            interval = int(row["server_aggregation_interval_per_client_samples"])
            if interval not in intervals:
                continue
            key = (row["dataset_name"], interval, int(row["random_seed"]))
            if key in runs:
                raise ValueError(f"同じ組のrunが2つあります: {configuration} {key}")
            runs[key] = (
                directory / configuration / "runs" / f"{row['run_identifier']}.npz",
                float(row["prediction_accuracy"]),
            )
    return runs


def read_run(path, accuracy):
    with np.load(path) as npz:
        correct = npz["sample_prediction.combined_prediction_is_correct"].astype(bool)
        concepts = npz["sample_prediction.observed_concept_id"]
        predicted_model = npz["sample_prediction.maximum_weight_model_id"]
        mass_model = npz["training_mass_by_concept.model_id"]
        mass_concept = npz["training_mass_by_concept.observed_concept_id"]
        mass = npz["training_mass_by_concept.training_mass"]
        alarm_count = int(npz["loss_change_alarm.alarm_sample_index"].size)
        switch_count = int(npz["training_model_switch.switch_sample_index"].size)
    if abs(correct.mean() - accuracy) > 1e-9:
        raise ValueError(f"標本ごとの正誤の平均が精度と違います: {path}")
    if correct.shape[0] != 1:
        raise ValueError(f"1クライアントのrunではありません: {path}")
    correct, concepts, predicted_model = correct[0], concepts[0], predicted_model[0]
    # 切り替わりからの経過標本数（先頭のコンセプトの区間は-1）。
    since_change = np.full(concepts.shape, -1, dtype=np.int64)
    start = None
    for index in range(1, concepts.size):
        if concepts[index] != concepts[index - 1]:
            start = index
        if start is not None:
            since_change[index] = index - start
    table = defaultdict(lambda: defaultdict(float))
    for m, k, a in zip(mass_model, mass_concept, mass, strict=True):
        table[int(m)][int(k)] += float(a)
    total = sum(sum(v.values()) for v in table.values())
    impurity = (
        1 - sum(max(v.values()) for v in table.values() if v) / total if total else None
    )
    majority = {m: max(v, key=v.get) for m, v in table.items() if v}
    known = np.array([int(m) in majority for m in predicted_model])
    mismatch = np.array(
        [majority.get(int(m), -999) != int(k) for m, k in zip(predicted_model, concepts, strict=True)]
    )
    return dict(
        correct=correct,
        concepts=concepts,
        since_change=since_change,
        impurity=impurity,
        mismatch=float(mismatch[known].mean()) if known.any() else None,
        coverage=float(known.mean()),
        alarms=alarm_count,
        switches=switch_count,
    )


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def fmt(value, digits=2, sign=False):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(command_arguments)
    lines = ["# 1クライアントで柔らかい学習が固い学習を上回る場面（診断）", ""]
    lines.append(
        "- 差は柔らかい−固い（pp）。セル（データ×間隔）ごとにseedの平均をとり、表の「平均」はセルを等しい重みで平均。"
        "経過標本数は、真のコンセプトが切り替わった標本から数える（系列の先頭のコンセプトの区間は除く）。"
        "固い学習と柔らかい学習の真のコンセプトの系列が全標本で一致することを確かめた。"
    )
    cell_rows = []
    for setting, (relative, intervals) in SETTINGS.items():
        directory = arguments.results_root / relative
        hard_runs, soft_runs = load(directory, HARD, intervals), load(directory, SOFT, intervals)
        if set(hard_runs) != set(soft_runs):
            raise ValueError(f"{setting}: runの組がそろっていません")
        keys = sorted(hard_runs)
        per_key = {}
        for key in keys:
            hard, soft = read_run(*hard_runs[key]), read_run(*soft_runs[key])
            if not np.array_equal(hard["concepts"], soft["concepts"]):
                raise ValueError(f"{setting} {key}: 真のコンセプトの系列が違います")
            bins = {}
            for low, high in BINS:
                mask = (hard["since_change"] >= low) & (
                    True if high is None else hard["since_change"] < high
                )
                bins[(low, high)] = (
                    100 * (soft["correct"][mask].mean() - hard["correct"][mask].mean())
                    if mask.any()
                    else None,
                    int(mask.sum()),
                )
            per_key[key] = dict(
                difference=100 * (soft["correct"].mean() - hard["correct"].mean()),
                bins=bins,
                hard=hard,
                soft=soft,
            )
        for interval in intervals:
            datasets = sorted({k[0] for k in keys if k[1] == interval})

            def cell(dataset, value):
                return mean_or_none([value(per_key[k]) for k in keys if k[:2] == (dataset, interval)])

            lines += ["", f"## {setting}・間隔{interval}", ""]
            lines.append("### 切り替わりからの経過標本数ごとの精度の差（柔らかい−固い）")
            lines.append("")
            lines.append(
                "| 経過標本数 | " + " | ".join(datasets) + " | 平均 | 標本の割合 |"
            )
            lines.append("|---|" + "---|" * (len(datasets) + 2))
            total_samples = sum(
                per_key[k]["bins"][b][1] for k in keys if k[1] == interval for b in per_key[k]["bins"]
            )
            for low, high in BINS:
                values = [cell(d, lambda r, b=(low, high): r["bins"][b][0]) for d in datasets]
                share = sum(per_key[k]["bins"][(low, high)][1] for k in keys if k[1] == interval)
                lines.append(
                    f"| {bin_label(low, high)} | "
                    + " | ".join(fmt(v, sign=True) for v in values)
                    + f" | {fmt(mean_or_none(values), sign=True)} | {100 * share / total_samples:.1f}% |"
                )
            lines += ["", "### データ別の精度の差と割り当ての誤り", ""]
            lines.append(
                "| データ | 精度の差 | 学習量の不純度（固い／柔らかい） | 予測のモデルの取り違え（固い／柔らかい） | 被覆（固い／柔らかい） | 警報の回数（固い／柔らかい） | 切り替えの回数（固い／柔らかい） |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for dataset in datasets:
                row = dict(
                    setting=setting,
                    interval=interval,
                    dataset=dataset,
                    difference=cell(dataset, lambda r: r["difference"]),
                    hard_impurity=cell(dataset, lambda r: r["hard"]["impurity"]),
                    soft_impurity=cell(dataset, lambda r: r["soft"]["impurity"]),
                    hard_mismatch=cell(dataset, lambda r: r["hard"]["mismatch"]),
                    soft_mismatch=cell(dataset, lambda r: r["soft"]["mismatch"]),
                    hard_coverage=cell(dataset, lambda r: r["hard"]["coverage"]),
                    soft_coverage=cell(dataset, lambda r: r["soft"]["coverage"]),
                    hard_alarms=cell(dataset, lambda r: r["hard"]["alarms"]),
                    soft_alarms=cell(dataset, lambda r: r["soft"]["alarms"]),
                    hard_switches=cell(dataset, lambda r: r["hard"]["switches"]),
                    soft_switches=cell(dataset, lambda r: r["soft"]["switches"]),
                )
                cell_rows.append(row)
                lines.append(
                    f"| {dataset} | {fmt(row['difference'], sign=True)} | "
                    f"{fmt(row['hard_impurity'], 3)}／{fmt(row['soft_impurity'], 3)} | "
                    f"{fmt(100 * row['hard_mismatch'], 1)}%／{fmt(100 * row['soft_mismatch'], 1)}% | "
                    f"{fmt(100 * row['hard_coverage'], 0)}%／{fmt(100 * row['soft_coverage'], 0)}% | "
                    f"{fmt(row['hard_alarms'], 1)}／{fmt(row['soft_alarms'], 1)} | "
                    f"{fmt(row['hard_switches'], 1)}／{fmt(row['soft_switches'], 1)} |"
                )
            rows = [r for r in cell_rows if r["setting"] == setting and r["interval"] == interval]
            lines += ["", "データ（セル）をまたいだ、精度の差との相関（記述値）:", ""]
            lines.append("| 量 | Pearsonの相関 |")
            lines.append("|---|---|")
            for name, label in (
                ("hard_impurity", "固い学習の学習量の不純度"),
                ("hard_mismatch", "固い学習の予測のモデルの取り違え"),
                ("hard_alarms", "固い学習の警報の回数"),
                ("hard_switches", "固い学習の切り替えの回数"),
            ):
                pairs = [(r[name], r["difference"]) for r in rows if r[name] is not None]
                if len(pairs) >= 3:
                    x, y = np.array(pairs).T
                    lines.append(f"| {label} | {float(np.corrcoef(x, y)[0, 1]):+.2f} |")
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "analysis_soft_training_advantage.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    with (arguments.output / "soft_training_advantage_cells.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    print(f"wrote {arguments.output / 'analysis_soft_training_advantage.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
