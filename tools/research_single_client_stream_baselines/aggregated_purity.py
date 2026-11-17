"""仮説2（集約で他のコンセプトの学習が混ざって広がる）の診断: クライアントの中の純度と、集約した後の純度を比べる。

各runの最後の学習量の記録`training_mass_by_concept`（client×model×真のコンセプト。model IDは全clientで共通）から、
- クライアントの中の純度: clientごとに、各モデルの学習量のうち最多のコンセプトの割合を、学習量で重み付けて合計した値
  （`results/fedsda-soft-training-v16/aggregate_soft_training.py`の`purity`と同じ定義）を、clientで平均する。
- 集約した後の純度: 全clientの学習量を同じmodel IDで足し合わせてから、同じ定義で求める（1つの値）。
を求める。model IDが負の行（保有をやめたモデルへ戻した学習量。-100など）は、集約した後の純度では別のモデルを
1つにまとめてしまうので除き、その割合を出す。クライアントの中の純度は、既存の定義どおり負のIDも含める。

固い学習（`fedsda-training-mass-diagnostics`）と柔らかい学習（`fedsda-soft-training`）を、データ・間隔・seedで対にし、
純度の差（固い−柔らかい）と精度の差を、セル（データ×間隔）ごとと全体で出す。追加の実験はしない。

使い方（プロジェクトのvenv、リポジトリのルートで）:
    python -m tools.research_single_client_stream_baselines.aggregated_purity --results-root <results> --output <出力先>
"""

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HARD = "fedsda-training-mass-diagnostics"
SOFT = "fedsda-soft-training"
SETTINGS = {
    "連合学習（10 client×5,000、seed 0〜9）": "fedsda-soft-training-v16/20261105_182656_soft-vs-hard-i50-i500-seeds0-9",
    "1クライアント×5,000（seed 0〜29）": "fedhasp-single-client-pilot-v16/20261117_101402_single-client-seeds0-29",
    "1クライアント×50,000（seed 0〜9）": "fedhasp-single-client-pilot-v16/20261117_195811_single-client-long-n50000-seeds0-9",
}


def purity_of(table):
    total = sum(table.values())
    if total == 0:
        return None
    models = {m for m, _ in table}
    concepts = {k for _, k in table}
    return sum(max(table.get((m, k), 0.0) for k in concepts) for m in models) / total


def run_purities(npz_path):
    with np.load(npz_path) as npz:
        clients = npz["training_mass_by_concept.client_id"]
        models = npz["training_mass_by_concept.model_id"]
        concepts = npz["training_mass_by_concept.observed_concept_id"]
        masses = npz["training_mass_by_concept.training_mass"]
    local_tables = defaultdict(lambda: defaultdict(float))
    global_table = defaultdict(float)
    excluded = 0.0
    total = 0.0
    for c, m, k, a in zip(clients, models, concepts, masses):
        local_tables[int(c)][(int(m), int(k))] += float(a)
        total += float(a)
        if int(m) < 0:
            excluded += float(a)
        else:
            global_table[(int(m), int(k))] += float(a)
    local = [p for p in (purity_of(t) for t in local_tables.values()) if p is not None]
    return (
        st.mean(local) if local else None,
        purity_of(global_table),
        excluded / total if total else None,
    )


def load(directory):
    runs = {}
    for configuration in (HARD, SOFT):
        with (directory / configuration / "run_summary.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["dataset_name"],
                    int(row["server_aggregation_interval_per_client_samples"]),
                    int(row["random_seed"]),
                )
                npz = directory / configuration / "runs" / f"{row['run_identifier']}.npz"
                local, global_purity, excluded = run_purities(npz)
                runs[(configuration, key)] = dict(
                    accuracy=100 * float(row["prediction_accuracy"]),
                    local=local,
                    global_purity=global_purity,
                    excluded=excluded,
                )
    keys = {key for (_, key) in runs}
    for key in keys:
        if (HARD, key) not in runs or (SOFT, key) not in runs:
            raise ValueError(f"対にならないrunがあります: {key}")
    return runs, sorted(keys)


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(command_arguments)
    lines = ["# クライアントの中の純度と集約した後の純度（仮説2の診断）", ""]
    lines.append(
        "- 値はセル（データ×間隔）ごとにseedの平均をとり、セルを等しい重みで平均。純度の差・精度の差は固い−柔らかい。"
        "集約した後の純度は、負のmodel IDの学習量を除いて求めた（除いた割合を併記）。"
    )
    rows = []
    for setting, relative in SETTINGS.items():
        runs, keys = load(arguments.results_root / relative)
        cells = defaultdict(list)
        for key in keys:
            cells[key[:2]].append(key)
        lines += ["", f"## {setting}", ""]
        lines.append(
            "| 間隔 | 固い: 中 | 柔らかい: 中 | 固い: 集約後 | 柔らかい: 集約後 | 除いた割合（固い／柔らかい） | "
            "精度の差 | 中の純度の差 | 集約後の純度の差 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for interval in sorted({k[1] for k in keys}):
            summary = defaultdict(list)
            for cell, cell_keys in cells.items():
                if cell[1] != interval:
                    continue
                values = defaultdict(list)
                for key in cell_keys:
                    h, s = runs[(HARD, key)], runs[(SOFT, key)]
                    for name in ("local", "global_purity", "excluded", "accuracy"):
                        values[f"hard_{name}"].append(h[name])
                        values[f"soft_{name}"].append(s[name])
                cell_mean = {name: st.mean(v for v in vals if v is not None) for name, vals in values.items()}
                for name, value in cell_mean.items():
                    summary[name].append(value)
                rows.append(dict(setting=setting, dataset=cell[0], interval=interval, **cell_mean))
            m = {name: st.mean(vals) for name, vals in summary.items()}
            lines.append(
                f"| {interval} | {m['hard_local']:.3f} | {m['soft_local']:.3f} | {m['hard_global_purity']:.3f} | "
                f"{m['soft_global_purity']:.3f} | {m['hard_excluded']:.3f}／{m['soft_excluded']:.3f} | "
                f"{m['hard_accuracy'] - m['soft_accuracy']:+.2f} | {m['hard_local'] - m['soft_local']:+.3f} | "
                f"{m['hard_global_purity'] - m['soft_global_purity']:+.3f} |"
            )
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "analysis_aggregated_purity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (arguments.output / "aggregated_purity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {arguments.output / 'analysis_aggregated_purity.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
