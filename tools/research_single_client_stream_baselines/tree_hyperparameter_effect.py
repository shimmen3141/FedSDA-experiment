"""木の版の、既定の木と、ハイパーパラメータを替えた木（`single_stream_tree_hyperparameters.py`）の結果を対にして比べる。

入力: --default（既定の木の木の版の出力）と--variant（替えた木の出力）。どちらも`runs/*.json`を持つ。
照合: (データ, seed)の組と、各runの系列のhashが両方で一致すること。
出力（--variantの下）: `analysis_tree_hyperparameter_effect.md`。データごとに、精度（%、seedの平均）、
差（替えた木−既定、pp、seedで対にした95%t区間、上回ったseed）、警報の回数、作ったモデルの数、最後の木の節の数、
概念の同定の内訳（全seedの合計）、1 runの秒。

使い方（プロジェクトのvenv）:
    python -m tools.research_single_client_stream_baselines.tree_hyperparameter_effect \\
        --default <既定の木の出力> --variant <替えた木の出力>
"""

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

_IDENTIFICATION = (
    "correct_reuse",
    "incorrect_reuse",
    "new_model_for_recurring_concept",
    "new_model_for_first_seen_concept",
    "stay",
)


def load_runs(directory):
    runs = {}
    for path in sorted((Path(directory) / "runs").glob("*.json")):
        if path.name.endswith(".tmp.json"):
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        key = (document["input"]["dataset_name"], int(document["input"]["random_seed"]))
        if key in runs:
            raise ValueError(f"同じ組の結果が2つあります: {key}")
        runs[key] = document
    return runs


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    arguments = parser.parse_args(command_arguments)
    default, variant = load_runs(arguments.default), load_runs(arguments.variant)
    if set(default) != set(variant):
        raise ValueError("(データ, seed)の組がそろっていません")
    for key in default:
        if default[key]["input"]["array_sha256"] != variant[key]["input"]["array_sha256"]:
            raise ValueError(f"系列のhashが違います: {key}")
    record = json.loads((arguments.variant / "tree_hyperparameters.json").read_text(encoding="utf-8"))
    datasets = sorted({k[0] for k in default})
    seeds = sorted({k[1] for k in default})
    lines = [
        "# 木のハイパーパラメータの効果（木の版）",
        "",
        f"- 替えた木: grace period {record['grace_period']}、delta {record['delta']}。既定の木: grace period 200、delta 1e-7。",
        f"- データ{len(datasets)}、seed {seeds[0]}〜{seeds[-1]}（{len(seeds)}）。照合: 組と全runの系列のhashが一致した。",
        "",
        "## 精度（%）と差（替えた木−既定、pp）",
        "",
        "| データ | 既定 | 替えた木 | 差 | 95%区間 | 上のseed |",
        "|---|---|---|---|---|---|",
    ]

    def accuracy(runs, d, s):
        return 100 * runs[(d, s)]["summary"]["prequential_accuracy"]

    for dataset in [*datasets, None]:
        selected = datasets if dataset is None else [dataset]
        per_seed_default = [st.mean(accuracy(default, d, s) for d in selected) for s in seeds]
        per_seed_variant = [st.mean(accuracy(variant, d, s) for d in selected) for s in seeds]
        mean, low, high = interval([v - b for v, b in zip(per_seed_variant, per_seed_default)])
        label = "全体（データの等重み）" if dataset is None else dataset
        wins = sum(v > b for v, b in zip(per_seed_variant, per_seed_default))
        lines.append(
            f"| {label} | {st.mean(per_seed_default):.2f} | {st.mean(per_seed_variant):.2f} | "
            f"{mean:+.2f} | [{low:+.2f}, {high:+.2f}] | {wins}/{len(seeds)} |"
        )
    lines += [
        "",
        "## 診断（seedの平均。概念の同定は全seedの合計。左が既定、右が替えた木）",
        "",
        "| データ | 警報 | 作ったモデル | 最後の木の節 | 正しい再利用 | 誤った再利用 | 再来に新しいモデル | 初出に新しいモデル | 留まる | 1 runの秒 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset in datasets:
        cells = []
        for name in ("alarm_count", "created_model_count", "final_tree_node_count"):
            values = [
                st.mean(runs[(dataset, s)]["summary"][name] for s in seeds)
                for runs in (default, variant)
            ]
            cells.append(f"{values[0]:.1f} / {values[1]:.1f}")
        totals = []
        for runs in (default, variant):
            counter = Counter()
            for s in seeds:
                counter.update(runs[(dataset, s)]["summary"]["concept_identification"])
            totals.append(counter)
        cells += [f"{totals[0][n]} / {totals[1][n]}" for n in _IDENTIFICATION]
        seconds = [st.mean(runs[(dataset, s)]["summary"]["elapsed_seconds"] for s in seeds) for runs in (default, variant)]
        cells.append(f"{seconds[0]:.1f} / {seconds[1]:.1f}")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    output = arguments.variant / "analysis_tree_hyperparameter_effect.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
