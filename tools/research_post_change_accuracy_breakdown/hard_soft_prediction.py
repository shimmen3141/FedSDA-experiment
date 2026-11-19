"""ソフト予測（FedHASPの既定）とハード予測・真の概念のOracleを、標準偏差つきで並べる研究ツール（追加の実験なし）。

連合学習（10 client×5000標本、10データ×間隔50・500×seed 0〜9）の既存の結果:
- soft: FedHASP（基準版16の既定、全モデルの重み付けと同期・手元の版の混合）。`fedhasp-oracle-v16`の`product-default`
  （表1のFedHASPと全runで一致することを、その記録で確かめ済み）。
- hard-current: 予測を今の学習先のモデル1つだけで行う構成`fedsda-single-model-prediction`（6データは
  `fedsda-component-ablation-v16`、新しい4データは`fedsda-component-ablation-new-datasets-v16`）。
- soft-oracle-assignment: FedHASPの警報・帰属先・統合を真の概念に合わせ、予測は混ぜたままの診断
  （`fedhasp-oracle-v16`の`product-default-oracle-concept-assignment`）。
- hard-oracle-assignment: 上の診断と同じ差し替えを、予測を今のモデル1つだけにした構成`fedsda-single-model-prediction`に掛けた
  ハード予測のOracle（`fedhasp-oracle-v16/20261119_hard-prediction-oracle-seeds0-9`、2026-11-19追加）。
- hard-oracle-feddrift: 真の概念でモデルを分け、その概念のモデル1つで予測するFedDrift（`feddrift-oracle-concept-assignment`）。
  学習の仕方（集約の時点だけでモデルを更新する、など）もFedHASPと違う。

各runで、標本ごとの正誤（FedHASP系は`sample_prediction.combined_prediction_is_correct`、FedDrift系は
`sample_prediction.prediction_is_correct`）の平均が要約の`prediction_accuracy`と一致することを確かめ、
「変化の直後」（各clientの真の概念の変化から集約の間隔I件）とそれ以外に分ける。
出力: `analysis_hard_soft_prediction.md`と`hard_soft_prediction_cells.csv`。

使い方（プロジェクトのvenv、worktreeのルートで）:
    python -m tools.research_post_change_accuracy_breakdown.hard_soft_prediction \\
        --results ../../results --output ../../results/hard-soft-prediction-v16 --workers 12
"""

import argparse
import csv
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from tools.research_post_change_accuracy_breakdown.analyze import post_change_mask

DATASETS = ["sine2", "circle2", "sea2", "sea4", "mnist2", "mnist4", "fmnist2", "fmnist4", "rotmnist2", "rotmnist4"]
INTERVALS = [50, 500]
SEEDS = list(range(10))
VARIANTS = {
    "soft": ["fedhasp-oracle-v16/20261102_035503_oracle-seeds0-9/product-default/fedsda"],
    "hard-current": [
        "fedsda-component-ablation-v16/20261029_231413_core-components-v16-seeds0-9/fedsda-single-model-prediction",
        "fedsda-component-ablation-new-datasets-v16/20261031_062435_warm-start-and-single-model-seeds0-9/fedsda-single-model-prediction",
    ],
    "soft-oracle-assignment": [
        "fedhasp-oracle-v16/20261102_035503_oracle-seeds0-9/product-default-oracle-concept-assignment/fedsda"
    ],
    "hard-oracle-feddrift": [
        "method-comparison-oracle-v16/20261101_194246_oracle-seeds0-9/feddrift-oracle-concept-assignment"
    ],
    "hard-oracle-assignment": [
        "fedhasp-oracle-v16/20261119_hard-prediction-oracle-seeds0-9/product-default-oracle-concept-assignment/fedsda-single-model-prediction"
    ],
}
LABELS = {
    "soft": "ソフト予測（FedHASP）",
    "hard-current": "ハード予測（今のモデル）",
    "soft-oracle-assignment": "割り当てだけ真の概念（予測はソフト）",
    "hard-oracle-feddrift": "真の概念のモデル1つ（FedDrift-Oracle）",
    "hard-oracle-assignment": "割り当てを真の概念・予測は今のモデル1つ（FedHASPのハード予測のOracle）",
}


def collect_tasks(results):
    tasks = []
    for variant, directories in VARIANTS.items():
        seen = set()
        for directory in directories:
            for row in csv.DictReader(open(Path(results) / directory / "run_summary.csv", encoding="utf-8")):
                key = (row["dataset_name"], int(row["server_aggregation_interval_per_client_samples"]), int(row["random_seed"]))
                if key[0] not in DATASETS or key[1] not in INTERVALS or key[2] not in SEEDS:
                    continue
                if key in seen:
                    raise ValueError(f"同じrunが2回あります: {variant} {key}")
                seen.add(key)
                npz = Path(results) / directory / "runs" / f"{row['run_identifier']}.npz"
                tasks.append((variant, *key, str(npz), float(row["prediction_accuracy"])))
        expected = {(d, i, s) for d in DATASETS for i in INTERVALS for s in SEEDS}
        if seen != expected:
            raise ValueError(f"{variant}: runが欠けています {sorted(expected - seen)[:5]}")
    return tasks


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
    mask = post_change_mask(concepts, interval)
    return (variant, dataset, interval, seed), dict(
        overall=100 * correct.mean(), post=100 * correct[mask].mean(), rest=100 * correct[~mask].mean()
    )


def mean_sd(values):
    return f"{st.mean(values):.2f} ± {st.stdev(values):.2f}"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(command_arguments)
    tasks = collect_tasks(arguments.results)
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        values = dict(executor.map(summarize, tasks, chunksize=8))
    arguments.output.mkdir(parents=True, exist_ok=True)
    variants = list(VARIANTS)
    lines = [
        "# ソフト予測とハード予測・真の概念のOracle（連合学習、基準版16）",
        "",
        "- 10 client×5000標本、10データ×間隔50・500×seed 0〜9。値は精度（%）のseedの平均 ± seedの標準偏差。",
        "- 差はソフト予測（FedHASP）との差（変種−ソフト、pp）で、seedで対にした差の平均 ± 標準偏差と95%区間。",
        "- 全体はデータを等しい重みで平均した値を各seedで作り、そのseedの間の平均 ± 標準偏差。",
        "- 各runで、標本ごとの正誤の平均が要約の精度と一致することを確かめた。",
        "",
    ]
    for scope, title in (("overall", "全標本"), ("post", "変化の直後（集約の間隔I件）"), ("rest", "それ以外")):
        lines += [f"## {title}", "", "| データ | 間隔 | " + " | ".join(LABELS[v] for v in variants) + " |", "|---|---|" + "---|" * len(variants)]
        for interval in INTERVALS:
            for dataset in [*DATASETS, None]:
                cells = []
                for variant in variants:
                    if dataset is None:
                        per_seed = [st.mean(values[(variant, d, interval, s)][scope] for d in DATASETS) for s in SEEDS]
                        reference = [st.mean(values[("soft", d, interval, s)][scope] for d in DATASETS) for s in SEEDS]
                    else:
                        per_seed = [values[(variant, dataset, interval, s)][scope] for s in SEEDS]
                        reference = [values[("soft", dataset, interval, s)][scope] for s in SEEDS]
                    text = mean_sd(per_seed)
                    if variant != "soft":
                        differences = np.array(per_seed) - np.array(reference)
                        half = stats.t.ppf(0.975, len(SEEDS) - 1) * differences.std(ddof=1) / np.sqrt(len(SEEDS))
                        text += f"（差 {differences.mean():+.2f} ± {differences.std(ddof=1):.2f}"
                        if dataset is None:
                            text += f"、[{differences.mean() - half:+.2f}, {differences.mean() + half:+.2f}]"
                        text += "）"
                    cells.append(text)
                label = "**全体**" if dataset is None else dataset
                lines.append(f"| {label} | {interval} | " + " | ".join(cells) + " |")
        lines.append("")
    (arguments.output / "analysis_hard_soft_prediction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (arguments.output / "hard_soft_prediction_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "dataset", "interval", "seed", "overall", "post_change", "rest"])
        for (variant, dataset, interval, seed), v in sorted(values.items()):
            writer.writerow([variant, dataset, interval, seed, v["overall"], v["post"], v["rest"]])
    print(f"wrote {arguments.output / 'analysis_hard_soft_prediction.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
