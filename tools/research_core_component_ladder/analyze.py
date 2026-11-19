"""FedHASPを「基盤」から1段ずつ積んだ精度と資源を、既存の結果から並べる研究ツール（追加の実験なし）。

段（連合学習、基準版16、10 client×5000標本、10データ×間隔50・100・200・500×seed 0〜9）:
- 0a 単一モデル（標本ごとに手元で更新）: 構成`fedavg`。
- 0b 単一モデル（集約の時点だけで更新）: 構成`fedavg-local-updates-per-aggregation`（論文の表1のFedAvg）。
- 1 基盤: 標本を1件ずつ予測→損失の監視→帰属の保留（FIFO）→警報でMDLにより留まる・再利用・新しいモデルを選ぶ→
  固い割り当てで学習。予測は今の学習先のモデル1つ（構成`fedsda-single-model-prediction`）。
- 2 基盤＋モデルの混合: Fixed-Shareの重みで全保有モデルを混ぜる。各モデルは手元のパラメータだけ
  （構成`fedsda-local-parameter-prediction`）。
- 3 FedHASP（既定）: さらに同期の時点と手元の2つの版を混ぜる。
段1〜3には、ablationで1つずつ外して±0.4pp以内だった細部（クラス別の監視など）が全部入っている。

1クライアント（予備実験、1 client×5000標本、10データ×間隔50・5000×seed 0〜29）も同じ段で並べる
（`fedavg`、`fedsda-single-model-prediction`、`fedsda-local-parameter-prediction`、FedHASP＝`fedsda-training-mass-diagnostics`）。

各runで、標本ごとの正誤の平均が要約の精度と一致することを確かめ、「変化の直後」（各clientの真の概念の変化から
集約の間隔I件。1クライアントの間隔5000は50件）とそれ以外に分ける。
出力（--output）: `analysis_core_component_ladder.md`と`core_component_ladder_runs.csv`。

使い方（プロジェクトのvenv、worktreeのルートで）:
    python -m tools.research_core_component_ladder.analyze --results ../../results \\
        --output ../../results/core-component-ladder-v16 --workers 12
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

csv.field_size_limit(10**9)
DATASETS = ["sine2", "circle2", "sea2", "sea4", "mnist2", "mnist4", "fmnist2", "fmnist4", "rotmnist2", "rotmnist4"]
_PILOT = "fedhasp-single-client-pilot-v16/20261117_101402_single-client-seeds0-29"
_SIMPLIFIED = "fedhasp-simplified-v16/20261119_simplified-seeds0-9"
SETTINGS = {
    "federated": dict(
        intervals=[50, 100, 200, 500],
        seeds=list(range(10)),
        levels={
            "0a": ["method-comparison-v15/20261029_204550_fedavg-seeds0-9/fedavg", "new-datasets-baselines/fedavg"],
            "0b": [
                "method-comparison-v15/20261029_204550_fedavg-seeds0-9/fedavg-local-updates-per-aggregation",
                "new-datasets-baselines/fedavg-local-updates-per-aggregation",
            ],
            "1": [
                "fedsda-component-ablation-v16/20261029_231413_core-components-v16-seeds0-9/fedsda-single-model-prediction",
                "fedsda-component-ablation-new-datasets-v16/20261031_062435_warm-start-and-single-model-seeds0-9/fedsda-single-model-prediction",
            ],
            "2": [
                "fedsda-component-ablation-v16/20261029_231413_core-components-v16-seeds0-9/fedsda-local-parameter-prediction",
                "fedsda-component-ablation-v16/20261030_181023_other-components-seeds0-9/fedsda-local-parameter-prediction",
            ],
            "3": [
                "fedsda-component-ablation-v16/20261029_231413_core-components-v16-seeds0-9/fedsda",
                "fedsda-every-sample-change-interval-search/20261029_172159_new-datasets-seeds0-9/fedsda-every-sample-change-interval-search",
            ],
        },
    ),
    # 細部を全部外した簡素版（研究ツール`research_simplified_fedhasp`、2026-11-19追加）。
    "federated-simplified": dict(
        intervals=[50, 500],
        seeds=list(range(10)),
        levels={
            "1s": [f"{_SIMPLIFIED}/simplified-hard-prediction/fedsda"],
            "2s": [f"{_SIMPLIFIED}/simplified-local-model-mixture/fedsda"],
            "3s": [f"{_SIMPLIFIED}/simplified/fedsda"],
            "3s-na": [f"{_SIMPLIFIED}/simplified-without-adapter/fedsda"],
        },
    ),
    "single-client": dict(
        intervals=[50, 5000],
        seeds=list(range(30)),
        levels={
            "0a": [f"{_PILOT}/fedavg"],
            "1": [f"{_PILOT}/fedsda-single-model-prediction"],
            "2": [f"{_PILOT}/fedsda-local-parameter-prediction"],
            "3": [f"{_PILOT}/fedsda-training-mass-diagnostics"],
        },
    ),
}
LEVEL_LABELS = {
    "0a": "0a 単一モデル（標本ごとに更新）",
    "0b": "0b 単一モデル（集約の時点で更新、表1のFedAvg）",
    "1": "1 基盤（予測は今のモデル1つ）",
    "2": "2 基盤＋モデルの混合（手元の版だけ）",
    "3": "3 FedHASP（＋同期・手元の版の混合）",
    "1s": "1s 簡素版の基盤（予測は今のモデル1つ）",
    "2s": "2s 簡素版＋モデルの混合",
    "3s": "3s 簡素版（＋版の混合）",
    "3s-na": "3s-na 簡素版からさらにadapterを外す",
}
# 簡素版と、細部を残した同じ段の対。
SIMPLIFIED_PAIRS = (("1s", "1"), ("2s", "2"), ("3s", "3"), ("3s-na", "3"))
_UPLOAD = "communication_volume.uploaded_byte_count"
_DOWNLOAD = "communication_volume.downloaded_byte_count"
_MAC = "computation_cost_summary.client_forward_and_backward_multiply_accumulate_count_per_processed_sample"


def collect_tasks(results):
    tasks = []
    for setting, spec in SETTINGS.items():
        for level, directories in spec["levels"].items():
            seen = set()
            for directory in directories:
                for row in csv.DictReader(open(Path(results) / directory / "run_summary.csv", encoding="utf-8")):
                    key = (row["dataset_name"], int(row["server_aggregation_interval_per_client_samples"]), int(row["random_seed"]))
                    if key[0] not in DATASETS or key[1] not in spec["intervals"] or key[2] not in spec["seeds"]:
                        continue
                    if key in seen:
                        raise ValueError(f"同じrunが2回あります: {setting} {level} {key}")
                    seen.add(key)
                    tasks.append(
                        (
                            setting,
                            level,
                            *key,
                            str(Path(results) / directory / "runs" / f"{row['run_identifier']}.npz"),
                            float(row["prediction_accuracy"]),
                            (float(row[_UPLOAD]) + float(row[_DOWNLOAD])) / 1e6,
                            float(row[_MAC]),
                        )
                    )
            expected = {(d, i, s) for d in DATASETS for i in spec["intervals"] for s in spec["seeds"]}
            if seen != expected:
                raise ValueError(f"{setting} {level}: runが欠けています {sorted(expected - seen)[:3]}")
    return tasks


def summarize(task):
    setting, level, dataset, interval, seed, npz, summary_accuracy, megabytes, mac = task
    with np.load(npz, allow_pickle=False) as loaded:
        name = (
            "sample_prediction.combined_prediction_is_correct"
            if "sample_prediction.combined_prediction_is_correct" in loaded.files
            else "sample_prediction.prediction_is_correct"
        )
        correct = loaded[name].astype(bool)
        concepts = loaded["sample_prediction.observed_concept_id"]
    if abs(correct.mean() - summary_accuracy) > 1e-9:
        raise ValueError(f"正誤の平均が要約の精度と違います: {npz}")
    mask = post_change_mask(concepts, 50 if interval == 5000 else interval)
    return (setting, level, dataset, interval, seed), dict(
        overall=100 * correct.mean(),
        post=100 * correct[mask].mean() if mask.any() else float("nan"),
        rest=100 * correct[~mask].mean(),
        megabytes=megabytes,
        mac=mac,
    )


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
    with (arguments.output / "core_component_ladder_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["setting", "level", "dataset", "interval", "seed", "overall", "post_change", "rest", "communication_megabytes", "mac_per_sample"])
        for key, v in sorted(values.items()):
            writer.writerow([*key, v["overall"], v["post"], v["rest"], v["megabytes"], v["mac"]])
    lines = [
        "# FedHASPを基盤から1段ずつ積んだ精度と資源（基準版16、既存の結果）",
        "",
        "- 段1〜3には、ablationで1つずつ外して±0.4pp以内だった細部が全部入っている（細部を全部外した版は未実行）。",
        "- 精度は10データの等重み平均（%）のseedの平均 ± seedの標準偏差。増分は前の段との差（pp）で、seedで対にした95%区間。",
        "- 「直後」は各clientの真の概念の変化から集約の間隔I件（1クライアントの間隔5000は50件）。",
        "- 通信は1 runの上り＋下り（MB）、MACは処理した標本あたりのclientの順＋逆の積和、どちらも10データ・seedの平均。",
        "- 各runで、標本ごとの正誤の平均が要約の精度と一致することを確かめた。",
        "",
    ]
    for setting, spec in SETTINGS.items():
        seeds = spec["seeds"]
        lines += [
            "## "
            + {
                "federated": "連合学習（10 client）",
                "federated-simplified": "連合学習（10 client）、細部を全部外した簡素版",
                "single-client": "1クライアント",
            }[setting],
            "",
            "| 段 | 間隔 | 精度（全標本） | 前の段からの増分 | 直後 | それ以外 | 通信（MB） | MAC/標本 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        levels = list(spec["levels"])
        for interval in spec["intervals"]:
            previous = None
            for level in levels:

                def per_seed(name, level=level):
                    return [st.mean(values[(setting, level, d, interval, s)][name] for d in DATASETS) for s in seeds]

                overall = per_seed("overall")
                increment = "—"
                if previous is not None and level != "0b":
                    differences = np.array(overall) - np.array(previous)
                    half = stats.t.ppf(0.975, len(seeds) - 1) * differences.std(ddof=1) / np.sqrt(len(seeds))
                    increment = f"{differences.mean():+.2f} [{differences.mean() - half:+.2f}, {differences.mean() + half:+.2f}]"
                lines.append(
                    f"| {LEVEL_LABELS[level]} | {interval} | {st.mean(overall):.2f} ± {st.stdev(overall):.2f} | {increment} | "
                    f"{st.mean(per_seed('post')):.2f} | {st.mean(per_seed('rest')):.2f} | "
                    f"{st.mean(per_seed('megabytes')):.0f} | {st.mean(per_seed('mac')):.3g} |"
                )
                if level != "0b":
                    previous = overall
        lines.append("")
        lines += [f"### データ別の精度（間隔{spec['intervals'][0]}、seedの平均）", "", "| データ | " + " | ".join(levels) + " |", "|---|" + "---|" * len(levels)]
        for dataset in DATASETS:
            lines.append(
                f"| {dataset} | "
                + " | ".join(
                    f"{st.mean(values[(setting, level, dataset, spec['intervals'][0], s)]['overall'] for s in seeds):.2f}"
                    for level in levels
                )
                + " |"
            )
        lines.append("")
    lines += [
        "## 簡素版−細部を残した同じ段（連合学習、pp、seedで対にした95%区間、簡素版が上回ったseed）",
        "",
        "| 対 | 間隔 | 差（全標本） | 95%区間 | 上のseed | 差（直後） | 差（それ以外） |",
        "|---|---|---|---|---|---|---|",
    ]
    for simplified_level, full_level in SIMPLIFIED_PAIRS:
        for interval in SETTINGS["federated-simplified"]["intervals"]:
            seeds = SETTINGS["federated-simplified"]["seeds"]

            def per_seed(setting, level, name):
                return np.array(
                    [st.mean(values[(setting, level, d, interval, s)][name] for d in DATASETS) for s in seeds]
                )

            cells = []
            for name in ("overall", "post", "rest"):
                cells.append(per_seed("federated-simplified", simplified_level, name) - per_seed("federated", full_level, name))
            differences = cells[0]
            half = stats.t.ppf(0.975, len(seeds) - 1) * differences.std(ddof=1) / np.sqrt(len(seeds))
            lines.append(
                f"| {LEVEL_LABELS[simplified_level]} − {LEVEL_LABELS[full_level]} | {interval} | {differences.mean():+.2f} | "
                f"[{differences.mean() - half:+.2f}, {differences.mean() + half:+.2f}] | {int((differences > 0).sum())}/{len(seeds)} | "
                f"{cells[1].mean():+.2f} | {cells[2].mean():+.2f} |"
            )
    lines.append("")
    output = arguments.output / "analysis_core_component_ladder.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
