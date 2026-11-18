"""単一ストリームの手法（Hoeffding treeの版）と、Riverの比較手法を、同じ系列・同じ条件で比べる集計。

入力:
- --single-stream: `single_stream_hard_assignment_soft_prediction.single_stream_command`の出力先（`runs/*.json`）。
- --river: `river_baselines.py`の出力先（`runs/*.json`）。繰り返して複数指定できる。事前学習ありの結果
  （`pretraining_samples_learned`がtrueか記録なし）は、条件が違うので止める（`--allow-pretrained-river`で
  別の手法として並べられる）。
- --streams: `stream_export.py`の書き出し先。全結果の系列のhashが書き出しの記録と一致することを確かめる。
照合（どれかが合わなければ止める）: 全手法で(データ, seed)の組がそろう。全結果の系列のhashが書き出しと一致する。
出力（--output）: `analysis_single_stream_comparison.md`と`single_stream_comparison_cells.csv`。
精度は%、差はpp。データごとにseedの平均±標準偏差。差の95%区間はseedで対にしたt区間。全体はデータを等しい重みで
平均し、区間はseedごとのデータの平均の差のt区間（多重比較の補正なし）。

使い方（プロジェクトのvenv）:
    python -m tools.research_single_client_stream_baselines.single_stream_comparison \\
        --single-stream <木の版の出力> --river <比較手法の出力> --streams <系列> --output <出力先>
"""

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SINGLE_STREAM_METHOD = "single-stream-hasp-hoeffding-tree"


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def load_stream_digests(streams_directory):
    digests = {}
    for path in sorted(Path(streams_directory).glob("*.json")):
        if path.name.startswith("export_provenance") or path.name.endswith(".tmp.json"):
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        digests[(metadata["dataset_name"], int(metadata["random_seed"]))] = metadata["array_sha256"]
    return digests


def load_single_stream(directory, digests):
    runs = {}
    for path in sorted((Path(directory) / "runs").glob("*.json")):
        if path.name.endswith(".tmp.json"):
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        key = (document["input"]["dataset_name"], int(document["input"]["random_seed"]))
        if digests.get(key) != document["input"]["array_sha256"]:
            raise ValueError(f"単一ストリームの手法の結果の系列が書き出しと一致しません: {path.name}")
        if key in runs:
            raise ValueError(f"同じ組の結果が2つあります: {key}")
        summary = document["summary"]
        runs[key] = dict(
            accuracy=100 * summary["prequential_accuracy"],
            elapsed_seconds=summary["elapsed_seconds"],
            kappa=summary.get("kappa"),
            kappa_temporal=summary.get("kappa_temporal"),
            alarm_count=summary.get("alarm_count"),
            final_held_model_count=summary.get("final_held_model_count"),
        )
    return runs


def load_river(directory, digests, allow_pretrained):
    runs_by_method = defaultdict(dict)
    for path in sorted((Path(directory) / "runs").glob("*.json")):
        if path.name.endswith(".tmp.json"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (result["dataset_name"], int(result["random_seed"]))
        if digests.get(key) != result["stream_array_sha256"]:
            raise ValueError(f"Riverの結果の系列が書き出しと一致しません: {path.name}")
        pretrained = result.get("pretraining_samples_learned", True)
        if pretrained and not allow_pretrained:
            raise ValueError(f"事前学習ありのRiverの結果です（条件が違う）: {path.name}")
        method = f"river:{result['method_name']}" + ("@with-pretraining" if pretrained else "")
        if key in runs_by_method[method]:
            raise ValueError(f"同じ組の結果が2つあります: {method} {key}")
        runs_by_method[method][key] = dict(
            accuracy=100 * result["prediction_accuracy"], elapsed_seconds=result["elapsed_seconds"]
        )
    return runs_by_method


def fmt(value, sign=False):
    if value is None:
        return "—"
    return f"{value:+.2f}" if sign else f"{value:.2f}"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--single-stream", type=Path, required=True)
    parser.add_argument("--river", type=Path, action="append", required=True)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-pretrained-river", action="store_true")
    arguments = parser.parse_args(command_arguments)
    digests = load_stream_digests(arguments.streams)
    runs = {SINGLE_STREAM_METHOD: load_single_stream(arguments.single_stream, digests)}
    for directory in arguments.river:
        for method, method_runs in load_river(directory, digests, arguments.allow_pretrained_river).items():
            if method in runs:
                raise ValueError(f"{method}の結果が複数の--riverにあります")
            runs[method] = method_runs
    keys = set(runs[SINGLE_STREAM_METHOD])
    for method, method_runs in runs.items():
        if set(method_runs) != keys:
            raise ValueError(f"{method}: (データ, seed)の組が単一ストリームの手法とそろっていません")
    datasets = sorted({k[0] for k in keys})
    seeds = sorted({k[1] for k in keys})
    methods = [SINGLE_STREAM_METHOD] + sorted(m for m in runs if m != SINGLE_STREAM_METHOD)
    lines = ["# 単一ストリームの手法（Hoeffding treeの版）と比較手法", ""]
    lines.append(
        f"- データ{len(datasets)}、seed {seeds[0]}〜{seeds[-1]}（{len(seeds)}）。精度は%、差はpp。"
        "照合: 全手法で組がそろい、全結果の系列のhashが書き出しと一致した。"
    )
    lines += ["", "## 精度（%、seedの平均 ± 標準偏差）", ""]
    lines.append("| データ | " + " | ".join(methods) + " |")
    lines.append("|---|" + "---|" * len(methods))
    cell_rows = []
    for dataset in datasets:
        cells = []
        for method in methods:
            values = [runs[method][(dataset, s)]["accuracy"] for s in seeds]
            sd = st.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{st.mean(values):.2f} ± {sd:.2f}")
            cell_rows.append(dict(dataset=dataset, method=method, mean=st.mean(values), sd=sd))
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines.append(
        "| 平均 | "
        + " | ".join(
            f"{st.mean(st.mean(runs[m][(d, s)]['accuracy'] for s in seeds) for d in datasets):.2f}" for m in methods
        )
        + " |"
    )
    lines += ["", "## 単一ストリームの手法との差（単一ストリームの手法−比較手法、pp）", ""]
    lines.append("| 比較手法 | 範囲 | 平均 | 95%区間 | 上のseed |")
    lines.append("|---|---|---|---|---|")
    for method in methods[1:]:
        for dataset in [None, *datasets]:
            selected = datasets if dataset is None else [dataset]
            per_seed = [
                st.mean(
                    runs[SINGLE_STREAM_METHOD][(d, s)]["accuracy"] - runs[method][(d, s)]["accuracy"]
                    for d in selected
                )
                for s in seeds
            ]
            mean, low, high = paired_interval(per_seed)
            label = "全体（データの等重み）" if dataset is None else dataset
            lines.append(
                f"| {method} | {label} | {fmt(mean, True)} | [{fmt(low, True)}, {fmt(high, True)}] | "
                f"{sum(v > 0 for v in per_seed)}/{len(per_seed)} |"
            )
    lines += ["", "## 単一ストリームの手法の診断と経過時間（seedの平均）", ""]
    lines.append("| データ | Kappa | Kappa-Temporal | 警報の回数 | 最後のモデルの数 | 1 runの秒（単一ストリームの手法） |")
    lines.append("|---|---|---|---|---|---|")
    for dataset in datasets:
        selected = [runs[SINGLE_STREAM_METHOD][(dataset, s)] for s in seeds]

        def mean_of(name, rows=selected):
            values = [r[name] for r in rows if r[name] is not None]
            return st.mean(values) if values else None

        lines.append(
            f"| {dataset} | {fmt(mean_of('kappa'))} | {fmt(mean_of('kappa_temporal'))} | "
            f"{fmt(mean_of('alarm_count'))} | {fmt(mean_of('final_held_model_count'))} | {fmt(mean_of('elapsed_seconds'))} |"
        )
    lines += ["", "比較手法の1 runの秒（seedの平均）:", ""]
    lines.append("| データ | " + " | ".join(methods[1:]) + " |")
    lines.append("|---|" + "---|" * (len(methods) - 1))
    for dataset in datasets:
        lines.append(
            f"| {dataset} | "
            + " | ".join(f"{st.mean(runs[m][(dataset, s)]['elapsed_seconds'] for s in seeds):.1f}" for m in methods[1:])
            + " |"
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "analysis_single_stream_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (arguments.output / "single_stream_comparison_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    print(f"wrote {arguments.output / 'analysis_single_stream_comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
