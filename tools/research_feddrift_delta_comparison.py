"""FedDriftのδをデータ×間隔ごとに選び、手法を同じ条件で並べる研究用の集計。

- `select`: δの調整の掃引の結果（調整用のseed）から、データ×間隔ごとに、seedの平均の全体の累積精度が最大のδを選ぶ
  （同じ値なら小さいδ）。幅を指定すると、最良の平均精度から幅に入るδのうち、平均の通信（上り＋下り）が最小のδを選ぶ。選んだδの表（CSV）と、本評価の掃引のコマンド（データとδの組ごとに1つ）を書き出す。
- `compare`: 結果の置き場所を`名前=ディレクトリ`で複数受け取り、手法（名前/構成）×データ×間隔ごとに、全体の累積精度、
  基準の手法との同じseedの対の差、上り・下りの通信byte、clientのMAC（順伝播＋推定逆伝播）を並べる。

精度はrunの要約（`run_metrics.prediction_accuracy`、5000標本の全体の累積精度。評価の設計の主指標）を使う。
期間別の精度は`python -m tools.experiments.accuracy_periods`で別に見る。
"""

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _run_key(path):
    name = Path(path).name
    dataset = name.split("_interval")[0]
    interval = int(name.split("_interval")[1].split("_")[0])
    seed = int(name.split("_seed")[1].split("_")[0])
    return dataset, interval, seed


def _read_runs(root, *, configuration_filter=None, include_computation=True):
    """`<root>/**/<構成>/runs/*.json`の要約を読む。同じ（構成、データ、間隔、seed、δ）が2つあれば止める。"""
    runs = {}
    for path in glob.glob(f"{root}/**/runs/*.json", recursive=True):
        configuration = Path(path).parent.parent.name
        if configuration_filter is not None and configuration != configuration_filter:
            continue
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        client_settings = (
            summary["run_settings"]
            .get("run_participant_settings", {})
            .get("run_client_settings", {})
        )
        delta = client_settings.get("loss_increase_threshold")
        key = (configuration, *_run_key(path), delta)
        if key in runs:
            raise ValueError(f"同じrunが2つあります: {key}")
        metrics = summary["run_metrics"]
        communication = metrics["communication_volume"]
        runs[key] = dict(
            accuracy=100 * float(metrics["prediction_accuracy"]),
            upload=communication["uploaded_byte_count"],
            download=communication["downloaded_byte_count"],
        )
        if include_computation:
            computation = metrics["computation_cost_summary"]
            runs[key]["client_mac"] = (
                computation["client_forward_multiply_accumulate_count"]
                + computation["client_estimated_backward_multiply_accumulate_count"]
            )
    return runs


def select(arguments):
    runs = _read_runs(arguments.tuning_root)
    by_cell = defaultdict(lambda: defaultdict(list))
    for (configuration, dataset, interval, seed, delta), run in runs.items():
        if configuration != "feddrift":
            continue
        by_cell[(dataset, interval)][delta].append((seed, run))
    if not by_cell:
        raise ValueError(f"FedDriftのrunがありません: {arguments.tuning_root}")
    rows = []
    for (dataset, interval), by_delta in sorted(by_cell.items()):
        seed_sets = {delta: sorted(s for s, _ in values) for delta, values in by_delta.items()}
        # 指定したδの格子のうち、1つもrunが無いδも欠けとして扱う（黙って候補から抜けないように）。
        for delta in arguments.deltas or ():
            seed_sets.setdefault(delta, [])
        expected = sorted(arguments.tuning_seeds)
        incomplete = {delta: seeds for delta, seeds in seed_sets.items() if seeds != expected}
        if incomplete and not arguments.exclude_incomplete_deltas:
            raise ValueError(
                f"{dataset}・間隔{interval}で調整用のseedがそろっていません: {incomplete}"
            )
        complete = {delta: values for delta, values in by_delta.items() if delta not in incomplete}
        if not complete:
            raise ValueError(f"{dataset}・間隔{interval}で調整用のseedがそろったδがありません")
        accuracies = {
            delta: [run["accuracy"] for _, run in values] for delta, values in complete.items()
        }
        means = {delta: float(np.mean(values)) for delta, values in accuracies.items()}
        communications = {
            delta: float(np.mean([(run["upload"] + run["download"]) / 1e6 for _, run in values]))
            for delta, values in complete.items()
        }
        best = max(means.values())
        best_delta = min(delta for delta, mean in means.items() if mean == best)
        # 幅: 最良の精度のδの、seed間の標準誤差の倍数か、固定のpp（指定が無ければ0で、最良の精度のδ）。
        if arguments.tolerance_standard_errors is not None:
            values = accuracies[best_delta]
            standard_error = (
                float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
            tolerance = arguments.tolerance_standard_errors * standard_error
        else:
            tolerance = arguments.tolerance_pp
        # 幅に入るδのうち通信が最小のδ（同じなら精度が高い方、さらに同じなら小さいδ）。
        chosen = min(
            (delta for delta, mean in means.items() if mean >= best - tolerance),
            key=lambda delta: (communications[delta], -means[delta], delta),
        )
        rows.append(
            dict(
                dataset=dataset,
                interval=interval,
                selected_delta=chosen,
                selected_mean_accuracy=means[chosen],
                selected_mean_communication_mb=communications[chosen],
                best_accuracy_delta=best_delta,
                best_mean_accuracy=best,
                best_mean_communication_mb=communications[best_delta],
                tolerance_pp=tolerance,
                # 欠けたδは、済んだseedの数と一緒に残す（例: `0.04:1/3`）。
                excluded_incomplete_deltas=" ".join(
                    f"{delta:.2f}:{len(seeds)}/{len(expected)}"
                    for delta, seeds in sorted(incomplete.items())
                ),
                **{
                    f"mean_accuracy_delta_{delta:.2f}": means.get(delta, "")
                    for delta in sorted(seed_sets)
                },
            )
        )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "|データ|間隔|選んだδ|平均精度%|平均通信MB|精度最良のδ|平均精度%|平均通信MB|幅pp|欠けて外したδ|"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"|{row['dataset']}|{row['interval']}|{row['selected_delta']:.2f}|{row['selected_mean_accuracy']:.2f}|"
            f"{row['selected_mean_communication_mb']:.1f}|{row['best_accuracy_delta']:.2f}|"
            f"{row['best_mean_accuracy']:.2f}|{row['best_mean_communication_mb']:.1f}|{row['tolerance_pp']:.3f}|"
            f"{row['excluded_incomplete_deltas']}|"
        )
    # 掃引は、データ×間隔の直積で流すので、データとδの組ごとに、そのδを選んだ間隔をまとめる。
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["selected_delta"])].append(row["interval"])
    commands = []
    for (dataset, delta), intervals in sorted(groups.items()):
        commands.append(
            " ".join(
                [
                    "python run_experiment_sweep.py --configurations feddrift",
                    f"--datasets {dataset}",
                    f"--seeds {arguments.evaluation_seeds}",
                    "--aggregation-intervals " + " ".join(str(i) for i in sorted(intervals)),
                    f"--feddrift-loss-increase-thresholds {delta:.2f}",
                    f"--client-count {arguments.client_count}",
                    f"--per-client-sample-count {arguments.per_client_sample_count}",
                    f"--workers {arguments.workers}",
                    f"--study {arguments.study}",
                    f"--label {dataset}-delta{int(round(delta * 100)):02d}",
                    f'--purpose "{arguments.purpose}"',
                ]
            )
        )
    command_path = output.with_suffix(".commands.txt")
    command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"\n選んだδ: {output}\n本評価のコマンド（{len(commands)}個）: {command_path}")
    return 0


def compare(arguments):
    groups = {}
    for item in arguments.group:
        name, root = item.split("=", 1)
        groups[name] = _read_runs(root)
    cells = defaultdict(dict)
    for name, runs in groups.items():
        for (configuration, dataset, interval, seed, _delta), run in runs.items():
            if arguments.configurations and configuration not in arguments.configurations:
                continue
            method = f"{name}/{configuration}"
            if seed in cells[(method, dataset, interval)]:
                raise ValueError(
                    f"同じ手法・条件・seedが2つあります: {method} {dataset} {interval} {seed}"
                )
            cells[(method, dataset, interval)][seed] = run
    methods = sorted({key[0] for key in cells})
    if arguments.reference not in methods:
        raise ValueError(f"基準の手法がありません: {arguments.reference}（ある手法: {methods}）")
    datasets = sorted({key[1] for key in cells})
    intervals = sorted({key[2] for key in cells})
    rows = []
    for dataset in datasets:
        for interval in intervals:
            reference = cells.get((arguments.reference, dataset, interval), {})
            for method in methods:
                runs = cells.get((method, dataset, interval))
                if not runs:
                    continue
                seeds = sorted(runs)
                paired = [s for s in seeds if s in reference]
                differences = [runs[s]["accuracy"] - reference[s]["accuracy"] for s in paired]

                def mean(field, seeds=seeds, runs=runs):
                    return float(np.mean([runs[s][field] for s in seeds]))

                rows.append(
                    dict(
                        dataset=dataset,
                        interval=interval,
                        method=method,
                        seeds=len(seeds),
                        accuracy=mean("accuracy"),
                        paired_seeds=len(paired),
                        difference=float(np.mean(differences)) if paired else float("nan"),
                        difference_se=float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
                        if len(paired) > 1
                        else float("nan"),
                        upload=mean("upload"),
                        download=mean("download"),
                        client_mac=mean("client_mac"),
                    )
                )
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"精度は%（全体の累積精度、seedの平均）。差は{arguments.reference}との同じseedの対の差（pp）とSE。通信はbyte/run、MACはclientの順伝播＋推定逆伝播/run。"
    )
    print("|データ|間隔|手法|seed|精度|差|SE|上り|下り|client MAC|")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"|{row['dataset']}|{row['interval']}|{row['method']}|{row['seeds']}|{row['accuracy']:.2f}|"
            f"{row['difference']:+.2f}|{row['difference_se']:.2f}|{row['upload']:.3g}|{row['download']:.3g}|{row['client_mac']:.3g}|"
        )
    return 0


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--tuning-root", required=True)
    choose.add_argument("--tuning-seeds", nargs="+", type=int, default=[100, 101, 102])
    choose.add_argument("--output", required=True)
    choose.add_argument(
        "--deltas", nargs="+", type=float, help="調整したδの格子（runが1つも無いδも欠けとして扱う）"
    )
    choose.add_argument(
        "--exclude-incomplete-deltas",
        action="store_true",
        help="調整用のseedがそろわないδを、そのデータ×間隔の候補から外す（外したδはCSVと表に残す）",
    )
    tolerance = choose.add_mutually_exclusive_group()
    tolerance.add_argument(
        "--tolerance-pp",
        type=float,
        default=0.0,
        help="最良の平均精度からこの幅（pp）に入るδのうち、通信が最小のδを選ぶ",
    )
    tolerance.add_argument(
        "--tolerance-standard-errors",
        type=float,
        help="幅を、最良の精度のδのseed間の標準誤差のこの倍数にする",
    )
    choose.add_argument("--evaluation-seeds", default="0-9")
    choose.add_argument("--client-count", type=int, default=10)
    choose.add_argument("--per-client-sample-count", type=int, default=5000)
    choose.add_argument("--workers", type=int, default=64)
    choose.add_argument("--study", default="feddrift-tuned-delta-v15")
    choose.add_argument(
        "--purpose",
        default="基準版15のFedDriftを、調整用のseedでデータ×間隔ごとに選んだδで評価する",
    )
    table = commands.add_parser("compare")
    table.add_argument("--group", action="append", required=True)
    table.add_argument("--reference", required=True, help="基準の手法（名前/構成）")
    table.add_argument("--configurations", nargs="*")
    table.add_argument("--output")
    arguments = parser.parse_args(command_arguments)
    try:
        return select(arguments) if arguments.command == "select" else compare(arguments)
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
