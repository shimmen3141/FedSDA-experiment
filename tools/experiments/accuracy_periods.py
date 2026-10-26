"""評価の補助の指標: 期間別の精度と、流れの長さごとの累積精度を、製品の掃引のrawから計算する。

主な指標は、今と同じ全体の累積精度（runの全標本の正解率。要約の`prediction_accuracy`）。補助として次を出す
（研究バックログの9、2026-10-27のユーザー判断）。どれも、全clientの標本を合わせた正解率。

- 期間（データの流れの真の概念で区切る。どの手法にも同じ区切りが当たる）:
  - 立ち上がり期（warmup）: runに出てくる概念の全部が、どこかのclientで一度現れ終わるまで（その標本の位置より前）。
  - 回復期（recovery）: 立ち上がり期の後で、clientごとに概念が変わった位置から`--recovery-window`標本（既定200。製品の回復期と同じ定義）。
  - 定常期（steady）: 立ち上がり期の後の、それ以外。
- 製品と同じ定義の回復期・定常期（立ち上がり期で分けない）も`product_recovery`・`product_stable`として出し、
  定常期の精度が要約の`stable_period_prediction_accuracy`と一致し、全体が`prediction_accuracy`と一致することを検算する。
- 流れの長さごとの累積精度: 先頭から`--prefixes`標本（clientあたり）までの正解率。既定はrunの長さの1/4・1/2・3/4・全部。

使い方:
    python -m tools.experiments.accuracy_periods --group v13=<raw の根> --group v12=<raw の根> --reference v12 [--output <csv>]
`--group 名前=ディレクトリ`は複数指定でき、各ディレクトリの下の`**/<構成>/runs/*.npz`を読む（構成はディレクトリ名）。
`--reference`を指定すると、同じ（構成、dataset、間隔、seed）の対の差（グループ−基準のグループ）も出す。
基準のグループに同じ構成が無いときは、`--reference-configuration`の構成と比べる（例: FedDriftと既定のFedSDAを比べる）。
"""

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_CORRECTNESS_ARRAY_NAMES = (
    # FedSDA: 結合した予測の正誤（要約の`prediction_accuracy`と同じ列）。
    "sample_prediction.combined_prediction_is_correct",
    # FedDrift・FedAvg。
    "sample_prediction.prediction_is_correct",
)


def _run_key(path):
    path = Path(path)
    name = path.name
    return (
        path.parent.parent.name,
        name.split("_")[0],
        int(name.split("_interval")[1].split("_")[0]),
        int(name.split("_seed")[1].split("_")[0]),
    )


def _correctness(z):
    for name in _CORRECTNESS_ARRAY_NAMES:
        if name in z.files:
            return z[name].astype(bool)
    raise ValueError("正誤の列がありません")


def _ratio(correct, mask):
    count = int(mask.sum())
    return (float(correct[mask].sum()) / count if count else float("nan")), count


def run_periods(path, *, recovery_window, prefixes):
    """1 runの期間別の精度と、長さごとの累積精度。"""
    with np.load(path) as z:
        correct = _correctness(z)
        concept = z["sample_prediction.observed_concept_id"]
    clients, n = concept.shape
    # 立ち上がり期の終わり: 全概念が、どこかのclientで一度現れた最初の標本の位置の、最大。
    first_seen = {}
    for c in range(clients):
        values, positions = np.unique(concept[c], return_index=True)
        for value, position in zip(values.tolist(), positions.tolist()):
            first_seen[value] = min(first_seen.get(value, n), position)
    warmup_end = max(first_seen.values()) + 1 if first_seen else 0
    in_recovery = np.zeros((clients, n), dtype=bool)
    for c in range(clients):
        for i in np.nonzero(concept[c, 1:] != concept[c, :-1])[0] + 1:
            in_recovery[c, i : min(n, i + recovery_window)] = True
    position = np.broadcast_to(np.arange(n), (clients, n))
    warmup = position < warmup_end
    result = dict(
        sample_count_per_client=n,
        warmup_end_sample_index=int(warmup_end),
    )
    for label, mask in (
        ("overall", np.ones_like(warmup)),
        ("warmup", warmup),
        ("recovery", ~warmup & in_recovery),
        ("steady", ~warmup & ~in_recovery),
        ("product_recovery", in_recovery),
        ("product_stable", ~in_recovery),
    ):
        value, count = _ratio(correct, mask)
        result[f"{label}_accuracy"] = value
        result[f"{label}_sample_count"] = count
    for prefix in prefixes(n):
        value, _ = _ratio(correct, position < prefix)
        result[f"cumulative_accuracy_at_{prefix}"] = value
    summary_path = Path(path).with_suffix(".json")
    if summary_path.exists():
        metrics = json.loads(summary_path.read_text(encoding="utf-8"))["run_metrics"]
        for key, ours in (("prediction_accuracy", "overall_accuracy"), ("stable_period_prediction_accuracy", "product_stable_accuracy")):
            if key in metrics and not np.isclose(metrics[key], result[ours], rtol=0, atol=1e-12):
                raise ValueError(f"要約の{key}と一致しません: {path}: {metrics[key]} != {result[ours]}")
        result["checked_against_summary"] = True
    return result


def _prefix_function(arguments):
    if arguments.prefixes:
        return lambda n: [p for p in arguments.prefixes if p <= n]
    return lambda n: sorted({max(1, n // 4), max(1, n // 2), max(1, 3 * n // 4), n})


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", required=True, help="名前=ディレクトリ（複数可）")
    parser.add_argument("--reference", help="対の差の基準にするグループの名前")
    parser.add_argument("--reference-configuration", help="基準のグループで比べる構成（省略時は同じ構成）")
    parser.add_argument("--recovery-window", type=int, default=200)
    parser.add_argument("--prefixes", type=int, nargs="+", help="累積精度を出す、clientあたりの標本数")
    parser.add_argument("--output", type=Path, help="runごとの値を書くCSV")
    arguments = parser.parse_args(command_arguments)
    prefixes = _prefix_function(arguments)
    rows = {}
    for spec in arguments.group:
        name, _, root = spec.partition("=")
        if not name or not root:
            raise SystemExit(f"--groupは 名前=ディレクトリ の形にしてください: {spec}")
        paths = sorted(glob.glob(f"{root}/**/runs/*.npz", recursive=True))
        if not paths:
            raise SystemExit(f"rawがありません: {root}")
        for path in paths:
            key = (name, *_run_key(path))
            value = run_periods(path, recovery_window=arguments.recovery_window, prefixes=prefixes)
            if key in rows and rows[key] != value:
                raise SystemExit(f"同じrunの、値の違うrawがあります: {key}")
            rows[key] = value
    metric_names = [k for k in next(iter(rows.values())) if k.endswith("_accuracy") or k.startswith("cumulative_accuracy_at_")]
    cells = defaultdict(list)
    for key, value in rows.items():
        cells[key[:4]].append((key, value))
    print("group  configuration  dataset  interval  seeds  " + "  ".join(metric_names))
    for cell, items in sorted(cells.items()):
        means = [100 * np.nanmean([v[m] for _, v in items]) if any(m in v for _, v in items) else float("nan") for m in metric_names]
        print(f"{cell[0]}  {cell[1]}  {cell[2]}  {cell[3]}  {len(items)}  " + "  ".join(f"{x:.3f}" for x in means))
    if arguments.reference:
        print(f"\npaired difference (pp) against group {arguments.reference}")
        print("group  configuration  dataset  interval  pairs  " + "  ".join(metric_names))
        for cell, items in sorted(cells.items()):
            if cell[0] == arguments.reference:
                continue
            diffs = defaultdict(list)
            for key, value in items:
                reference_key = (arguments.reference, arguments.reference_configuration or key[1], *key[2:])
                if reference_key not in rows:
                    continue
                for m in metric_names:
                    if m in value and m in rows[reference_key]:
                        diffs[m].append(100 * (value[m] - rows[reference_key][m]))
            if diffs:
                pairs = len(next(iter(diffs.values())))
                print(f"{cell[0]}  {cell[1]}  {cell[2]}  {cell[3]}  {pairs}  " + "  ".join(f"{np.nanmean(diffs[m]):+.3f}" if m in diffs else "nan" for m in metric_names))
    if arguments.output:
        names = sorted({k for v in rows.values() for k in v})
        with arguments.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["group", "configuration", "dataset", "interval", "seed", *names])
            for key, value in sorted(rows.items()):
                writer.writerow([*key, *[value.get(n, "") for n in names]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
