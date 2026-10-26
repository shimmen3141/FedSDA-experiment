"""基準版13の候補の比較: 基準版12と、上限の撤廃だけ・案Aだけ・両方を合わせた構成の、精度と通信量の差を条件ごとに並べる。

- 基準版12・案A・組み合わせた構成: 製品の掃引のraw（`fedsda/runs/*.npz`と要約の`*.json`）。
- 上限の撤廃だけ: rawがないので、v12・v13比較の集計（`results/version12-version13-comparison-20261026/comparison.json`の
  `v13_upper_limit_only`）の、seedごとの精度と通信量を使う（同じ集計の研究用CLIの基準版12は、製品の基準版12と精度・通信が一致することを確認済み）。
- 差は同じ（dataset、間隔、seed）の対の差のseed平均。「足し合わせ」は、上限の撤廃だけの差＋案Aだけの差。

使い方:
    python -m tools.research_loss_monitor_rebaseline_v13_comparison --combined <組み合わせた構成の出力先…>
（既定の場所は下の定数。別の場所なら --default・--rebaseline・--upper-limit-only で指定する）
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_DEFAULT = [
    "results/method-comparison-v12/20261023_220120_fedsda-v12-default-seeds0-9",
    "results/method-comparison-v12/20261024_002727_fedsda-v12-mnist4-seeds10-19",
]
_REBASELINE = ["results/loss-monitor-rebaseline/20261027_uranium-decrease-rebaseline-n5000"]
_UPPER_LIMIT_ONLY = "results/version12-version13-comparison-20261026/comparison.json"
_DATASETS = ("sea2", "sea4", "circle2", "sine2", "mnist2", "mnist4")


def _run_key(path):
    name = Path(path).name
    return name.split("_")[0], int(name.split("_interval")[1].split("_")[0]), int(name.split("_seed")[1].split("_")[0])


def _collect_raw(roots):
    runs = {}
    for root in roots:
        for path in glob.glob(f"{root}/**/fedsda/runs/*.npz", recursive=True):
            key = _run_key(path)
            summary = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))["run_metrics"]
            with np.load(path) as z:
                accuracy = float(z["sample_prediction.combined_prediction_is_correct"].mean())
            value = dict(
                accuracy_pct=100 * accuracy,
                bytes=summary["communication_volume"]["uploaded_byte_count"] + summary["communication_volume"]["downloaded_byte_count"],
            )
            if key in runs and runs[key] != value:
                raise ValueError(f"同じrunの、中身の違う結果があります: {key}")
            runs[key] = value
    return runs


def _collect_upper_limit_only(path):
    cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
    return {
        (c["dataset"], c["interval"], c["seed"]): dict(accuracy_pct=c["accuracy_pct"], bytes=c["uploaded_bytes"] + c["downloaded_bytes"])
        for c in cells
        if c["arm"] == "v13_upper_limit_only"
    }


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", nargs="+", required=True)
    parser.add_argument("--default", nargs="+", default=_DEFAULT)
    parser.add_argument("--rebaseline", nargs="+", default=_REBASELINE)
    parser.add_argument("--upper-limit-only", default=_UPPER_LIMIT_ONLY)
    arguments = parser.parse_args(command_arguments)
    arms = dict(
        default=_collect_raw(arguments.default),
        upper_limit_only=_collect_upper_limit_only(arguments.upper_limit_only),
        rebaseline=_collect_raw(arguments.rebaseline),
        combined=_collect_raw(arguments.combined),
    )
    keys = set(arms["combined"])
    for name, runs in arms.items():
        missing = keys - set(runs)
        if missing:
            raise SystemExit(f"{name}に、組み合わせた構成と対になるrunがありません: {sorted(missing)[:5]}（ほか{len(missing)}件）")
    cells = defaultdict(list)
    for key in keys:
        cells[key[:2]].append(key)
    print("精度の差（pp、基準版12との対の差のseed平均）と、組み合わせた構成の通信量の比")
    print("dataset  interval seeds  上限の撤廃だけ  案Aだけ  足し合わせ  組み合わせ  (負のseed)  通信量の比")
    by_interval = defaultdict(list)
    by_dataset = defaultdict(list)
    for dataset in _DATASETS:
        for interval in (50, 100, 200, 500):
            run_keys = cells.get((dataset, interval))
            if not run_keys:
                continue

            def diff(arm, run_keys=run_keys):
                return [arms[arm][k]["accuracy_pct"] - arms["default"][k]["accuracy_pct"] for k in run_keys]

            upper, rebaseline, combined = (np.mean(diff(a)) for a in ("upper_limit_only", "rebaseline", "combined"))
            ratio = np.mean([arms["combined"][k]["bytes"] / arms["default"][k]["bytes"] for k in run_keys])
            negative = sum(x < 0 for x in diff("combined"))
            print(f"{dataset:8s} {interval:5d} {len(run_keys):5d}   {upper:+9.3f}   {rebaseline:+8.3f}  {upper + rebaseline:+9.3f}  {combined:+9.3f}  ({negative}/{len(run_keys)})  {ratio:8.4f}")
            by_interval[interval].append((upper, rebaseline, combined, ratio))
            by_dataset[dataset].append((upper, rebaseline, combined, ratio))
    print("\n間隔ごと（6データの等重み平均）: 上限の撤廃だけ / 案Aだけ / 組み合わせ / 通信量の比")
    for interval, values in sorted(by_interval.items()):
        u, r, c, q = np.mean(values, axis=0)
        print(f"  {interval:4d}: {u:+.3f} / {r:+.3f} / {c:+.3f} / {q:.4f}")
    print("データごと（4間隔の等重み平均）: 上限の撤廃だけ / 案Aだけ / 組み合わせ / 通信量の比")
    for dataset, values in by_dataset.items():
        u, r, c, q = np.mean(values, axis=0)
        print(f"  {dataset:8s}: {u:+.3f} / {r:+.3f} / {c:+.3f} / {q:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
