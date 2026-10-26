"""IMPROVE-031の指標の集計: 既定と変種のraw（製品の掃引のコマンドの出力）を、同じrunの対で比べる。

定義はIMPROVE-031の「指標の定義」節。
- 見逃しの割合: 概念の変化（真の概念が前の標本と違う位置）のうち、その位置から300標本以内に警報（`loss_change_alarm`）がない割合。
  runの最後の300標本未満の変化は数えない。
- 検出の遅れで説明できない混入: 適応の記録（`adaptation`）の間の区間ごとに真の概念の多数派を決め、多数派でない概念の標本のうち、
  直前の概念の変化から150標本以上たったものの数を、全標本数で割った割合。
- 精度の差: 同じ（dataset、間隔、seed）の対の差（変種−既定）。

使い方:
    python -m tools.research_loss_monitor_rebaseline_metrics --default <rawの集まり> --variant <rawの集まり> [--variant-diagnostics <出力の根>]
rawの集まりは、`*/fedsda/runs/*.npz`か`fedsda/runs/*.npz`を含むディレクトリ（複数可）。runはファイル名の（dataset、間隔、seed）で対にする。
"""

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DETECTION_WINDOW = 300
DELAY_ALLOWANCE = 150


def _table(z, name):
    columns = [n for n in z.files if n.startswith(name + ".")]
    if not columns:
        return []
    return [{c.split(".", 1)[1]: z[c][i].item() for c in columns} for i in range(len(z[columns[0]]))]


def _run_key(path):
    name = Path(path).name
    dataset = name.split("_")[0]
    interval = int(name.split("_interval")[1].split("_")[0])
    seed = int(name.split("_seed")[1].split("_")[0])
    return dataset, interval, seed


def _collect(roots):
    runs = {}
    for root in roots:
        for path in glob.glob(f"{root}/**/fedsda/runs/*.npz", recursive=True):
            key = _run_key(path)
            if key in runs:
                raise ValueError(f"同じrunが2つあります: {key}")
            runs[key] = path
    return runs


def run_metrics(path):
    """1 runの指標。"""
    z = np.load(path)
    concept = z["sample_prediction.observed_concept_id"]
    correct = z["sample_prediction.combined_prediction_is_correct"]
    clients, n = concept.shape
    alarms = defaultdict(list)
    for r in _table(z, "loss_change_alarm"):
        alarms[r["client_id"]].append(r["alarm_sample_index"])
    adaptations = defaultdict(list)
    for r in _table(z, "adaptation"):
        adaptations[r["client_id"]].append(r["adaptation_sample_index"])
    changes = missed = persistent = 0
    for c in range(clients):
        client_alarms = np.asarray(sorted(alarms[c]))
        change_positions = np.nonzero(concept[c, 1:] != concept[c, :-1])[0] + 1
        for i in change_positions:
            if i + DETECTION_WINDOW > n:
                continue
            changes += 1
            if not ((client_alarms >= i) & (client_alarms < i + DETECTION_WINDOW)).any():
                missed += 1
        # 直前の概念の変化からの標本数（最初の概念の間は大きい値）。
        since_change = np.full(n, 10**9)
        last = None
        for i in range(n):
            if i > 0 and concept[c, i] != concept[c, i - 1]:
                last = i
            if last is not None:
                since_change[i] = i - last
        bounds = [0, *sorted(adaptations[c]), n]
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end <= start:
                continue
            segment = concept[c, start:end]
            majority = Counter(segment.tolist()).most_common(1)[0][0]
            persistent += int(((segment != majority) & (since_change[start:end] >= DELAY_ALLOWANCE)).sum())
    selections = Counter(r["selected_hypothesis"] for r in _table(z, "alarm_interval_assignment_selection"))
    return dict(
        accuracy=float(correct.mean()),
        changes=changes,
        missed=missed,
        persistent=persistent,
        samples=int(concept.size),
        alarms=sum(len(v) for v in alarms.values()),
        selections=dict(selections),
    )


def _diagnostics(roots):
    found = {}
    for root in roots or []:
        for path in glob.glob(f"{root}/**/loss_monitor_rebaseline_diagnostics.json", recursive=True):
            npz = glob.glob(str(Path(path).parent / "fedsda/runs/*.npz"))
            if len(npz) == 1:
                found[_run_key(npz[0])] = json.loads(Path(path).read_text(encoding="utf-8"))["diagnostics"]
    return found


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", nargs="+", required=True)
    parser.add_argument("--variant", nargs="+", required=True)
    parser.add_argument("--variant-diagnostics", nargs="*")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(command_arguments)
    default_runs = _collect(arguments.default)
    variant_runs = _collect(arguments.variant)
    keys = sorted(set(default_runs) & set(variant_runs))
    if not keys:
        raise SystemExit("対にできるrunがありません")
    missing = sorted(set(variant_runs) - set(default_runs))
    diagnostics = _diagnostics(arguments.variant_diagnostics)
    rows = []
    for key in keys:
        d = run_metrics(default_runs[key])
        v = run_metrics(variant_runs[key])
        diag = diagnostics.get(key, {})
        rows.append(dict(dataset=key[0], interval=key[1], seed=key[2], default=d, variant=v,
                         rebaseline_count=diag.get("rebaseline_count"), decrease_alarm_count=diag.get("decrease_alarm_count")))
    print(f"paired runs: {len(rows)}" + (f"  (variant runs without default: {len(missing)})" if missing else ""))
    print("dataset  interval seeds  acc_diff_pp  negative_seeds  missed% (default->variant)  persistent% (default->variant)  alarms (default->variant)  rebaselines/run")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["dataset"], r["interval"])].append(r)
    by_dataset = defaultdict(list)
    for (dataset, interval), rs in sorted(cells.items()):
        diffs = [100 * (r["variant"]["accuracy"] - r["default"]["accuracy"]) for r in rs]
        def pooled(arm, num, den, rs=rs):
            return 100 * sum(r[arm][num] for r in rs) / max(sum(r[arm][den] for r in rs), 1)
        rebaselines = [r["rebaseline_count"] for r in rs if r["rebaseline_count"] is not None]
        print(f"{dataset:8s} {interval:5d} {len(rs):5d}  {np.mean(diffs):+10.4f}  {sum(x < 0 for x in diffs):5d}/{len(rs)}"
              f"   {pooled('default', 'missed', 'changes'):6.2f} -> {pooled('variant', 'missed', 'changes'):6.2f}"
              f"      {pooled('default', 'persistent', 'samples'):6.3f} -> {pooled('variant', 'persistent', 'samples'):6.3f}"
              f"     {np.mean([r['default']['alarms'] for r in rs]):7.1f} -> {np.mean([r['variant']['alarms'] for r in rs]):7.1f}"
              f"   {np.mean(rebaselines) if rebaselines else float('nan'):8.1f}")
        by_dataset[dataset].append(np.mean(diffs))
    print("\ndataset mean accuracy difference (intervals equally weighted):")
    for dataset, values in sorted(by_dataset.items()):
        print(f"  {dataset:8s} {np.mean(values):+.4f} pp over {len(values)} intervals")
    if arguments.output:
        arguments.output.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
