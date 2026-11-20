"""警報を、真の変化の後300件以内とそれ以外に分け、それ以外の警報の集約からの位相と、選ばれた仮説を数える。

使い方: python -m tools.research_false_alarm_control_alpha.alarm_breakdown <名前> <構成のフォルダ> [<名前> <構成のフォルダ> ...]
対象はagrawal4・randomrbf4のseed 0〜4。runs/かruns.zipから読む。
"""

import collections
import csv
import io
import sys
import zipfile

import numpy as np

csv.field_size_limit(10**9)
WINDOW = 300


def runs(directory):
    path = f"{directory}/runs.zip"
    try:
        archive = zipfile.ZipFile(path)
        read = archive.read
    except FileNotFoundError:
        archive = None

        def read(name):
            return open(f"{directory}/{name}", "rb").read()

    for row in csv.DictReader(open(f"{directory}/run_summary.csv", encoding="utf-8")):
        yield row, np.load(io.BytesIO(read(f"runs/{row['run_identifier']}.npz")))


for label, directory in zip(sys.argv[1::2], sys.argv[2::2]):
    stats = collections.defaultdict(collections.Counter)
    for row, a in runs(directory):
        interval = int(row["server_aggregation_interval_per_client_samples"])
        key = (row["dataset_name"], interval)
        if key[0] not in ("agrawal4", "randomrbf4") or int(row["random_seed"]) > 4:
            continue
        changes = collections.defaultdict(list)
        for c, s in zip(a["concept_change.client_id"], a["concept_change.first_new_concept_sample_index"]):
            changes[int(c)].append(int(s))
        concepts = a["sample_prediction.observed_concept_id"]
        sel_alarm = dict(
            ((int(c), int(s)), (h, int(m), int(n), bool(r)))
            for c, s, h, m, n, r in zip(
                a["alarm_interval_assignment_selection.client_id"],
                a["alarm_interval_assignment_selection.alarm_sample_index"],
                a["alarm_interval_assignment_selection.selected_hypothesis"],
                a["alarm_interval_assignment_selection.destination_model_id"],
                a["alarm_interval_assignment_selection.change_interval_sample_count"],
                a["alarm_interval_assignment_selection.alarm_raised"],
            )
        )
        stats[key]["true_changes"] += sum(len(v) for v in changes.values())
        for c, s in zip(a["loss_change_alarm.client_id"], a["loss_change_alarm.alarm_sample_index"]):
            c, s = int(c), int(s)
            near = any(0 <= s - t < WINDOW for t in changes[c])
            kind = "near" if near else "other"
            stats[key][kind] += 1
            if not near:
                phase = s % interval
                stats[key][f"other_phase<{interval // 5}"] += phase < interval // 5
                h = sel_alarm.get((c, s))
                if h:
                    stats[key][f"other_hyp:{h[0]}"] += 1
        stats[key]["runs"] += 1
        stats[key]["max_index"] = max(stats[key]["max_index"], int(a["loss_change_alarm.alarm_sample_index"].max(initial=0)))
    print("==", label)
    for key in sorted(stats):
        s = stats[key]
        r = s["runs"]
        hyps = {k.split(":")[1]: round(v / r, 1) for k, v in s.items() if k.startswith("other_hyp:")}
        print(
            key,
            f"真の変化 {s['true_changes'] / r:.0f}",
            f"変化後{WINDOW}件以内の警報 {s['near'] / r:.1f}",
            f"それ以外 {s['other'] / r:.1f}",
            f"（集約から最初の1/5に {s[f'other_phase<{key[1] // 5}'] / max(s['other'], 1):.2f}）",
            "仮説", hyps,
            "最大index", s["max_index"],
        )
