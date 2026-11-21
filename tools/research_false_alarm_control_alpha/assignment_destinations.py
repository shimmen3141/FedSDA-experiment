"""保存済みのrunから、標本ごとの学習先を組み立て、割り当ての誤りと真の変化のときの行き先を数える（追加のrunなし）。

組み立て方:
- 学習先: `adaptation`の切り替え（新しいモデルの採用・保有モデルの再利用）の`current_training_model_id`を、
  推定された変化点の標本から、次の切り替えの変化点の前まで当てはめる。最初はモデル0（事前学習の初期モデル）。
  帰属保留のFIFOによる学習の遅れは無視する（標本がどのモデルの学習へ入るかだけを見る）。
- 番号: 登録前の手元の候補（負の番号）は、同じclientの次の切り替えの「前の学習先」（登録の後の番号）に対応させる。
  次の切り替えが無いなどで分からない候補は、そのclientが採用した後の、最初のまだ対応していない登録
  （`model_registration`、採用した標本の後の同期）の番号に対応させる。どちらも無い候補は、そのclientだけの系統とする。
  検算: 真の概念で割り当てる版（`research_oracle_concept_assignment`）では、誤りが全条件で0になる。
  統合で吸収されたモデル（`model_clustering.absorbed_into_representative`）は代表へたどる（最後の代表を系統とする）。
- 系統の概念: その系統で学んだ全clientの標本の、真の概念の多数派。標本の学習先の系統の概念が、真の概念と違えば割り当ての誤り。

真の変化の行き先: 各clientの真の変化（最初の標本t、新しい概念k）について、tからmin(t+300, 次の変化)までに起きた
最初の切り替えを見る。そのclientがkを前に見たか（再来か初出か）で分け、無ければ「切り替えなし」（変化を見逃したか、遅れた）、
再利用なら、行き先の系統の概念がkかどうかで正誤を分ける。初出の「再利用（正）」は、行き先の系統の多数派がkになった場合
（ほかのclientが先にkを学んだモデル）。
サーバの統合: `model_pair_clustering`で同じ集まりにした対と、そのうち真の概念が違う対（`true_concepts_match`が0）。

使い方: python -m tools.research_false_alarm_control_alpha.assignment_destinations <名前> <構成のフォルダ> [...]
  --datasets・--intervals・--seeds で対象を絞る。結果は標準出力（Markdownの表）。
"""

import argparse
import collections
import csv
import io
import zipfile
from pathlib import Path

import numpy as np

csv.field_size_limit(10**9)
WINDOW = 300
_SWITCH_ACTIONS = ("adopt_candidate_model", "reuse_held_model")


def _runs(directory, datasets, intervals, seeds, row_filter=None):
    archive_path = directory / "runs.zip"
    archive = zipfile.ZipFile(archive_path) if archive_path.exists() else None
    with open(directory / "run_summary.csv", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row_filter and row.get(row_filter[0]) != row_filter[1]:
                continue
            interval = int(row["server_aggregation_interval_per_client_samples"])
            key = (row["dataset_name"], interval, int(row["random_seed"]))
            if key[0] not in datasets or interval not in intervals or key[2] not in seeds:
                continue
            name = f"runs/{row['run_identifier']}.npz"
            data = archive.read(name) if archive else (directory / name).read_bytes()
            with np.load(io.BytesIO(data)) as arrays:
                yield key, {k: arrays[k] for k in arrays.files}


def _lineage_function(a, interval):
    """(client, 学習先の番号, 標本の位置) → 系統の番号（登録と統合をたどる）。"""
    registrations = collections.defaultdict(list)
    for model_id, round_index, client_id in zip(
        a["model_registration.model_id"],
        a["model_registration.registered_server_round_index"],
        a["model_registration.registering_client_id"],
    ):
        if int(client_id) >= 0:
            registrations[int(client_id)].append((int(round_index), int(model_id)))
    for values in registrations.values():
        values.sort()
    # 手元の候補（負の番号）は、まず、同じclientの次の切り替えの「前の学習先」（登録の後の番号）で対応させる。
    local_to_global = {}
    used = set()
    events = collections.defaultdict(list)
    for s, c, act, previous, current in zip(
        a["adaptation.adaptation_sample_index"],
        a["adaptation.client_id"],
        a["adaptation.adaptation_action"],
        a["adaptation.previous_training_model_id"],
        a["adaptation.current_training_model_id"],
    ):
        events[int(c)].append((int(s), int(previous), int(current)))
    for client_id, values in events.items():
        values.sort()
        for (_, _, current), (_, following_previous, _) in zip(values, values[1:]):
            if current < 0 and following_previous >= 0:
                local_to_global.setdefault((client_id, current), following_previous)
    # 残りは、採用の後の最初の未対応の登録へ対応させる（同じclientの登録の順番による近似）。
    adoptions = sorted(
        (int(s), int(c), int(m))
        for s, c, act, m in zip(
            a["adaptation.adaptation_sample_index"],
            a["adaptation.client_id"],
            a["adaptation.adaptation_action"],
            a["adaptation.current_training_model_id"],
        )
        if act == "adopt_candidate_model" and int(m) < 0
    )
    used.update(local_to_global.values())
    for sample_index, client_id, local_id in adoptions:
        if (client_id, local_id) in local_to_global:
            continue
        for round_index, model_id in registrations[client_id]:
            if round_index * interval > sample_index and model_id not in used:
                local_to_global[(client_id, local_id)] = model_id
                used.add(model_id)
                break
    absorbed = {
        int(m): int(r)
        for m, r, flag in zip(
            a["model_clustering.model_id"],
            a["model_clustering.representative_model_id"],
            a["model_clustering.absorbed_into_representative"],
        )
        if flag
    }

    def lineage(client_id, model_id):
        if model_id < 0:
            if (client_id, model_id) not in local_to_global:
                return ("local", client_id, model_id)
            model_id = local_to_global[(client_id, model_id)]
        seen = set()
        while model_id in absorbed and model_id not in seen:
            seen.add(model_id)
            model_id = absorbed[model_id]
        return ("global", model_id)

    return lineage


def analyze_run(a, interval):
    concepts = a["sample_prediction.observed_concept_id"]
    client_count, sample_count = concepts.shape
    lineage = _lineage_function(a, interval)
    switches = collections.defaultdict(list)
    for s, c, act, cp, m in zip(
        a["adaptation.adaptation_sample_index"],
        a["adaptation.client_id"],
        a["adaptation.adaptation_action"],
        a["adaptation.estimated_change_point_sample_index"],
        a["adaptation.current_training_model_id"],
    ):
        if act in _SWITCH_ACTIONS:
            switches[int(c)].append((int(s), int(cp), str(act), int(m)))
    assigned = np.empty((client_count, sample_count), dtype=object)
    for c in range(client_count):
        events = sorted(switches[c])
        current = lineage(c, 0)
        start = 0
        for _, change_point, _, model_id in events:
            change_point = max(change_point, start)
            assigned[c, start:change_point] = [current] * (change_point - start)
            current = lineage(c, model_id)
            start = change_point
        assigned[c, start:] = [current] * (sample_count - start)
    counts = collections.defaultdict(collections.Counter)
    for c in range(client_count):
        for model, concept in zip(assigned[c], concepts[c]):
            counts[model][int(concept)] += 1
    label = {model: counter.most_common(1)[0][0] for model, counter in counts.items()}
    wrong = np.array(
        [[label[assigned[c, i]] != concepts[c, i] for i in range(sample_count)] for c in range(client_count)]
    )
    changes = collections.defaultdict(list)
    for c, s in zip(a["concept_change.client_id"], a["concept_change.first_new_concept_sample_index"]):
        changes[int(c)].append(int(s))
    post = np.zeros_like(wrong)
    outcomes = collections.Counter()
    for c in range(client_count):
        times = sorted(changes[c])
        for index, t in enumerate(times):
            end = times[index + 1] if index + 1 < len(times) else sample_count
            post[c, t : min(t + interval, end)] = True
            k = int(concepts[c, t])
            seen_before = k in set(int(x) for x in concepts[c, :t])
            first = next(
                (e for e in sorted(switches[c]) if t <= e[0] < min(t + WINDOW, end)), None
            )
            prefix = "再来" if seen_before else "初出"
            if first is None:
                outcomes[f"{prefix}・切り替えなし"] += 1
            elif first[2] == "adopt_candidate_model":
                outcomes[f"{prefix}・新しいモデル"] += 1
            else:
                correct = label.get(lineage(c, first[3])) == k
                outcomes[f"{prefix}・再利用（{'正' if correct else '誤'}）"] += 1
    merged = a["model_pair_clustering.assigned_to_same_cluster"].astype(bool)
    concepts_match = a["model_pair_clustering.true_concepts_match"]
    return dict(
        # 系統の概念（系統→真の概念の多数派）と、(client, 番号)→系統の関数（`research_mdl_hypothesis_trace`が使う）。
        label=label,
        # (真の概念, 学習先の系統の概念)ごとの標本の数。
        confusion=collections.Counter(
            (int(concepts[c, i]), label[assigned[c, i]])
            for c in range(client_count)
            for i in range(sample_count)
        ),
        lineage=lambda client_id, model_id: lineage(client_id, model_id),
        merged_pairs=int(merged.sum()),
        wrongly_merged_pairs=int((merged & (concepts_match == 0)).sum()),
        wrong=100 * wrong.mean(),
        wrong_post=100 * wrong[post].mean(),
        wrong_stable=100 * wrong[~post].mean(),
        lineages=len(label),
        outcomes=outcomes,
        changes=sum(len(v) for v in changes.values()),
    )


_OUTCOME_ORDER = (
    "初出・新しいモデル",
    "初出・再利用（誤）",
    "初出・再利用（正）",
    "初出・切り替えなし",
    "再来・再利用（正）",
    "再来・再利用（誤）",
    "再来・新しいモデル",
    "再来・切り替えなし",
)


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "sources",
        nargs="+",
        help="<名前> <構成のフォルダ> の組。フォルダの後に@列=値を付けると、その列がその値のrunだけを読む",
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    arguments = parser.parse_args(command_arguments)
    if len(arguments.sources) % 2:
        raise SystemExit("名前とフォルダを組にしてください")
    print(
        f"seed {min(arguments.seeds)}〜{max(arguments.seeds)}の平均。割り当ての誤りは標本の%。"
        f"真の変化の行き先は、変化の後{WINDOW}件以内の最初の切り替え（1 runあたりの件数）。"
    )
    print()
    print(
        "| データ | 間隔 | 構成 | 誤り（全体） | 直後 | 安定 | 系統の数 | 統合した対（うち概念違い） | 真の変化 | "
        + " | ".join(_OUTCOME_ORDER)
        + " |"
    )
    print("|---|---|---|---|---|---|---|---|---|" + "---|" * len(_OUTCOME_ORDER))
    results = collections.defaultdict(list)
    for name, directory in zip(arguments.sources[::2], arguments.sources[1::2]):
        directory, _, condition = directory.partition("@")
        row_filter = tuple(condition.split("=", 1)) if condition else None
        for key, a in _runs(
            Path(directory), arguments.datasets, arguments.intervals, arguments.seeds, row_filter
        ):
            results[(key[0], key[1], name)].append(analyze_run(a, key[1]))
    names = arguments.sources[::2]
    for dataset in arguments.datasets:
        for interval in arguments.intervals:
            for name in names:
                values = results[(dataset, interval, name)]
                if len(values) != len(arguments.seeds):
                    raise SystemExit(f"{name}の{dataset}・{interval}のrunの数が合いません: {len(values)}")
                n = len(values)
                outcome = {o: sum(v["outcomes"][o] for v in values) / n for o in _OUTCOME_ORDER}
                print(
                    f"| {dataset} | {interval} | {name} | "
                    f"{np.mean([v['wrong'] for v in values]):.2f} | {np.mean([v['wrong_post'] for v in values]):.2f} | "
                    f"{np.mean([v['wrong_stable'] for v in values]):.2f} | {np.mean([v['lineages'] for v in values]):.1f} | "
                    f"{np.mean([v['merged_pairs'] for v in values]):.1f}（{np.mean([v['wrongly_merged_pairs'] for v in values]):.1f}） | "
                    f"{np.mean([v['changes'] for v in values]):.0f} | "
                    + " | ".join(f"{outcome[o]:.1f}" for o in _OUTCOME_ORDER)
                    + " |"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
