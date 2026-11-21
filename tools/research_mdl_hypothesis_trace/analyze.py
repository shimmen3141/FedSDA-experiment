"""記録した最小記述長の比較から、真の変化の後の最初の警報で、各仮説がどれだけの符号長だったかを集計する。

各警報で、記録した入力から仮説ごとの符号長を計算し直す（製品の`select_assignment_by_minimum_description_length`と
同じ式: 今のモデルのままは列の対数損失の和、変化区間の長さkで切り替わる仮説は、前のn−k件の今のモデルの和
＋切り替えの符号長＋後のk件の行き先の和（新しいモデルは記録した予測的符号長））。最短の仮説の種類と長さが、記録した
選択と合うことを確かめる。
保有モデルの概念は、`tools.research_false_alarm_control_alpha.assignment_destinations`と同じ組み立て（系統の多数派）。

対象の警報: 各clientの真の変化t（新しい概念k）の後、min(t+300, 次の変化)までの最初の警報。そのclientがkを前に
見たかで、初出と再来に分ける。
"""

import argparse
import collections
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from tools.research_false_alarm_control_alpha.assignment_destinations import analyze_run

csv.field_size_limit(10**9)
WINDOW = 300


def _hypotheses(alarm):
    """仮説ごとの最短の符号長: 今のモデル、行き先の保有モデルごと（最良のk）、新しいモデル（最良のk）。"""
    current = np.array(alarm["current_model_log_losses"])
    n = len(current)
    switching = alarm["switching_code_length"]
    grid = alarm["change_interval_grid_sample_count"]
    step = alarm["change_interval_step_sample_count"]
    ks = [k for k in range(grid, n + 1, step)]
    prefix = {k: float(current[: n - k].sum()) for k in ks}
    result = dict(current=(float(current.sum()), None))
    held = {}
    for model_id, losses in alarm["held_model_log_losses"]:
        losses = np.array(losses)
        best = min(((prefix[k] + switching + float(losses[n - k :].sum()), k) for k in ks), default=None)
        if best is not None:
            held[model_id] = best
    result["held"] = held
    new = [
        (prefix[k] + switching + alarm["new_model_code_lengths"][str(k)], k)
        for k in ks
        if str(k) in alarm["new_model_code_lengths"]
    ]
    result["new"] = min(new) if new else None
    return result


def _read(directory):
    archive_path = directory / "fedsda" / "runs.zip"
    archive = zipfile.ZipFile(archive_path) if archive_path.exists() else None
    with open(directory / "fedsda" / "run_summary.csv", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        name = f"runs/{row['run_identifier']}.npz"
        path = directory / "fedsda" / name
        data = archive.read(name) if archive else path.read_bytes()
        with np.load(io.BytesIO(data)) as arrays:
            a = {k: arrays[k] for k in arrays.files}
        trace = json.loads(
            (directory / "mdl_hypothesis_trace" / f"{row['run_identifier']}.json").read_text(encoding="utf-8")
        )
        yield row, a, trace["alarms"]


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-name", default="analysis_mdl_hypothesis_trace.md")
    arguments = parser.parse_args(command_arguments)
    groups = collections.defaultdict(list)
    mismatches = 0
    alarm_total = 0
    for row, a, alarms in _read(arguments.root):
        interval = int(row["server_aggregation_interval_per_client_samples"])
        destinations = analyze_run(a, interval)
        label, lineage = destinations["label"], destinations["lineage"]
        concepts = a["sample_prediction.observed_concept_id"]
        by_client = collections.defaultdict(list)
        for alarm in alarms:
            alarm_total += 1
            hypotheses = _hypotheses(alarm)
            candidates = [("current_model", hypotheses["current"][0])]
            candidates += [("held_model", v[0]) for v in hypotheses["held"].values()]
            if hypotheses["new"]:
                candidates.append(("new_model", hypotheses["new"][0]))
            best_kind, best_length = min(candidates, key=lambda item: item[1])
            if best_kind != alarm["selected_hypothesis"] or abs(
                best_length - alarm["selected_description_length"]
            ) > 1e-6 * max(1.0, best_length):
                mismatches += 1
            by_client[alarm["client_id"]].append((alarm["alarm_sample_index"], alarm, hypotheses))
        changes = collections.defaultdict(list)
        for c, s in zip(a["concept_change.client_id"], a["concept_change.first_new_concept_sample_index"]):
            changes[int(c)].append(int(s))
        for c, times in changes.items():
            times.sort()
            for index, t in enumerate(times):
                end = times[index + 1] if index + 1 < len(times) else concepts.shape[1]
                k = int(concepts[c, t])
                kind = "再来" if k in set(int(x) for x in concepts[c, :t]) else "初出"
                first = next(
                    (item for item in sorted(by_client[c], key=lambda x: x[0]) if t <= item[0] < min(t + WINDOW, end)),
                    None,
                )
                key = (row["dataset_name"], kind)
                if first is None:
                    groups[key].append(dict(no_alarm=True))
                    continue
                alarm_index, alarm, hypotheses = first
                n = len(alarm["current_model_log_losses"])
                after = alarm_index - t + 1
                correct_held = [
                    v for m, v in hypotheses["held"].items() if label.get(lineage(c, m)) == k
                ]
                wrong_held = [
                    v for m, v in hypotheses["held"].items() if label.get(lineage(c, m)) != k
                ]
                current_is_correct = label.get(lineage(c, alarm["current_model_id"])) == k
                post = min(after, n)
                current_losses = np.array(alarm["current_model_log_losses"])
                record = dict(
                    no_alarm=False,
                    selected=alarm["selected_hypothesis"],
                    n=n,
                    after=after,
                    current=hypotheses["current"][0],
                    current_post_loss=float(current_losses[n - post :].mean()),
                    current_is_correct=current_is_correct,
                    new=hypotheses["new"][0] if hypotheses["new"] else None,
                    new_k=hypotheses["new"][1] if hypotheses["new"] else None,
                    new_per_sample=(
                        alarm["new_model_code_lengths"][str(hypotheses["new"][1])] / hypotheses["new"][1]
                        if hypotheses["new"]
                        else None
                    ),
                    correct_held=min(correct_held)[0] if correct_held else None,
                    wrong_held=min(wrong_held)[0] if wrong_held else None,
                    switching=alarm["switching_code_length"],
                )
                # 変化の後の標本だけで測る、標本1件あたりの損失（行き先ごと）。
                if post >= 1:
                    for model_id, losses in alarm["held_model_log_losses"]:
                        mean = float(np.mean(losses[n - post :]))
                        tag = "correct_held_post_loss" if label.get(lineage(c, model_id)) == k else "wrong_held_post_loss"
                        record[tag] = min(record.get(tag, np.inf), mean)
                    code = alarm["new_model_code_lengths"].get(str(post))
                    record["new_post_per_sample"] = code / post if code is not None else None
                # 選ばれた行き先の正誤と、変化の後の標本での、選ばれた行き先の1件あたりの損失。
                if alarm["selected_hypothesis"] == "held_model":
                    destination = alarm["selected_destination_model_id"]
                    record["selected_kind"] = (
                        "保有・正" if label.get(lineage(c, destination)) == k else "保有・誤"
                    )
                    losses = dict((m, l) for m, l in alarm["held_model_log_losses"])[destination]
                    record["selected_post_loss"] = float(np.mean(losses[n - post :]))
                elif alarm["selected_hypothesis"] == "current_model":
                    record["selected_kind"] = "今のモデル・正" if current_is_correct else "今のモデル・誤"
                    record["selected_post_loss"] = record["current_post_loss"]
                else:
                    record["selected_kind"] = "新しいモデル"
                    record["selected_post_loss"] = record.get("new_post_per_sample")
                groups[key].append(record)
    lines = [
        f"警報 {alarm_total}件。記録から計算し直した最短の仮説と符号長が、記録した選択と合わない警報: {mismatches}件。",
        "",
        "真の変化の後300件以内の最初の警報。符号長はnats。差は「新しいモデルの最良−ほかの最良」で、正なら新しいモデルが負ける。",
        "",
        "| データ | 変化 | 変化の数 | 警報なし | 選ばれた: 今のモデル / 保有 / 新しい | 列の長さ n | 変化からの件数 | 正しい保有モデルがある | "
        "新しい−今のモデル | 新しい−違う保有の最良 | 新しい−正しい保有 | 切り替えの符号長 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    loss_lines = [
        "",
        "変化の後の標本だけの、標本1件あたりの損失（nats、平均）。新しいモデルは、変化の後の件数での予測的符号長÷件数。",
        "",
        "| データ | 変化 | 今のモデル（違う概念のとき） | 違う概念の保有モデルの最良 | 正しい概念の保有モデル | 新しいモデル |",
        "|---|---|---|---|---|---|",
    ]

    def mean(values):
        values = [v for v in values if v is not None and np.isfinite(v)]
        return f"{np.mean(values):.2f}（{len(values)}）" if values else "–"

    for key in sorted(groups):
        records = groups[key]
        alarmed = [r for r in records if not r["no_alarm"]]
        selected = collections.Counter(r["selected"] for r in alarmed)
        lines.append(
            f"| {key[0]} | {key[1]} | {len(records)} | {len(records) - len(alarmed)} | "
            f"{selected['current_model']} / {selected['held_model']} / {selected['new_model']} | "
            f"{mean([r['n'] for r in alarmed])} | {mean([r['after'] for r in alarmed])} | "
            f"{sum(r['correct_held'] is not None for r in alarmed)} | "
            f"{mean([r['new'] - r['current'] for r in alarmed if r['new'] is not None and not r['current_is_correct']])} | "
            f"{mean([r['new'] - r['wrong_held'] for r in alarmed if r['new'] is not None and r['wrong_held'] is not None])} | "
            f"{mean([r['new'] - r['correct_held'] for r in alarmed if r['new'] is not None and r['correct_held'] is not None])} | "
            f"{mean([r['switching'] for r in alarmed])} |"
        )
        loss_lines.append(
            f"| {key[0]} | {key[1]} | {mean([r['current_post_loss'] for r in alarmed if not r['current_is_correct']])} | "
            f"{mean([r.get('wrong_held_post_loss') for r in alarmed])} | {mean([r.get('correct_held_post_loss') for r in alarmed])} | "
            f"{mean([r.get('new_post_per_sample') for r in alarmed])} |"
        )
    kinds = ("保有・正", "保有・誤", "今のモデル・誤", "今のモデル・正", "新しいモデル")
    kind_lines = [
        "",
        "選ばれた行き先ごとの件数と、変化の後の標本での1件あたりの損失（nats、平均）。"
        "「件数: 選ばれた行き先 / 同じ警報の新しいモデル」。",
        "",
        "| データ | 変化 | " + " | ".join(kinds) + " |",
        "|---|---|" + "---|" * len(kinds),
    ]
    for key in sorted(groups):
        alarmed = [r for r in groups[key] if not r["no_alarm"]]
        cells = []
        for kind in kinds:
            chosen = [r for r in alarmed if r.get("selected_kind") == kind]
            selected_loss = [
                r["selected_post_loss"] for r in chosen if r.get("selected_post_loss") is not None
            ]
            new_loss = [
                r["new_post_per_sample"] for r in chosen if r.get("new_post_per_sample") is not None
            ]
            if selected_loss and new_loss:
                cells.append(f"{len(chosen)}: {np.mean(selected_loss):.2f} / {np.mean(new_loss):.2f}")
            else:
                cells.append(str(len(chosen)))
        kind_lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    text = "\n".join(lines + loss_lines + kind_lines) + "\n\n括弧は対象の警報の数。\n"
    (arguments.root / arguments.output_name).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
