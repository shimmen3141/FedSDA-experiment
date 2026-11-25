"""`run.py` の変種を `product-default` と対にして比べる（再実験なし）。

- 学習が同じことの確認: 予測と重みの診断を除く全列（通信量、計算量、検出・適応、モデルの数）が、対の run で一致するか。
- 精度: 全体、変化の後の期間（各clientで概念が変わってから `--post-change-window` 標本）とそれ以外、概念別。
- 標準指標: 上り・下りの通信量（byte）、client の MAC（順伝播＋推定の逆伝播）とサーバの MAC。経過時間は run の記録に
  残らないので、`--log` に掃引のコマンドの標準出力を渡したときだけ、表示の `(<秒>s)` を集める。
- 診断: 共有率を上げた回数と警報の数（`weight_rule_trace`）、重みの1位の入れ替わり、有効なモデルの数。
結果は `<出力先>/analysis_prediction_weight_rules.md` に書く。
"""

import argparse
import csv
import io
import json
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

BASELINE = "product-default"
_PREDICTION_DEPENDENT_PREFIXES = (
    "prediction_accuracy",
    "stable_period_prediction_accuracy",
    "prediction_diagnostic_metrics.",
    "model_registration",
)
_PREDICTION_DEPENDENT_COLUMNS = {
    "code_commit",
    "has_uncommitted_changes",
    # 登録の時点の「重みが最大のモデルとして予測した標本の数」などの診断。
}
_RESOURCE_COLUMNS = {
    "upload_bytes": ["communication_volume.uploaded_byte_count"],
    "download_bytes": ["communication_volume.downloaded_byte_count"],
    "client_mac": [
        "computation_cost_summary.client_forward_multiply_accumulate_count",
        "computation_cost_summary.client_estimated_backward_multiply_accumulate_count",
    ],
    "server_mac": ["computation_cost_summary.server_multiply_accumulate_count"],
}


def _read_summary(directory):
    with open(directory / "fedsda" / "run_summary.csv", encoding="utf-8", newline="") as stream:
        return {row["run_identifier"]: row for row in csv.DictReader(stream)}


def _read_tables(directory):
    archive = directory / "fedsda" / "runs.zip"
    tables = {}
    if archive.exists():
        with zipfile.ZipFile(archive) as packed:
            for name in packed.namelist():
                if name.endswith(".npz"):
                    tables[Path(name).stem] = dict(np.load(io.BytesIO(packed.read(name))))
    else:
        for path in (directory / "fedsda" / "runs").glob("*.npz"):
            tables[path.stem] = dict(np.load(path))
    return tables


def _read_traces(directory):
    traces = {}
    for path in (directory / "weight_rule_trace").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        traces[record["run_identifier"]] = record
    return traces


def _read_seconds(log_paths):
    seconds = defaultdict(dict)
    variant = None
    for path in log_paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            found = re.search(r"output directory: .*[\\/]([^\\/]+)$", line)
            if found:
                variant = found.group(1).strip()
            found = re.search(r"\] fedsda/(\S+): completed \((\d+(?:\.\d+)?)s\)", line)
            if found and variant:
                seconds[variant][found.group(1)] = float(found.group(2))
    return seconds


def _period_masks(table, window):
    correct = table["sample_prediction.combined_prediction_is_correct"].astype(bool)
    post_change = np.zeros_like(correct, dtype=bool)
    for client_id, start in zip(
        table["concept_change.client_id"], table["concept_change.first_new_concept_sample_index"], strict=True
    ):
        post_change[int(client_id), int(start) : int(start) + window] = True
    return correct, post_change


def _condition(run_identifier):
    return re.sub(r"_seed\d+_[0-9a-f]+$", "", run_identifier)


def _non_prediction_columns(row):
    return {
        key: value
        for key, value in row.items()
        if key not in _PREDICTION_DEPENDENT_COLUMNS
        and not key.startswith(_PREDICTION_DEPENDENT_PREFIXES)
    }


def analyze(output, window, log_paths):
    variants = sorted(path.name for path in output.iterdir() if (path / "fedsda" / "run_summary.csv").exists())
    if BASELINE not in variants:
        raise ValueError(f"{BASELINE} の結果がありません: {output}")
    summaries = {variant: _read_summary(output / variant) for variant in variants}
    tables = {variant: _read_tables(output / variant) for variant in variants}
    traces = {variant: _read_traces(output / variant) for variant in variants}
    seconds = _read_seconds(log_paths)
    baseline_summary = summaries[BASELINE]
    lines = [
        "# モデルの混合の重みの規則の変種と既定の対比較",
        "",
        f"- 入力: `{output}`。対は同じ run の識別子（条件・seed が同じ）。変化の後の期間は、各clientの概念の変化から {window} 標本。",
        "- 差は（変種 − 既定）、精度は pp。条件ごとに seed で平均し、全体の行は条件の等重みの平均。",
        "",
    ]
    for variant in variants:
        if variant == BASELINE:
            continue
        shared = sorted(set(summaries[variant]) & set(baseline_summary))
        mismatched = []
        for run_identifier in shared:
            left = _non_prediction_columns(summaries[variant][run_identifier])
            right = _non_prediction_columns(baseline_summary[run_identifier])
            different = [key for key in left if left[key] != right.get(key)]
            if different:
                mismatched.append((run_identifier, different[:5]))
        per_condition = defaultdict(lambda: defaultdict(list))
        for run_identifier in shared:
            condition = _condition(run_identifier)
            variant_table = tables[variant][run_identifier]
            baseline_table = tables[BASELINE][run_identifier]
            variant_correct, post_change = _period_masks(variant_table, window)
            baseline_correct, baseline_post_change = _period_masks(baseline_table, window)
            if not np.array_equal(post_change, baseline_post_change):
                mismatched.append((run_identifier, ["concept_change"]))
            values = per_condition[condition]
            values["all"].append(100 * (variant_correct.mean() - baseline_correct.mean()))
            if post_change.any():
                values["post"].append(
                    100 * (variant_correct[post_change].mean() - baseline_correct[post_change].mean())
                )
            values["other"].append(
                100 * (variant_correct[~post_change].mean() - baseline_correct[~post_change].mean())
            )
            values["baseline_accuracy"].append(100 * baseline_correct.mean())
            concepts = baseline_table["sample_prediction.observed_concept_id"]
            for concept in np.unique(concepts):
                mask = concepts == concept
                values[f"concept{int(concept)}"].append(
                    100 * (variant_correct[mask].mean() - baseline_correct[mask].mean())
                )
            trace = traces[variant].get(run_identifier, {})
            values["share_boosts"].append(trace.get("share_boosts", float("nan")))
            values["alarms"].append(trace.get("alarms", float("nan")))
            for name in (
                "prediction_diagnostic_metrics.fixed_share_weight_leader_switch_count",
                "prediction_diagnostic_metrics.mean_fixed_share_effective_model_count",
            ):
                values[name.split(".")[-1]].append(
                    float(summaries[variant][run_identifier][name]) - float(baseline_summary[run_identifier][name])
                )
            for resource, columns in _RESOURCE_COLUMNS.items():
                values[resource].append(sum(float(baseline_summary[run_identifier][column]) for column in columns))
            if run_identifier in seconds.get(variant, {}) and run_identifier in seconds.get(BASELINE, {}):
                values["seconds_ratio"].append(seconds[variant][run_identifier] / seconds[BASELINE][run_identifier])
        lines += [f"## {variant}", ""]
        lines.append(
            f"- 学習が同じことの確認（予測・重みの診断を除く全列と概念の変化）: 対 {len(shared)} のうち不一致 {len(mismatched)}"
            + (f"（例 {mismatched[:3]}）" if mismatched else "")
        )
        lines.append(
            "- 上り・下りの通信量と MAC は、確認の列に含まれるので、不一致が0なら既定と同じ（下の表は既定の値の平均）。"
        )
        lines.append(
            "- 経過時間: "
            + ("掃引のログからの比（変種/既定）を表に示す。" if seconds else "記録なし（`--log` を渡していない）。")
        )
        lines.append("")
        lines.append(
            "| 条件 | seed数 | 既定の精度 | 差 全体 | 差 変化後 | 差 それ以外 | 概念別の差 | 共有率を上げた回数 | 警報 | 1位の入替の差 | 有効モデル数の差 | 上り byte | 下り byte | client MAC | server MAC | 時間比 |"
        )
        lines.append("|" + "---|" * 16)
        totals = defaultdict(list)
        for condition in sorted(per_condition):
            values = per_condition[condition]
            concept_text = " ".join(
                f"{key[7:]}:{statistics.mean(values[key]):+.3f}" for key in sorted(values) if key.startswith("concept")
            )
            row = [
                condition,
                str(len(values["all"])),
                f"{statistics.mean(values['baseline_accuracy']):.3f}",
                f"{statistics.mean(values['all']):+.3f}",
                f"{statistics.mean(values['post']):+.3f}" if values["post"] else "対象なし",
                f"{statistics.mean(values['other']):+.3f}",
                concept_text,
                f"{statistics.mean(values['share_boosts']):.1f}",
                f"{statistics.mean(values['alarms']):.1f}",
                f"{statistics.mean(values['fixed_share_weight_leader_switch_count']):+.1f}",
                f"{statistics.mean(values['mean_fixed_share_effective_model_count']):+.3f}",
                f"{statistics.mean(values['upload_bytes']):.4g}",
                f"{statistics.mean(values['download_bytes']):.4g}",
                f"{statistics.mean(values['client_mac']):.4g}",
                f"{statistics.mean(values['server_mac']):.4g}",
                f"{statistics.mean(values['seconds_ratio']):.3f}" if values["seconds_ratio"] else "記録なし",
            ]
            lines.append("| " + " | ".join(row) + " |")
            for key in ("all", "post", "other"):
                if values[key]:
                    totals[key].append(statistics.mean(values[key]))
            totals["positive"].append(statistics.mean(values["all"]) > 0)
        lines.append("")
        lines.append(
            f"- 条件の等重みの平均: 全体 {statistics.mean(totals['all']):+.4f} pp、"
            f"変化後 {statistics.mean(totals['post']):+.4f} pp、それ以外 {statistics.mean(totals['other']):+.4f} pp。"
            f"全体の差が正の条件 {sum(totals['positive'])}/{len(totals['positive'])}。"
            if totals["post"]
            else f"- 条件の等重みの平均: 全体 {statistics.mean(totals['all']):+.4f} pp。"
        )
        lines.append("")
    path = output / "analysis_prediction_weight_rules.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--post-change-window", type=int, default=100)
    parser.add_argument("--log", nargs="*", default=[])
    arguments = parser.parse_args(command_arguments)
    print(analyze(arguments.output.resolve(), arguments.post_change_window, arguments.log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
