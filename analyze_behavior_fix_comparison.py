"""`run_behavior_fix_comparison.py`の結果を、同じdataset・集約間隔・seedのrunどうしの対で比べる（検証用）。

使い方:
    python analyze_behavior_fix_comparison.py <出力先> [--chains]

出すもの（Markdownの表。標準出力）:
1. datasetと集約間隔ごとの、4通りの組合せの平均（精度、定常精度、通信量、最終のグローバルモデル数、検出数、短い警報の数）。
2. 対の差（同じseedのrunどうしの差の、平均±標準誤差、上がったseedの数/下がったseedの数）:
   - LEGACY-016の効果（短い警報の扱いを固定して、登録だけを替える。2通り）
   - LEGACY-002の効果（登録の扱いを固定して、短い警報だけを替える。2通り）
   - 両方の効果（どちらも旧と同じ→どちらも直した）
   - 警報を待つ扱い（IMPROVE-006）の効果（ほかの2つを固定して、待たない→待つ。結果がある組合せだけ）
   平均の差の絶対値が、標準誤差の2倍を超えるものに、`*`を付ける（seedの揺らぎより大きい目安。検定ではない）。
3. `--chains`: 短い警報の連鎖（同じclientで、短い警報が、ほかの結果の警報を挟まずに続く並び）の数と長さ。
   runごとの記録の表（NPZ）の、適応の記録から数える。
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

LEVELS = ("legacy", "fixed")
SHORT_ALARM_OUTCOME = "alarm_change_interval_too_short"
# 一覧の表の列の名前→（表示名、表示の倍率、小数の桁数）。
METRICS = {
    "prediction_accuracy": ("精度（%）", 100.0, 2),
    "stable_period_prediction_accuracy": ("定常精度（%）", 100.0, 2),
    "communication_volume.downloaded_byte_count": ("下りの通信量（MB）", 1e-6, 2),
    "communication_volume.uploaded_byte_count": ("上りの通信量（MB）", 1e-6, 2),
    "final_global_model_count": ("最終のグローバルモデル数", 1.0, 2),
    "training_model_switch_detection_metrics.detection_count": ("検出数", 1.0, 1),
    "change_response_metrics.adaptation_action_metrics.skip_too_short_change_interval_count": (
        "短い警報の数",
        1.0,
        1,
    ),
    "training_model_switch_detection_metrics.detection_recall": ("切替の検出の再現率", 1.0, 3),
    "training_model_switch_detection_metrics.detection_precision": ("切替の検出の適合率", 1.0, 3),
    "change_response_metrics.training_model_switch_timing_metrics.mean_matched_switch_delay_sample_count": (
        "変化から切替までの遅れ（標本）",
        1.0,
        1,
    ),
    "change_response_metrics.adaptation_action_metrics.adopt_candidate_model_count": (
        "候補の採用の数",
        1.0,
        1,
    ),
}


ALARM_WAIT_LEVELS = ("off", "on")


def combination_name(registration_level, short_alarm_level, alarm_wait_level="off"):
    name = f"registration-{registration_level}__short-alarm-{short_alarm_level}"
    return name + ("__alarm-wait-on" if alarm_wait_level == "on" else "")


def read_combination(output_directory, name):
    """組合せの一覧の表を読み、（dataset, 集約間隔, seed）→（指標の値、識別子）を返す。無ければ空。"""
    table_path = output_directory / name / "run_summary.csv"
    if not table_path.is_file():
        return {}
    runs = {}
    with table_path.open(encoding="utf-8", newline="") as table_file:
        for row in csv.DictReader(table_file):
            key = (
                row["dataset_name"],
                int(row["server_aggregation_interval_per_client_samples"]),
                int(row["random_seed"]),
            )
            runs[key] = (
                {
                    column: float(row[column])
                    for column in METRICS
                    if row.get(column) not in (None, "")
                },
                row["run_identifier"],
            )
    return runs


def mean_and_standard_error(values):
    count = len(values)
    mean = sum(values) / count
    if count < 2:
        return mean, float("nan")
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, math.sqrt(variance / count)


def format_paired_difference(before_runs, after_runs, group_keys, column):
    _, scale, digits = METRICS[column]
    differences = [
        (after_runs[key][0][column] - before_runs[key][0][column]) * scale
        for key in group_keys
        if key in before_runs
        and key in after_runs
        and column in before_runs[key][0]
        and column in after_runs[key][0]
    ]
    if not differences:
        return "—"
    mean, standard_error = mean_and_standard_error(differences)
    increased = sum(difference > 0 for difference in differences)
    decreased = sum(difference < 0 for difference in differences)
    marker = "*" if standard_error == standard_error and abs(mean) > 2 * standard_error > 0 else ""
    error_text = "" if standard_error != standard_error else f"±{standard_error:.{digits}f}"
    return (
        f"{mean:+.{digits}f}{error_text}{marker}（↑{increased}/↓{decreased}/n={len(differences)}）"
    )


def collect_short_alarm_chains(record_table_path):
    """1 runの記録の表から、clientごとの短い警報の連鎖の長さの一覧と、警報の総数を返す。"""
    import numpy as np

    with np.load(record_table_path, allow_pickle=False) as record_tables:
        client_ids = record_tables["adaptation.client_id"].tolist()
        sample_indices = record_tables["adaptation.adaptation_sample_index"].tolist()
        outcomes = record_tables["adaptation.adaptation_outcome"].tolist()
    outcomes_by_client = defaultdict(list)
    for client_id, _sample_index, outcome in sorted(zip(client_ids, sample_indices, outcomes)):
        outcomes_by_client[client_id].append(outcome)
    chain_lengths = []
    alarm_count = 0
    for client_outcomes in outcomes_by_client.values():
        length = 0
        for outcome in client_outcomes:
            if not outcome.startswith("alarm_"):
                continue
            alarm_count += 1
            if outcome == SHORT_ALARM_OUTCOME:
                length += 1
            elif length:
                chain_lengths.append(length)
                length = 0
        if length:
            chain_lengths.append(length)
    return chain_lengths, alarm_count


def main(arguments):
    if not arguments or arguments[0] in ("-h", "--help"):
        print(__doc__)
        return 1
    output_directory = Path(arguments[0])
    report_chains = "--chains" in arguments[1:]
    runs_by_combination = {
        (registration_level, short_alarm_level, alarm_wait_level): read_combination(
            output_directory,
            combination_name(registration_level, short_alarm_level, alarm_wait_level),
        )
        for registration_level in LEVELS
        for short_alarm_level in LEVELS
        for alarm_wait_level in ALARM_WAIT_LEVELS
    }
    available = {levels: runs for levels, runs in runs_by_combination.items() if runs}
    if not available:
        print(f"no run_summary.csv under {output_directory}")
        return 1
    groups = sorted({key[:2] for runs in available.values() for key in runs})
    print("# 挙動の修正（LEGACY-016・LEGACY-002）の、適用の有無の比較\n")
    print(
        f"出力先: `{output_directory}`。組合せ: "
        + "、".join(
            f"{combination_name(*levels)}（{len(runs)} run）" for levels, runs in available.items()
        )
    )
    print("\n## 1. 組合せごとの平均\n")
    print(
        "| dataset | 集約間隔 | 登録（016） | 短い警報（002） | 警報を待つ（006） | n | "
        + " | ".join(display_name for display_name, _, _ in METRICS.values())
        + " |"
    )
    print("|---|---|---|---|---|---|" + "---|" * len(METRICS))
    for dataset_name, aggregation_interval in groups:
        for levels, runs in available.items():
            group_runs = [
                metric_values
                for key, (metric_values, _) in runs.items()
                if key[:2] == (dataset_name, aggregation_interval)
            ]
            if not group_runs:
                continue
            cells = []
            for column, (_, scale, digits) in METRICS.items():
                values = [run[column] * scale for run in group_runs if column in run]
                cells.append(f"{sum(values) / len(values):.{digits}f}" if values else "—")
            print(
                f"| {dataset_name} | {aggregation_interval} | {levels[0]} | {levels[1]} | {levels[2]} | {len(group_runs)} | "
                + " | ".join(cells)
                + " |"
            )
    comparisons = [
        (
            "LEGACY-016の効果（短い警報は旧と同じ）",
            ("legacy", "legacy", "off"),
            ("fixed", "legacy", "off"),
        ),
        (
            "LEGACY-016の効果（短い警報は直した側）",
            ("legacy", "fixed", "off"),
            ("fixed", "fixed", "off"),
        ),
        (
            "LEGACY-002の効果（登録は旧と同じ）",
            ("legacy", "legacy", "off"),
            ("legacy", "fixed", "off"),
        ),
        (
            "LEGACY-002の効果（登録は直した側）",
            ("fixed", "legacy", "off"),
            ("fixed", "fixed", "off"),
        ),
        ("両方の効果", ("legacy", "legacy", "off"), ("fixed", "fixed", "off")),
        *[
            (
                f"警報を待つ扱い（IMPROVE-006）の効果（登録は{registration_level}、短い警報は{short_alarm_level}）",
                (registration_level, short_alarm_level, "off"),
                (registration_level, short_alarm_level, "on"),
            )
            for registration_level in reversed(LEVELS)
            for short_alarm_level in reversed(LEVELS)
        ],
    ]
    print(
        "\n## 2. 対の差（同じseedのrunどうし。後−前の、平均±標準誤差。`*`は、平均の絶対値が標準誤差の2倍を超えるもの）\n"
    )
    for title, before_levels, after_levels in comparisons:
        if before_levels not in available or after_levels not in available:
            continue
        print(f"### {title}\n")
        print(
            "| dataset | 集約間隔 | "
            + " | ".join(display_name for display_name, _, _ in METRICS.values())
            + " |"
        )
        print("|---|---|" + "---|" * len(METRICS))
        before_runs, after_runs = available[before_levels], available[after_levels]
        for group in [*groups, None]:
            group_keys = sorted(
                key
                for key in before_runs
                if key in after_runs and (group is None or key[:2] == group)
            )
            if not group_keys:
                continue
            label = "| 全部 | — | " if group is None else f"| {group[0]} | {group[1]} | "
            print(
                label
                + " | ".join(
                    format_paired_difference(before_runs, after_runs, group_keys, column)
                    for column in METRICS
                )
                + " |"
            )
        print()
    if report_chains:
        print("## 3. 短い警報の連鎖（同じclientで、短い警報が続く並び）\n")
        print(
            "| dataset | 集約間隔 | 登録（016） | 短い警報（002） | 警報を待つ（006） | run | 警報の総数（平均） | 短い警報（平均） | 連鎖の数（平均） | 連鎖の長さの平均 | 連鎖の長さの最大 | 長さ5以上の連鎖にある短い警報の割合 |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for dataset_name, aggregation_interval in groups:
            for levels, runs in available.items():
                all_chain_lengths, alarm_counts, run_count = [], [], 0
                for key, (_, run_identifier) in sorted(runs.items()):
                    if key[:2] != (dataset_name, aggregation_interval):
                        continue
                    record_table_path = (
                        output_directory
                        / combination_name(*levels)
                        / "runs"
                        / f"{run_identifier}.npz"
                    )
                    if not record_table_path.is_file():
                        continue
                    chain_lengths, alarm_count = collect_short_alarm_chains(record_table_path)
                    all_chain_lengths.extend(chain_lengths)
                    alarm_counts.append(alarm_count)
                    run_count += 1
                if not run_count:
                    continue
                short_alarm_count = sum(all_chain_lengths)
                long_chain_share = (
                    sum(length for length in all_chain_lengths if length >= 5) / short_alarm_count
                    if short_alarm_count
                    else 0.0
                )
                print(
                    f"| {dataset_name} | {aggregation_interval} | {levels[0]} | {levels[1]} | {levels[2]} | {run_count} | "
                    f"{sum(alarm_counts) / run_count:.1f} | {short_alarm_count / run_count:.1f} | "
                    f"{len(all_chain_lengths) / run_count:.1f} | "
                    f"{(short_alarm_count / len(all_chain_lengths)) if all_chain_lengths else 0:.1f} | "
                    f"{max(all_chain_lengths, default=0)} | {long_chain_share:.0%} |"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
