"""`run_behavior_fix_comparison.py`の結果を、同じdataset・集約間隔・seedのrunどうしの対で比べる（検証用）。

使い方:
    python analyze_behavior_fix_comparison.py <出力先> [--chains] [--stdout]

結果は、`<出力先>/analysis_<作成日時>.md`へ書く（同じ名前があれば`_2`・`_3`…）。`--stdout`では、同じ中身を
標準出力へ出し、ファイルを作らない。先頭に、出どころ（作成の日時、分析のコマンドの引数、分析のコードの版、入力の
ディレクトリ、組合せごとのrun数と、入力のrunのcommit・基準版）を書く。入力のrunの基準版が揃っていなければ、
見出しの直後に警告を書く。

出すもの（Markdownの表。種類の違うデータを越えて平均した行は出さない）:
1. datasetと集約間隔ごとの、組合せの絶対値（平均±標準偏差。runが1つなら標準偏差は「—」）。
2. 対の差（datasetと集約間隔ごとに、同じseedのrunどうしの差の、平均±標準誤差、上がったseedの数/下がったseedの数）:
   - 登録の扱いの効果（短い警報・警報を待つ扱いを固定して、登録だけを替える。登録の水準の対の全部）
   - LEGACY-002の効果（登録・警報を待つ扱いを固定して、短い警報だけを替える）
   - 両方の効果（どちらも旧と同じ→登録を直した側か現行として待つ側、短い警報を直した側）
   - 警報を待つ扱い（IMPROVE-006）の効果（ほかの2つを固定して、待たない→待つ）
   どれも、両方の組合せの結果があるものだけを出す。平均の差の絶対値が、標準誤差の2倍を超えるものに、`*`を付ける
   （seedの揺らぎより大きい目安。検定ではない）。
3. `--chains`: 短い警報の連鎖（同じclientで、短い警報が、ほかの結果の警報を挟まずに続く並び）の数と長さ。
   runごとの記録の表（NPZ）の、適応の記録から数える。
"""

import csv
import math
import sys
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPOSITORY_DIRECTORY = Path(__file__).resolve().parent
ANALYSIS_FILE_NAME_PREFIX = "analysis_"
# 一覧の表の、入力のrunの来歴の列（`run_experiment_sweep.py`と同じ。前の形式の表には無い）。
PROVENANCE_NOT_RECORDED = "記録なし"

# 登録の水準と、短い警報の水準（`run_behavior_fix_comparison.py`と同じ）。
REGISTRATION_LEVELS = ("legacy", "fixed", "waited-as-current")
SHORT_ALARM_LEVELS = ("legacy", "fixed")
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


def mean_and_standard_deviation(values):
    """平均と、標本の標準偏差（n−1で割った分散の平方根）。値が1つなら、標準偏差はNaN。"""
    count = len(values)
    mean = sum(values) / count
    if count < 2:
        return mean, float("nan")
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (count - 1))


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


def build_comparisons():
    """（表題、前の組合せの水準、後の組合せの水準）の列。軸の水準から生成する。"""
    comparisons = []
    for alarm_wait_level in ALARM_WAIT_LEVELS:
        for short_alarm_level in reversed(SHORT_ALARM_LEVELS):
            for before_index, before_level in enumerate(REGISTRATION_LEVELS):
                for after_level in REGISTRATION_LEVELS[before_index + 1 :]:
                    comparisons.append(
                        (
                            f"登録の扱いの効果（{before_level}→{after_level}。短い警報は{short_alarm_level}、"
                            f"警報を待つ扱いは{alarm_wait_level}）",
                            (before_level, short_alarm_level, alarm_wait_level),
                            (after_level, short_alarm_level, alarm_wait_level),
                        )
                    )
    for alarm_wait_level in ALARM_WAIT_LEVELS:
        for registration_level in reversed(REGISTRATION_LEVELS):
            comparisons.append(
                (
                    f"LEGACY-002の効果（登録は{registration_level}、警報を待つ扱いは{alarm_wait_level}）",
                    (registration_level, "legacy", alarm_wait_level),
                    (registration_level, "fixed", alarm_wait_level),
                )
            )
    for registration_level in REGISTRATION_LEVELS[1:]:
        comparisons.append(
            (
                f"両方の効果（旧と同じ→登録は{registration_level}、短い警報は直した側）",
                ("legacy", "legacy", "off"),
                (registration_level, "fixed", "off"),
            )
        )
    for registration_level in reversed(REGISTRATION_LEVELS):
        for short_alarm_level in reversed(SHORT_ALARM_LEVELS):
            comparisons.append(
                (
                    f"警報を待つ扱い（IMPROVE-006）の効果（登録は{registration_level}、短い警報は{short_alarm_level}）",
                    (registration_level, short_alarm_level, "off"),
                    (registration_level, short_alarm_level, "on"),
                )
            )
    return tuple(comparisons)


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


def read_combination_provenance(output_directory, name):
    """組合せの一覧の表の全行の、入力のrunのcommitと基準版の一覧を返す。列が無い・空なら「記録なし」。"""
    table_path = output_directory / name / "run_summary.csv"
    provenances = []
    with table_path.open(encoding="utf-8", newline="") as table_file:
        for row in csv.DictReader(table_file):
            provenances.append(
                (
                    row.get("code_commit") or PROVENANCE_NOT_RECORDED,
                    row.get("behavior_baseline_version") or PROVENANCE_NOT_RECORDED,
                )
            )
    return provenances


def describe_analysis_code_revision():
    """分析のコードの版（commit、未commitの変更の有無、基準版）の1行。実行のときだけ、torchほかを読む。"""
    sys.path.insert(0, str(REPOSITORY_DIRECTORY / "src"))
    from federated_learning_experiments.sweep.experiment_sweep_records import (
        build_run_provenance,
    )

    run_provenance = build_run_provenance(repository_directory=REPOSITORY_DIRECTORY)
    code_commit_text = (
        run_provenance.code_commit
        if run_provenance.code_commit is not None
        else f"（取れなかった: {run_provenance.code_revision_unavailable_reason}）"
    )
    return (
        f"commit {code_commit_text}、未commitの変更 {run_provenance.has_uncommitted_changes}、"
        f"基準版 {run_provenance.behavior_baseline_version}"
    )


def build_provenance_lines(*, output_directory, available, command_arguments, created_at):
    """見出しの直後の警告（基準版が揃っていなければ）と、出どころの節の行を返す。"""
    provenance_by_combination = {
        levels: read_combination_provenance(output_directory, combination_name(*levels))
        for levels in available
    }
    baseline_versions = sorted(
        {
            version
            for provenances in provenance_by_combination.values()
            for _, version in provenances
        }
    )
    lines = []
    if len(baseline_versions) != 1:
        lines.append(
            f"> 警告: 入力のrunの基準版が揃っていない（{'、'.join(baseline_versions)}）。"
            "版の違う結果を比べている可能性がある。\n"
        )
    lines.append("## 出どころ\n")
    lines.append(f"- 作成の日時: {created_at}")
    lines.append(f"- 分析のコマンドの引数: `{' '.join(command_arguments)}`")
    lines.append(f"- 分析のコードの版: {describe_analysis_code_revision()}")
    lines.append(f"- 入力のディレクトリ: `{output_directory}`\n")
    lines.append("| 組合せ | run | 入力のrunのcommit | 入力のrunの基準版 |")
    lines.append("|---|---|---|---|")
    for levels, runs in available.items():
        provenances = provenance_by_combination[levels]
        commits = sorted(
            {
                commit if commit == PROVENANCE_NOT_RECORDED else commit[:12]
                for commit, _ in provenances
            }
        )
        versions = sorted({version for _, version in provenances})
        lines.append(
            f"| {combination_name(*levels)} | {len(runs)} | {'、'.join(commits)} | {'、'.join(versions)} |"
        )
    return lines


def write_analysis_file(output_directory, report_text, created_time):
    """`analysis_<作成日時>.md`へ排他的に書く（同じ名前があれば、次の番号）。書いた場所を返す。"""
    file_stem = f"{ANALYSIS_FILE_NAME_PREFIX}{created_time.strftime('%Y%m%d_%H%M%S')}"
    file_number = 1
    while True:
        analysis_file_path = output_directory / (
            file_stem + ("" if file_number == 1 else f"_{file_number}") + ".md"
        )
        try:
            with analysis_file_path.open("x", encoding="utf-8", newline="\n") as analysis_file:
                analysis_file.write(report_text)
        except FileExistsError:
            file_number += 1
            continue
        return analysis_file_path


def main(arguments):
    argument_parser = ArgumentParser(
        prog="analyze_behavior_fix_comparison.py",
        description=__doc__.split("\n\n")[0],
    )
    argument_parser.add_argument("output_directory", type=Path, help="比較scriptの出力先")
    argument_parser.add_argument("--chains", action="store_true", help="短い警報の連鎖の表も出す")
    argument_parser.add_argument(
        "--stdout", action="store_true", help="ファイルを作らず、結果を標準出力へ出す"
    )
    parsed_arguments = argument_parser.parse_args(arguments)
    output_directory = parsed_arguments.output_directory
    report_chains = parsed_arguments.chains
    created_time = datetime.now().astimezone().replace(microsecond=0)
    report_lines = []
    emit = report_lines.append
    runs_by_combination = {
        (registration_level, short_alarm_level, alarm_wait_level): read_combination(
            output_directory,
            combination_name(registration_level, short_alarm_level, alarm_wait_level),
        )
        for registration_level in REGISTRATION_LEVELS
        for short_alarm_level in SHORT_ALARM_LEVELS
        for alarm_wait_level in ALARM_WAIT_LEVELS
    }
    available = {levels: runs for levels, runs in runs_by_combination.items() if runs}
    if not available:
        print(f"no run_summary.csv under {output_directory}")
        return 1
    groups = sorted({key[:2] for runs in available.values() for key in runs})
    emit("# 挙動の修正（LEGACY-016・LEGACY-002）の、適用の有無の比較\n")
    report_lines.extend(
        build_provenance_lines(
            output_directory=output_directory,
            available=available,
            command_arguments=list(arguments),
            created_at=created_time.isoformat(),
        )
    )
    emit("\n## 1. 組合せごとの絶対値（平均±標準偏差）\n")
    emit(
        "| dataset | 集約間隔 | 登録（016） | 短い警報（002） | 警報を待つ（006） | n | "
        + " | ".join(display_name for display_name, _, _ in METRICS.values())
        + " |"
    )
    emit("|---|---|---|---|---|---|" + "---|" * len(METRICS))
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
                if not values:
                    cells.append("—")
                    continue
                mean, standard_deviation = mean_and_standard_deviation(values)
                deviation_text = (
                    "—"
                    if standard_deviation != standard_deviation
                    else f"{standard_deviation:.{digits}f}"
                )
                cells.append(f"{mean:.{digits}f}±{deviation_text}")
            emit(
                f"| {dataset_name} | {aggregation_interval} | {levels[0]} | {levels[1]} | {levels[2]} | {len(group_runs)} | "
                + " | ".join(cells)
                + " |"
            )
    comparisons = build_comparisons()
    emit(
        "\n## 2. 対の差（datasetと集約間隔ごと。同じseedのrunどうしの、後−前の平均±標準誤差。"
        "`*`は、平均の絶対値が標準誤差の2倍を超えるもの）\n"
    )
    for title, before_levels, after_levels in comparisons:
        if before_levels not in available or after_levels not in available:
            continue
        emit(f"### {title}\n")
        emit(
            "| dataset | 集約間隔 | "
            + " | ".join(display_name for display_name, _, _ in METRICS.values())
            + " |"
        )
        emit("|---|---|" + "---|" * len(METRICS))
        before_runs, after_runs = available[before_levels], available[after_levels]
        for group in groups:
            group_keys = sorted(
                key for key in before_runs if key in after_runs and key[:2] == group
            )
            if not group_keys:
                continue
            label = f"| {group[0]} | {group[1]} | "
            emit(
                label
                + " | ".join(
                    format_paired_difference(before_runs, after_runs, group_keys, column)
                    for column in METRICS
                )
                + " |"
            )
        emit("")
    if report_chains:
        emit("## 3. 短い警報の連鎖（同じclientで、短い警報が続く並び）\n")
        emit(
            "| dataset | 集約間隔 | 登録（016） | 短い警報（002） | 警報を待つ（006） | run | 警報の総数（平均） | 短い警報（平均） | 連鎖の数（平均） | 連鎖の長さの平均 | 連鎖の長さの最大 | 長さ5以上の連鎖にある短い警報の割合 |"
        )
        emit("|---|---|---|---|---|---|---|---|---|---|---|---|")
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
                emit(
                    f"| {dataset_name} | {aggregation_interval} | {levels[0]} | {levels[1]} | {levels[2]} | {run_count} | "
                    f"{sum(alarm_counts) / run_count:.1f} | {short_alarm_count / run_count:.1f} | "
                    f"{len(all_chain_lengths) / run_count:.1f} | "
                    f"{(short_alarm_count / len(all_chain_lengths)) if all_chain_lengths else 0:.1f} | "
                    f"{max(all_chain_lengths, default=0)} | {long_chain_share:.0%} |"
                )
    report_text = "\n".join(report_lines) + "\n"
    if parsed_arguments.stdout:
        sys.stdout.write(report_text)
    else:
        print(f"analysis: {write_analysis_file(output_directory, report_text, created_time)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
