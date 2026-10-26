"""同じ条件の対を集計し、変種ごとの集計（例: analysis_oracle_concept_assignment.{md,json}）へ保存する。"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tools.experiments.accuracy_periods import main as accuracy_periods_main
from tools.research_loss_monitor_rebaseline_metrics import _collect, _table, run_metrics
from tools.research_oracle_concept_assignment.policy import (
    ORACLE_CHANGE_ALARM_VARIANT,
    ORACLE_VARIANT,
    ORACLE_VARIANTS,
)

# 変種ごとの集計の保存名。全体の正解のoracleは既存の名前のまま。
_ANALYSIS_NAMES = {
    ORACLE_VARIANT: ("analysis_oracle_concept_assignment", "analysis_accuracy_periods"),
    ORACLE_CHANGE_ALARM_VARIANT: (
        "analysis_oracle_change_alarm",
        "analysis_accuracy_periods_oracle_change_alarm",
    ),
}


def _read_run(path, diagnostics_directory):
    metrics = run_metrics(path)
    summary = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    record = json.loads((diagnostics_directory / f"{path.stem}.json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as raw:
        for name, column in (
            ("a", "combined_prediction_is_correct"),
            ("b", "true_concept_diagnostic_prediction_is_correct"),
            ("c", "maximum_weight_model_prediction_is_correct"),
        ):
            correct = raw[f"sample_prediction.{column}"]
            metrics[f"accuracy_{name}"] = float(correct[correct >= 0].mean())
        metrics["adaptation_outcomes"] = dict(
            Counter(r["adaptation_outcome"] for r in _table(raw, "adaptation"))
        )
        metrics["absorbed_models"] = int(raw["model_clustering.absorbed_into_representative"].sum())
        concepts = raw["sample_prediction.observed_concept_id"]
        expected = sorted(
            (client, int(i + 4))
            for client, row in enumerate(concepts)
            for i in np.flatnonzero(row[1:] != row[:-1]) + 1
            if i + 4 < len(row)
        )
        metrics["true_changes"] = int((concepts[:, 1:] != concepts[:, :-1]).sum())
        metrics["alarm_eligible_changes"] = len(expected)
        if record["variant"] in ORACLE_VARIANTS:
            actual = sorted(
                (r["client_id"], r["alarm_sample_index"]) for r in _table(raw, "loss_change_alarm")
            )
            if actual != expected:
                raise ValueError(f"oracleの警報が真の変化の5標本目と一致しません: {path}")
            if metrics["adaptation_outcomes"].get("alarm_change_interval_too_short", 0):
                raise ValueError(f"oracleに短すぎる区間の応答があります: {path}")
        if record["variant"] == ORACLE_VARIANT:
            diagnostics = record["diagnostics"]
            if diagnostics["mixed_assignment_count"] or metrics["persistent"]:
                raise ValueError(f"oracleに学習標本の混入があります: {path}")
            if (
                diagnostics["assigned_sample_count"] + diagnostics["pending_sample_count"]
                != diagnostics["processed_sample_count"]
            ):
                raise ValueError(f"oracleの帰属・保留の件数が全標本数と一致しません: {path}")
            for event in diagnostics["consolidations"]:
                for cluster in event["model_clusters"]:
                    if (
                        len(cluster) > 1
                        and len({event["majority_concept_ids"][str(model)] for model in cluster})
                        != 1
                    ):
                        raise ValueError(f"違う概念の統合があります: {path}")
    return dict(
        **metrics,
        standard_metrics=summary["run_metrics"],
        elapsed_seconds=record["elapsed_seconds"],
        diagnostics=record["diagnostics"],
    )


def analyze(root, *, expected_pairs=16, variant=ORACLE_VARIANT):
    if variant not in ORACLE_VARIANTS:
        raise ValueError(f"変種の名前が不正です: {variant}")
    report_name, periods_name = _ANALYSIS_NAMES[variant]
    roots = {"default": root / "product-default", "oracle": root / variant}
    paths = {name: _collect([directory]) for name, directory in roots.items()}
    if set(paths["default"]) != set(paths["oracle"]) or len(paths["default"]) != expected_pairs:
        raise ValueError(
            f"同じ条件の対が{expected_pairs}件必要です（既定{len(paths['default'])}、oracle{len(paths['oracle'])}）"
        )
    # 指定のCLIと同じ集計。全体・定常期を保存済みの要約へ照合し、期間別のCSVを保存する。
    accuracy_periods_main(
        [
            "--group",
            f"default={roots['default']}",
            "--group",
            f"oracle={roots['oracle']}",
            "--reference",
            "default",
            "--output",
            str(root / f"{periods_name}.csv"),
        ]
    )
    rows = []
    for key in sorted(paths["default"]):
        pair = {}
        for name in roots:
            path = Path(paths[name][key])
            pair[name] = _read_run(path, roots[name] / "oracle_concept_assignment_diagnostics")
        with (
            np.load(paths["default"][key]) as default,
            np.load(paths["oracle"][key]) as oracle,
        ):
            for column in ("observed_concept_id", "observed_class_id"):
                name = f"sample_prediction.{column}"
                if not np.array_equal(default[name], oracle[name]):
                    raise ValueError(f"入力の系列が一致しません: {key}: {column}")
        rows.append(dict(dataset=key[0], interval=key[1], seed=key[2], **pair))

    lines = [
        f"# {variant}の集計",
        "",
        "精度は%・差はpp。各cellはseedを等重みで平均。bは真の概念別の証拠を持つAdaHedgeの混合、cは最大重みのモデル。",
        "",
        "|データ|間隔|seed数|既定a|oracle a|差a|既定b|oracle b|既定c|oracle c|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cells = defaultdict(list)
    for row in rows:
        cells[(row["dataset"], row["interval"])].append(row)
    for (dataset, interval), items in sorted(cells.items()):

        def mean(arm, key):
            return float(np.mean([row[arm][key] for row in items]))

        values = [mean(arm, f"accuracy_{column}") * 100 for column in "abc" for arm in roots]
        lines.append(
            f"|{dataset}|{interval}|{len(items)}|{values[0]:.4f}|{values[1]:.4f}|{values[1] - values[0]:+.4f}|{values[2]:.4f}|{values[3]:.4f}|{values[4]:.4f}|{values[5]:.4f}|"
        )

    decisions = {}
    lines += [
        "",
        "データごとに、同じseedの間隔間の差を先に平均し、そのseed間の平均と標本標準偏差（ddof=1）を計算する。",
        "",
    ]
    for dataset in sorted({row["dataset"] for row in rows}):
        by_seed = defaultdict(list)
        for row in rows:
            if row["dataset"] == dataset:
                by_seed[row["seed"]].append(
                    100 * (row["oracle"]["accuracy_a"] - row["default"]["accuracy_a"])
                )
        differences = {seed: float(np.mean(values)) for seed, values in by_seed.items()}
        mean = float(np.mean(list(differences.values())))
        sd = float(np.std(list(differences.values()), ddof=1)) if len(differences) > 1 else None
        decisions[dataset] = dict(
            seed_mean_differences_pp=differences,
            mean_pp=mean,
            seed_standard_deviation_pp=sd,
        )
        lines.append(
            f"- {dataset}: 平均{mean:+.4f}pp、seed間SD {sd if sd is not None else '対象なし'}pp。"
        )
    candidate = any(value["mean_pp"] >= 0.3 for value in decisions.values())
    decision_eligible = set(paths["default"]) == {
        (dataset, interval, seed)
        for dataset in ("sea2", "sea4")
        for interval in (100, 500)
        for seed in range(4)
    } and all(row[arm]["samples"] == 50000 for row in rows for arm in roots)
    decision = (
        "全体の正解のoracleとの分解に使う（基準の判断は全体の正解のoracleで行った）"
        if variant != ORACLE_VARIANT
        else (
            (
                "保有モデルとの比較による検出へ進む候補"
                if candidate
                else "今回の基準では検出の改善へ進まない"
            )
            if decision_eligible
            else "判断は保留（設計の2データ×2間隔×4seed・5000標本の条件が未完了）"
        )
    )
    lines += [
        "",
        f"判断: {decision}（どちらかのデータで平均+0.3pp以上）。",
        "",
        "|データ|間隔|変種|見逃し/対象変化|遅れで説明できない混入/全標本|警報/真の変化（警報対象）|選択の内訳|警報を含む適応の結果の内訳|統合で吸収したモデル数|",
        "|---|---:|---|---|---|---|---|---|---:|",
    ]
    for (dataset, interval), items in sorted(cells.items()):
        for arm in roots:

            def total(key):
                return sum(row[arm][key] for row in items)

            selections = Counter()
            outcomes = Counter()
            for row in items:
                selections.update(row[arm]["selections"])
                outcomes.update(row[arm]["adaptation_outcomes"])
            lines.append(
                f"|{dataset}|{interval}|{arm}|{total('missed')}/{total('changes')}|{total('persistent')}/{total('samples')}|{total('alarms')}/{total('true_changes')} ({total('alarm_eligible_changes')})|{dict(selections)}|{dict(outcomes)}|{total('absorbed_models')}|"
            )
    lines += [
        "",
        "|データ|間隔|変種|上りbyte/run|下りbyte/run|client順伝播MAC/run|client推定逆伝播MAC/run|server MAC/run|監視の候補×賭け率/run|経過秒/run|",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (dataset, interval), items in sorted(cells.items()):
        for arm in roots:
            costs = []
            for row in items:
                metrics = row[arm]["standard_metrics"]
                communication = metrics["communication_volume"]
                computation = metrics["computation_cost_summary"]
                costs.append(
                    [
                        communication["uploaded_byte_count"],
                        communication["downloaded_byte_count"],
                        computation["client_forward_multiply_accumulate_count"],
                        computation["client_estimated_backward_multiply_accumulate_count"],
                        computation["server_multiply_accumulate_count"],
                        metrics["loss_monitoring_computation_counts"][
                            "evaluated_candidate_bet_count"
                        ],
                        row[arm]["elapsed_seconds"],
                    ]
                )
            lines.append(
                f"|{dataset}|{interval}|{arm}|"
                + "|".join(f"{value:.2f}" for value in np.mean(costs, axis=0))
                + "|"
            )
    lines += [
        "",
        f"通信・MACは製品と同じ計数で、概念の参照・診断記録のPython処理はMACに含まない。経過秒は診断の処理とraw保存を含み、並列実行の負荷の影響を受ける。期間別の精度は{periods_name}.csvを参照。",
        "理論の上限: 概念別90%、境界1本SEA2 87.13%・SEA4 85.09%。4seedの記述統計による診断で、本実装の採用判定ではない。",
    ]
    (root / f"{report_name}.json").write_text(
        json.dumps(dict(rows=rows, decisions=decisions), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = "\n".join(lines) + "\n"
    (root / f"{report_name}.md").write_text(report, encoding="utf-8")
    return report


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=16)
    parser.add_argument("--variant", choices=ORACLE_VARIANTS, default=ORACLE_VARIANT)
    arguments = parser.parse_args(command_arguments)
    print(
        analyze(
            arguments.root,
            expected_pairs=arguments.expected_pairs,
            variant=arguments.variant,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
