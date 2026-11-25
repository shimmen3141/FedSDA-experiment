"""確率の記録（`probability_trace.py`）から、モデルの混合の重みの規則を、再実験なしに手元で計算し直す（研究ツール）。

記録した変種の run ごとに、clientごとの重みの制御（製品の `FixedShareWeightController` に、`run.py` の
`install_weight_rule` で規則を差し込んだもの）へ、記録した呼出しの列を同じ順で渡し、予測ごとに、その規則の重みで
同期の版のモデル別の確率を混ぜた予測（Fixed Share の混合の予測）の正誤と log loss を求める。版の混合（同期の版と
手元の版）は計算し直さない（手元の版の確率は記録していない）ので、実際の予測の精度ではなく、混合FS の精度を比べる。

確かめること（`analysis_replay_<記録した変種>.md` の冒頭）:
- 記録した変種の規則で計算し直した重みが、記録した重みと全予測で同じか（差の最大）。
- どの規則でも、同じ出力先に実際の run がある変種なら、計算し直した混合FS の正誤が、実際の run の記録
  （`sample_prediction.fixed_share_weighted_prediction_is_correct`）と全標本で同じか（学習が重みの規則に依らないので、
  同じになるはず）。

使い方: `python -m tools.research_prediction_weight_rules.replay <出力先> --recorded-variant <変種> [--rules …]`
"""

import argparse
import io
import math
import statistics
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

import torch  # noqa: E402

from federated_learning_experiments.learning.prediction.class_probability_calculations import (  # noqa: E402
    combine_model_prediction_probabilities,
    normalize_model_prediction_weights,
    predict_class_labels_from_prediction_scores,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (  # noqa: E402
    FixedShareWeightController,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_and_local_prediction_mixture import (  # noqa: E402
    true_class_loss_of_prediction_probabilities,
)
from tools.research_prediction_weight_rules.probability_trace import iterate_probability_traces  # noqa: E402
from tools.research_prediction_weight_rules.run import (  # noqa: E402
    ALARM_SHARE_BOOST_VARIANTS,
    VARIANTS,
    install_weight_rule,
    restore_weight_rules,
)
from tools.research_prediction_weight_rules.selection_versus_mixture import PERIOD_NAMES, _period_index  # noqa: E402

# 版の混合だけを変える変種は、混合FS の重みが `log-loss-weights` と同じなので、計算し直す規則に入れない。
REPLAYABLE_RULES = tuple(
    variant
    for variant in VARIANTS
    if variant not in ALARM_SHARE_BOOST_VARIANTS and variant != "log-loss-weights-log-loss-version-mixture"
)
_FIXED_SHARE_CORRECT = "sample_prediction.fixed_share_weighted_prediction_is_correct"


def _class_count(probabilities):
    return 2 if probabilities.shape[-1] == 1 else probabilities.shape[-1]


def replay_client(events, settings, rule):
    """1 client の呼出しの列を、規則 `rule` の制御で計算し直す。予測ごとに (標本, 正誤, log loss, 重み, 記録との重みの差)。"""
    install_weight_rule(rule, {"alarms": 0, "share_boosts": 0})
    try:
        controller = FixedShareWeightController(prediction_combination_settings=settings)
        outputs = []
        pending_weights = None
        in_prediction = False
        for event in events:
            kind = event[0]
            if kind == "weights":
                weights = controller.get_weights_before_label_observation(model_ids=event[1])
                if in_prediction and pending_weights is None:
                    pending_weights = normalize_model_prediction_weights(prediction_weights_by_model_id=weights)
            elif kind == "update":
                losses = dict(event[1])
                # 呼出し元は、直前に同じ ID の組で観測前の重みを求め、正規化して渡す（予測の経路）。
                prediction_weights = normalize_model_prediction_weights(
                    prediction_weights_by_model_id=controller.get_weights_before_label_observation(model_ids=tuple(losses))
                )
                controller.update_weights_after_loss_observation(
                    observed_losses_by_model_id=losses, prediction_weights_by_model_id=prediction_weights
                )
            elif kind == "replay":
                controller.replay_observed_losses(observed_loss_sequence=[dict(row) for row in event[1]])
            elif kind == "replay_after_aggregation":
                controller.replay_observed_losses_after_aggregation(observed_loss_sequence=[dict(row) for row in event[1]])
            elif kind == "reset_after_aggregation":
                controller.reset_weights_after_aggregation()
            elif kind == "predict_begin":
                in_prediction = True
                pending_weights = None
            elif kind == "predict":
                _, sample_index, label, model_ids, probabilities, recorded_weights = event
                if pending_weights is None or tuple(sorted(pending_weights)) != model_ids:
                    raise RuntimeError(f"予測の重みが計算し直せません（標本 {sample_index}）")
                class_count = _class_count(probabilities)
                combined = combine_model_prediction_probabilities(
                    prediction_probabilities_by_model_id={
                        model_id: torch.from_numpy(probabilities[position])
                        for position, model_id in enumerate(model_ids)
                    },
                    prediction_weights_by_model_id=pending_weights,
                    class_count=class_count,
                    prediction_probabilities_are_validated=True,
                )
                predicted = predict_class_labels_from_prediction_scores(
                    prediction_scores=combined, class_count=class_count, prediction_scores_are_validated=True
                )
                true_class_probability = 1.0 - true_class_loss_of_prediction_probabilities(
                    prediction_probabilities=combined, observed_class_id=label, class_count=class_count
                )
                weight_gap = max(
                    abs(pending_weights[model_id] - recorded)
                    for model_id, recorded in zip(model_ids, recorded_weights, strict=True)
                )
                outputs.append(
                    (
                        sample_index,
                        bool(predicted.view(-1)[0].item() == label),
                        -math.log(max(true_class_probability, 1e-12)),
                        weight_gap,
                    )
                )
                in_prediction = False
                pending_weights = None
            else:
                raise RuntimeError(f"知らない記録です: {kind}")
        return outputs
    finally:
        restore_weight_rules()


def _read_tables(variant_directory):
    archive = Path(variant_directory) / "fedsda" / "runs.zip"
    tables = {}
    if archive.exists():
        with zipfile.ZipFile(archive) as packed:
            for name in packed.namelist():
                if name.endswith(".npz"):
                    tables[Path(name).stem] = name
    return archive, tables


def _load_table(archive, name):
    with zipfile.ZipFile(archive) as packed:
        return dict(np.load(io.BytesIO(packed.read(name))))


def _replay_run(arguments):
    run_identifier, payload, rules, recorded_variant, actual_sources = arguments
    shape = None
    results = {}
    for rule in rules:
        correct_by_position = {}
        log_loss_by_position = {}
        maximum_weight_gap = 0.0
        for client_id, events in payload["clients"].items():
            for sample_index, correct, log_loss, weight_gap in replay_client(events, payload["settings"], rule):
                correct_by_position[(client_id, sample_index)] = correct
                log_loss_by_position[(client_id, sample_index)] = log_loss
                maximum_weight_gap = max(maximum_weight_gap, weight_gap)
        results[rule] = dict(correct=correct_by_position, log_loss=log_loss_by_position, weight_gap=maximum_weight_gap)
    own_archive, own_name = actual_sources[recorded_variant]
    table = _load_table(own_archive, own_name)
    shape = table[_FIXED_SHARE_CORRECT].shape
    period = _period_index(table)
    summary = {}
    for rule, result in results.items():
        correct = np.zeros(shape, dtype=bool)
        log_loss = np.zeros(shape, dtype=float)
        covered = np.zeros(shape, dtype=bool)
        for (client_id, sample_index), value in result["correct"].items():
            correct[client_id, sample_index] = value
            log_loss[client_id, sample_index] = result["log_loss"][(client_id, sample_index)]
            covered[client_id, sample_index] = True
        mismatch = None
        if rule in actual_sources:
            actual = _load_table(*actual_sources[rule])[_FIXED_SHARE_CORRECT].astype(bool)
            mismatch = int(np.sum((actual != correct) & covered))
        summary[rule] = dict(
            covered=int(covered.sum()),
            uncovered=int((~covered).sum()),
            accuracy={
                name: (100.0 * correct[covered & (period == index)].mean() if np.any(covered & (period == index)) else None)
                for index, name in enumerate(PERIOD_NAMES)
            },
            overall_accuracy=100.0 * correct[covered].mean(),
            mean_log_loss=float(log_loss[covered].mean()),
            mismatch_with_actual=mismatch,
            weight_gap=result["weight_gap"] if rule == recorded_variant else None,
        )
    return run_identifier, summary


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recorded-variant", required=True, choices=REPLAYABLE_RULES)
    parser.add_argument("--rules", nargs="+", choices=REPLAYABLE_RULES, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="最初の n run だけ（試しに流すとき）")
    arguments = parser.parse_args(command_arguments)
    output = arguments.output.resolve()
    rules = list(arguments.rules or REPLAYABLE_RULES)
    if arguments.recorded_variant not in rules:
        rules.insert(0, arguments.recorded_variant)
    variant_tables = {}
    for variant in REPLAYABLE_RULES:
        archive, tables = _read_tables(output / variant)
        if tables:
            variant_tables[variant] = (archive, tables)
    if arguments.recorded_variant not in variant_tables:
        raise SystemExit(f"記録した変種の run の成果物がありません: {output / arguments.recorded_variant}")
    jobs = []
    for run_identifier, payload in iterate_probability_traces(output / arguments.recorded_variant):
        actual_sources = {
            variant: (archive, tables[run_identifier])
            for variant, (archive, tables) in variant_tables.items()
            if run_identifier in tables
        }
        jobs.append((run_identifier, payload, rules, arguments.recorded_variant, actual_sources))
        if arguments.limit is not None and len(jobs) >= arguments.limit:
            break
    if not jobs:
        raise SystemExit("確率の記録がありません")
    if arguments.workers > 1:
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            summaries = dict(executor.map(_replay_run, jobs))
    else:
        summaries = dict(map(_replay_run, jobs))
    lines = _report(output, arguments.recorded_variant, rules, summaries)
    path = output / f"analysis_replay_{arguments.recorded_variant}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    return 0


def _condition(run_identifier):
    import re

    return re.sub(r"_seed\d+_[0-9a-f]+$", "", run_identifier)


def _report(output, recorded_variant, rules, summaries):
    lines = [
        f"# 確率の記録からの重みの規則の計算し直し（記録した変種 {recorded_variant}、再実験なし）",
        "",
        f"- 入力: `{output}`。run {len(summaries)}。精度は混合FS（同期の版のモデル別の確率を、規則の重みで混ぜた予測）の正解率で、",
        "  版の混合を含む実際の予測の精度ではない。条件ごとに seed で平均し、条件の等重みで平均する。",
        "",
        "## 確かめたこと",
        "",
        "| 規則 | 予測の数 | 記録の無い標本 | 実際の run と正誤が違う標本 | 記録した重みとの差の最大 |",
        "|---|---|---|---|---|",
    ]
    for rule in rules:
        covered = sum(summary[rule]["covered"] for summary in summaries.values())
        uncovered = sum(summary[rule]["uncovered"] for summary in summaries.values())
        mismatches = [summary[rule]["mismatch_with_actual"] for summary in summaries.values()]
        mismatch_text = (
            "実際の run なし"
            if all(value is None for value in mismatches)
            else f"{sum(value for value in mismatches if value is not None)}（run {sum(value is not None for value in mismatches)}）"
        )
        gaps = [summary[rule]["weight_gap"] for summary in summaries.values() if summary[rule]["weight_gap"] is not None]
        gap_text = f"{max(gaps):.3e}" if gaps else "—"
        lines.append(f"| {rule} | {covered} | {uncovered} | {mismatch_text} | {gap_text} |")
    by_condition = defaultdict(lambda: defaultdict(list))
    for run_identifier, summary in summaries.items():
        for rule in rules:
            by_condition[_condition(run_identifier)][rule].append(summary[rule])
    lines += [
        "",
        "## 混合FS の精度（条件の等重みの平均、%）と、1予測あたりの log loss",
        "",
        "| 規則 | 全体 | " + " | ".join(PERIOD_NAMES) + " | log loss |",
        "|---|---|" + "---|" * len(PERIOD_NAMES) + "---|",
    ]
    for rule in rules:
        overall = statistics.mean(
            statistics.mean(item["overall_accuracy"] for item in records[rule]) for records in by_condition.values()
        )
        period_values = []
        for name in PERIOD_NAMES:
            condition_means = []
            for records in by_condition.values():
                values = [item["accuracy"][name] for item in records[rule] if item["accuracy"][name] is not None]
                if values:
                    condition_means.append(statistics.mean(values))
            period_values.append(f"{statistics.mean(condition_means):.3f}" if condition_means else "—")
        log_loss = statistics.mean(
            statistics.mean(item["mean_log_loss"] for item in records[rule]) for records in by_condition.values()
        )
        lines.append(f"| {rule} | {overall:.3f} | " + " | ".join(period_values) + f" | {log_loss:.4f} |")
    lines += ["", "## 条件別の混合FS の全体の精度（%）", "", "| 条件 | " + " | ".join(rules) + " |", "|---|" + "---|" * len(rules)]
    for condition in sorted(by_condition):
        records = by_condition[condition]
        lines.append(
            f"| {condition} | "
            + " | ".join(f"{statistics.mean(item['overall_accuracy'] for item in records[rule]):.3f}" for rule in rules)
            + " |"
        )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
