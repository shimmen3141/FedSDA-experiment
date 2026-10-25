"""S3寄与診断の非介入性・入力一致・符号長を検算し、寄与と全体差を分ける。"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

from tools.research_s3_contribution.run import KEYS, ROOT, digest, reference_rows

sys.path.insert(0, str(ROOT / "src"))


def validate_prediction_coverage(predictions, sample_count):
    """件数だけでなく、10clientsの全sample indexとの完全一致を要求する。"""
    keys = [(p["client_id"], p["sample_index"]) for p in predictions]
    expected = {(c, s) for c in range(10) for s in range(sample_count)}
    if (
        len(keys) != len(expected)
        or len(set(keys)) != len(keys)
        or set(keys) != expected
    ):
        raise ValueError("予測のclient/sample indexの完全な被覆がありません")


def summarize(samples):
    defined = [
        s
        for s in samples
        if s.get("correctness_benefit") is not None and s.get("status") in ("defined",)
    ]
    return dict(
        observations=len(samples),
        defined=len(defined),
        undefined=len(samples) - len(defined),
        helped=sum(s["correctness_benefit"] > 0 for s in defined),
        harmed=sum(s["correctness_benefit"] < 0 for s in defined),
        net_correct=sum(s["correctness_benefit"] for s in defined),
        mean_log_loss_benefit=mean(s["log_loss_benefit"] for s in defined)
        if defined
        else None,
    )


def validate_decision(record):
    from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
        select_assignment_by_minimum_description_length,
    )

    args = dict(record["mdl_inputs"])
    args["current_model_log_losses"] = tuple(args["current_model_log_losses"])
    args["held_model_log_losses_by_model_id"] = tuple(
        (m, tuple(v)) for m, v in args["held_model_log_losses_by_model_id"]
    )
    for name in (
        "new_model_code_lengths_by_change_interval_sample_count",
        "new_model_learner_names_by_change_interval_sample_count",
    ):
        args[name] = {int(k): v for k, v in args[name].items()}
    if (
        asdict(select_assignment_by_minimum_description_length(**args))
        != record["selection"]
    ):
        raise ValueError("保存したMDL入力から選択を再現できません")


def period_flags(predictions):
    """製品既定の変化後200標本。概念は診断だけに使い、採否へ戻さない。"""
    flags = {}
    for client in range(10):
        ps = sorted(
            (p for p in predictions if p["client_id"] == client),
            key=lambda p: p["sample_index"],
        )
        last_change = -100000
        for i, p in enumerate(ps):
            if i and p["concept"] != ps[i - 1]["concept"]:
                last_change = p["sample_index"]
            flags[(client, p["sample_index"])] = p["sample_index"] < last_change + 200
    return flags


def compare_traces(base, measured, head_adoption_order=None):
    if len(base) != len(measured):
        raise ValueError("traceの長さが異なります")
    bmap = {(p["client_id"], p["sample_index"]): p for p in base}
    mkeys = [(p["client_id"], p["sample_index"]) for p in measured]
    if (
        len(bmap) != len(base)
        or len(set(mkeys)) != len(measured)
        or set(mkeys) != set(bmap)
    ):
        raise ValueError("入力のclient/sample keyが一致しません")
    same_order = list(bmap) == mkeys
    counts = defaultdict(lambda: dict(n=0, helped=0, harmed=0, net=0))
    first_probability = None
    flags = period_flags(base)
    before_head_net = 0
    for order, m in enumerate(measured):
        b = bmap[m["client_id"], m["sample_index"]]
        if any(
            b[k] != m[k]
            for k in ("client_id", "sample_index", "concept", "label", "input_hash")
        ):
            raise ValueError("入力列が異なるため同標本の比較ができません")
        if first_probability is None and b["probabilities"] != m["probabilities"]:
            first_probability = dict(
                prediction_order=order,
                client_id=b["client_id"],
                sample_index=b["sample_index"],
            )
        benefit = int(m["correct"]) - int(b["correct"])
        period = "recovery" if flags[b["client_id"], b["sample_index"]] else "stable"
        for key in (
            "all",
            f"concept:{b['concept']}",
            f"client:{b['client_id']}",
            period,
        ):
            d = counts[key]
            d["n"] += 1
            d["helped"] += benefit > 0
            d["harmed"] += benefit < 0
            d["net"] += benefit
        if head_adoption_order is not None and order <= head_adoption_order:
            before_head_net += benefit
    return dict(
        same_processing_order=same_order,
        strata={
            k: dict(v, delta_pp=100 * v["net"] / v["n"]) for k, v in counts.items()
        },
        first_probability_difference=first_probability,
        net_correct_before_first_head_adoption=before_head_net
        if head_adoption_order is not None
        else None,
    )


def first_decision_difference(base, measured):
    bmap = {(r["client_id"], r["sample_index"]): r for r in base}
    mmap = {(r["client_id"], r["sample_index"]): r for r in measured}
    fields = (
        "selected_hypothesis",
        "change_interval_sample_count",
        "destination_model_id",
        "new_model_learner_name",
    )
    # 基準側だけで起きた判断も含め、共通の予測処理順で最初を探す。
    keys = sorted(
        set(bmap) | set(mmap),
        key=lambda key: min(
            r["prediction_order"]
            for r in (bmap.get(key), mmap.get(key))
            if r is not None
        ),
    )
    for key in keys:
        b, m = bmap.get(key), mmap.get(key)
        if (
            b is None
            or m is None
            or any(b["selection"][k] != m["selection"][k] for k in fields)
        ):
            first = (
                m
                if b is None
                else b
                if m is None
                else min((b, m), key=lambda r: r["prediction_order"])
            )
            return dict(
                client_id=key[0],
                sample_index=key[1],
                prediction_order=first["prediction_order"],
                baseline=None if b is None else b["selection"],
                measured=None if m is None else m["selection"],
                same_held_parameters=b is not None
                and m is not None
                and b["held_parameter_hashes"] == m["held_parameter_hashes"],
                same_window=b is not None
                and m is not None
                and b["observation_indices"] == m["observation_indices"],
            )
    return None


def analyze(output):
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    for p, h in metadata["source_hashes"].items():
        if digest(ROOT / p) != h:
            raise ValueError(f"実行ソースが異なります: {p}")
    reference = Path(metadata["reference"])
    if digest(reference / "results.json") != metadata["reference_results_sha256"]:
        raise ValueError("計測なし参照が変化しています")
    refs = reference_rows(reference)
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    complete = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    actual = [tuple(r["job"][k] for k in KEYS) for r in rows]
    expected = [tuple(j[k] for k in KEYS) for j in metadata["jobs"]]
    if (
        Counter(actual) != Counter(expected)
        or len(set(actual)) != len(actual)
        or complete.get("source_hashes_verified") is not True
        or complete["run_count"] != len(rows)
    ):
        raise ValueError("実行の欠落・重複・未完了があります")
    seeds = {r["job"]["seed"] for r in rows}
    if len(seeds) < 4:
        raise ValueError("4seed以上が必要です")
    details = {}
    run_summaries = []
    for row in rows:
        metrics = row["metrics"]
        ref = refs[tuple(row["job"][k] for k in KEYS)]
        if any(metrics[k] != v for k, v in ref.items() if k != "seconds"):
            raise ValueError("計測なし参照の値と一致しません")
        path = output / row["diagnostic_file"]
        if (
            path.resolve().parent != output.resolve()
            or digest(path) != row["diagnostic_sha256"]
        ):
            raise ValueError("診断ファイルが変更されています")
        d = json.loads(path.read_text(encoding="utf-8"))
        ps = d["predictions"]
        validate_prediction_coverage(ps, row["job"]["sample_count"])
        for p in ps:
            probabilities = p["probabilities"]
            if len(probabilities) != 10 or not all(
                math.isfinite(v) and 0 <= v <= 1 for v in probabilities
            ):
                raise ValueError("MNISTの予測確率が不正です")
            correct = max(range(10), key=probabilities.__getitem__) == p["label"]
            if correct != p["correct"]:
                raise ValueError("予測確率から復元した正誤が異なります")
        pmap = {(p["client_id"], p["sample_index"]): p for p in ps}
        if len(ps) != 10 * row["job"]["sample_count"] or len(pmap) != len(ps):
            raise ValueError("全標本traceが不完全です")
        prediction_digest = hashlib.sha256(
            json.dumps([p["correct"] for _, p in sorted(pmap.items())]).encode()
        ).hexdigest()
        if (
            prediction_digest != metrics["prediction_record_digest"]
            or mean(p["correct"] for p in ps) != metrics["accuracy"]
        ):
            raise ValueError("標準精度とtraceが異なります")
        for decision in d["decisions"]:
            validate_decision(decision)
        candidate_summaries = []
        warm_events = iter(
            e for e in metrics["research_events"] if e["event"] == "matched_candidate"
        )
        for r in d["candidates"]:
            samples = r["removal_samples"]
            for sample in samples:
                p = pmap[sample["client_id"], sample["sample_index"]]
                if (
                    sample["actual_probabilities"] != p["probabilities"]
                    or sample["label"] != p["label"]
                    or sample["observed_concept_id"] != p["concept"]
                ):
                    raise ValueError("候補除去と実予測の標本が異なります")
            warm = r["learner"] == "warm_started_learner"
            event = (
                next(warm_events)
                if warm and row["job"]["research_variant"] != "product-default"
                else None
            )
            decision = next(
                x
                for x in d["decisions"]
                if (x["client_id"], x["sample_index"])
                == (r["client_id"], r["sample_index"])
            )
            origin30 = [
                s
                for s in samples
                if s["is_origin_client"] and 1 <= s["adoption_age"] <= 30
            ]
            candidate_summaries.append(
                dict(
                    token=r["token"],
                    client_id=r["client_id"],
                    sample_index=r["sample_index"],
                    learner=r["learner"],
                    is_head_only=r["is_head_only"],
                    registered_model_id=r["registered_model_id"],
                    first_merge=r["first_merge"],
                    pre_adoption_features=dict(
                        selected_k=r["selection"]["change_interval_sample_count"],
                        selected_description_length=r["selection"][
                            "description_length"
                        ],
                        advantage_over_current=sum(
                            decision["mdl_inputs"]["current_model_log_losses"]
                        )
                        - r["selection"]["description_length"],
                        trained_samples=r["trained_samples"],
                        update_steps=r["update_steps"],
                        initialization_changed=None
                        if event is None
                        else event["initialization_changed"],
                        initial_matches_scoring_current=r["initial_classifier_hash"]
                        == decision["held_parameter_hashes"][
                            str(decision["current_model_id"])
                        ],
                    ),
                    adoption_parameter_hash_changed=r["pre_adoption_parameter_hash"]
                    != r["post_adoption_parameter_hash"],
                    contribution=summarize(samples),
                    origin_first30=summarize(origin30),
                    origin_first30_complete=len(origin30) == 30,
                    by_scope={
                        scope: summarize([s for s in samples if s["scope"] == scope])
                        for scope in ("temporary", "global")
                    },
                    by_origin={
                        name: summarize(
                            [s for s in samples if s["is_origin_client"] == flag]
                        )
                        for name, flag in (("origin", True), ("other", False))
                    },
                    by_concept={
                        str(c): summarize(
                            [s for s in samples if s["observed_concept_id"] == c]
                        )
                        for c in sorted({s["observed_concept_id"] for s in samples})
                    },
                )
            )
        if next(warm_events, None) is not None:
            raise ValueError("warm採用の件数が研究イベントと一致しません")
        warm = [
            r for r in candidate_summaries if r["learner"] == "warm_started_learner"
        ]
        joint = summarize(d["joint_removals"])
        joint["net_correct_over_run_pp"] = (
            None if joint["undefined"] else 100 * joint["net_correct"] / len(ps)
        )
        run_summaries.append(
            dict(
                job=row["job"],
                metrics=metrics,
                joint_warm_direct=joint,
                joint_warm_by_concept={
                    str(c): summarize(
                        [s for s in d["joint_removals"] if s["concept"] == c]
                    )
                    for c in sorted({s["concept"] for s in d["joint_removals"]})
                },
                warm_candidates=warm,
                all_candidate_count=len(candidate_summaries),
                all_candidates=candidate_summaries,
                prediction_count=len(ps),
                decisions=len(d["decisions"]),
            )
        )
        # 重い枝別確率の全記録はディスクに残す。比較には軽い全予測・判断だけを保持する。
        details[row["job"]["seed"], row["job"]["research_variant"]] = dict(
            predictions=d["predictions"],
            decisions=d["decisions"],
            candidates=[
                dict(
                    is_head_only=r["is_head_only"],
                    prediction_order=r["prediction_order"],
                )
                for r in d["candidates"]
            ],
        )
    comparisons = []
    for seed in sorted(seeds):
        for reference_variant, variant in (
            ("product-default", "warm-matched"),
            ("product-default", "output-layer-matched"),
            ("warm-matched", "output-layer-matched"),
        ):
            b = details[seed, reference_variant]
            d = details[seed, variant]
            heads = [
                r["prediction_order"] for r in d["candidates"] if r["is_head_only"]
            ]
            comparison = compare_traces(
                b["predictions"], d["predictions"], min(heads) if heads else None
            )
            comparison.update(
                seed=seed,
                reference_variant=reference_variant,
                variant=variant,
                first_selection_difference=first_decision_difference(
                    b["decisions"], d["decisions"]
                ),
            )
            # 安定期・回復期の定義も製品の指標へ照合する。
            for name, native in (
                ("stable", "stable_period_accuracy"),
                ("recovery", "recovery_period_accuracy"),
            ):
                bm = next(
                    r["metrics"]
                    for r in rows
                    if r["job"]["seed"] == seed
                    and r["job"]["research_variant"] == reference_variant
                )
                mm = next(
                    r["metrics"]
                    for r in rows
                    if r["job"]["seed"] == seed
                    and r["job"]["research_variant"] == variant
                )
                if (
                    abs(
                        comparison["strata"][name]["delta_pp"]
                        - 100
                        * (
                            mm["prediction_period_accuracies"][native]
                            - bm["prediction_period_accuracies"][native]
                        )
                    )
                    > 1e-10
                ):
                    raise ValueError("安定期・回復期の定義が異なります")
            comparisons.append(comparison)
    result = dict(
        runs=run_summaries,
        comparisons=comparisons,
        analysis_source_sha256=digest(Path(__file__)),
        input_result_sha256=digest(output / "results.json"),
        extra_model_forward_count=0,
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            [
                dict(
                    seed=r["job"]["seed"],
                    variant=r["job"]["research_variant"],
                    joint=r["joint_warm_direct"],
                    warm_candidates=len(r["warm_candidates"]),
                )
                for r in run_summaries
            ],
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
