"""実採用候補を、最初の統合まで実際の両枝混合から一つずつ外して診断する。"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from tools.research_shadow_model_lifecycle import LifecycleTracker  # noqa: E402
from tools.research_current_mixture_benefit import probability_statistics  # noqa: E402


def combine_branches(synchronized, local, weights, branches, classes):
    from federated_learning_experiments.learning.prediction.class_probability_calculations import (
        combine_model_prediction_probabilities,
    )

    sync = combine_model_prediction_probabilities(
        prediction_probabilities_by_model_id=synchronized,
        prediction_weights_by_model_id=weights,
        class_count=classes,
    )
    loc = combine_model_prediction_probabilities(
        prediction_probabilities_by_model_id=local,
        prediction_weights_by_model_id=weights,
        class_count=classes,
    )
    return sync * branches[0] + loc * branches[1]


def without_candidate(synchronized, local, weights, branches, classes, removed):
    """両枝の内側だけを再正規化する。外側係数を保持し、残余なしはNoneとする。"""
    if (
        set(synchronized) != set(local)
        or set(weights) != set(local)
        or removed not in weights
    ):
        raise ValueError("枝別のモデル集合が一致しません")
    remaining = {k: v for k, v in weights.items() if k != removed}
    mass = math.fsum(remaining.values())
    if not remaining or mass <= 1e-12:
        return None
    normalized = {k: v / mass for k, v in remaining.items()}
    return combine_branches(
        {k: synchronized[k] for k in remaining},
        {k: local[k] for k in remaining},
        normalized,
        branches,
        classes,
    )


class RemovalTracker(LifecycleTracker):
    """来歴を再利用し、通常の確率結合をcopyする。モデルの呼出しは増やさない。"""

    def __init__(self):
        super().__init__()
        self.collecting = False
        self.last_combination = None
        self.captured = None

    def adopted(self, state, arguments):
        super().adopted(state, arguments)
        self.records[-1]["removal_samples"] = []

    def direct_tokens(self, client, model):
        tokens = (
            self.local.get((client, model), ())
            if model < 0
            else self.global_ids.get(model, ())
        )
        return [t for t in tokens if self.records[t]["first_merge"] is None]

    def predicted(self, client, sample, result, branch_weights):
        import torch

        super().predicted(client, sample, result, branch_weights)
        targets = [
            (m, t)
            for m in result.prediction_weights_by_model_id
            for t in self.direct_tokens(client, m)
        ]
        if not targets:
            return
        if self.captured is None:
            raise ValueError("実際の局所混合の入力を取得できませんでした")
        local, used_weights, branches = self.captured
        if used_weights != result.prediction_weights_by_model_id or tuple(
            branches
        ) != tuple(branch_weights):
            raise ValueError("観測前の混合係数が一致しません")
        sync = result.prediction_probabilities_by_model_id
        weights = result.prediction_weights_by_model_id
        classes = (
            self.frames[-1]["held_model_training_state_registry"]
            .get_held_model_training_state(model_id=next(iter(weights)))
            .classifier.class_count
        )
        actual = result.combined_prediction_probabilities.detach()
        recombined = combine_branches(sync, local, weights, branches, classes)
        if not torch.equal(actual, recombined):
            raise ValueError("枝別の再結合と実際の最終予測が一致しません")
        observation = self.frames[-1]["indexed_observation"]
        label = int(observation.training_sample.observed_class_labels.item())
        actual_stats = probability_statistics(actual, label=label, classes=classes)
        for model, token in targets:
            record = self.records[token]
            data = dict(
                client_id=client,
                sample_index=sample,
                model_id=model,
                scope="temporary" if model < 0 else "global",
                is_origin_client=client == record["client_id"],
                adoption_age=sample - record["sample_index"],
                label=label,
                observed_concept_id=observation.observed_concept_id,
                weight=weights[model],
                branch_weights=list(branches),
                model_weights={str(k): v for k, v in weights.items()},
                synchronized_probabilities={
                    str(k): v.detach().reshape(-1).tolist() for k, v in sync.items()
                },
                local_probabilities={
                    str(k): v.detach().reshape(-1).tolist() for k, v in local.items()
                },
                actual_probabilities=actual.reshape(-1).tolist(),
                actual=actual_stats,
            )
            removed = (
                actual.clone()
                if weights[model] == 0
                else without_candidate(sync, local, weights, branches, classes, model)
            )
            if removed is None:
                data.update(status="undefined_no_remaining_mass")
            else:
                stats = probability_statistics(removed, label=label, classes=classes)
                data.update(
                    status="defined",
                    without_probabilities=removed.reshape(-1).tolist(),
                    without=stats,
                    log_loss_benefit=stats["log_loss"] - actual_stats["log_loss"],
                    bounded_loss_benefit=stats["bounded_loss"]
                    - actual_stats["bounded_loss"],
                    correctness_benefit=int(actual_stats["correct"])
                    - int(stats["correct"]),
                )
            record["removal_samples"].append(data)

    @contextmanager
    def activated(self):
        import federated_learning_experiments.runtime.observed_sample_prediction as prediction
        import federated_learning_experiments.runtime.observed_sample_processing as processing
        from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_and_local_prediction_mixture import (
            SynchronizedAndLocalPredictionMixture,
        )

        combine = prediction.combine_model_prediction_probabilities
        mix = SynchronizedAndLocalPredictionMixture.mix_prediction_probabilities
        with super().activated():
            original_predict = (
                processing.predict_observed_sample_and_update_prediction_weights
            )

            def combine_capture(**kwargs):
                result = combine(**kwargs)
                if self.collecting:
                    self.last_combination = (
                        {
                            m: p.detach().clone()
                            for m, p in kwargs[
                                "prediction_probabilities_by_model_id"
                            ].items()
                        },
                        dict(kwargs["prediction_weights_by_model_id"]),
                    )
                return result

            def mix_capture(owner, **kwargs):
                if self.collecting:
                    if self.last_combination is None:
                        raise ValueError("局所結合の直前の入力がありません")
                    local, weights = self.last_combination
                    self.captured = (local, weights, owner.snapshot_weights())
                return mix(owner, **kwargs)

            def predict(**kwargs):
                frame = self.frames[-1]
                ids = [
                    s.model_id
                    for s in frame[
                        "held_model_training_state_registry"
                    ].snapshot_ordered_held_model_training_states()
                ]
                self.collecting = any(
                    self.direct_tokens(frame["client_id"], m) for m in ids
                )
                self.last_combination = self.captured = None
                try:
                    return original_predict(**kwargs)
                finally:
                    self.collecting = False
                    self.last_combination = self.captured = None

            with (
                patch.object(
                    prediction,
                    "combine_model_prediction_probabilities",
                    combine_capture,
                ),
                patch.object(
                    SynchronizedAndLocalPredictionMixture,
                    "mix_prediction_probabilities",
                    mix_capture,
                ),
                patch.object(
                    processing,
                    "predict_observed_sample_and_update_prediction_weights",
                    predict,
                ),
            ):
                yield


def execute_job(job):
    from tools.research_shadow_learner_diagnostic.run_followup import execute_job as run

    tracker = RemovalTracker()
    with tracker.activated():
        result = run(job)
    result["model_lifecycle"] = tracker.records
    result["extra_model_forward_count"] = 0
    return result


def summarize_samples(samples):
    defined = [s for s in samples if s["status"] == "defined"]
    return dict(
        evaluated=len(samples),
        defined=len(defined),
        undefined=len(samples) - len(defined),
        helped=sum(s["correctness_benefit"] > 0 for s in defined),
        harmed=sum(s["correctness_benefit"] < 0 for s in defined),
        mean_log_loss_benefit=mean(s["log_loss_benefit"] for s in defined)
        if defined
        else None,
        delta_accuracy_pp=100 * mean(s["correctness_benefit"] for s in defined)
        if defined
        else None,
    )


def summarize_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row["job"][k] for k in ("dataset", "stream", "interval"))].append(
            row
        )
    result = []
    for key, rs in sorted(groups.items()):
        seeds = []
        for row in sorted(rs, key=lambda r: r["job"]["seed"]):
            records = row["model_lifecycle"]
            samples = [s for r in records for s in r["removal_samples"]]
            candidates = [summarize_samples(r["removal_samples"]) for r in records]
            valid = [c for c in candidates if c["defined"]]
            seeds.append(
                dict(
                    seed=row["job"]["seed"],
                    adoptions=len(records),
                    candidates_with_defined_samples=len(valid),
                    no_observations=sum(not r["removal_samples"] for r in records),
                    **summarize_samples(samples),
                    candidate_mean_log_loss_benefit=mean(
                        c["mean_log_loss_benefit"] for c in valid
                    )
                    if valid
                    else None,
                    candidate_mean_accuracy_delta_pp=mean(
                        c["delta_accuracy_pp"] for c in valid
                    )
                    if valid
                    else None,
                    by_scope={
                        scope: summarize_samples(
                            [s for s in samples if s["scope"] == scope]
                        )
                        for scope in ("temporary", "global")
                    },
                    by_client={
                        name: summarize_samples(
                            [s for s in samples if s["is_origin_client"] == flag]
                        )
                        for name, flag in (("origin", True), ("other", False))
                    },
                    by_age={
                        name: summarize_samples(
                            [s for s in samples if lo <= s["adoption_age"] <= hi]
                        )
                        for name, lo, hi in (
                            ("1-30", 1, 30),
                            ("31-100", 31, 100),
                            ("101+", 101, 10**9),
                        )
                    },
                )
            )
        valid = [s for s in seeds if s["defined"]]
        result.append(
            dict(
                dataset=key[0],
                stream=key[1],
                interval=key[2],
                seeds=seeds,
                eligible_seed_count=len(valid),
                all_seed_count=len(seeds),
                seed_balanced_sample_mean_log_loss_benefit=mean(
                    s["mean_log_loss_benefit"] for s in valid
                )
                if valid
                else None,
                seed_balanced_sample_accuracy_delta_pp=mean(
                    s["delta_accuracy_pp"] for s in valid
                )
                if valid
                else None,
                seed_balanced_candidate_mean_log_loss_benefit=mean(
                    s["candidate_mean_log_loss_benefit"] for s in valid
                )
                if valid
                else None,
                seed_balanced_candidate_accuracy_delta_pp=mean(
                    s["candidate_mean_accuracy_delta_pp"] for s in valid
                )
                if valid
                else None,
            )
        )
    return result


def analyze(output):
    from tools.research_shadow_learner_diagnostic.run_followup import (
        analyze as standard_analysis,
    )

    metadata = json.loads((output / "research-source.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    if Counter(json.dumps(r["job"], sort_keys=True) for r in rows) != Counter(
        json.dumps(j, sort_keys=True) for j in metadata["jobs"]
    ):
        raise ValueError("run集合が不正です")
    for name, h in metadata["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != h:
            raise ValueError(f"実行ソース不一致: {name}")
    reference = Path(metadata["removal"]["reference_result"])
    ref_rows = json.loads((reference / "results.json").read_text(encoding="utf-8"))
    refs = {
        tuple(r["job"][k] for k in ("dataset", "stream", "interval", "seed")): r
        for r in ref_rows
    }
    if len(refs) != len(ref_rows):
        raise ValueError("参照が重複しています")
    for row in rows:
        ref = refs[
            tuple(row["job"][k] for k in ("dataset", "stream", "interval", "seed"))
        ]
        for k in (
            "dataset",
            "stream",
            "fixed_concept",
            "interval",
            "seed",
            "sample_count",
            "horizon_samples",
            "task",
            "disabled",
        ):
            if row["job"][k] != ref["job"][k]:
                raise ValueError("参照の条件不一致")
        for part, original in (
            ("baseline", ref["baseline"]),
            ("measured", ref["immediate"]),
        ):
            if {k: v for k, v in row[part].items() if k != "seconds"} != {
                k: v for k, v in original.items() if k != "seconds"
            }:
                raise ValueError("計測なしと全保存値が一致しません")
        if (
            row["shadow_diagnostic"] != ref["immediate_diagnostic"]
            or row["extra_model_forward_count"] != 0
        ):
            raise ValueError("採用経路が変化しました")
        expected = {
            (
                e["client_id"],
                e["episode_id"],
                e["adoption_sample_index"],
                e["adopted_model_id"],
            )
            for e in row["shadow_diagnostic"]["episodes"]
            if "adopted_model_id" in e
        }
        actual = [
            (
                r["client_id"],
                r["episode_id"],
                r["sample_index"],
                r["temporary_model_id"],
            )
            for r in row["model_lifecycle"]
        ]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("採用来歴が対応しません")
        for r in row["model_lifecycle"]:
            merges = [e for e in r["events"] if e["kind"] == "consolidated"]
            for s in r["removal_samples"]:
                if (
                    s["sample_index"] <= r["sample_index"]
                    or merges
                    and s["sample_index"] > merges[0]["origin_sample_index"]
                ):
                    raise ValueError("採用前または統合後を計測しています")
                if s["status"] == "defined" and (
                    not math.isfinite(s["log_loss_benefit"])
                    or s["correctness_benefit"] not in (-1, 0, 1)
                ):
                    raise ValueError("除去差が不正です")
    standard_analysis(output)
    groups = summarize_rows(rows)
    lines = [
        "# 採用候補の統合前の除去差",
        "",
        "正の値は候補を含む実際の混合が優れること。採用×client×標本の延べ対であり、重複する標本や候補別効果を全体精度へ加算しない。",
        "",
        "| データ | 入力 | 間隔 | 対象seed/全seed | 採用 | 定義可能な対 | 正解化/不正解化 | seed均等の損失利得 | seed均等の局所精度差pp |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in groups:
        eligible = g["eligible_seed_count"]
        loss = (
            f"{g['seed_balanced_sample_mean_log_loss_benefit']:+.6f}"
            if eligible
            else "対象なし"
        )
        accuracy = (
            f"{g['seed_balanced_sample_accuracy_delta_pp']:+.4f}"
            if eligible
            else "対象なし"
        )
        lines.append(
            f"| {g['dataset']} | {g['stream']} | {g['interval']} | {eligible}/{g['all_seed_count']} | {sum(s['adoptions'] for s in g['seeds'])} | {sum(s['defined'] for s in g['seeds'])} | {sum(s['helped'] for s in g['seeds'])}/{sum(s['harmed'] for s in g['seeds'])} | {loss} | {accuracy} |"
        )
    (output / "analysis_removal.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_removal.json").write_text(
        json.dumps(
            dict(
                groups=groups,
                reference_pairs_verified=len(rows),
                input_sha256={
                    n: hashlib.sha256((output / n).read_bytes()).hexdigest()
                    for n in ("results.json", "research-source.json")
                },
                reference_input_sha256={
                    n: hashlib.sha256((reference / n).read_bytes()).hexdigest()
                    for n in ("results.json", "research-source.json")
                },
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    from tools.research_shadow_learner_diagnostic import run_followup

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--task", choices=["adoption"], default="adoption")
    args, remaining = parser.parse_known_args()
    if "--disabled" in remaining:
        parser.error("実際の即時採用経路だけを計測します")
    sys.argv = [sys.argv[0], *remaining, "--task", "adoption"]
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    planned = "--print-plan" in sys.argv
    sources = [
        Path(__file__),
        ROOT / "tools/research_shadow_model_lifecycle.py",
        ROOT / "tools/research_current_mixture_benefit.py",
        ROOT / "tools/research_reconnected_candidate_holdout.py",
    ]
    hashes = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    # 参照を読むだけで計画確認中には成果物を作らない。
    meta = json.loads(
        (args.reference_result / "research-source.json").read_text(encoding="utf-8")
    )
    if any(
        hashlib.sha256((ROOT / n).read_bytes()).hexdigest() != h
        for n, h in meta["sha256"].items()
    ):
        raise ValueError("参照の実行ソースが一致しません")
    existed = output.exists()
    try:
        with patch.object(run_followup, "execute_job", execute_job):
            run_followup.main()
    finally:
        path = output / "research-source.json"
        if not planned and not existed and path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta["sha256"].update(hashes)
            meta["removal"] = dict(
                reference_result=str(args.reference_result), extra_model_forward_count=0
            )
            path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if not planned:
        analyze(output)
