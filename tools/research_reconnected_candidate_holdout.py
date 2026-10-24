"""再接続後の新しい標本を使う採用保留の研究対照。製品経路には登録しない。"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import mean
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def concept_specific_digest(learner):
    h = hashlib.sha256()
    for name, p in learner.named_parameters():
        if not name.startswith("feature_extractor."):
            h.update(name.encode())
            h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


class ReconnectedHoldout:
    """候補の概念固有部を固定し、共有部を現行へ追随させて新標本だけで評価する。"""

    def __init__(self, *, window, gate, adopter=None):
        from tools.research_shadow_learner_diagnostic.adoption import adopt_shadow

        if type(window) is not int or window < 0:
            raise ValueError("待機窓は非負整数が必要です")
        self.window = window
        self.gate = gate
        self.adopter = adopter or adopt_shadow
        self.records = []
        self.active = {}
        self.handoffs = []
        self.frames = []
        self.counts = Counter()

    def start(self, *, state, arguments):
        a, source = arguments, state["episode"]
        client = a["client_id"]
        if self.window == 0:
            # 専用test用。同じstateへ既存採用を一度呼ぶので、旧S1との一致を検査できる。
            self.adopter(state=state, arguments=a)
            return
        self.close(client, "replaced", a["indexed_observation"].sample_index)
        learner, optimizer = deepcopy((state["learner"], state["optimizer"]))
        current_id = a["current_training_model_assignment"].current_training_model_id
        current = (
            a["held_model_training_state_registry"]
            .get_held_model_training_state(model_id=current_id)
            .classifier
        )
        learner.attach_shared_feature_extractor(
            shared_feature_extractor=current.feature_extractor
        )
        r = dict(
            client_id=client,
            source_episode_id=source["episode_id"],
            start_sample_index=a["indexed_observation"].sample_index,
            origin_current_model_id=current_id,
            source_score=source["last_score"],
            window=self.window,
            gate=self.gate,
            score=0.0,
            samples=[],
            end_reason=None,
            end_sample_index=None,
            adopted_model_id=None,
        )
        self.records.append(r)
        self.active[client] = dict(
            state=dict(
                learner=learner,
                optimizer=optimizer,
                observations=list(state["observations"]),
                episode=dict(source),
            ),
            record=r,
            head_digest=concept_specific_digest(learner),
        )
        self.handoffs.append(source)

    def close(self, client, reason, sample_index):
        pending = self.active.pop(client, None)
        if pending is not None:
            if (
                concept_specific_digest(pending["state"]["learner"])
                != pending["head_digest"]
            ):
                raise ValueError("保留候補の概念固有部が更新されました")
            pending["record"].update(end_reason=reason, end_sample_index=sample_index)
        return pending

    def observe(self, arguments):
        import torch
        from federated_learning_experiments.learning.models.model_computation_measurement import (
            measure_model_computation,
        )
        from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
            evaluate_classifier_per_sample_log_losses,
        )

        a = arguments
        pending = self.active.get(a["client_id"])
        if pending is None:
            return
        r = pending["record"]
        o = a["indexed_observation"]
        if o.sample_index <= r["start_sample_index"]:
            raise ValueError("評価開始後の新標本だけを使います")
        current_id = a["current_training_model_assignment"].current_training_model_id
        if current_id != r["origin_current_model_id"]:
            self.close(a["client_id"], "assignment_changed", o.sample_index)
            return
        if (
            o.sample_index != r["start_sample_index"] + len(r["samples"]) + 1
            or len(r["samples"]) >= self.window
        ):
            raise ValueError("保留中の標本順序または窓の上限が不正です")
        current = (
            a["held_model_training_state_registry"]
            .get_held_model_training_state(model_id=current_id)
            .classifier
        )
        candidate = pending["state"]["learner"]
        # aliasを共有するので、この標本の通常学習で共有部が変わっても双方へ同じ更新が届く。
        candidate.attach_shared_feature_extractor(
            shared_feature_extractor=current.feature_extractor
        )
        with torch.random.fork_rng(devices=[]), measure_model_computation() as meter:
            losses = [
                float(
                    evaluate_classifier_per_sample_log_losses(
                        classifier=model,
                        input_features=o.training_sample.input_features,
                        observed_class_labels=o.training_sample.observed_class_labels,
                    ).sum()
                )
                for model in (current, candidate)
            ]
        self.counts.update(asdict(meter.get_model_computation_counts()))
        r["score"] += losses[0] - losses[1]
        r["samples"].append(
            dict(
                sample_index=o.sample_index,
                current_model_id=current_id,
                current_loss=losses[0],
                candidate_loss=losses[1],
                score=r["score"],
                observed_concept_id=o.observed_concept_id,
            )
        )

    def after_sample(self, arguments, result):
        a = arguments
        client = a["client_id"]
        pending = self.active.get(client)
        if pending is None:
            return
        o = a["indexed_observation"]
        if result.alarm_occurrence_handling is not None:
            self.close(client, "alarm", o.sample_index)
            return
        current_id = a["current_training_model_assignment"].current_training_model_id
        if current_id != pending["record"]["origin_current_model_id"]:
            self.close(client, "assignment_changed", o.sample_index)
            return
        if len(pending["record"]["samples"]) != self.window:
            return
        if self.gate and pending["record"]["score"] <= 0:
            self.close(client, "rejected", o.sample_index)
            return
        pending = self.close(client, "adopted", o.sample_index)
        self.adopter(state=pending["state"], arguments=a)
        pending["record"]["adopted_model_id"] = pending["state"]["episode"][
            "adopted_model_id"
        ]

    @contextmanager
    def activated(self):
        import federated_learning_experiments.runtime.fedsda_run_client as client
        import federated_learning_experiments.runtime.observed_sample_processing as processing

        original_process = client.process_observed_sample
        original_predict = (
            processing.predict_observed_sample_and_update_prediction_weights
        )

        def predict(**kw):
            result = original_predict(**kw)
            # 通常予測とS1の記録の後、通常の現行モデル学習より前に評価する。
            self.observe(self.frames[-1])
            return result

        def process(**kw):
            handoff_count = len(self.handoffs)
            self.frames.append(kw)
            try:
                result = original_process(**kw)
                # S1はhandlerへ渡すと元の影を閉じる。実採用ではなく評価開始として区別する。
                for source in self.handoffs[handoff_count:]:
                    if source["end_reason"] != "shadow_adopted":
                        raise ValueError("S1の引渡し記録が予想と異なります")
                    source["end_reason"] = "reconnected_validation_started"
                self.after_sample(kw, result)
                return result
            finally:
                self.frames.pop()

        try:
            with (
                patch.object(client, "process_observed_sample", process),
                patch.object(
                    processing,
                    "predict_observed_sample_and_update_prediction_weights",
                    predict,
                ),
            ):
                yield
        finally:
            for client_id, pending in list(self.active.items()):
                r = pending["record"]
                self.close(
                    client_id,
                    "run_end",
                    r["samples"][-1]["sample_index"]
                    if r["samples"]
                    else r["start_sample_index"],
                )


def run_holdout_arm(job, *, window, gate):
    from tools.research_provisional_candidate_expert.run import _execute_job as run
    from tools.research_shadow_learner_diagnostic.controls import (
        stationary_concept_policy,
    )
    from tools.research_shadow_learner_diagnostic.diagnostic import (
        shadow_learner_diagnostic_policy,
    )

    common = {k: job[k] for k in ("dataset", "interval", "seed", "sample_count")}
    common.update(
        variant="product-default", check=False, record_prediction_period_metrics=True
    )
    diagnostics, trace = {}, []
    control = ReconnectedHoldout(window=window, gate=gate)
    with (
        stationary_concept_policy(
            dataset=job["dataset"], concept_id=job["fixed_concept"]
        ),
        shadow_learner_diagnostic_policy(
            diagnostics=diagnostics,
            trace=trace,
            horizon_samples=job["horizon_samples"],
            adoption_handler=control.start,
        ),
        control.activated(),
    ):
        measured = run(common)
    diagnostics["trace_row_count"] = len(trace)
    if "output" in job:
        label = "gate" if gate else "wait"
        name = f"{job['dataset']}_{job['stream']}_i{job['interval']}_s{job['seed']}_{label}.jsonl"
        (Path(job["output"]) / name).write_text(
            "".join(json.dumps(t) + "\n" for t in trace), encoding="utf-8"
        )
        diagnostics["trace_file"] = name
    return dict(
        measured=measured,
        s1_diagnostic=diagnostics,
        holdouts=control.records,
        validation_counts=dict(control.counts),
    )


def execute_job(job):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    from tools.research_shadow_learner_diagnostic.run_followup import (
        execute_job as immediate,
    )

    old = immediate(job)
    arms = {
        label: run_holdout_arm(job, window=job["window"], gate=label == "gate")
        for label in ("wait", "gate")
    }
    return dict(
        job=job,
        baseline=old["baseline"],
        immediate=old["measured"],
        immediate_diagnostic=old["shadow_diagnostic"],
        **arms,
    )


def analyze(output):
    from tools.research_shadow_learner_diagnostic.summarize_standard_metrics import (
        total_mac,
    )

    meta = json.loads((output / "research-source.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    if (
        Counter(json.dumps(r["job"], sort_keys=True) for r in rows)
        != Counter(json.dumps(j, sort_keys=True) for j in meta["jobs"])
        or (output / "failures.json").exists()
    ):
        raise ValueError("runの欠落・失敗があります")
    for name, h in meta["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != h:
            raise ValueError(f"実行ソースが異なります: {name}")
    groups = defaultdict(list)
    for row in rows:
        j = row["job"]
        groups[tuple(j[k] for k in ("dataset", "stream", "interval"))].append(row)
        for label in ("wait", "gate"):
            arm = row[label]
            for r in arm["holdouts"]:
                score = 0.0
                for n, t in enumerate(r["samples"], 1):
                    score += t["current_loss"] - t["candidate_loss"]
                    if (
                        t["sample_index"] != r["start_sample_index"] + n
                        or abs(score - t["score"]) > 1e-10
                    ):
                        raise ValueError("保留の逐次記録が不正です")
                if (
                    abs(score - r["score"]) > 1e-10
                    or len(r["samples"]) > j["window"]
                    or r["end_reason"]
                    not in {
                        "adopted",
                        "rejected",
                        "alarm",
                        "assignment_changed",
                        "replaced",
                        "run_end",
                    }
                ):
                    raise ValueError("保留の終端が不正です")
                if (
                    r["end_reason"] in ("adopted", "rejected")
                    and len(r["samples"]) != j["window"]
                ):
                    raise ValueError("窓が完了する前に判定しました")
                if (
                    label == "gate"
                    and r["end_reason"] == "adopted"
                    and score <= 0
                    or r["end_reason"] == "rejected"
                    and (label != "gate" or score > 0)
                ):
                    raise ValueError("保留の採用条件が不正です")
                if (r["adopted_model_id"] is not None) != (
                    r["end_reason"] == "adopted"
                ):
                    raise ValueError("採用IDの記録が不正です")
            if any(
                v
                for k, v in arm["validation_counts"].items()
                if "training" in k or "optimizer_step" in k
            ):
                raise ValueError("保留候補を学習しました")
    lines = [
        "# 再接続後の新標本による採用保留",
        "",
        "各行seed平均。差ppは既定との差。MB=10^6byte、G MAC=10^9。追加評価を含むMAC、負荷差を含む秒。",
        "",
        "| データ | 入力 | 間隔 | 構成 | 精度% | 差pp | 上りMB | 下りMB | G MAC | 秒 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = []
    for key, rs in sorted(groups.items()):
        for label in ("baseline", "immediate", "wait", "gate"):
            ms = [
                r[label] if label in ("baseline", "immediate") else r[label]["measured"]
                for r in rs
            ]
            s = dict(
                dataset=key[0],
                stream=key[1],
                interval=key[2],
                variant=label,
                accuracy=mean(m["accuracy"] for m in ms),
                delta_pp=100
                * mean(
                    m["accuracy"] - r["baseline"]["accuracy"] for m, r in zip(ms, rs)
                ),
                uploaded_mb=mean(m["uploaded_bytes"] for m in ms) / 1e6,
                downloaded_mb=mean(m["downloaded_bytes"] for m in ms) / 1e6,
                gmac=mean(total_mac(m["model_computation_counts"]) for m in ms) / 1e9,
                seconds=mean(m["seconds"] for m in ms),
            )
            if label in ("wait", "gate"):
                s["outcomes"] = dict(
                    Counter(t["end_reason"] for r in rs for t in r[label]["holdouts"])
                )
            summary.append(s)
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {label} | {100 * s['accuracy']:.3f} | {s['delta_pp']:+.4f} | {s['uploaded_mb']:.3f} | {s['downloaded_mb']:.3f} | {s['gmac']:.3f} | {s['seconds']:.2f} |"
            )
    (output / "analysis_holdout.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_holdout.json").write_text(
        json.dumps(
            dict(
                groups=summary,
                input_sha256={
                    n: hashlib.sha256((output / n).read_bytes()).hexdigest()
                    for n in ("results.json", "research-source.json")
                },
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--datasets", nargs="+", choices=["sea4", "sine2"], default=["sea4", "sine2"]
    )
    p.add_argument("--intervals", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--sample-count", type=int, default=2500)
    p.add_argument("--horizon-samples", type=int, default=240)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--print-plan", action="store_true")
    a = p.parse_args()
    if (
        min(a.intervals + [a.sample_count, a.horizon_samples, a.window, a.workers]) < 1
        or min(a.seeds) < 0
        or any(a.sample_count % i for i in a.intervals)
    ):
        p.error("正の値・間隔の倍数の標本数・非負seedが必要です")
    jobs = [
        dict(
            dataset=d,
            stream="drift" if c is None else f"fixed{c}",
            fixed_concept=c,
            interval=i,
            seed=s,
            sample_count=a.sample_count,
            horizon_samples=a.horizon_samples,
            window=a.window,
            task="adoption",
            disabled=False,
            output=str(a.output),
        )
        for d in a.datasets
        for c in ([None, 0, 2] if d == "sea4" else [None, 0, 1])
        for i in a.intervals
        for s in a.seeds
    ]
    if len(
        {(j["dataset"], j["stream"], j["interval"], j["seed"]) for j in jobs}
    ) != len(jobs):
        p.error("条件が重複しています")
    if a.print_plan:
        print(
            json.dumps(
                dict(
                    jobs=jobs,
                    job_count=len(jobs),
                    run_count=4 * len(jobs),
                    workers=a.workers,
                    exists=a.output.exists(),
                ),
                indent=2,
            )
        )
        return
    a.output.mkdir(parents=True, exist_ok=False)
    sources = (
        list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "tools/research_shadow_learner_diagnostic").glob("*.py"))
        + list((ROOT / "tools/research_provisional_candidate_expert").glob("*.py"))
        + [Path(__file__)]
    )
    (a.output / "research-source.json").write_text(
        json.dumps(
            dict(
                jobs=jobs,
                workers=a.workers,
                sha256={
                    f.resolve().relative_to(ROOT).as_posix(): hashlib.sha256(
                        f.read_bytes()
                    ).hexdigest()
                    for f in sources
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = []
    errors = []
    with ProcessPoolExecutor(
        max_workers=a.workers, mp_context=get_context("spawn")
    ) as pool:
        pending = {pool.submit(execute_job, j): j for j in jobs}
        for f in as_completed(pending):
            try:
                rows.append(f.result())
            except Exception:
                errors.append(dict(job=pending[f], traceback=traceback.format_exc()))
                (a.output / "failures.json").write_text(
                    json.dumps(errors, indent=2), encoding="utf-8"
                )
                continue
            temp = a.output / "results.json.tmp"
            temp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            temp.replace(a.output / "results.json")
            print(
                pending[f]["dataset"],
                pending[f]["stream"],
                pending[f]["seed"],
                "complete",
                flush=True,
            )
    if errors:
        raise SystemExit(1)
    analyze(a.output)


if __name__ == "__main__":
    main()
