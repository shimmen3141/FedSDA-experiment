"""同一のwait30候補で、現行単体差と混合への仮想追加利得を非介入で測る。"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
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
from tools import research_reconnected_candidate_holdout as holdout  # noqa: E402

ALPHAS = (0.1, 0.01, 0.5)


def probability_statistics(probabilities, *, label, classes):
    """旧wait30と同じfloat32演算順序の対数損失と製品の正誤を返す。"""
    import torch
    from federated_learning_experiments.learning.prediction.class_probability_calculations import (
        predict_class_labels_from_prediction_scores,
    )

    predicted = int(
        predict_class_labels_from_prediction_scores(
            prediction_scores=probabilities,
            class_count=classes,
            prediction_scores_are_validated=False,
        ).item()
    )
    bounded = (
        torch.abs(probabilities.reshape(-1) - float(label))
        if classes == 2
        else 1.0 - probabilities[:, label]
    )
    loss = float((-torch.log(torch.clamp(1.0 - bounded, min=1e-12))).item())
    true_probability = (
        (float(probabilities.item()) if label else 1.0 - float(probabilities.item()))
        if classes == 2
        else float(probabilities[0, label])
    )
    return dict(
        log_loss=loss,
        bounded_loss=float(bounded.item()),
        true_class_probability=true_probability,
        predicted_class=predicted,
        correct=predicted == label,
    )


def compare_probabilities(current, candidate, mixture, *, label, classes):
    """同じ標本の確率から対数損失差と仮想混合の正誤を計算する。"""
    predictions = dict(current=current, candidate=candidate, mixture=mixture)
    predictions.update(
        {
            f"add_{alpha}": (1.0 - alpha) * mixture + alpha * candidate
            for alpha in ALPHAS
        }
    )
    stats = {
        name: probability_statistics(p, label=label, classes=classes)
        for name, p in predictions.items()
    }
    return dict(
        probabilities={
            name: p.detach().reshape(-1).tolist() for name, p in predictions.items()
        },
        statistics=stats,
        gains=dict(
            current=stats["current"]["log_loss"] - stats["candidate"]["log_loss"],
            mixture_alone=stats["mixture"]["log_loss"] - stats["candidate"]["log_loss"],
            **{
                f"add_{alpha}": stats["mixture"]["log_loss"]
                - stats[f"add_{alpha}"]["log_loss"]
                for alpha in ALPHAS
            },
        ),
    )


class MixtureBenefitObserver(holdout.ReconnectedHoldout):
    """既存評価のforward出力をcopyする。追加forwardも採否への介入も行わない。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.gate or self.window <= 0:
            raise ValueError("正の窓を持つ非ゲートwait構成だけを計測します")
        self.queued = None

    def start(self, **kwargs):
        super().start(**kwargs)
        self.records[-1]["mixture_samples"] = []

    def observe(self, arguments):
        from federated_learning_experiments.learning.prediction.class_probability_calculations import (
            convert_model_outputs_to_prediction_probabilities,
        )

        self.queued = None
        pending = self.active.get(arguments["client_id"])
        if pending is None:
            return super().observe(arguments)
        current_id = arguments[
            "current_training_model_assignment"
        ].current_training_model_id
        if current_id != pending["record"]["origin_current_model_id"]:
            return super().observe(arguments)
        current = (
            arguments["held_model_training_state_registry"]
            .get_held_model_training_state(model_id=current_id)
            .classifier
        )
        candidate = pending["state"]["learner"]
        outputs = []

        def capture(module, inputs, output):
            outputs.append(output.detach().clone())

        hooks = [model.register_forward_hook(capture) for model in (current, candidate)]
        try:
            super().observe(arguments)
        finally:
            for hook in hooks:
                hook.remove()
        if len(outputs) != 2:
            raise ValueError("元のwait評価の2出力を取得できませんでした")
        probabilities = convert_model_outputs_to_prediction_probabilities(
            model_outputs_by_model_id=dict(enumerate(outputs)),
            class_count=current.class_count,
        )
        self.queued = (pending["record"], probabilities, current.class_count, arguments)

    @contextmanager
    def activated(self):
        import federated_learning_experiments.runtime.observed_sample_processing as processing

        with super().activated():
            original = processing.predict_observed_sample_and_update_prediction_weights

            def predict(**kwargs):
                mixture = kwargs.get("synchronized_and_local_prediction_mixture")
                if mixture is None:
                    raise ValueError("同期/局所の混合が必要です")
                branch_weights = mixture.snapshot_weights()
                result = original(**kwargs)
                if self.queued is not None:
                    record, probabilities, classes, arguments = self.queued
                    self.queued = None
                    observation = arguments["indexed_observation"]
                    label = int(
                        observation.training_sample.observed_class_labels.item()
                    )
                    data = compare_probabilities(
                        probabilities[0],
                        probabilities[1],
                        result.combined_prediction_probabilities.detach().clone(),
                        label=label,
                        classes=classes,
                    )
                    old = record["samples"][-1]
                    if (
                        data["statistics"]["current"]["log_loss"] != old["current_loss"]
                        or data["statistics"]["candidate"]["log_loss"]
                        != old["candidate_loss"]
                    ):
                        raise ValueError("元wait30と標本別対数損失が一致しません")
                    data.update(
                        sample_index=observation.sample_index,
                        observed_class_id=label,
                        observed_concept_id=observation.observed_concept_id,
                        branch_weights=list(branch_weights),
                        model_weights={
                            str(k): v
                            for k, v in result.prediction_weights_by_model_id.items()
                        },
                    )
                    record["mixture_samples"].append(data)
                return result

            with patch.object(
                processing,
                "predict_observed_sample_and_update_prediction_weights",
                predict,
            ):
                yield


def verify_reference(arm, reference):
    """追加の確率記録と秒数以外は元wait30と完全一致させる。"""
    for name in arm:
        if name == "measured":
            left = {k: v for k, v in arm[name].items() if k != "seconds"}
            right = {k: v for k, v in reference[name].items() if k != "seconds"}
        elif name == "holdouts":
            left = [
                {k: v for k, v in r.items() if k != "mixture_samples"}
                for r in arm[name]
            ]
            right = reference[name]
        else:
            left, right = arm[name], reference[name]
        if left != right:
            raise ValueError(f"参照wait30との不一致: {name}")


def execute_job(job):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    from tools.research_provisional_candidate_expert.run import _execute_job as run
    from tools.research_shadow_learner_diagnostic.controls import (
        stationary_concept_policy,
    )

    refs = json.loads(
        (Path(job["reference_result"]) / "results.json").read_text(encoding="utf-8")
    )
    matches = [
        r
        for r in refs
        if all(
            r["job"][k] == job[k]
            for k in (
                "dataset",
                "stream",
                "fixed_concept",
                "interval",
                "seed",
                "sample_count",
                "window",
                "horizon_samples",
            )
        )
    ]
    if len(matches) != 1:
        raise ValueError("参照runは一意に対応する必要があります")
    reference = matches[0]
    common = {k: job[k] for k in ("dataset", "interval", "seed", "sample_count")}
    common.update(
        variant="product-default", check=False, record_prediction_period_metrics=True
    )
    with stationary_concept_policy(
        dataset=job["dataset"], concept_id=job["fixed_concept"]
    ):
        baseline = run(common)
    if {k: v for k, v in baseline.items() if k != "seconds"} != {
        k: v for k, v in reference["baseline"].items() if k != "seconds"
    }:
        raise ValueError("既定runの参照不一致")
    with patch.object(holdout, "ReconnectedHoldout", MixtureBenefitObserver):
        arm = holdout.run_holdout_arm(job, window=job["window"], gate=False)
    verify_reference(arm, reference["wait"])
    return dict(
        job=job,
        baseline=baseline,
        wait=arm,
        reference_verified=True,
        extra_model_forward_count=0,
    )


def analyze(output):
    from tools.research_shadow_learner_diagnostic.summarize_standard_metrics import (
        total_mac,
    )

    metadata = json.loads((output / "research-source.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    if (
        Counter(json.dumps(r["job"], sort_keys=True) for r in rows)
        != Counter(json.dumps(j, sort_keys=True) for j in metadata["jobs"])
        or (output / "failures.json").exists()
    ):
        raise ValueError("run欠落・重複・失敗があります")
    for name, digest in metadata["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"実行ソース不一致: {name}")
    groups = defaultdict(list)
    for row in rows:
        if not row["reference_verified"] or row["extra_model_forward_count"] != 0:
            raise ValueError("非介入の証拠がありません")
        for record in row["wait"]["holdouts"]:
            samples, old = record["mixture_samples"], record["samples"]
            if [s["sample_index"] for s in samples] != [s["sample_index"] for s in old]:
                raise ValueError("比較の標本が一致しません")
            score = sum(s["gains"]["current"] for s in samples)
            if score != record["score"]:
                raise ValueError("現行差の累積値が一致しません")
        groups[tuple(row["job"][k] for k in ("dataset", "stream", "interval"))].append(
            row
        )
    summary = []
    lines = [
        "# 現行単体差と混合への追加利得",
        "",
        "追加forward=0。非ゲートwait30の予測・学習・採用は参照と秒以外一致。alpha=0.1が主解析。",
        "",
        "| データ | 入力 | 間隔 | 構成 | 精度% | 既定差pp | 上りMB | 下りMB | G MAC | 秒 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, rs in sorted(groups.items()):
        g = dict(dataset=key[0], stream=key[1], interval=key[2], metrics=[], seeds=[])
        for variant in ("baseline", "wait"):
            ms = [
                r["baseline"] if variant == "baseline" else r["wait"]["measured"]
                for r in rs
            ]
            m = dict(
                variant=variant,
                accuracy=mean(x["accuracy"] for x in ms),
                delta_pp=100
                * mean(
                    x["accuracy"] - r["baseline"]["accuracy"] for x, r in zip(ms, rs)
                ),
                uploaded_mb=mean(x["uploaded_bytes"] for x in ms) / 1e6,
                downloaded_mb=mean(x["downloaded_bytes"] for x in ms) / 1e6,
                gmac=mean(total_mac(x["model_computation_counts"]) for x in ms) / 1e9,
                seconds=mean(x["seconds"] for x in ms),
            )
            g["metrics"].append(m)
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {variant} | {100 * m['accuracy']:.3f} | {m['delta_pp']:+.4f} | {m['uploaded_mb']:.3f} | {m['downloaded_mb']:.3f} | {m['gmac']:.3f} | {m['seconds']:.2f} |"
            )
        complete = []
        for row in sorted(rs, key=lambda r: r["job"]["seed"]):
            records = row["wait"]["holdouts"]
            done = [r for r in records if r["end_reason"] == "adopted"]
            quadrants = Counter()
            for r in done:
                scores = {
                    name: sum(s["gains"][name] for s in r["mixture_samples"])
                    for name in (
                        "current",
                        "mixture_alone",
                        "add_0.1",
                        "add_0.01",
                        "add_0.5",
                    )
                }
                quadrants[
                    f"current_{'positive' if scores['current'] > 0 else 'nonpositive'}:add_{'positive' if scores['add_0.1'] > 0 else 'nonpositive'}"
                ] += 1
                complete.append(
                    dict(
                        client=r["client_id"],
                        episode=r["source_episode_id"],
                        seed=row["job"]["seed"],
                        scores=scores,
                        samples=r["mixture_samples"],
                    )
                )
            g["seeds"].append(
                dict(
                    seed=row["job"]["seed"],
                    started=len(records),
                    completed=len(done),
                    outcomes=dict(Counter(r["end_reason"] for r in records)),
                    quadrants=dict(quadrants),
                    partial_samples=sum(
                        len(r["samples"])
                        for r in records
                        if r["end_reason"] != "adopted"
                    ),
                )
            )
        g["completed_windows"] = [
            {k: v for k, v in r.items() if k != "samples"} for r in complete
        ]
        g["quadrants"] = dict(
            sum((Counter(s["quadrants"]) for s in g["seeds"]), Counter())
        )
        g["completed_sample_count"] = sum(len(r["samples"]) for r in complete)
        g["virtual_accuracy_change_counts"] = {
            name: dict(
                helped=sum(
                    not s["statistics"]["mixture"]["correct"]
                    and s["statistics"][name]["correct"]
                    for r in complete
                    for s in r["samples"]
                ),
                harmed=sum(
                    s["statistics"]["mixture"]["correct"]
                    and not s["statistics"][name]["correct"]
                    for r in complete
                    for s in r["samples"]
                ),
            )
            for name in ("add_0.1", "add_0.01", "add_0.5")
        }
        summary.append(g)
    lines.extend(
        [
            "",
            "## 完了窓の四象限",
            "",
            "候補・標本を揃えた記述集計。部分窓は含めない。候補0件は対象なし。",
            "",
            "| データ | 入力 | 間隔 | 開始/完了 | 現行+/追加+ | 現行+/追加≤0 | 現行≤0/追加+ | 現行≤0/追加≤0 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for g in summary:
        q = g["quadrants"]
        cells = [
            q.get(f"current_{a}:add_{b}", 0)
            for a, b in (
                ("positive", "positive"),
                ("positive", "nonpositive"),
                ("nonpositive", "positive"),
                ("nonpositive", "nonpositive"),
            )
        ]
        lines.append(
            f"| {g['dataset']} | {g['stream']} | {g['interval']} | {sum(s['started'] for s in g['seeds'])}/{sum(s['completed'] for s in g['seeds'])} | "
            + " | ".join(map(str, cells))
            + " |"
        )
    (output / "analysis_mixture.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_mixture.json").write_text(
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
    p.add_argument("--reference-result", type=Path, required=True)
    p.add_argument(
        "--datasets", nargs="+", choices=["sea4", "sine2"], default=["sea4", "sine2"]
    )
    p.add_argument("--intervals", nargs="+", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--sample-count", type=int, default=2500)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--print-plan", action="store_true")
    a = p.parse_args()
    if (
        min(a.intervals + [a.sample_count, a.workers]) < 1
        or min(a.seeds) < 0
        or any(a.sample_count % i for i in a.intervals)
    ):
        p.error("正の値・非負seed・間隔の倍数の標本数が必要です")
    jobs = [
        dict(
            dataset=d,
            stream="drift" if c is None else f"fixed{c}",
            fixed_concept=c,
            interval=i,
            seed=s,
            sample_count=a.sample_count,
            horizon_samples=240,
            window=30,
            task="adoption",
            disabled=False,
            output=str(a.output),
            reference_result=str(a.reference_result),
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
    ref_meta = json.loads(
        (a.reference_result / "research-source.json").read_text(encoding="utf-8")
    )
    for name, h in ref_meta["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != h:
            p.error(f"参照ソース不一致: {name}")
    if a.print_plan:
        print(
            json.dumps(
                dict(
                    job_count=len(jobs),
                    run_count=2 * len(jobs),
                    workers=a.workers,
                    output_exists=a.output.exists(),
                    jobs=jobs,
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
        + [Path(holdout.__file__), Path(__file__)]
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
                reference_input_sha256={
                    n: hashlib.sha256((a.reference_result / n).read_bytes()).hexdigest()
                    for n in ("results.json", "research-source.json")
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    rows, errors = [], []
    with ProcessPoolExecutor(
        max_workers=a.workers, mp_context=get_context("spawn")
    ) as pool:
        futures = {pool.submit(execute_job, j): j for j in jobs}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception:
                errors.append(
                    dict(job=futures[future], traceback=traceback.format_exc())
                )
                (a.output / "failures.json").write_text(
                    json.dumps(errors, indent=2), encoding="utf-8"
                )
                continue
            temp = a.output / "results.json.tmp"
            temp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            temp.replace(a.output / "results.json")
            print(
                futures[future]["dataset"],
                futures[future]["stream"],
                futures[future]["seed"],
                "complete",
                flush=True,
            )
    if errors:
        raise SystemExit(1)
    analyze(a.output)


if __name__ == "__main__":
    main()
