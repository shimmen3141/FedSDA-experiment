"""影の採用時の共有部接続をコピーで測定し、採用方式そのものは変えない研究CLI。"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from functools import partial
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


class ReconnectionProbe:
    """接続時点で固定した二つのモデルを、過去と将来の同じ標本で比較する。"""

    def __init__(self, future_samples):
        if type(future_samples) is not int or future_samples < 0:
            raise ValueError("将来の観測数は非負整数が必要です")
        self.future_samples = future_samples
        self.records = []
        self.active = {}
        self.counts = Counter()
        self.seconds = 0.0

    def measure_pair(self, frozen, connected, features, labels):
        import torch
        from federated_learning_experiments.learning.models.model_computation_measurement import (
            measure_model_computation,
        )
        from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
            evaluate_classifier_per_sample_log_losses,
        )

        with torch.random.fork_rng(devices=[]), measure_model_computation() as meter:
            losses = [
                evaluate_classifier_per_sample_log_losses(
                    classifier=model,
                    input_features=features,
                    observed_class_labels=labels,
                ).tolist()
                for model in (frozen, connected)
            ]
        self.counts.update(asdict(meter.get_model_computation_counts()))
        return losses

    def adopt(self, original, *, state, arguments):
        import torch

        start = perf_counter()
        a = arguments
        current_id = a["current_training_model_assignment"].current_training_model_id
        current = (
            a["held_model_training_state_registry"]
            .get_held_model_training_state(model_id=current_id)
            .classifier
        )
        frozen = deepcopy(state["learner"])
        connected = deepcopy(frozen)
        connected.attach_shared_feature_extractor(
            shared_feature_extractor=deepcopy(current.feature_extractor)
        )
        old = frozen.feature_extractor.state_dict()
        new = connected.feature_extractor.state_dict()
        difference = sum(
            float(((old[n].double() - new[n].double()) ** 2).sum()) for n in old
        )
        magnitude = sum(float((v.double() ** 2).sum()) for v in old.values())
        observations = state["observations"]
        losses = self.measure_pair(
            frozen,
            connected,
            torch.cat([o.training_sample.input_features for o in observations]),
            torch.cat([o.training_sample.observed_class_labels for o in observations]),
        )
        record = dict(
            episode_id=state["episode"]["episode_id"],
            client_id=a["client_id"],
            sample_index=a["indexed_observation"].sample_index,
            concept_id=a["indexed_observation"].observed_concept_id,
            shared_squared_parameter_difference=difference,
            shared_relative_parameter_difference=sqrt(
                difference / max(magnitude, 1e-24)
            ),
            frozen_losses=losses[0],
            connected_losses=losses[1],
            recent_sample_count=min(30, len(observations)),
            future=[],
            end_reason=None,
        )
        self.seconds += perf_counter() - start
        # コピーの計測を済ませてから、変更していない既存の採用処理を実行する。
        original(state=state, arguments=arguments)
        record["adopted_model_id"] = state["episode"]["adopted_model_id"]
        previous = self.active.pop(a["client_id"], None)
        if previous is not None:
            previous[2]["end_reason"] = "next_adoption"
        self.records.append(record)
        if self.future_samples:
            self.active[a["client_id"]] = (frozen, connected, record)
        else:
            record["end_reason"] = "disabled"

    def observe(self, arguments):
        pair = self.active.get(arguments["client_id"])
        if pair is None:
            return
        start = perf_counter()
        frozen, connected, record = pair
        o = arguments["indexed_observation"]
        if o.sample_index != record["sample_index"] + len(record["future"]) + 1:
            raise ValueError("将来標本の順序が一致しません")
        losses = self.measure_pair(
            frozen,
            connected,
            o.training_sample.input_features,
            o.training_sample.observed_class_labels,
        )
        record["future"].append(
            dict(
                sample_index=o.sample_index,
                concept_id=o.observed_concept_id,
                frozen_loss=losses[0][0],
                connected_loss=losses[1][0],
            )
        )
        if len(record["future"]) == self.future_samples:
            record["end_reason"] = "horizon"
            self.active.pop(arguments["client_id"])
        self.seconds += perf_counter() - start

    @contextmanager
    def activated(self):
        import tools.research_shadow_learner_diagnostic.adoption as adoption
        import federated_learning_experiments.runtime.fedsda_run_client as client

        original_adopt, original_process = (
            adoption.adopt_shadow,
            client.process_observed_sample,
        )

        def adopt(**kwargs):
            self.adopt(original_adopt, **kwargs)

        def process(**kwargs):
            self.observe(kwargs)
            return original_process(**kwargs)

        try:
            with (
                patch.object(adoption, "adopt_shadow", adopt),
                patch.object(client, "process_observed_sample", process),
            ):
                yield
        finally:
            for _, _, record in self.active.values():
                record["end_reason"] = "run_end"
            self.active.clear()


def execute_probe_job(job, *, future_samples=30):
    from tools.research_shadow_learner_diagnostic.run_followup import execute_job

    if job["task"] != "adoption":
        raise ValueError("採用実験だけが対象です")
    probe = ReconnectionProbe(future_samples)
    with probe.activated():
        result = execute_job(job)
    measured = result["measured"]
    inclusive = measured["model_computation_counts"]
    adjusted = {k: v - probe.counts.get(k, 0) for k, v in inclusive.items()}
    if min(adjusted.values()) < 0 or any(
        v for k, v in probe.counts.items() if "training" in k or "optimizer_step" in k
    ):
        raise ValueError("計測費用の分離、または非学習の条件に違反しました")
    measured["model_computation_counts"] = adjusted
    result["feature_reconnection_probe"] = dict(
        records=probe.records,
        model_computation_counts=dict(probe.counts),
        probe_inclusive_model_computation_counts=inclusive,
        seconds=probe.seconds,
        future_samples=future_samples,
    )
    return result


def analyze_probe(output):
    import math
    from tools.research_shadow_learner_diagnostic.run_followup import analyze

    meta = json.loads((output / "research-source.json").read_text(encoding="utf-8"))
    for name, expected in meta["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"実行ソースが異なります: {name}")
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    if (
        Counter(json.dumps(r["job"], sort_keys=True) for r in rows)
        != Counter(json.dumps(j, sort_keys=True) for j in meta["jobs"])
        or (output / "failures.json").exists()
    ):
        raise ValueError("実験が未完了または不一致です")
    groups = defaultdict(list)
    reference_path = meta["feature_reconnection_probe"].get("reference_result")
    reference = None
    if reference_path:
        reference = {
            tuple(r["job"][k] for k in ("dataset", "stream", "interval", "seed")): r
            for r in json.loads(
                (Path(reference_path) / "results.json").read_text(encoding="utf-8")
            )
        }
    for row in rows:
        key = tuple(row["job"][k] for k in ("dataset", "stream", "interval", "seed"))
        groups.setdefault(key[:3], [])
        if reference is not None:
            original = reference[key]
            if any(
                row["job"][k] != original["job"][k] for k in row["job"] if k != "output"
            ):
                raise ValueError("参照runの条件が一致しません")
            for part in ("baseline", "measured"):
                if {k: v for k, v in row[part].items() if k != "seconds"} != {
                    k: v for k, v in original[part].items() if k != "seconds"
                }:
                    raise ValueError("計測なしの実験経路と一致しません")
            if row["shadow_diagnostic"] != original["shadow_diagnostic"]:
                raise ValueError("影の生成・採用の記録が変わっています")
        probe = row["feature_reconnection_probe"]
        adopted = {
            (e["client_id"], e["adoption_sample_index"], e["episode_id"])
            for e in row["shadow_diagnostic"]["episodes"]
            if "adopted_model_id" in e
        }
        actual = [
            (e["client_id"], e["sample_index"], e["episode_id"])
            for e in probe["records"]
        ]
        if len(actual) != len(set(actual)) or set(actual) != adopted:
            raise ValueError("計測と実際の採用が一致しません")
        for e in probe["records"]:
            n = len(e["frozen_losses"])
            if (
                not n
                or len(e["connected_losses"]) != n
                or e["recent_sample_count"] != min(30, n)
            ):
                raise ValueError("同一標本の対応が不正です")
            if not all(
                math.isfinite(v) for v in e["frozen_losses"] + e["connected_losses"]
            ):
                raise ValueError("非有限の損失です")
            if len(e["future"]) > probe["future_samples"] or e["end_reason"] not in {
                "horizon",
                "next_adoption",
                "run_end",
                "disabled",
            }:
                raise ValueError("将来観測の打切りが不正です")
            for i, t in enumerate(e["future"], 1):
                if t["sample_index"] != e["sample_index"] + i or not all(
                    math.isfinite(t[f]) for f in ("frozen_loss", "connected_loss")
                ):
                    raise ValueError("将来観測の順序が不正です")
            if (
                e["end_reason"] == "horizon"
                and len(e["future"]) != probe["future_samples"]
            ):
                raise ValueError("将来観測の上限が不正です")
            groups[
                tuple(row["job"][k] for k in ("dataset", "stream", "interval"))
            ].append(e)
    analyze(output)
    summary = []
    lines = [
        "# 共有部接続の非介入計測",
        "",
        "正の損失差は接続による悪化。採用単位の平均で、seed均等平均ではない。過去の標本は影が学習済み。将来は接続時点で固定した二つのコピーの比較で、学習を続ける実モデルの精度ではない。",
        "",
        "| データ | 入力 | 間隔 | 採用 | 共有部差あり | 直近損失差 nats/標本 | 悪化/同値/改善 | 将来損失差 nats/標本 | 将来観測あり |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, es in sorted(groups.items()):
        recent = [
            mean(e["connected_losses"][-e["recent_sample_count"] :])
            - mean(e["frozen_losses"][-e["recent_sample_count"] :])
            for e in es
        ]
        future = [
            mean(t["connected_loss"] - t["frozen_loss"] for t in e["future"])
            for e in es
            if e["future"]
        ]
        g = dict(
            dataset=key[0],
            stream=key[1],
            interval=key[2],
            adoptions=len(es),
            shared_changed=sum(
                e["shared_squared_parameter_difference"] > 0 for e in es
            ),
            recent_delta=mean(recent) if recent else None,
            worse=sum(d > 1e-10 for d in recent),
            equal=sum(abs(d) <= 1e-10 for d in recent),
            better=sum(d < -1e-10 for d in recent),
            future_delta=mean(future) if future else None,
            future_count=len(future),
        )
        summary.append(g)
        f = f"{g['future_delta']:+.6f}" if future else "対象無し"
        r = f"{g['recent_delta']:+.6f}" if recent else "対象無し"
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(es)} | {g['shared_changed']} | {r} | {g['worse']}/{g['equal']}/{g['better']} | {f} | {len(future)} |"
        )
    (output / "analysis_reconnection.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_reconnection.json").write_text(
        json.dumps(
            dict(
                groups=summary,
                reference_pairs_verified=len(rows) if reference is not None else None,
                input_sha256={
                    n: hashlib.sha256((output / n).read_bytes()).hexdigest()
                    for n in ("results.json", "research-source.json")
                },
                analysis_source_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    from tools.research_shadow_learner_diagnostic import run_followup

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-future-samples", type=int, default=30)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--task", choices=["adoption"], default="adoption")
    args, remaining = parser.parse_known_args()
    if args.probe_future_samples < 0:
        parser.error("将来の観測数は非負です")
    sys.argv = [sys.argv[0], *remaining]
    if "--help" in sys.argv or "--output" not in sys.argv:
        run_followup.main()
        raise SystemExit()
    if "--task" not in sys.argv:
        sys.argv.extend(["--task", "adoption"])
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    planned = "--print-plan" in sys.argv
    existed = (output / "research-source.json").exists()
    source = Path(__file__).read_bytes()
    try:
        with patch.object(
            run_followup,
            "execute_job",
            partial(execute_probe_job, future_samples=args.probe_future_samples),
        ):
            run_followup.main()
    finally:
        path = output / "research-source.json"
        if not planned and not existed and path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta["feature_reconnection_probe"] = dict(
                future_samples=args.probe_future_samples,
                recent_samples=30,
                measured_mac_excludes_probe=True,
                seconds_include_probe=True,
                reference_result=str(args.reference_result)
                if args.reference_result
                else None,
            )
            meta["sha256"][Path(__file__).resolve().relative_to(ROOT).as_posix()] = (
                hashlib.sha256(source).hexdigest()
            )
            path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if not planned:
        analyze_probe(output)
