"""接続前後の影と採用直前の現行モデルを、固定コピーで非介入比較する研究CLI。"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from unittest.mock import patch

from tools import research_shadow_feature_reconnection as reconnection

ROOT = reconnection.ROOT


class CurrentComparisonProbe(reconnection.ReconnectionProbe):
    """共有部接続の二つのコピーに、同じ時点の現行モデルのコピーを追加する。"""

    def __init__(self, future_samples):
        super().__init__(future_samples)
        self.current_snapshots = {}

    def measure_current(self, classifier, features, labels):
        import torch
        from federated_learning_experiments.learning.models.model_computation_measurement import (
            measure_model_computation,
        )
        from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
            evaluate_classifier_per_sample_log_losses,
        )

        with torch.random.fork_rng(devices=[]), measure_model_computation() as meter:
            losses = evaluate_classifier_per_sample_log_losses(
                classifier=classifier,
                input_features=features,
                observed_class_labels=labels,
            ).tolist()
        self.counts.update(asdict(meter.get_model_computation_counts()))
        return losses

    def adopt(self, original, *, state, arguments):
        import torch

        start = perf_counter()
        a = arguments
        current_id = a["current_training_model_assignment"].current_training_model_id
        current = deepcopy(
            a["held_model_training_state_registry"]
            .get_held_model_training_state(model_id=current_id)
            .classifier
        )
        observations = state["observations"]
        losses = self.measure_current(
            current,
            torch.cat([o.training_sample.input_features for o in observations]),
            torch.cat([o.training_sample.observed_class_labels for o in observations]),
        )
        self.seconds += perf_counter() - start
        # 同じ時点の現行コピーを採用前に固定する。既存の実採用はsuperで一度だけ行う。
        super().adopt(original, state=state, arguments=arguments)
        record = self.records[-1]
        record["current_model_id"] = current_id
        record["current_losses"] = losses
        if self.future_samples:
            self.current_snapshots[a["client_id"]] = current

    def observe(self, arguments):
        client = arguments["client_id"]
        if client not in self.active:
            return
        record = self.active[client][2]
        o = arguments["indexed_observation"]
        if o.sample_index != record["sample_index"] + len(record["future"]) + 1:
            raise ValueError("現行コピーの将来観測順序が不正です")
        start = perf_counter()
        loss = self.measure_current(
            self.current_snapshots[client],
            o.training_sample.input_features,
            o.training_sample.observed_class_labels,
        )[0]
        self.seconds += perf_counter() - start
        super().observe(arguments)
        record["future"][-1]["current_loss"] = loss
        if client not in self.active:
            self.current_snapshots.pop(client)

    @contextmanager
    def activated(self):
        try:
            with super().activated():
                yield
        finally:
            self.current_snapshots.clear()


def execute_comparison_job(job, *, future_samples=30):
    # 既存の費用差引きと全保存形式を再利用し、古い計測ツール自体は変更しない。
    with patch.object(reconnection, "ReconnectionProbe", CurrentComparisonProbe):
        return reconnection.execute_probe_job(job, future_samples=future_samples)


def comparison_values(record):
    """同じ採用・同じ標本の平均差。負は影の方が良く、0は同値。"""
    n = record["recent_sample_count"]
    current = mean(record["current_losses"][-n:])
    frozen = mean(record["frozen_losses"][-n:]) - current
    connected = mean(record["connected_losses"][-n:]) - current
    future = record["future"]
    return dict(
        recent_frozen_minus_current=frozen,
        recent_connected_minus_current=connected,
        recent_advantage_lost=frozen < -1e-10 and connected >= -1e-10,
        future_frozen_minus_current=mean(
            t["frozen_loss"] - t["current_loss"] for t in future
        )
        if future
        else None,
        future_connected_minus_current=mean(
            t["connected_loss"] - t["current_loss"] for t in future
        )
        if future
        else None,
    )


def analyze_comparison(output):
    metadata = json.loads((output / "research-source.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    reference = metadata["feature_reconnection_probe"]["reference_result"]
    if reference:
        refs = json.loads(
            (Path(reference) / "results.json").read_text(encoding="utf-8")
        )
        keys = [
            tuple(r["job"][k] for k in ("dataset", "stream", "interval", "seed"))
            for r in refs
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("参照runが重複しています")
    groups = defaultdict(list)
    details = []
    for row in rows:
        job = row["job"]
        key = tuple(job[k] for k in ("dataset", "stream", "interval"))
        groups.setdefault(key, [])
        for record in row["feature_reconnection_probe"]["records"]:
            losses = record["current_losses"]
            if len(losses) != len(record["frozen_losses"]) or not all(
                math.isfinite(v) for v in losses
            ):
                raise ValueError("現行と影の同一標本の対応が不正です")
            if not all(math.isfinite(t["current_loss"]) for t in record["future"]):
                raise ValueError("現行コピーの将来損失が不正です")
            values = comparison_values(record)
            groups[key].append(values)
            details.append(
                dict(
                    **{k: job[k] for k in ("dataset", "stream", "interval", "seed")},
                    client_id=record["client_id"],
                    episode_id=record["episode_id"],
                    sample_index=record["sample_index"],
                    **values,
                )
            )
    # 全ソース・job・採用対応・将来順序・参照との全保存値/採用経路を既存の検査で照合。
    reconnection.analyze_probe(output)
    lines = [
        "# 接続した影と現行モデルの固定コピー比較",
        "",
        "差=影−現行（nats/標本）。負は影の方が良い。採用単位平均、seed均等平均ではない。",
        "直近は学習済み最大30標本。将来は三つとも採用直前に固定したコピーで、実際の混合予測や学習継続の精度ではない。採用条件は変えていない。",
        "",
        "| データ | 入力 | 間隔 | 採用 | 直近 接続前−現行 | 直近 接続後−現行 | 直近 接続後に影が良い件数 | 接続で直近優位が消えた件数 | 将来 接続後−現行 | 将来 接続後に影が良い件数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = []
    for key, values in sorted(groups.items()):
        g = dict(dataset=key[0], stream=key[1], interval=key[2], adoptions=len(values))
        for field in (
            "recent_frozen_minus_current",
            "recent_connected_minus_current",
            "future_frozen_minus_current",
            "future_connected_minus_current",
        ):
            vs = [v[field] for v in values if v[field] is not None]
            g[field] = dict(
                mean=mean(vs) if vs else None,
                better=sum(x < -1e-10 for x in vs),
                equal=sum(abs(x) <= 1e-10 for x in vs),
                worse=sum(x > 1e-10 for x in vs),
                count=len(vs),
            )
        g["recent_advantage_lost"] = sum(v["recent_advantage_lost"] for v in values)
        summary.append(g)

        def fmt(field):
            v = g[field]["mean"]
            return f"{v:+.6f}" if v is not None else "対象無し"

        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(values)} | {fmt('recent_frozen_minus_current')} | {fmt('recent_connected_minus_current')} | {g['recent_connected_minus_current']['better']} | {g['recent_advantage_lost']} | {fmt('future_connected_minus_current')} | {g['future_connected_minus_current']['better']} |"
        )
    (output / "analysis_current_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_current_comparison.json").write_text(
        json.dumps(
            dict(
                groups=summary,
                episodes=details,
                input_sha256={
                    n: hashlib.sha256((output / n).read_bytes()).hexdigest()
                    for n in ("research-source.json", "results.json")
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


def main():
    from tools.research_shadow_learner_diagnostic import run_followup

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-future-samples", type=int, default=30)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--task", choices=["adoption"], default="adoption")
    args, remaining = parser.parse_known_args()
    if args.probe_future_samples < 0:
        parser.error("将来標本数は非負が必要です")
    sys.argv = [sys.argv[0], *remaining]
    if "--help" in sys.argv or "--output" not in sys.argv:
        run_followup.main()
        return
    sys.argv.extend(["--task", "adoption"])
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    planned = "--print-plan" in sys.argv
    existed = (output / "research-source.json").exists()
    sources = {p: p.read_bytes() for p in (Path(__file__), Path(reconnection.__file__))}
    try:
        with patch.object(
            run_followup,
            "execute_job",
            partial(execute_comparison_job, future_samples=args.probe_future_samples),
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
            meta["current_comparison"] = dict(
                current_snapshot="before_adoption", adoption_policy_changed=False
            )
            for p, source in sources.items():
                meta["sha256"][p.resolve().relative_to(ROOT).as_posix()] = (
                    hashlib.sha256(source).hexdigest()
                )
            path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if not planned:
        analyze_comparison(output)


if __name__ == "__main__":
    main()
