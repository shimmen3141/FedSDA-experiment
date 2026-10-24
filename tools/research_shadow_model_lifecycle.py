"""採用モデルのID来歴と予測係数を読み取る。追加の推論・モデル更新を行わない。"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


class LifecycleTracker:
    """負IDはclient内、正式IDはrun全体で追う。統合後は来歴の追跡であり寄与率ではない。"""

    def __init__(self):
        self.records = []
        self.local = {}
        self.global_ids = {}
        self.positions = {}
        self.frames = []
        self.round_index = None
        self.event_index = 0
        self.owners = {}

    def event(self, token, kind, **values):
        self.event_index += 1
        r = self.records[token]
        event = dict(
            event_index=self.event_index,
            kind=kind,
            round_index=self.round_index,
            origin_sample_index=self.positions.get(r["client_id"]),
            **values,
        )
        r["events"].append(event)
        return event

    def adopted(self, state, arguments):
        e = state["episode"]
        client = arguments["client_id"]
        model = e["adopted_model_id"]
        token = len(self.records)
        if model >= 0 or (client, model) in self.local:
            raise ValueError("採用モデルIDのscopeが不正です")
        self.records.append(
            dict(
                token=token,
                client_id=client,
                temporary_model_id=model,
                episode_id=e["episode_id"],
                sample_index=e["adoption_sample_index"],
                registered_model_id=None,
                first_merge=None,
                usage={},
                events=[],
            )
        )
        self.local[(client, model)] = {token}
        self.owners[client] = (
            arguments["held_model_training_state_registry"],
            arguments["pending_model_upload_state"],
        )
        self.event(token, "adopted", model_id=model)

    def registered(self, client, previous, current):
        tokens = self.local.pop((client, previous), set())
        self.global_ids.setdefault(current, set()).update(tokens)
        for token in sorted(tokens):
            self.records[token]["registered_model_id"] = current
            self.event(
                token, "registered", previous_model_id=previous, model_id=current
            )

    def consolidated(self, clusters, mapping):
        before = {k: set(v) for k, v in self.global_ids.items()}
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            representative = mapping[cluster[0]]
            for model in cluster:
                for token in sorted(before.get(model, ())):
                    event = self.event(
                        token,
                        "consolidated",
                        previous_model_id=model,
                        model_id=representative,
                        cluster=list(cluster),
                        absorbed=model != representative,
                    )
                    if self.records[token]["first_merge"] is None:
                        self.records[token]["first_merge"] = event["event_index"]
        after = {}
        for model, tokens in before.items():
            after.setdefault(mapping.get(model, model), set()).update(tokens)
        self.global_ids = after

    def predicted(self, client, sample, result, branch_weights):
        for model, weight in result.prediction_weights_by_model_id.items():
            tokens = (
                self.local.get((client, model), ())
                if model < 0
                else self.global_ids.get(model, ())
            )
            for token in sorted(tokens):
                record = self.records[token]
                phase = "direct" if record["first_merge"] is None else "descendant"
                scope = "temporary" if model < 0 else "global"
                key = f"{phase}:{scope}:{client}"
                usage = record["usage"].setdefault(
                    key,
                    dict(
                        phase=phase,
                        scope=scope,
                        client_id=client,
                        evaluated_samples=0,
                        positive_weight_samples=0,
                        weight_sum=0.0,
                        synchronized_weight_sum=0.0,
                        local_weight_sum=0.0,
                        maximum_weight_samples=0,
                        first_sample_index=sample,
                        last_sample_index=sample,
                    ),
                )
                usage["evaluated_samples"] += 1
                usage["positive_weight_samples"] += int(weight > 0)
                usage["weight_sum"] += weight
                usage["synchronized_weight_sum"] += weight * branch_weights[0]
                usage["local_weight_sum"] += weight * branch_weights[1]
                usage["maximum_weight_samples"] += int(
                    result.sample_prediction_record.maximum_weight_model_id == model
                )
                usage["last_sample_index"] = sample

    def distributed(self, run_clients, applications):
        for client, application in zip(run_clients, applications):
            for model in application.held_model_ids:
                if model >= 0:
                    for token in sorted(self.global_ids.get(model, ())):
                        self.event(
                            token,
                            "distributed",
                            receiving_client_id=client.client_id,
                            model_id=model,
                            phase="direct"
                            if self.records[token]["first_merge"] is None
                            else "descendant",
                        )
            self.remove_missing_temporary(
                client.client_id, set(application.held_model_ids)
            )

    def remove_missing_temporary(self, client, held):
        for key in list(self.local):
            if key[0] == client and key[1] not in held:
                for token in sorted(self.local.pop(key)):
                    self.event(token, "temporary_removed", model_id=key[1])

    def finish(self):
        for r in self.records:
            registry, pending = self.owners[r["client_id"]]
            r["end_origin_held_ids"] = [
                s.model_id
                for s in registry.snapshot_ordered_held_model_training_states()
            ]
            r["end_origin_pending_ids"] = [
                s.model_id for s in pending.get_all_pending_model_uploads()
            ]
            r["end_global_alias_ids"] = [
                model
                for model, tokens in self.global_ids.items()
                if r["token"] in tokens
            ]

    @contextmanager
    def activated(self):
        from contextlib import ExitStack
        import tools.research_shadow_learner_diagnostic.adoption as adoption
        import federated_learning_experiments.runtime.fedsda_run_client as client
        import federated_learning_experiments.runtime.fedsda_run_server as server
        import federated_learning_experiments.runtime.observed_sample_processing as processing
        import federated_learning_experiments.runtime.server_model_registration_and_aggregation as registration
        import federated_learning_experiments.runtime.server_round_synchronization as sync

        original_adopt = adoption.adopt_shadow
        original_process = client.process_observed_sample
        original_predict = (
            processing.predict_observed_sample_and_update_prediction_weights
        )
        original_sync = server.synchronize_models_in_server_round
        original_confirm = registration.confirm_held_model_registration
        original_aggregate = sync.aggregate_client_models_into_global_models
        original_cluster = sync.cluster_and_consolidate_global_models
        original_distribute = sync.distribute_global_models_to_clients

        def adopt(**kw):
            result = original_adopt(**kw)
            self.adopted(kw["state"], kw["arguments"])
            return result

        def process(**kw):
            self.frames.append(kw)
            self.positions[kw["client_id"]] = kw["indexed_observation"].sample_index
            try:
                result = original_process(**kw)
                if self.records:
                    held = {
                        s.model_id
                        for s in kw[
                            "held_model_training_state_registry"
                        ].snapshot_ordered_held_model_training_states()
                    }
                    self.remove_missing_temporary(kw["client_id"], held)
                return result
            finally:
                self.frames.pop()

        def predict(**kw):
            mixture = kw.get("synchronized_and_local_prediction_mixture")
            branches = (
                mixture.snapshot_weights()
                if mixture is not None
                else (
                    (1.0, 0.0)
                    if kw.get("synchronized_prediction_model_snapshot") is not None
                    else (0.0, 1.0)
                )
            )
            result = original_predict(**kw)
            if self.records:
                self.predicted(
                    self.frames[-1]["client_id"],
                    kw["indexed_observation"].sample_index,
                    result,
                    branches,
                )
            return result

        def synchronize(**kw):
            self.round_index = kw["round_index"]
            try:
                return original_sync(**kw)
            finally:
                self.round_index = None

        def confirm(**kw):
            registry = kw["held_model_training_state_registry"]
            before = {
                s.model_id
                for s in registry.snapshot_ordered_held_model_training_states()
            }
            result = original_confirm(**kw)
            after = {
                s.model_id
                for s in registry.snapshot_ordered_held_model_training_states()
            }
            removed = before - after
            if len(removed) > 1:
                raise ValueError("登録時のID対応が一意ではありません")
            if removed:
                # 登録処理はclientを順に回す。ownerのidentityで対応するclientを見つける。
                matches = [
                    c for c, (owner, _) in self.owners.items() if owner is registry
                ]
                if matches:
                    if len(matches) != 1:
                        raise ValueError("client ownerが一意ではありません")
                    self.registered(
                        matches[0],
                        next(iter(removed)),
                        kw["registered_global_model_id"],
                    )
            return result

        def aggregate(**kw):
            uploads = []
            for c in kw["run_clients"]:
                held = {
                    s.model_id
                    for s in c.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
                }
                for samples in c.owners.training_sample_store.snapshot_ordered_model_training_samples():
                    if (
                        samples.model_id in held
                        and self.global_ids.get(samples.model_id)
                        and samples.training_samples
                    ):
                        uploads.append(
                            (
                                c.client_id,
                                samples.model_id,
                                len(samples.training_samples),
                            )
                        )
            result = original_aggregate(**kw)
            counts = Counter()
            for c, model, n in uploads:
                counts[model] += n
                for token in sorted(self.global_ids[model]):
                    self.event(
                        token,
                        "uploaded_update",
                        sending_client_id=c,
                        model_id=model,
                        training_samples=n,
                    )
            if any(
                result.aggregated_training_sample_counts_by_model_id[m] != n
                for m, n in counts.items()
            ):
                raise ValueError("送信対象と集約計数が一致しません")
            return result

        def cluster(**kw):
            result = original_cluster(**kw)
            self.consolidated(result.model_clusters, result.model_id_mapping)
            return result

        def distribute(**kw):
            result = original_distribute(**kw)
            self.distributed(kw["run_clients"], result)
            return result

        with ExitStack() as stack:
            for module, name, wrapper in (
                (adoption, "adopt_shadow", adopt),
                (client, "process_observed_sample", process),
                (
                    processing,
                    "predict_observed_sample_and_update_prediction_weights",
                    predict,
                ),
                (server, "synchronize_models_in_server_round", synchronize),
                (registration, "confirm_held_model_registration", confirm),
                (sync, "aggregate_client_models_into_global_models", aggregate),
                (sync, "cluster_and_consolidate_global_models", cluster),
                (sync, "distribute_global_models_to_clients", distribute),
            ):
                stack.enter_context(
                    patch.object(module, name, wraps(getattr(module, name))(wrapper))
                )
            yield
        self.finish()


def execute_tracking_job(job):
    from tools.research_shadow_learner_diagnostic.run_followup import execute_job

    if job["task"] != "adoption":
        raise ValueError("採用実験が必要です")
    tracker = LifecycleTracker()
    with tracker.activated():
        result = execute_job(job)
    result["model_lifecycle"] = tracker.records
    return result


def analyze(output):
    import math
    from tools.research_shadow_learner_diagnostic.run_followup import (
        analyze as standard_analysis,
    )

    meta = json.loads((output / "research-source.json").read_text(encoding="utf-8"))
    for name, h in meta["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != h:
            raise ValueError(f"実行ソースが異なります: {name}")
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    if (
        Counter(json.dumps(r["job"], sort_keys=True) for r in rows)
        != Counter(json.dumps(j, sort_keys=True) for j in meta["jobs"])
        or (output / "failures.json").exists()
    ):
        raise ValueError("runの欠落・失敗があります")
    reference_path = meta["model_lifecycle"]["reference_result"]
    reference = {}
    if reference_path:
        refs = json.loads(
            (Path(reference_path) / "results.json").read_text(encoding="utf-8")
        )
        for r in refs:
            key = tuple(r["job"][k] for k in ("dataset", "stream", "interval", "seed"))
            if key in reference:
                raise ValueError("参照runが重複しています")
            reference[key] = r
    groups = defaultdict(list)
    for row in rows:
        job = row["job"]
        key = tuple(job[k] for k in ("dataset", "stream", "interval", "seed"))
        if reference_path:
            original = reference[key]
            if any(job[k] != original["job"][k] for k in job if k != "output"):
                raise ValueError("参照のrun条件が違います")
            for part in ("baseline", "measured"):
                if {k: v for k, v in row[part].items() if k != "seconds"} != {
                    k: v for k, v in original[part].items() if k != "seconds"
                }:
                    raise ValueError("計測なしと全保存指標が一致しません")
            if row["shadow_diagnostic"] != original["shadow_diagnostic"]:
                raise ValueError("影・採用経路が変わりました")
        records = row["model_lifecycle"]
        expected = {
            (
                e["client_id"],
                e["adoption_sample_index"],
                e["adopted_model_id"],
                e["episode_id"],
            )
            for e in row["shadow_diagnostic"]["episodes"]
            if "adopted_model_id" in e
        }
        actual = [
            (
                r["client_id"],
                r["sample_index"],
                r["temporary_model_id"],
                r["episode_id"],
            )
            for r in records
        ]
        if set(actual) != expected or len(actual) != len(set(actual)):
            raise ValueError("採用と追跡の対応が不正です")
        for r in records:
            for u in r["usage"].values():
                if not 0 <= u["positive_weight_samples"] <= u[
                    "evaluated_samples"
                ] or not math.isclose(
                    u["weight_sum"],
                    u["synchronized_weight_sum"] + u["local_weight_sum"],
                    abs_tol=1e-8,
                ):
                    raise ValueError("予測係数の集計が不正です")
        groups[key[:3]].extend(records)
    standard_analysis(output)
    lines = [
        "# 採用モデルの来歴と予測利用",
        "",
        "直接=最初の統合より前（代表として残る場合も統合と数える）。統合後の利用は代表IDの来歴であり、元の採用モデルの予測寄与率ではない。",
        "標本数は採用モデル×client×標本の延べ数。同じ代表へ統合された複数の来歴は同じ予測をそれぞれ数えるため、uniqueな予測数として合算しない。",
        "",
        "| データ | 入力 | 間隔 | 採用 | 登録 | 統合あり | 統合前global正重み予測0件 | 直接の正重み標本数 | 統合後代表の正重み標本数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = []
    for key, rs in sorted(groups.items()):

        def direct(r):
            return sum(
                u["positive_weight_samples"]
                for u in r["usage"].values()
                if u["phase"] == "direct"
            )

        def global_direct(r):
            return sum(
                u["positive_weight_samples"]
                for u in r["usage"].values()
                if u["phase"] == "direct" and u["scope"] == "global"
            )

        g = dict(
            dataset=key[0],
            stream=key[1],
            interval=key[2],
            adoptions=len(rs),
            registered=sum(r["registered_model_id"] is not None for r in rs),
            merged=sum(r["first_merge"] is not None for r in rs),
            merged_without_global_use=sum(
                r["first_merge"] is not None and global_direct(r) == 0 for r in rs
            ),
            direct_samples=sum(direct(r) for r in rs),
            descendant_samples=sum(
                u["positive_weight_samples"]
                for r in rs
                for u in r["usage"].values()
                if u["phase"] == "descendant"
            ),
        )
        for phase in ("direct", "descendant"):
            usages = [u for r in rs for u in r["usage"].values() if u["phase"] == phase]
            count = sum(u["evaluated_samples"] for u in usages)
            weight_sum = sum(u["weight_sum"] for u in usages)
            g[phase + "_evaluated_samples"] = count
            g[phase + "_weight_sum"] = weight_sum
            g[phase + "_mean_weight"] = weight_sum / count if count else None
        summary.append(g)
        lines.append(
            "| "
            + " | ".join(
                str(g[k])
                for k in (
                    "dataset",
                    "stream",
                    "interval",
                    "adoptions",
                    "registered",
                    "merged",
                    "merged_without_global_use",
                    "direct_samples",
                    "descendant_samples",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 混合予測内の平均係数",
            "",
            "各phaseの係数の和÷評価された延べ標本数。統合後は代表モデルの係数であり、元モデルへ帰属する寄与率ではない。",
            "",
            "| データ | 入力 | 間隔 | 統合前の平均係数 | 統合後代表の平均係数 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for g in summary:
        values = [
            f"{g[p + '_mean_weight']:.6f}"
            if g[p + "_mean_weight"] is not None
            else "対象無し"
            for p in ("direct", "descendant")
        ]
        lines.append(
            f"| {g['dataset']} | {g['stream']} | {g['interval']} | {values[0]} | {values[1]} |"
        )
    (output / "analysis_lifecycle.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_lifecycle.json").write_text(
        json.dumps(
            dict(
                groups=summary,
                reference_pairs_verified=len(rows) if reference_path else None,
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


if __name__ == "__main__":
    from tools.research_shadow_learner_diagnostic import run_followup

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--task", choices=["adoption"], default="adoption")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if "--help" in sys.argv or "--output" not in sys.argv:
        run_followup.main()
        raise SystemExit()
    sys.argv.extend(["--task", "adoption"])
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    planned = "--print-plan" in sys.argv
    existed = (output / "research-source.json").exists()
    source = Path(__file__).read_bytes()
    try:
        with patch.object(run_followup, "execute_job", execute_tracking_job):
            run_followup.main()
    finally:
        path = output / "research-source.json"
        if not planned and not existed and path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta["model_lifecycle"] = dict(
                reference_result=str(args.reference_result)
                if args.reference_result
                else None,
                extra_model_forward_count=0,
            )
            meta["sha256"][Path(__file__).resolve().relative_to(ROOT).as_posix()] = (
                hashlib.sha256(source).hexdigest()
            )
            path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if not planned:
        analyze(output)
