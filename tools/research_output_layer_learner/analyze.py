"""標準指標をseed対応で集計し、保存済みの既定結果も来歴照合して利用する。"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def mac(row):
    return sum(
        v
        for k, v in row["model_computation_counts"].items()
        if k.endswith("multiply_accumulate_count")
    )


def analyze(*, inputs, baselines, output):
    rows = []
    evidence = []
    root = Path(__file__).resolve().parents[2]
    for directory, reused in [(p, False) for p in inputs] + [
        (p, True) for p in baselines
    ]:
        metadata = json.loads(
            (
                directory / ("research-source.json" if reused else "metadata.json")
            ).read_text(encoding="utf-8")
        )
        hashes = {
            k.replace("\\", "/"): v
            for k, v in metadata["sha256" if reused else "source_hashes"].items()
        }
        execution_tools = {
            "tools/research_output_layer_learner/" + name
            for name in ("__init__.py", "policy.py", "run.py")
        }
        # 再利用時にも、製品全体・標準runner・固定入力の同一性を確認する。
        relevant = {
            k: v
            for k, v in hashes.items()
            if k.startswith("src/")
            or k
            in (
                "tools/research_provisional_candidate_expert/run.py",
                "tools/research_shadow_learner_diagnostic/controls.py",
            )
            or (not reused and k in execution_tools)
        }
        expected = {
            str(p.relative_to(root)).replace("\\", "/")
            for p in (root / "src/federated_learning_experiments").rglob("*.py")
        }
        expected.update(
            (
                "tools/research_provisional_candidate_expert/run.py",
                "tools/research_shadow_learner_diagnostic/controls.py",
            )
        )
        if not reused:
            expected.update(execution_tools)
        if not expected.issubset(relevant):
            raise ValueError(f"入力ソースのhash不足: {directory}")
        for name, digest in relevant.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"入力ソース相違: {directory}: {name}")
        values = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        if not reused:
            completion = json.loads(
                (directory / "complete.json").read_text(encoding="utf-8")
            )
            if completion.get("source_hashes_verified") is not True:
                raise ValueError("実行終了時のソース照合が未完了です")
            keys = (
                "dataset",
                "fixed_concept",
                "interval",
                "sample_count",
                "seed",
                "research_variant",
            )
            if {tuple(j[k] for k in keys) for j in metadata["jobs"]} != {
                tuple(r[k] for k in keys) for r in values
            }:
                raise ValueError("要求したrun条件と結果が一致しません")
            if len(values) != len(metadata["jobs"]) or completion["run_count"] != len(
                values
            ):
                raise ValueError("run数が不完全です")
        for value in values:
            if reused:
                row = dict(
                    value["baseline"],
                    research_variant="product-default",
                    fixed_concept=value["job"]["fixed_concept"],
                    research_events=[],
                )
            else:
                row = value
            rows.append(row)
        evidence.append(
            dict(
                directory=str(directory),
                reused=reused,
                verified_hash_count=len(relevant),
                result_sha256=hashlib.sha256(
                    (directory / "results.json").read_bytes()
                ).hexdigest(),
            )
        )
    keyed = {}
    for row in rows:
        key = tuple(
            row[k]
            for k in (
                "dataset",
                "fixed_concept",
                "interval",
                "sample_count",
                "seed",
                "research_variant",
            )
        )
        if key in keyed:
            raise ValueError(f"重複したrun: {key}")
        keyed[key] = row
    groups = defaultdict(list)
    pairs = []
    for key, row in keyed.items():
        if row["research_variant"] == "product-default":
            continue
        base = keyed[key[:-1] + ("product-default",)]
        comparison = dict(
            dataset=key[0],
            fixed_concept=key[1],
            interval=key[2],
            sample_count=key[3],
            seed=key[4],
            variant=key[5],
            accuracy_pct=100 * row["accuracy"],
            accuracy_delta_pp=100 * (row["accuracy"] - base["accuracy"]),
            baseline_accuracy_pct=100 * base["accuracy"],
            uploaded_bytes=row["uploaded_bytes"],
            downloaded_bytes=row["downloaded_bytes"],
            mac=mac(row),
            seconds=row["seconds"],
            baseline_seconds=base["seconds"],
            baseline_uploaded_bytes=base["uploaded_bytes"],
            baseline_downloaded_bytes=base["downloaded_bytes"],
            baseline_mac=mac(base),
            matched_candidates=sum(
                e["event"] == "matched_candidate" for e in row["research_events"]
            ),
            warm_scores=sum(e["event"] == "warm_score" for e in row["research_events"]),
            concept_delta_pp={
                k: 100 * (v - base["concept_accuracies"][k])
                for k, v in row["concept_accuracies"].items()
            },
        )
        pairs.append(comparison)
        groups[key[:4] + (key[5],)].append(comparison)
    summaries = []
    for key, values in groups.items():
        if len(values) < 4:
            raise ValueError(f"4 seeds未満: {key}")
        summaries.append(
            dict(
                dataset=key[0],
                fixed_concept=key[1],
                interval=key[2],
                sample_count=key[3],
                variant=key[4],
                n=len(values),
                **{
                    field: mean(r[field] for r in values)
                    for field in (
                        "accuracy_pct",
                        "accuracy_delta_pp",
                        "baseline_accuracy_pct",
                        "uploaded_bytes",
                        "downloaded_bytes",
                        "mac",
                        "seconds",
                        "baseline_seconds",
                        "baseline_uploaded_bytes",
                        "baseline_downloaded_bytes",
                        "baseline_mac",
                        "matched_candidates",
                        "warm_scores",
                    )
                },
                delta_sd=stdev(r["accuracy_delta_pp"] for r in values),
                positive_seeds=sum(r["accuracy_delta_pp"] > 0 for r in values),
                concept_delta_pp={
                    k: mean(r["concept_delta_pp"][k] for r in values)
                    for k in values[0]["concept_delta_pp"]
                },
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    result = dict(
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence=evidence,
        pairs=pairs,
        summaries=summaries,
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--baselines", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(inputs=args.inputs, baselines=args.baselines, output=args.output)


if __name__ == "__main__":
    main()
