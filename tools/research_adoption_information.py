"""採用時までの保存情報で、Bの候補別利益を識別できるかseed単位で評価する。"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "initial_score",
    "adoption_score",
    "current_code_length",
    "new_code_length",
    "change_span",
    "observations",
    "score_gain_per_observation",
    "warm_started",
    "pending_samples",
    "recent5_current_loss",
    "recent5_candidate_loss",
    "recent5_gap",
    "recent30_current_loss",
    "recent30_candidate_loss",
    "recent30_gap",
    "recent30_gap_std",
)
TARGETS = ("full_log", "full_correctness", "origin30_log", "origin30_correctness")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def label(value, log_loss):
    if value is None:
        return None
    tolerance = 1e-10 if log_loss else 0
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def adoption_features(episode, trace):
    """採用標本を含む過去だけを参照し、将来・真概念fieldは使わない。"""
    observations = [
        t for t in trace if t["sample_index"] <= episode["adoption_sample_index"]
    ]
    n = episode["observed_sample_count"]
    if len(observations) != n or [t["age"] for t in observations] != list(
        range(1, n + 1)
    ):
        raise ValueError("採用前のtraceが完全でありません")
    score = episode["initial_score"]
    for t in observations:
        score += t["current_log_loss"] - t["shadow_log_loss"]
        if abs(score - t["score"]) > 1e-9:
            raise ValueError("Dの逐次記録が不正です")
    if (
        not observations
        or observations[-1]["sample_index"] != episode["adoption_sample_index"]
        or abs(score - episode["last_score"]) > 1e-9
    ):
        raise ValueError("採用時点のDが一致しません")
    f = dict(
        initial_score=episode["initial_score"],
        adoption_score=score,
        current_code_length=episode["current_code_length"],
        new_code_length=episode["new_code_length"],
        change_span=episode["change_interval_sample_count"],
        observations=n,
        score_gain_per_observation=(score - episode["initial_score"]) / n,
        warm_started=int(episode["learner_name"] == "warm_started_learner"),
        pending_samples=episode["adoption_pending_sample_count"],
    )
    for size in (5, 30):
        recent = observations[-size:]
        f[f"recent{size}_current_loss"] = mean(t["current_log_loss"] for t in recent)
        f[f"recent{size}_candidate_loss"] = mean(t["shadow_log_loss"] for t in recent)
        f[f"recent{size}_gap"] = mean(
            t["current_log_loss"] - t["shadow_log_loss"] for t in recent
        )
    f["recent30_gap_std"] = pstdev(
        t["current_log_loss"] - t["shadow_log_loss"] for t in observations[-30:]
    )
    if set(f) != set(FEATURES) or not all(math.isfinite(v) for v in f.values()):
        raise ValueError("特徴が不正です")
    return f


def candidate_rows(result_root):
    metadata = load(result_root / "research-source.json")
    if (result_root / "failures.json").exists():
        raise ValueError("失敗のある結果です")
    for name, h in metadata["sha256"].items():
        if sha(ROOT / name) != h:
            raise ValueError(f"実行ソース不一致: {name}")
    rows = load(result_root / "results.json")
    if Counter(json.dumps(r["job"], sort_keys=True) for r in rows) != Counter(
        json.dumps(j, sort_keys=True) for j in metadata["jobs"]
    ):
        raise ValueError("run集合が不正です")
    candidates = []
    for row in rows:
        j = row["job"]
        if (
            j["sample_count"] != 2500
            or j["interval"] != 100
            or row["extra_model_forward_count"] != 0
        ):
            raise ValueError("今回の固定条件と一致しません")
        traces = defaultdict(list)
        path = result_root / row["shadow_diagnostic"]["trace_file"]
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                t = json.loads(line)
                traces[(t["client_id"], t["episode_id"])].append(t)
        if (
            sum(map(len, traces.values()))
            != row["shadow_diagnostic"]["trace_row_count"]
        ):
            raise ValueError("trace件数不一致")
        episodes = {
            (e["client_id"], e["episode_id"]): e
            for e in row["shadow_diagnostic"]["episodes"]
            if "adopted_model_id" in e
        }
        records = row["model_lifecycle"]
        if set(episodes) != {(r["client_id"], r["episode_id"]) for r in records} or len(
            records
        ) != len(episodes):
            raise ValueError("採用と来歴が対応しません")
        for r in records:
            e = episodes[(r["client_id"], r["episode_id"])]
            features = adoption_features(e, traces[(r["client_id"], r["episode_id"])])
            defined = [s for s in r["removal_samples"] if s["status"] == "defined"]
            origin = [
                s
                for s in defined
                if s["is_origin_client"] and 1 <= s["adoption_age"] <= 30
            ]
            if sorted(s["adoption_age"] for s in origin) != list(range(1, 31)):
                origin = []
            targets = {}
            for name, samples in (("full", defined), ("origin30", origin)):
                targets[name + "_log"] = (
                    mean(s["log_loss_benefit"] for s in samples) if samples else None
                )
                targets[name + "_correctness"] = (
                    mean(s["correctness_benefit"] for s in samples) if samples else None
                )
            candidates.append(
                dict(
                    seed=j["seed"],
                    dataset=j["dataset"],
                    stream=j["stream"],
                    client=r["client_id"],
                    episode=r["episode_id"],
                    adoption_sample=e["adoption_sample_index"],
                    features=features,
                    targets=targets,
                    labels={
                        name: label(value, name.endswith("_log"))
                        for name, value in targets.items()
                    },
                    defined_samples=len(defined),
                    origin30_samples=len(origin),
                )
            )
    return rows, candidates


def predict(model, features):
    if model["feature"] is None:
        return model["constant"]
    return (
        1
        if model["direction"] * (features[model["feature"]] - model["threshold"]) > 0
        else -1
    )


def evaluate(model, candidates, target):
    valid = [r for r in candidates if r["labels"][target] in (-1, 1)]
    counts = Counter(r["seed"] for r in valid)
    # 各seedの総重みを同じにし、その中では各候補を等重みにする。
    weights = [1 / counts[r["seed"]] for r in valid]
    predictions = [predict(model, r["features"]) for r in valid]
    recalls = {}
    for cls in (-1, 1):
        total = sum(w for r, w in zip(valid, weights) if r["labels"][target] == cls)
        recalls[str(cls)] = (
            sum(
                w
                for r, w, p in zip(valid, weights, predictions)
                if r["labels"][target] == cls and p == cls
            )
            / total
            if total
            else None
        )
    total = sum(weights)
    return dict(
        evaluated=len(valid),
        eligible_seed_count=len(counts),
        beneficial=sum(r["labels"][target] == 1 for r in valid),
        harmful=sum(r["labels"][target] == -1 for r in valid),
        neutral=sum(r["labels"][target] == 0 for r in candidates),
        missing=sum(r["labels"][target] is None for r in candidates),
        balanced_accuracy=mean(recalls.values())
        if all(v is not None for v in recalls.values())
        else None,
        seed_weighted_accuracy=sum(
            w * (p == r["labels"][target])
            for r, w, p in zip(valid, weights, predictions)
        )
        / total
        if total
        else None,
        recall_by_class=recalls,
        confusion={
            f"actual_{a}:predicted_{p}": sum(
                r["labels"][target] == a and prediction == p
                for r, prediction in zip(valid, predictions)
            )
            for a in (-1, 1)
            for p in (-1, 1)
        },
        predictions=[
            dict(
                seed=r["seed"],
                dataset=r["dataset"],
                stream=r["stream"],
                client=r["client"],
                episode=r["episode"],
                actual=r["labels"][target],
                predicted=p,
            )
            for r, p in zip(valid, predictions)
        ],
    )


def fit(candidates, target):
    valid = [r for r in candidates if r["labels"][target] in (-1, 1)]
    best = dict(feature=None, constant=1)
    best_value = -1
    options = [best, dict(feature=None, constant=-1)]
    for feature in FEATURES:
        values = sorted({r["features"][feature] for r in valid})
        options.extend(
            dict(
                feature=feature,
                threshold=left + (right - left) / 2,
                direction=direction,
            )
            for left, right in zip(values, values[1:])
            for direction in (1, -1)
        )
    for model in options:
        report = evaluate(model, candidates, target)
        value = report["balanced_accuracy"]
        if value is None:
            value = report["seed_weighted_accuracy"]
        if value is not None and value > best_value + 1e-12:
            best, best_value = model, value
    return best


def lock(training, output):
    rows, candidates = candidate_rows(training)
    if {r["job"]["seed"] for r in rows} != {0, 1}:
        raise ValueError("固定判定の学習にはseed0/1だけを使います")
    output.mkdir(parents=True, exist_ok=False)
    models = {target: fit(candidates, target) for target in TARGETS}
    record = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        training_result=str(training),
        training_input_sha256={
            n: sha(training / n) for n in ("results.json", "research-source.json")
        },
        features=FEATURES,
        primary_target="full_log",
        models=models,
        training_scores={
            target: evaluate(models[target], candidates, target) for target in TARGETS
        },
        analysis_source_sha256=sha(Path(__file__)),
    )
    (output / "locked-models.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                models=models,
                training=[
                    dict(
                        target=t,
                        balanced_accuracy=record["training_scores"][t][
                            "balanced_accuracy"
                        ],
                        n=record["training_scores"][t]["evaluated"],
                    )
                    for t in TARGETS
                ],
            ),
            indent=2,
        )
    )


def analyze(inputs, output):
    locked = load(output / "locked-models.json")
    if locked["analysis_source_sha256"] != sha(Path(__file__)):
        raise ValueError("固定判定後に解析コードが変わっています")
    training = Path(locked["training_result"])
    if any(sha(training / n) != h for n, h in locked["training_input_sha256"].items()):
        raise ValueError("学習入力が変化しています")
    rows, candidates = [], []
    for result_root in inputs:
        r, c = candidate_rows(result_root)
        rows.extend(r)
        candidates.extend(c)
    keys = [
        tuple(r["job"][k] for k in ("dataset", "stream", "interval", "seed"))
        for r in rows
    ]
    expected = {
        (d, s, 100, seed)
        for d, streams in (
            ("sea4", ("drift", "fixed0", "fixed2")),
            ("sine2", ("drift", "fixed0", "fixed1")),
        )
        for s in streams
        for seed in range(4)
    }
    if Counter(keys) != Counter({k: 1 for k in expected}):
        raise ValueError("4seed24条件の完全一致が必要です")
    results = {}
    for target in TARGETS:
        test = [r for r in candidates if r["seed"] in (2, 3)]
        model = locked["models"][target]
        folds = []
        for seed in range(4):
            train = [r for r in candidates if r["seed"] != seed]
            held = [r for r in candidates if r["seed"] == seed]
            fold_model = fit(train, target)
            folds.append(
                dict(
                    seed=seed,
                    model=fold_model,
                    evaluation=evaluate(fold_model, held, target),
                )
            )
        results[target] = dict(
            locked_model=model,
            training=locked["training_scores"][target],
            prospective_test=evaluate(model, test, target),
            test_by_seed={
                str(seed): evaluate(
                    model, [r for r in test if r["seed"] == seed], target
                )
                for seed in (2, 3)
            },
            always_beneficial_test=evaluate(
                dict(feature=None, constant=1), test, target
            ),
            always_harmful_test=evaluate(dict(feature=None, constant=-1), test, target),
            leave_one_seed_out=folds,
        )
    from tools.research_shadow_learner_diagnostic.summarize_standard_metrics import (
        total_mac,
    )

    groups = defaultdict(list)
    for row in rows:
        groups[(row["job"]["dataset"], row["job"]["stream"])].append(row)
    metrics = []
    for key, rs in sorted(groups.items()):
        for part in ("baseline", "measured"):
            ms = [r[part] for r in rs]
            metrics.append(
                dict(
                    dataset=key[0],
                    stream=key[1],
                    variant=part,
                    accuracy=mean(m["accuracy"] for m in ms),
                    delta_pp=100
                    * mean(
                        m["accuracy"] - r["baseline"]["accuracy"]
                        for m, r in zip(ms, rs)
                    ),
                    uploaded_mb=mean(m["uploaded_bytes"] for m in ms) / 1e6,
                    downloaded_mb=mean(m["downloaded_bytes"] for m in ms) / 1e6,
                    gmac=mean(total_mac(m["model_computation_counts"]) for m in ms)
                    / 1e9,
                    seconds=mean(m["seconds"] for m in ms),
                )
            )
    result = dict(
        candidates=candidates,
        results=results,
        standard_metrics=metrics,
        input_sha256={
            str(p): {n: sha(p / n) for n in ("results.json", "research-source.json")}
            for p in inputs
        },
        locked_models_sha256=sha(output / "locked-models.json"),
        analysis_source_sha256=sha(Path(__file__)),
    )
    (output / "summary_information.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 採用時情報の識別性能",
        "",
        "seed0/1で固定、seed2/3は未使用test。中立/観測なしは除外し、候補とseedの分母を明記する。性能は局所利益ラベルの識別で、全体精度ではない。",
        "",
        "| 目的 | train BA | test BA | test 有益/有害/中立/欠測 | 固定判定 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for target, r in results.items():
        train = r["training"]
        test = r["prospective_test"]
        lines.append(
            f"| {target} | {train['balanced_accuracy']} | {test['balanced_accuracy']} | {test['beneficial']}/{test['harmful']}/{test['neutral']}/{test['missing']} | {json.dumps(r['locked_model'])} |"
        )
    lines.extend(
        [
            "",
            "## 標準指標（4seed平均）",
            "",
            "| データ | 入力 | 構成 | 精度% | 既定差pp | 上りMB | 下りMB | G MAC | 秒 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for m in metrics:
        lines.append(
            f"| {m['dataset']} | {m['stream']} | {m['variant']} | {100 * m['accuracy']:.3f} | {m['delta_pp']:+.4f} | {m['uploaded_mb']:.3f} | {m['downloaded_mb']:.3f} | {m['gmac']:.3f} | {m['seconds']:.2f} |"
        )
    (output / "analysis_information.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                t: dict(
                    model=r["locked_model"],
                    train_ba=r["training"]["balanced_accuracy"],
                    test_ba=r["prospective_test"]["balanced_accuracy"],
                    test_n=r["prospective_test"]["evaluated"],
                )
                for t, r in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["lock", "analyze"])
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "lock":
        if len(a.inputs) != 1:
            p.error("学習入力はseed0/1の1ディレクトリだけです")
        lock(a.inputs[0], a.output)
    else:
        analyze(a.inputs, a.output)
