"""IMPROVE-030の製品外の確認: 履歴を塊に分け、塊単位の事後の再帰属（硬いEM）で概念ごとのモデルに分かれるか。

製品は使わない。概念の系列だけを既存のrawから取り、SEAの特徴とラベルは製品と同じ仕様（x∈U[0,10]³、x1+x2≤θならラベル1、
10%反転）で生成し直す。モデルはロジスティック回帰（SEAの境界を正確に表せる）。1 clientの履歴の接頭辞ごと、
または10 clientの塊をまとめた場合（連合の上限。通信・秘匿は考えない）に、次を求める。

- 混ざり方: 標本の概念が、その標本の塊を割り当てたモデルの多数派の概念と違う割合（製品の診断のmixed_assignment_countに近い）。
- 概念に合わせた精度: 各概念を、その概念が多数派のモデルのうち標本の最も多いモデルで予測した、新しい標本での精度（概念の割合で重み付け）。
  単一モデル（K=1）と、真の概念で学習帰属したモデル（oracle）の間で、どれだけ取り戻せたかを見る。
"""

import argparse
import glob
from collections import Counter

import numpy as np

_THRESHOLDS = (9.0, 8.0, 7.0, 9.5)
_NOISE = 0.10


def _generate(concepts, rng):
    features = rng.uniform(0.0, 10.0, size=(len(concepts), 3))
    thresholds = np.asarray(_THRESHOLDS)[concepts]
    labels = (features[:, 0] + features[:, 1] <= thresholds).astype(float)
    flip = rng.random(len(concepts)) < _NOISE
    labels[flip] = 1.0 - labels[flip]
    return features, labels


def _design(features):
    # 中心化した特徴＋切片。
    return np.column_stack([features - 5.0, np.ones(len(features))])


def _fit(design, labels, weights=None, iterations=25, ridge=1e-3):
    """ロジスティック回帰のIRLS（小さなL2）。"""
    w = np.zeros(design.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ w, -30, 30)))
        s = p * (1 - p) + 1e-9
        hessian = design.T @ (design * s[:, None]) + ridge * np.eye(design.shape[1])
        gradient = design.T @ (p - labels) + ridge * w
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _log_losses(design, labels, w):
    z = np.clip(design @ w, -30, 30)
    return np.logaddexp(0.0, z) - labels * z


def _hard_em(design, labels, block_ids, block_count, assignment, iterations=30):
    """塊の割当を固定して学習し、塊ごとの損失の和が最小のモデルへ割り当て直す。空のモデルは消す。"""
    for _ in range(iterations):
        models = []
        for k in np.unique(assignment):
            mask = assignment[block_ids] == k
            models.append(_fit(design[mask], labels[mask]))
        losses = np.stack([_log_losses(design, labels, w) for w in models])
        block_losses = np.stack(
            [np.bincount(block_ids, weights=row, minlength=block_count) for row in losses]
        )
        new_assignment = block_losses.argmin(axis=0)
        if np.array_equal(new_assignment, assignment):
            break
        assignment = np.unique(new_assignment, return_inverse=True)[1]
    total = float(block_losses.min(axis=0).sum())
    return assignment, models, total


def _segment(
    design, labels, block_ids, block_count, max_models, rng, restarts, *, fixed_models=None
):
    """K=1から、損失の大きい塊を種に分割しながらKを増やし、罰則つきの符号長が最小のKを選ぶ。

    fixed_modelsを与えると、罰則を使わず、そのKの解（探索で最小の損失）を返す。
    """
    n = len(labels)
    parameter_count = design.shape[1]
    best = None
    assignment = np.zeros(block_count, dtype=int)
    assignment, models, total = _hard_em(design, labels, block_ids, block_count, assignment)
    candidates = [(assignment, models, total)]
    for k in range(2, max_models + 1):
        trials = []
        # 種1: 今の解の各モデルで、損失の大きい半分の塊を新しいモデルへ。
        losses = np.stack([_log_losses(design, labels, w) for w in models])
        block_losses = np.stack(
            [np.bincount(block_ids, weights=row, minlength=block_count) for row in losses]
        )
        own = block_losses[assignment, np.arange(block_count)]
        for split in np.unique(assignment):
            members = np.flatnonzero(assignment == split)
            if len(members) < 2:
                continue
            seeded = assignment.copy()
            seeded[members[own[members] > np.median(own[members])]] = assignment.max() + 1
            trials.append(_hard_em(design, labels, block_ids, block_count, seeded))
        # 種2: でたらめな割当。
        for _ in range(restarts):
            trials.append(
                _hard_em(design, labels, block_ids, block_count, rng.integers(0, k, block_count))
            )
        trials = [t for t in trials if len(t[1]) <= k]
        if not trials:
            break
        assignment, models, total = min(trials, key=lambda t: t[2])
        candidates.append((assignment, models, total))
    if fixed_models is not None:
        fixed = [c for c in candidates if len(c[1]) == fixed_models]
        chosen = fixed[-1] if fixed else candidates[-1]
        return chosen[0], chosen[1]
    for assignment, models, total in candidates:
        k = len(models)
        # 2部の符号長: 損失（nats）＋パラメータ（d/2 ln n）×K＋塊ごとの割当（ln K）。
        code_length = total + k * parameter_count / 2 * np.log(n) + block_count * np.log(k)
        if best is None or code_length < best[0]:
            best = (code_length, assignment, models)
    return best[1], best[2]


def _concept_matched_accuracy(models_by_concept, concept_weights, rng, test_count=20000):
    accuracy = 0.0
    for concept, weight in concept_weights.items():
        features, labels = _generate(np.full(test_count, concept), rng)
        prediction = (_design(features) @ models_by_concept[concept] > 0).astype(float)
        accuracy += weight * float((prediction == labels).mean())
    return accuracy


def _evaluate(design, labels, concepts, block_ids, assignment, models, rng):
    sample_models = assignment[block_ids]
    majority = {}
    for k in np.unique(sample_models):
        counts = Counter(concepts[sample_models == k].tolist())
        majority[k] = (counts.most_common(1)[0][0], sum(counts.values()))
    mixed = float(np.mean([majority[k][0] != c for k, c in zip(sample_models, concepts)]))
    weights = {c: n / len(concepts) for c, n in Counter(concepts.tolist()).items()}
    models_by_concept = {}
    fallback = _fit(design, labels)
    for concept in weights:
        owned = [(n, k) for k, (c, n) in majority.items() if c == concept]
        # 多数派のモデルが無い概念は、全体の単一モデルで予測する（分けられなかった扱い）。
        models_by_concept[concept] = models[max(owned)[1]] if owned else fallback
    single = {c: fallback for c in weights}
    oracle = {c: _fit(design[concepts == c], labels[concepts == c]) for c in weights}
    return dict(
        model_count=len(models),
        mixed=mixed,
        accuracy=_concept_matched_accuracy(models_by_concept, weights, rng),
        single=_concept_matched_accuracy(single, weights, rng),
        oracle=_concept_matched_accuracy(oracle, weights, rng),
    )


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--prefixes", nargs="+", type=int, default=[1000, 2000, 3000, 5000])
    parser.add_argument("--max-models", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=3)
    # 切り分け: fixed（Kを真の概念数に固定）、oracle-init（塊の多数派の真の概念で初期化し、EMだけ回す）。
    parser.add_argument("--mode", choices=("select", "fixed", "oracle-init"), default="select")
    parser.add_argument("--scopes", nargs="+", default=["client", "pooled"])
    arguments = parser.parse_args(command_arguments)
    rows = []
    for path in sorted(glob.glob(f"{arguments.raw_root}/fedsda/runs/*_interval100_*.npz")):
        name = path.replace("\\", "/").split("/")[-1]
        dataset, seed = name.split("_")[0], int(name.split("_seed")[1].split("_")[0])
        with np.load(path) as raw:
            schedules = raw["sample_prediction.observed_concept_id"].astype(int)
        for scope in arguments.scopes:
            for prefix in arguments.prefixes:
                rng = np.random.default_rng([seed, prefix, scope == "pooled"])
                parts = (
                    [schedules[c, :prefix] for c in range(len(schedules))]
                    if scope == "client"
                    else [schedules[:, :prefix]]
                )
                for part in parts:
                    rows_of_part = part if part.ndim == 2 else part[None, :]
                    concepts = rows_of_part.reshape(-1)
                    # 塊はclientの時間順に切る（clientをまたがない）。
                    per_client_blocks = -(-rows_of_part.shape[1] // arguments.block_size)
                    block_ids = np.concatenate(
                        [
                            i * per_client_blocks
                            + np.arange(rows_of_part.shape[1]) // arguments.block_size
                            for i in range(len(rows_of_part))
                        ]
                    )
                    block_count = per_client_blocks * len(rows_of_part)
                    features, labels = _generate(concepts, rng)
                    design = _design(features)
                    concept_count = 2 if dataset == "sea2" else 4
                    if arguments.mode == "oracle-init":
                        initial = np.array(
                            [
                                Counter(concepts[block_ids == b].tolist()).most_common(1)[0][0]
                                for b in range(block_count)
                            ]
                        )
                        initial = np.unique(initial, return_inverse=True)[1]
                        assignment, models, _ = _hard_em(
                            design, labels, block_ids, block_count, initial
                        )
                    else:
                        assignment, models = _segment(
                            design,
                            labels,
                            block_ids,
                            block_count,
                            concept_count if arguments.mode == "fixed" else arguments.max_models,
                            rng,
                            arguments.restarts,
                            fixed_models=concept_count if arguments.mode == "fixed" else None,
                        )
                    result = _evaluate(design, labels, concepts, block_ids, assignment, models, rng)
                    rows.append(dict(dataset=dataset, seed=seed, scope=scope, prefix=prefix, **result))
    print(
        "|データ|範囲|接頭辞|件数|モデル数|混ざり方%|単一%|EM%|oracle%|取り戻した割合|"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    keys = sorted({(r["dataset"], r["scope"], r["prefix"]) for r in rows})
    for dataset, scope, prefix in keys:
        items = [r for r in rows if (r["dataset"], r["scope"], r["prefix"]) == (dataset, scope, prefix)]

        def mean(key):
            return float(np.mean([r[key] for r in items]))

        gap = mean("oracle") - mean("single")
        share = (mean("accuracy") - mean("single")) / gap if gap > 0 else float("nan")
        print(
            f"|{dataset}|{scope}|{prefix}|{len(items)}|{mean('model_count'):.2f}|{100 * mean('mixed'):.1f}|"
            f"{100 * mean('single'):.2f}|{100 * mean('accuracy'):.2f}|{100 * mean('oracle'):.2f}|{share:.2f}|"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
