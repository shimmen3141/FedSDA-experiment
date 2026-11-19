"""真の概念で割り当てる版（オラクル）: 木の版の警報・同定が完璧なときの目安を測る研究ツール。

各真の概念に専用の概念モデル（木の版と同じ`HoeffdingTreeConceptModel`、事前学習なし）を1つ持ち、各標本で、
今の真の概念のモデルだけで予測し（確率が最大のクラス、同点は小さいクラス。木の版と同じ規則）、そのモデルで1回
学習する。初めて現れた概念のモデルは空の木から始める。警報・MDL・Fixed-Shareの混合は使わない。
木の版との差は「仕組み（検出・同定・混合）で失っている分」の目安になる。ただし混合予測が真の概念のモデルより
当たる場面もあるので、厳密な上限ではない。

--grace-period・--deltaを付けると、木をその値の`HoeffdingTreeClassifier`にする（付けなければ木の版と同じ既定）。
--tree-versionに木の版の結果（`runs/*.json`）を渡すと、(データ, seed)の組と系列のhashを照合し、オラクル−木の版の
差（seedで対にした95%区間）を`analysis_oracle_gap.md`に書く。

使い方（WSLの~/river-env、worktreeのルートで）:
    PYTHONPATH=src python tools/research_single_client_stream_baselines/oracle_concept_assignment.py \\
        --streams <系列> --output <出力先> [--grace-period 50 --delta 0.01] --workers 8 --tree-version <木の版の出力>
"""

import argparse
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

for _thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

_SMOOTHING_WEIGHT = 0.01  # 木の版の既定の平滑化（予測するクラスは変えない）。


def _create_model(class_count, grace_period, delta):
    from river.tree import HoeffdingTreeClassifier

    from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (
        HoeffdingTreeConceptModel,
    )

    if grace_period is None:
        return HoeffdingTreeConceptModel.create_empty(
            class_count=class_count, probability_smoothing_weight=_SMOOTHING_WEIGHT
        )
    return HoeffdingTreeConceptModel(
        tree=HoeffdingTreeClassifier(grace_period=grace_period, delta=delta),
        class_count=class_count,
        probability_smoothing_weight=_SMOOTHING_WEIGHT,
    )


def run_oracle(*, stream, grace_period, delta):
    """系列1本を処理し、予測したクラスの列を返す。"""
    import numpy as np

    from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (
        convert_features_to_dictionary,
    )

    models = {}
    predicted = np.empty(len(stream.labels), dtype=np.int64)
    for index, (row, label, concept_id) in enumerate(
        zip(stream.features, stream.labels, stream.concept_ids)
    ):
        model = models.get(int(concept_id))
        if model is None:
            model = models[int(concept_id)] = _create_model(stream.class_count, grace_period, delta)
        x = convert_features_to_dictionary(row=row)
        probabilities = model.predict_class_probabilities(features=x)
        predicted[index] = max(
            range(stream.class_count), key=lambda class_id: (probabilities[class_id], -class_id)
        )
        model.learn_one(features=x, observed_class_id=int(label))
    return predicted, sum(model.node_count for model in models.values())


def _run_one(task):
    from single_stream_hard_assignment_soft_prediction.single_stream_input import (
        load_single_stream_input,
    )
    from single_stream_hard_assignment_soft_prediction.single_stream_metrics import (
        compute_prequential_metrics,
    )

    stream_path, output, grace_period, delta = task
    result_path = Path(output) / "runs" / f"{Path(stream_path).stem}.json"
    if result_path.exists():
        return f"skip {result_path.stem}"
    stream = load_single_stream_input(stream_path=Path(stream_path))
    started = time.perf_counter()
    predicted, node_count = run_oracle(stream=stream, grace_period=grace_period, delta=delta)
    elapsed = time.perf_counter() - started
    summary = compute_prequential_metrics(
        predicted_class_ids=predicted,
        observed_class_ids=stream.labels,
        concept_ids=stream.concept_ids,
        class_count=stream.class_count,
    )
    summary.update(elapsed_seconds=elapsed, final_tree_node_count=node_count)
    document = dict(
        method_name="oracle_concept_assignment_hoeffding_tree",
        tree=dict(grace_period=grace_period, delta=delta),
        input=dict(
            dataset_name=stream.dataset_name,
            random_seed=stream.random_seed,
            array_sha256=stream.array_sha256,
        ),
        summary=summary,
    )
    temporary = result_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    return f"done {result_path.stem} {summary['prequential_accuracy']:.4f}"


def _load(directory):
    runs = {}
    for path in sorted((Path(directory) / "runs").glob("*.json")):
        if path.name.endswith(".tmp.json"):
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        runs[(document["input"]["dataset_name"], int(document["input"]["random_seed"]))] = document
    return runs


def write_gap_analysis(*, oracle_directory, tree_version_directory):
    import numpy as np
    from scipy import stats

    oracle, tree = _load(oracle_directory), _load(tree_version_directory)
    if set(oracle) != set(tree):
        raise ValueError("(データ, seed)の組がそろっていません")
    for key in oracle:
        if oracle[key]["input"]["array_sha256"] != tree[key]["input"]["array_sha256"]:
            raise ValueError(f"系列のhashが違います: {key}")
    datasets = sorted({k[0] for k in oracle})
    seeds = sorted({k[1] for k in oracle})
    tree_setting = next(iter(oracle.values()))["tree"]
    lines = [
        "# オラクル（真の概念で割り当て）と木の版",
        "",
        f"- 木: {tree_setting}（Noneは既定のgrace period 200、delta 1e-7）。木の版の結果: `{tree_version_directory}`。",
        f"- データ{len(datasets)}、seed {seeds[0]}〜{seeds[-1]}（{len(seeds)}）。照合: 組と全runの系列のhashが一致した。",
        "- 差はオラクル−木の版（pp）。正なら仕組み（検出・同定・混合）で失っている分の目安。",
        "",
        "| データ | 木の版 | オラクル | 差 | 95%区間 | オラクルが上のseed | 変化の後100件の精度（木の版 / オラクル） |",
        "|---|---|---|---|---|---|---|",
    ]

    def value(runs, d, s, name="prequential_accuracy"):
        return 100 * runs[(d, s)]["summary"][name]

    for dataset in [*datasets, None]:
        selected = datasets if dataset is None else [dataset]
        per_tree = [st.mean(value(tree, d, s) for d in selected) for s in seeds]
        per_oracle = [st.mean(value(oracle, d, s) for d in selected) for s in seeds]
        differences = np.array(per_oracle) - np.array(per_tree)
        half = stats.t.ppf(0.975, len(seeds) - 1) * differences.std(ddof=1) / np.sqrt(len(seeds))
        after = [
            st.mean(
                st.mean(value(runs, d, s, "accuracy_first_100_after_change") for d in selected)
                for s in seeds
            )
            for runs in (tree, oracle)
        ]
        label = "全体（データの等重み）" if dataset is None else dataset
        lines.append(
            f"| {label} | {st.mean(per_tree):.2f} | {st.mean(per_oracle):.2f} | {differences.mean():+.2f} | "
            f"[{differences.mean() - half:+.2f}, {differences.mean() + half:+.2f}] | "
            f"{int((differences > 0).sum())}/{len(seeds)} | {after[0]:.2f} / {after[1]:.2f} |"
        )
    output = Path(oracle_directory) / "analysis_oracle_gap.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grace-period", type=int)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tree-version", type=Path)
    arguments = parser.parse_args(command_arguments)
    if (arguments.grace_period is None) != (arguments.delta is None):
        parser.error("--grace-periodと--deltaは両方付けるか、両方付けないでください。")
    paths = sorted(
        path
        for path in arguments.streams.glob("*.npz")
        if not path.name.endswith(".tmp.npz") and path.with_suffix(".json").exists()
    )
    if not paths:
        raise SystemExit("処理する系列がありません。")
    (arguments.output / "runs").mkdir(parents=True, exist_ok=True)
    (arguments.output / "oracle_provenance.json").write_text(
        json.dumps(
            dict(
                grace_period=arguments.grace_period,
                delta=arguments.delta,
                streams=str(arguments.streams),
                tool_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tasks = [
        (str(path), str(arguments.output), arguments.grace_period, arguments.delta) for path in paths
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for message in executor.map(_run_one, tasks):
            print(message, flush=True)
    if arguments.tree_version is not None:
        print(
            f"wrote {write_gap_analysis(oracle_directory=arguments.output, tree_version_directory=arguments.tree_version)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
