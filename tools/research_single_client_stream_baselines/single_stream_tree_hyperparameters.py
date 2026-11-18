"""単一ストリームの手法（木の版）の、概念モデルのHoeffding treeのハイパーパラメータだけを替えて流す研究ツール。

問い: 木の版が10本の木のアンサンブル（ARF・SRP・LB）に負けるのは、学習器（既定のHoeffding treeは
grace period 200、delta 1e-7で育ちが遅い）のためか、警報・同定の仕組みのためか。ARFの木と同じ値
（grace period 50、delta 0.01）にした木で、ほかは同じ手法・同じ系列を流し、既定の木の結果と比べる。

やり方: 各worker processの初期化で、パッケージの概念モデル`HoeffdingTreeConceptModel.create_empty`を、値を指定した
`HoeffdingTreeClassifier`を作るものに差し替える（パッケージのコードは変えない）。既定の値を指定すると、既定の
結果と全run一致することを対照で確かめる。各runで、作った概念モデルの木が
指定の値を持つことを確かめる。成果物は木の版のコマンドと同じ形式（`runs/*.json`・`.npz`と`run_summary.csv`）で、
出力先に`tree_hyperparameters.json`（替えた値とこのファイルのhash）を書く。

使い方（WSLの~/river-env、worktreeのルートで）:
    PYTHONPATH=src python tools/research_single_client_stream_baselines/single_stream_tree_hyperparameters.py \\
        --streams <系列> --output <出力先> --grace-period 50 --delta 0.01 --workers 4
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

for _thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

_TREE_ARGUMENTS = {}


def _install_tree(grace_period, delta):
    """パッケージの概念モデルが作る空の木の値を替える（このprocessだけ）。

    木は本物の`HoeffdingTreeClassifier`のまま、値を引数で渡す（部分クラスにすると、Riverの内部の計算が
    最後の桁で変わり、既定の値でも既定の結果と一致しなかったため、2026-11-19に部分クラスをやめた）。
    写しは`deepcopy`、空の新しいモデルは`create_empty`を通るので、どちらも替えた値になる。
    """
    from river.tree import HoeffdingTreeClassifier

    from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (
        HoeffdingTreeConceptModel,
    )

    def create_empty(cls, *, class_count, probability_smoothing_weight):
        return cls(
            tree=HoeffdingTreeClassifier(grace_period=grace_period, delta=delta),
            class_count=class_count,
            probability_smoothing_weight=probability_smoothing_weight,
        )

    HoeffdingTreeConceptModel.create_empty = classmethod(create_empty)
    _TREE_ARGUMENTS.update(grace_period=grace_period, delta=delta)


def _check_installed():
    from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (
        HoeffdingTreeConceptModel,
    )

    model = HoeffdingTreeConceptModel.create_empty(class_count=2, probability_smoothing_weight=0.01)
    tree = model.tree
    if (tree.grace_period, tree.delta) != (
        _TREE_ARGUMENTS["grace_period"],
        _TREE_ARGUMENTS["delta"],
    ):
        raise RuntimeError("木のハイパーパラメータが差し替わっていません")
    if (model.copy().tree.grace_period, model.create_empty_like().tree.grace_period) != (
        _TREE_ARGUMENTS["grace_period"],
    ) * 2:
        raise RuntimeError("写し・空の新しいモデルの木が差し替わっていません")


def _run_one(task):
    from single_stream_hard_assignment_soft_prediction.single_stream_artifacts import (
        save_single_stream_run,
    )
    from single_stream_hard_assignment_soft_prediction.single_stream_input import (
        load_single_stream_input,
    )
    from single_stream_hard_assignment_soft_prediction.single_stream_run import run_single_stream
    from single_stream_hard_assignment_soft_prediction.single_stream_settings import (
        build_default_single_stream_settings,
    )

    _check_installed()
    stream_path, output_directory = Path(task[0]), Path(task[1])
    runs_directory = output_directory / "runs"
    json_path = runs_directory / f"{stream_path.stem}.json"
    npz_path = runs_directory / f"{stream_path.stem}.npz"
    if json_path.exists() and npz_path.exists():
        return f"skip {stream_path.stem}"
    if json_path.exists() or npz_path.exists():
        raise RuntimeError(f"成果物が片方だけあります: {stream_path.stem}")
    stream = load_single_stream_input(stream_path=stream_path)
    settings = build_default_single_stream_settings()
    result = run_single_stream(stream=stream, settings=settings)
    save_single_stream_run(
        result=result,
        settings=settings,
        stream=stream,
        output_directory=runs_directory,
        file_stem=stream_path.stem,
    )
    return f"done {stream_path.stem} {result.summary['prequential_accuracy']:.4f}"


def main(command_arguments=None):
    from single_stream_hard_assignment_soft_prediction.single_stream_command import (
        select_stream_paths,
        write_run_summary_table,
    )

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grace-period", type=int, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(command_arguments)
    if arguments.grace_period < 1 or not 0 < arguments.delta < 1 or arguments.workers < 1:
        parser.error("grace-periodは1以上、deltaは0と1の間、workersは1以上にしてください。")
    paths = select_stream_paths(streams=arguments.streams, datasets=arguments.datasets, seeds=None)
    if not paths:
        raise SystemExit("処理する系列がありません。")
    arguments.output.mkdir(parents=True, exist_ok=True)
    record_path = arguments.output / "tree_hyperparameters.json"
    record = dict(
        grace_period=arguments.grace_period,
        delta=arguments.delta,
        note="木の版の概念モデルのHoeffding treeだけを替えた（ほかは既定）。ARFの木と同じ値",
        tool_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if record_path.exists() and json.loads(record_path.read_text(encoding="utf-8"))[
        "grace_period"
    ] != arguments.grace_period:
        raise SystemExit("出力先に別の値の結果があります。")
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tasks = [(str(path), str(arguments.output)) for path in paths]
    with ProcessPoolExecutor(
        max_workers=arguments.workers,
        initializer=_install_tree,
        initargs=(arguments.grace_period, arguments.delta),
    ) as executor:
        for message in executor.map(_run_one, tasks):
            print(message, flush=True)
    print(f"wrote {write_run_summary_table(output_directory=arguments.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
