"""IMPROVE-031の掃引入口。製品の掃引のコマンドを、変種を有効にして変種ごとに1回実行する。

研究用CLI（`tools.research_provisional_candidate_expert.run`）と同じ形で指定する:
    python -m tools.research_loss_monitor_rebaseline --output <出力先> --datasets sine2 --intervals 100
        --seeds 0 1 --sample-count 5000 --variants product-default-loss-monitor-decrease-rebaseline --workers 64
変種ごとに`<出力先>/<変種>/`を製品の掃引の出力先にする（README、実行の記録、`fedsda/runs/*.npz`は製品と同じ置き方）。
変種の診断は`<出力先>/<変種>/loss_monitor_rebaseline_diagnostics/<runの識別子>.json`へ書く。
照合: python -m tools.research_loss_monitor_rebaseline compare --expected <保存済みの結果> --actual <出力先>
（`--actual`の各runを、`--expected`の下から同じファイル名で探して、全raw配列と要約を比べる）

変種は、製品の掃引のrunの実行の関数（`experiment_sweep_execution.execute_planned_experiment_run`）を、
このモジュールの`_execute_run_with_variant`へ差し替えて有効にする。並列のworker（spawn）は、差し替えた関数を
名前で読み込み、変種の名前を環境変数から受け取るので、worker の中でも変種が有効になる。
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

for _thread_count_variable_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from federated_learning_experiments.runtime import (  # noqa: E402
    fedsda_final_configuration_run_settings,
)
from federated_learning_experiments.sweep import experiment_sweep_execution  # noqa: E402
from tools.research_provisional_candidate_expert.loss_monitor_rebaseline import (  # noqa: E402
    ALL_EVALUATING_CLIENTS_LOSS_MONITOR_REBASELINE_VARIANTS,
    LOSS_MONITOR_REBASELINE_VARIANTS,
    loss_monitor_rebaseline_policy,
)

_VARIANTS = ("product-default", *LOSS_MONITOR_REBASELINE_VARIANTS)
# workerのprocessへ変種の名前を渡す環境変数（spawnのworkerは親の環境を引き継ぐ）。
_VARIANT_ENVIRONMENT_VARIABLE = "FDE_LOSS_MONITOR_REBASELINE_VARIANT"
_DIAGNOSTICS_DIRECTORY_NAME = "loss_monitor_rebaseline_diagnostics"
_VARIANT_RECORD_NAME = "loss_monitor_rebaseline_variant.json"
# 差し替える前の、製品のrunの実行の関数（このモジュールを読み込んだ時点の値）。
_execute_planned_experiment_run = experiment_sweep_execution.execute_planned_experiment_run


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _provenance():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPOSITORY_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=_REPOSITORY_ROOT,
        text=True,
    )
    return dict(commit=commit, has_uncommitted_changes=bool(dirty.strip()))


def _execute_run_with_variant(*, planned_run, output_directory, run_provenance):
    """変種を有効にして製品の1 runを実行し、済んだら変種の診断を書く（workerのprocessでも呼ばれる）。"""
    variant = os.environ[_VARIANT_ENVIRONMENT_VARIABLE]
    if variant not in _VARIANTS:
        raise ValueError(f"変種の名前が不正です: {variant}")
    diagnostics = {}
    with loss_monitor_rebaseline_policy(variant=variant, diagnostics=diagnostics):
        outcome = _execute_planned_experiment_run(
            planned_run=planned_run,
            output_directory=output_directory,
            run_provenance=run_provenance,
        )
    if outcome.run_status == "completed":
        directory = output_directory / _DIAGNOSTICS_DIRECTORY_NAME
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            directory / f"{planned_run.run_identifier}.json",
            dict(variant=variant, run_identifier=planned_run.run_identifier, diagnostics=diagnostics),
        )
    return outcome


def _sweep_arguments(arguments, variant, output_directory):
    return [
        "--configurations",
        "fedsda",
        "--datasets",
        *arguments.datasets,
        "--seeds",
        *[str(seed) for seed in arguments.seeds],
        "--aggregation-intervals",
        *[str(interval) for interval in arguments.intervals],
        "--per-client-sample-count",
        str(arguments.sample_count),
        "--client-count",
        str(arguments.client_count),
        "--workers",
        str(arguments.workers),
        "--output-directory",
        str(output_directory),
        "--purpose",
        f"{arguments.purpose}（変種 {variant}）",
    ]


def _run_variant(arguments, variant):
    """1つの変種で、製品の掃引のコマンドを実行する。完了・再開・飛ばしの判定は製品のコマンドに任せる。"""
    from federated_learning_experiments.sweep.experiment_sweep_command import main as run_sweep

    output_directory = arguments.output.resolve() / variant
    record_path = output_directory / _VARIANT_RECORD_NAME
    if record_path.exists():
        if _read_json(record_path).get("variant") != variant:
            raise ValueError(f"出力先に別の変種の記録があります: {record_path}")
    elif output_directory.exists() and any(output_directory.iterdir()):
        # 変種の記録は製品のコマンドを呼ぶ前に書くので、記録のない出力先は、この入口で作ったものではない。
        raise ValueError(f"変種の記録のない出力先は使えません（別の出力先を指定してください）: {output_directory}")
    sweep_arguments = _sweep_arguments(arguments, variant, output_directory)
    # 変種の記録を、製品のコマンドを呼ぶ前に置く（途中で止まっても、出力先と変種の対応が残る）。
    if not record_path.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_json(record_path, dict(variant=variant, **_provenance()))
    previous_variant = os.environ.get(_VARIANT_ENVIRONMENT_VARIABLE)
    os.environ[_VARIANT_ENVIRONMENT_VARIABLE] = variant
    experiment_sweep_execution.execute_planned_experiment_run = _execute_run_with_variant
    # 上限の撤廃は、runの設定に入る（計画を作るのは親のprocessで、workerは計画の設定のまま実行する）。
    # 設定が変わるので、runの識別子も既定と別になる。
    previous_maximum = fedsda_final_configuration_run_settings._MAXIMUM_EVALUATING_CLIENT_COUNT_PER_MODEL
    if variant in ALL_EVALUATING_CLIENTS_LOSS_MONITOR_REBASELINE_VARIANTS:
        fedsda_final_configuration_run_settings._MAXIMUM_EVALUATING_CLIENT_COUNT_PER_MODEL = (
            arguments.client_count
        )
    try:
        exit_code = run_sweep(sweep_arguments)
    finally:
        fedsda_final_configuration_run_settings._MAXIMUM_EVALUATING_CLIENT_COUNT_PER_MODEL = (
            previous_maximum
        )
        experiment_sweep_execution.execute_planned_experiment_run = _execute_planned_experiment_run
        if previous_variant is None:
            del os.environ[_VARIANT_ENVIRONMENT_VARIABLE]
        else:
            os.environ[_VARIANT_ENVIRONMENT_VARIABLE] = previous_variant
    return exit_code


def _run(arguments):
    if min(arguments.workers, arguments.sample_count, arguments.client_count) < 1:
        raise ValueError("workers・sample-count・client-countは正の整数にしてください")
    if not arguments.purpose.strip() or any(c in arguments.purpose for c in "\r\n"):
        raise ValueError("purposeは空白だけでない1行にしてください")
    if len(set(arguments.variants)) != len(arguments.variants):
        raise ValueError("変種に重複があります")
    exit_codes = []
    for variant in arguments.variants:
        print(f"== {variant}", flush=True)
        exit_codes.append(_run_variant(arguments, variant))
    # 製品のコマンドの終了コード（0: 全部済んだか飛ばした、1: 失敗したrunがある、2: 引数・計画の誤り）の最大。
    return max(exit_codes)


def _comparable_summary(value):
    if isinstance(value, dict):
        return {
            key: _comparable_summary(item)
            for key, item in value.items()
            if key
            not in (
                "started_at",
                "finished_at",
                "record_table_file_name",
                "output_directory",
                # 実行したコードの来歴。保存済みの結果（別のcommit）とも照合できるように外す。基準版は比べる。
                "code_commit",
                "has_uncommitted_changes",
                "code_revision_unavailable_reason",
            )
            and not key.endswith("_seconds")
        }
    if isinstance(value, list):
        return [_comparable_summary(item) for item in value]
    return value


def _file_digest(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_paths(root, names=None):
    """rootの下のrawを名前で引く。同じ名前が複数あれば、raw・要約の中身が全部同じときだけ1つとして扱う。

    `names`を渡すと、その名前だけを調べる（保存済みの結果から、照合に使うrunだけを引く）。
    """
    candidates = {}
    for path in sorted(root.glob("**/fedsda/runs/*.npz")):
        if names is None or path.name in names:
            candidates.setdefault(path.name, []).append(path)
    paths = {}
    for name, found in candidates.items():
        digests = {(_file_digest(path), _file_digest(path.with_suffix(".json"))) for path in found}
        if len(digests) != 1:
            raise ValueError(f"同じ名前で中身の違うrawが2つ以上あります: {[str(path) for path in found]}")
        paths[name] = found[0]
    return paths


def compare_run_artifacts(expected_directory, actual_directory):
    """`actual_directory`の下の全runを、`expected_directory`の下の同じ名前のrunと照合する。

    全raw配列と要約を比べ、秒・時刻・出力先の名前だけを除く。`expected_directory`は保存済みの結果の
    ディレクトリをそのまま渡せる（対応するrunがあればよく、ほかのrunは使わない）。
    """
    import numpy as np

    actual_paths = _run_paths(actual_directory)
    expected_paths = _run_paths(expected_directory, names=set(actual_paths))
    if not actual_paths:
        raise ValueError(f"照合するrawがありません: {actual_directory}")
    missing = sorted(set(actual_paths) - set(expected_paths))
    if missing:
        raise ValueError(f"照合の相手のrawがありません: {missing}")
    array_count = 0
    for name, actual_path in actual_paths.items():
        expected_path = expected_paths[name]
        with (
            np.load(expected_path, allow_pickle=False) as expected,
            np.load(actual_path, allow_pickle=False) as actual,
        ):
            if set(expected.files) != set(actual.files):
                raise ValueError(f"rawの列が一致しません: {name}")
            for key in expected.files:
                a, b = expected[key], actual[key]
                equal_nan = a.dtype.kind in "fc" and b.dtype.kind in "fc"
                if not np.array_equal(a, b, equal_nan=equal_nan):
                    raise ValueError(f"rawが一致しません: {name}: {key}")
                array_count += 1
        if _comparable_summary(
            _read_json(expected_path.with_suffix(".json"))
        ) != _comparable_summary(_read_json(actual_path.with_suffix(".json"))):
            raise ValueError(f"要約が一致しません: {name}")
    return dict(run_count=len(actual_paths), array_count=array_count)


def _run_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--client-count", type=int, default=10)
    parser.add_argument("--variants", nargs="+", choices=_VARIANTS, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--purpose", default="IMPROVE-031の監視の基準の再設定の実験")
    return parser


def main(command_arguments=None):
    command_arguments = sys.argv[1:] if command_arguments is None else list(command_arguments)
    try:
        if command_arguments[:1] == ["compare"]:
            parser = argparse.ArgumentParser(prog="compare", allow_abbrev=False)
            parser.add_argument("--expected", type=Path, required=True)
            parser.add_argument("--actual", type=Path, required=True)
            arguments = parser.parse_args(command_arguments[1:])
            print(json.dumps(compare_run_artifacts(arguments.expected, arguments.actual)))
            return 0
        return _run(_run_parser().parse_args(command_arguments))
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
