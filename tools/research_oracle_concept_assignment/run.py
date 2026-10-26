"""基準版13の既定と、真の概念で帰属・統合を決める診断の掃引入口。"""

import argparse
import json
import os
import sys
from hashlib import sha256
from pathlib import Path
from time import perf_counter

for _thread_count_variable_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from federated_learning_experiments.sweep import (  # noqa: E402
    experiment_sweep_execution,
)
from tools.research_oracle_concept_assignment.policy import (
    VARIANTS as _VARIANTS,
    oracle_concept_assignment_policy,
)
from tools.research_loss_monitor_rebaseline import (
    _provenance,
    _read_json,
    _write_json,
    _sweep_arguments,
    compare_run_artifacts,
)

_VARIANT_ENVIRONMENT_VARIABLE = "FDE_ORACLE_CONCEPT_ASSIGNMENT_VARIANT"
_DIAGNOSTICS_DIRECTORY_NAME = "oracle_concept_assignment_diagnostics"
_VARIANT_RECORD_NAME = "oracle_concept_assignment_variant.json"
# 差し替える前の、製品のrunの実行の関数（このモジュールを読み込んだ時点の値）。
_execute_planned_experiment_run = experiment_sweep_execution.execute_planned_experiment_run


def _execute_run_with_variant(*, planned_run, output_directory, run_provenance):
    """変種を有効にして製品の1 runを実行し、済んだら変種の診断を書く（workerのprocessでも呼ばれる）。"""
    variant = os.environ[_VARIANT_ENVIRONMENT_VARIABLE]
    if variant not in _VARIANTS:
        raise ValueError(f"変種の名前が不正です: {variant}")
    diagnostics = {}
    started = perf_counter()
    with oracle_concept_assignment_policy(variant=variant, diagnostics=diagnostics):
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
            dict(
                variant=variant,
                run_identifier=planned_run.run_identifier,
                elapsed_seconds=perf_counter() - started,
                research_source_sha256={
                    path.name: sha256(path.read_bytes()).hexdigest()
                    for path in sorted(Path(__file__).parent.glob("*.py"))
                },
                diagnostics=diagnostics,
            ),
        )
    return outcome


def _run_variant(arguments, variant):
    """1つの変種で、製品の掃引のコマンドを実行する。完了・再開・飛ばしの判定は製品のコマンドに任せる。"""
    from federated_learning_experiments.sweep.experiment_sweep_command import (
        main as run_sweep,
    )

    output_directory = arguments.output.resolve() / variant
    record_path = output_directory / _VARIANT_RECORD_NAME
    if record_path.exists():
        if _read_json(record_path).get("variant") != variant:
            raise ValueError(f"出力先に別の変種の記録があります: {record_path}")
    elif output_directory.exists() and any(output_directory.iterdir()):
        # 変種の記録は製品のコマンドを呼ぶ前に書くので、記録のない出力先は、この入口で作ったものではない。
        raise ValueError(
            f"変種の記録のない出力先は使えません（別の出力先を指定してください）: {output_directory}"
        )
    sweep_arguments = _sweep_arguments(arguments, variant, output_directory)
    if arguments.print_plan:
        return run_sweep([*sweep_arguments, "--print-plan"])
    # 変種の記録を、製品のコマンドを呼ぶ前に置く（途中で止まっても、出力先と変種の対応が残る）。
    if not record_path.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_json(record_path, dict(variant=variant, **_provenance()))
    previous_variant = os.environ.get(_VARIANT_ENVIRONMENT_VARIABLE)
    os.environ[_VARIANT_ENVIRONMENT_VARIABLE] = variant
    experiment_sweep_execution.execute_planned_experiment_run = _execute_run_with_variant
    try:
        exit_code = run_sweep(sweep_arguments)
    finally:
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
    parser.add_argument("--purpose", default="IMPROVE-030の概念を分けたときの天井の診断")
    parser.add_argument("--print-plan", action="store_true")
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
