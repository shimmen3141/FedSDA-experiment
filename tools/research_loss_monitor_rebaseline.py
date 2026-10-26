"""IMPROVE-031の掃引入口。run-oneは1 processで1 run、launchは子processを並列に動かす。

例: python -m tools.research_loss_monitor_rebaseline launch --variant
    product-default-loss-monitor-decrease-rebaseline --datasets sine2 --intervals 100
    --seeds 0 --jobs 1 --output-root <dir> --purpose "基準の再設定の機能確認"
照合: compare --expected <既定の出力先> --actual <observe-onlyの出力先>
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

for _thread_count_variable_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from tools.research_provisional_candidate_expert.loss_monitor_rebaseline import (  # noqa: E402
    LOSS_MONITOR_REBASELINE_VARIANTS,
    loss_monitor_rebaseline_policy,
)

_VARIANTS = ("product-default", *LOSS_MONITOR_REBASELINE_VARIANTS)
_DIAGNOSTICS_NAME = "loss_monitor_rebaseline_diagnostics.json"


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


def _check_variant(output_directory, variant, *, required=False):
    path = output_directory / _DIAGNOSTICS_NAME
    if not path.exists():
        if required:
            raise ValueError(f"済んだrunの診断がありません: {path}")
        return
    if _read_json(path).get("variant") != variant:
        raise ValueError(f"出力先に別の変種の診断があります: {path}")


def _validate_sweep_arguments(arguments):
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    for name in (
        "--workers",
        "--datasets",
        "--seeds",
        "--aggregation-intervals",
        "--configurations",
        "--output-directory",
    ):
        parser.add_argument(name, nargs="+", action="append", required=True)
    parsed, _ = parser.parse_known_args(arguments)
    for name, values in vars(parsed).items():
        if len(values) != 1 or len(values[0]) != 1:
            raise ValueError(f"{name}は1回だけ、1つの値を指定してください")
    if parsed.workers != [["1"]] or parsed.configurations != [["fedsda"]]:
        raise ValueError("--workers 1と--configurations fedsdaが必要です")
    int(parsed.seeds[0][0])
    int(parsed.aggregation_intervals[0][0])
    _validate_single_run_plan(arguments)
    return Path(parsed.output_directory[0][0])


def _validate_single_run_plan(arguments):
    from federated_learning_experiments.sweep.experiment_sweep_command import (
        _build_argument_parser,
        _build_sweep_plan,
    )
    from federated_learning_experiments.sweep.experiment_sweep_plan import (
        build_planned_experiment_runs,
    )

    parser = _build_argument_parser()
    parser.allow_abbrev = False
    parsed = parser.parse_args(arguments)
    if parsed.print_plan:
        raise ValueError("run-oneでは--print-planは使えません")
    if len(build_planned_experiment_runs(sweep_plan=_build_sweep_plan(parsed))) != 1:
        raise ValueError("1 processで実行できるのは1 runだけです")


def _new_run_status(output_directory, previous_records):
    records = set(output_directory.glob("sweep_record_*.json")) - previous_records
    if len(records) != 1:
        raise ValueError("今回の掃引の記録が1つに定まりません")
    outcomes = _read_json(next(iter(records)))["run_outcomes"]
    if len(outcomes) != 1 or outcomes[0]["configuration_name"] != "fedsda":
        raise ValueError("今回の掃引の結果がfedsdaの1 runに定まりません")
    return outcomes[0]["run_status"]


def _run_one(variant, sweep_arguments):
    from federated_learning_experiments.sweep.experiment_sweep_command import (
        main as run_sweep,
    )

    output_directory = _validate_sweep_arguments(sweep_arguments)
    _check_variant(output_directory, variant)
    previous_records = set(output_directory.glob("sweep_record_*.json"))
    diagnostics = {}
    with loss_monitor_rebaseline_policy(variant=variant, diagnostics=diagnostics):
        exit_code = run_sweep(sweep_arguments)
    if exit_code != 0:
        return exit_code
    status = _new_run_status(output_directory, previous_records)
    if status == "skipped_existing":
        _check_variant(output_directory, variant, required=True)
    elif status == "completed":
        _write_json(
            output_directory / _DIAGNOSTICS_NAME,
            dict(
                variant=variant,
                sweep_arguments=sweep_arguments,
                diagnostics=diagnostics,
                **_provenance(),
            ),
        )
    else:
        raise ValueError(f"成功した掃引に想定外のrun状態があります: {status}")
    return 0


def _child_command(arguments, dataset, interval, seed, output_directory):
    return [
        sys.executable,
        "-m",
        "tools.research_loss_monitor_rebaseline",
        "run-one",
        "--variant",
        arguments.variant,
        "--",
        "--configurations",
        "fedsda",
        "--datasets",
        dataset,
        "--seeds",
        str(seed),
        "--aggregation-intervals",
        str(interval),
        "--per-client-sample-count",
        str(arguments.sample_count),
        "--client-count",
        str(arguments.client_count),
        "--workers",
        "1",
        "--output-directory",
        str(output_directory),
        "--purpose",
        f"{arguments.purpose}（変種 {arguments.variant}）",
    ]


def _launch(arguments):
    if min(arguments.jobs, arguments.sample_count, arguments.client_count) < 1:
        raise ValueError("jobs・sample-count・client-countは正の整数にしてください")
    if not arguments.purpose.strip() or any(c in arguments.purpose for c in "\r\n"):
        raise ValueError("purposeは空白だけでない1行にしてください")
    # 製品の計画で、datasetや間隔の値の検査も実行前に済ませる。出力先は作らない。
    runs = []
    for dataset, interval, seed in product(
        arguments.datasets, arguments.intervals, arguments.seeds
    ):
        output_directory = (
            arguments.output_root.resolve() / f"{dataset}-i{interval}-s{seed}"
        )
        command = _child_command(arguments, dataset, interval, seed, output_directory)
        _validate_single_run_plan(command[command.index("--") + 1 :])
        _check_variant(output_directory, arguments.variant)
        runs.append((output_directory, command))
    if len({directory for directory, _ in runs}) != len(runs):
        raise ValueError("dataset・間隔・seedに重複があります")
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    record = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(arguments).items()
        },
        variant=arguments.variant,
        **_provenance(),
        started_at=datetime.now(timezone.utc).isoformat(),
        run_outcomes=[],
    )
    record_path = arguments.output_root / "launch.json"
    _write_json(record_path, record)

    def execute(run):
        directory, command = run
        previous = set(directory.glob("sweep_record_*.json"))
        try:
            with directory.with_suffix(".log").open("w", encoding="utf-8") as log:
                code = subprocess.run(
                    command, cwd=_REPOSITORY_ROOT, stdout=log, stderr=log
                ).returncode
                status = _new_run_status(directory, previous) if code == 0 else "failed"
                # 子processの失敗の終了コードは、そのまま残す。成功の終了コードで想定外の状態のときだけ2にする。
                if code == 0 and status not in ("completed", "skipped_existing"):
                    code, status = 2, "failed"
        except (OSError, ValueError, KeyError) as error:
            print(f"{directory}: {error}", file=sys.stderr)
            code, status = 2, "failed"
        return dict(output_directory=str(directory), run_status=status, exit_code=code)

    with ThreadPoolExecutor(max_workers=arguments.jobs) as pool:
        futures = [pool.submit(execute, run) for run in runs]
        for future in as_completed(futures):
            record["run_outcomes"].append(future.result())
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(record_path, record)
    return int(any(run["exit_code"] != 0 for run in record["run_outcomes"]))


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
            )
            and not key.endswith("_seconds")
        }
    if isinstance(value, list):
        return [_comparable_summary(item) for item in value]
    return value


def compare_run_artifacts(expected_directory, actual_directory):
    """全raw配列と要約を照合する。秒・時刻・出力先の名前だけを除く。"""
    import numpy as np

    expected_paths = {
        path.name: path for path in expected_directory.glob("fedsda/runs/*.npz")
    }
    actual_paths = {
        path.name: path for path in actual_directory.glob("fedsda/runs/*.npz")
    }
    if not expected_paths or expected_paths.keys() != actual_paths.keys():
        raise ValueError("照合するrawの名前の集合が一致しません（空集合も不可）")
    array_count = 0
    for name, expected_path in expected_paths.items():
        actual_path = actual_paths[name]
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
    return dict(run_count=len(expected_paths), array_count=array_count)


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    run_one = commands.add_parser("run-one", allow_abbrev=False)
    run_one.add_argument("--variant", choices=_VARIANTS, required=True)
    run_one.add_argument("sweep_arguments", nargs=argparse.REMAINDER)
    launch = commands.add_parser("launch", allow_abbrev=False)
    launch.add_argument("--variant", choices=_VARIANTS, required=True)
    launch.add_argument("--datasets", nargs="+", required=True)
    launch.add_argument("--intervals", nargs="+", type=int, required=True)
    launch.add_argument("--seeds", nargs="+", type=int, required=True)
    launch.add_argument("--sample-count", type=int, default=5000)
    launch.add_argument("--client-count", type=int, default=10)
    launch.add_argument("--jobs", type=int, required=True)
    launch.add_argument("--output-root", type=Path, required=True)
    launch.add_argument("--purpose", required=True)
    compare = commands.add_parser("compare", allow_abbrev=False)
    compare.add_argument("--expected", type=Path, required=True)
    compare.add_argument("--actual", type=Path, required=True)
    arguments = parser.parse_args(command_arguments)
    try:
        if arguments.command == "run-one":
            if not arguments.sweep_arguments or arguments.sweep_arguments[0] != "--":
                raise ValueError(
                    "run-one --variant <変種> -- <掃引の引数>と指定してください"
                )
            return _run_one(arguments.variant, arguments.sweep_arguments[1:])
        if arguments.command == "launch":
            return _launch(arguments)
        print(json.dumps(compare_run_artifacts(arguments.expected, arguments.actual)))
        return 0
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
