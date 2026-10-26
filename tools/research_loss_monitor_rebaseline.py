"""IMPROVE-031の掃引入口。製品の掃引のコマンドを、変種を有効にして1 runずつ子processで実行する。

研究用CLI（`tools.research_provisional_candidate_expert.run`）と同じ形で指定する:
    python -m tools.research_loss_monitor_rebaseline --output <出力先> --datasets sine2 --intervals 100
        --seeds 0 1 --sample-count 5000 --variants product-default-loss-monitor-decrease-rebaseline --workers 64
各runは`<出力先>/<変種>/<dataset>-i<間隔>-s<seed>/`へ保存する（製品と同じraw）。
照合: python -m tools.research_loss_monitor_rebaseline compare --expected <保存済みの結果> --actual <出力先>
（`--actual`の各runを、`--expected`の下から同じファイル名で探して、全raw配列と要約を比べる）
1 runだけの実行（子processの中身）: python -m tools.research_loss_monitor_rebaseline run-one --variant <変種> -- <掃引の引数>
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


def _child_command(arguments, variant, dataset, interval, seed, output_directory):
    return [
        sys.executable,
        "-m",
        "tools.research_loss_monitor_rebaseline",
        "run-one",
        "--variant",
        variant,
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
        f"{arguments.purpose}（変種 {variant}）",
    ]


def _launch(arguments):
    if min(arguments.workers, arguments.sample_count, arguments.client_count) < 1:
        raise ValueError("workers・sample-count・client-countは正の整数にしてください")
    if not arguments.purpose.strip() or any(c in arguments.purpose for c in "\r\n"):
        raise ValueError("purposeは空白だけでない1行にしてください")
    if len(set(arguments.variants)) != len(arguments.variants):
        raise ValueError("変種に重複があります")
    # 製品の計画で、datasetや間隔の値の検査も実行前に済ませる。出力先は作らない。
    output = arguments.output.resolve()
    runs = []
    for variant, dataset, interval, seed in product(
        arguments.variants, arguments.datasets, arguments.intervals, arguments.seeds
    ):
        output_directory = output / variant / f"{dataset}-i{interval}-s{seed}"
        command = _child_command(arguments, variant, dataset, interval, seed, output_directory)
        _validate_single_run_plan(command[command.index("--") + 1 :])
        _check_variant(output_directory, variant)
        runs.append((output_directory, command))
    if len({directory for directory, _ in runs}) != len(runs):
        raise ValueError("dataset・間隔・seedに重複があります")
    for variant in arguments.variants:
        (output / variant).mkdir(parents=True, exist_ok=True)
    # 同じ出力先への実行（例: MNIST4の追加seed）は、実行ごとの記録を足していく。
    record_path = output / "launch.json"
    history = _read_json(record_path)["launches"] if record_path.exists() else []
    record = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(arguments).items()
        },
        **_provenance(),
        started_at=datetime.now(timezone.utc).isoformat(),
        run_outcomes=[],
    )
    history.append(record)
    _write_json(record_path, dict(launches=history))

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

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = [pool.submit(execute, run) for run in runs]
        for future in as_completed(futures):
            outcome = future.result()
            record["run_outcomes"].append(outcome)
            print(f"[{len(record['run_outcomes'])}/{len(runs)}] {outcome['output_directory']}: {outcome['run_status']}", flush=True)
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(record_path, dict(launches=history))
    counts = {s: sum(r["run_status"] == s for r in record["run_outcomes"]) for s in ("completed", "skipped_existing", "failed")}
    print(f"completed: {counts['completed']}, skipped (existing): {counts['skipped_existing']}, failed: {counts['failed']}")
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


def _launch_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--client-count", type=int, default=10)
    parser.add_argument("--variants", nargs="+", choices=_VARIANTS, required=True)
    parser.add_argument("--workers", type=int, required=True, help="同時に動かす子processの数")
    parser.add_argument("--purpose", default="IMPROVE-031の監視の基準の再設定の実験")
    return parser


def main(command_arguments=None):
    command_arguments = sys.argv[1:] if command_arguments is None else list(command_arguments)
    try:
        if command_arguments[:1] == ["run-one"]:
            parser = argparse.ArgumentParser(prog="run-one", allow_abbrev=False)
            parser.add_argument("--variant", choices=_VARIANTS, required=True)
            parser.add_argument("sweep_arguments", nargs=argparse.REMAINDER)
            arguments = parser.parse_args(command_arguments[1:])
            if not arguments.sweep_arguments or arguments.sweep_arguments[0] != "--":
                raise ValueError("run-one --variant <変種> -- <掃引の引数>と指定してください")
            return _run_one(arguments.variant, arguments.sweep_arguments[1:])
        if command_arguments[:1] == ["compare"]:
            parser = argparse.ArgumentParser(prog="compare", allow_abbrev=False)
            parser.add_argument("--expected", type=Path, required=True)
            parser.add_argument("--actual", type=Path, required=True)
            arguments = parser.parse_args(command_arguments[1:])
            print(json.dumps(compare_run_artifacts(arguments.expected, arguments.actual)))
            return 0
        return _launch(_launch_parser().parse_args(command_arguments))
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
