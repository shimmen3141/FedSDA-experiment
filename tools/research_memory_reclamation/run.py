"""同期末の解放済み領域回収を比較する独立研究CLI。製品既定へ登録しない。"""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import runpy
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "src/federated_learning_experiments").is_dir()
    and (p / "run_experiment_sweep.py").is_file()
)
_EXPECTED_SOURCE = "3861f16d77fad2449baf39e37c433bb5343b56f4291efa315ce9c3aea8cf71e2"
_ARMS = ("control", "reclaim")
_EXPECTED_ARRAY_COUNT = 133
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def _source_identity():
    digest = hashlib.sha256()
    for path in sorted((_ROOT / "src").rglob("*.py")):
        digest.update(path.relative_to(_ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return dict(
        source_sha256=digest.hexdigest(),
        runner_sha256=hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        cli_sha256=hashlib.sha256(
            (_ROOT / "run_experiment_sweep.py").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    )


def _validate_arguments(args):
    if args.sample_count <= 0 or args.sample_count % 50:
        raise ValueError("sample-countは正の50の倍数が必要です。")
    if not args.purpose.strip() or "\n" in args.purpose or "\r" in args.purpose:
        raise ValueError("purposeには空でない1行を指定してください。")
    output = args.output_directory.resolve()
    if any(output.is_relative_to(_ROOT / name) for name in ("src", "tools", ".git", ".kiro")):
        raise ValueError("コードやGit管理領域の中を出力先にできません。")


def _build_reclaimer():
    if sys.platform != "linux" or platform.libc_ver()[0] != "glibc":
        raise RuntimeError("この試作の実行対象はLinux/glibcです。")
    if not Path("/usr/bin/time").is_file() or not Path("/proc/self/status").is_file():
        raise RuntimeError("GNU timeまたは/proc/self/statusがありません。")
    function = ctypes.CDLL(None).malloc_trim
    function.argtypes = [ctypes.c_size_t]
    function.restype = ctypes.c_int
    return lambda: function(0)


def _read_memory():
    fields = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            fields[key] = int(value.split()[0]) * 1024
    return dict(rss_bytes=fields["VmRSS"], high_water_bytes=fields["VmHWM"])


def _make_sync_wrapper(original, reclaim, emit):
    def synchronize(owner, **kwargs):
        result = original(owner, **kwargs)
        before = _read_memory()
        started = perf_counter()
        value = reclaim() if reclaim is not None else None
        seconds = perf_counter() - started
        after = _read_memory()
        emit(
            dict(
                stage="synchronization_end",
                round_index=kwargs["round_index"],
                before_rss_bytes=before["rss_bytes"],
                after_rss_bytes=after["rss_bytes"],
                before_high_water_bytes=before["high_water_bytes"],
                after_high_water_bytes=after["high_water_bytes"],
                trim_result=value,
                trim_seconds=seconds,
            )
        )
        return result

    return synchronize


def _environment():
    import numpy as np
    import torch

    return dict(
        identity=_source_identity(),
        python=sys.version,
        executable=sys.executable,
        numpy=np.__version__,
        torch=torch.__version__,
        libc=platform.libc_ver(),
        threads=torch.get_num_threads(),
        thread_environment={k: os.environ.get(k) for k in _THREAD_VARIABLES},
        allocator_environment={
            k: os.environ.get(k)
            for k in (
                "MALLOC_ARENA_MAX",
                "MALLOC_TRIM_THRESHOLD_",
                "MALLOC_MMAP_THRESHOLD_",
                "PYTHONMALLOC",
            )
        },
    )


def _parameter_hashes(result):
    repository = result.measured_run.participants.server_operations.owners.global_model_repository
    return {
        str(model_id): {
            key: dict(
                dtype=str(value.dtype),
                shape=list(value.shape),
                sha256=hashlib.sha256(
                    value.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            )
            for key, value in repository.get_global_model_parameters(model_id=model_id).items()
        }
        for model_id in repository.global_model_ids
    }


def _arm(args):
    for variable in _THREAD_VARIABLES:
        os.environ[variable] = "1"
    reclaimer = _build_reclaimer()
    sys.path.insert(0, str(_ROOT / "src"))
    from federated_learning_experiments.runtime.fedsda_measured_run_execution import (
        _RoundMeasuringServerOperations,
    )
    from federated_learning_experiments.sweep import (
        experiment_sweep_execution as execution,
    )

    directory = args.output_directory.resolve()
    environment = _environment()
    if environment["identity"]["source_sha256"] != _EXPECTED_SOURCE:
        raise RuntimeError("sourceが基準と異なります。")
    if environment["threads"] != 1:
        raise RuntimeError("torch thread数が1ではありません。")
    _write_json(directory / "environment.json", environment)
    original_run = execution.execute_fedsda_experiment_run
    started = perf_counter()
    with (directory / "events.jsonl").open("x", encoding="utf-8", buffering=1) as stream:

        def emit(value):
            stream.write(json.dumps(dict(seconds=perf_counter() - started, **value)) + "\n")

        def execute(**kwargs):
            result = original_run(**kwargs)
            emit(dict(stage="run_return", **_read_memory()))
            _write_json(directory / "parameters.json", _parameter_hashes(result))
            emit(dict(stage="parameters_saved", **_read_memory()))
            return result

        synchronization = _make_sync_wrapper(
            _RoundMeasuringServerOperations.synchronize_models,
            reclaimer if args._arm == "reclaim" else None,
            emit,
        )
        argv = [
            str(_ROOT / "run_experiment_sweep.py"),
            "--configurations",
            "fedsda",
            "--datasets",
            "mnist2",
            "--seeds",
            "0",
            "--client-count",
            "10",
            "--per-client-sample-count",
            str(args.sample_count),
            "--aggregation-intervals",
            "50",
            "--workers",
            "1",
            "--output-directory",
            str(directory / "cli"),
            "--purpose",
            args.purpose,
        ]
        emit(dict(stage="run_start", **_read_memory()))
        with (
            patch.object(_RoundMeasuringServerOperations, "synchronize_models", synchronization),
            patch.object(execution, "execute_fedsda_experiment_run", execute),
            patch.object(sys, "argv", argv),
        ):
            try:
                runpy.run_path(argv[0], run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise RuntimeError(f"元CLIが非成功で終了しました: {exc.code}") from exc
        after = _environment()
        _write_json(directory / "environment-after.json", after)
        if after != environment:
            raise RuntimeError("実行前後でコード/環境が変わりました。")
        emit(dict(stage="finished", **_read_memory()))


def _parse_time(path):
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition(": ")
        if separator:
            fields[key] = value
    seconds = 0.0
    for part in fields["Elapsed (wall clock) time (h:mm:ss or m:ss)"].split(":"):
        seconds = seconds * 60 + float(part)
    if int(fields["Exit status"]) != 0:
        raise ValueError("GNU timeの終了状態が非成功です。")
    return dict(
        peak_rss_mib=int(fields["Maximum resident set size (kbytes)"]) / 1024,
        elapsed_seconds=seconds,
        user_seconds=float(fields["User time (seconds)"]),
        system_seconds=float(fields["System time (seconds)"]),
    )


def _read_arm(directory):
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    environment = read("environment.json")
    if environment != read("environment-after.json"):
        raise ValueError("前後のコード/環境が一致しません。")
    summaries = list((directory / "cli/fedsda/runs").glob("*.json"))
    if len(summaries) != 1:
        raise ValueError("CLIのrun要約が1件ではありません。")
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    table = summary["record_table_file_name"]
    if Path(table).name != table:
        raise ValueError("記録ファイル名が不正です。")
    rows = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    if not rows or rows[-1]["stage"] != "finished":
        raise ValueError("完了記録がありません。")
    returned = [row for row in rows if row["stage"] == "run_return"]
    synchronized = [row for row in rows if row["stage"] == "synchronization_end"]
    conditions = summary["run_settings"]["execution_settings"]["experiment_run_conditions"]
    expected = conditions["per_client_sample_count"] // 50
    if len(returned) != 1 or [r["round_index"] for r in synchronized] != list(range(expected)):
        raise ValueError("run/同期記録の件数や順序が不正です。")
    return dict(
        environment=environment,
        summary=summary,
        table=summaries[0].with_name(table),
        parameters=read("parameters.json"),
        events=synchronized,
        run_return_peak_mib=returned[0]["high_water_bytes"] / 2**20,
        time=_parse_time(directory / "gnu-time.txt"),
    )


def _compare(directory):
    import numpy as np

    a, b = (_read_arm(directory / name) for name in _ARMS)
    for field in ("environment", "parameters"):
        if a[field] != b[field]:
            raise ValueError(f"{field}が対照間で一致しません。")
    for field in ("run_settings", "run_metrics"):
        if a["summary"][field] != b["summary"][field]:
            raise ValueError(f"{field}が対照間で一致しません。")
    with (
        np.load(a["table"], allow_pickle=False) as av,
        np.load(b["table"], allow_pickle=False) as bv,
    ):
        if len(av.files) != _EXPECTED_ARRAY_COUNT or len(bv.files) != _EXPECTED_ARRAY_COUNT:
            raise ValueError("固定した構成の133配列が揃っていません。")
        if set(av.files) != set(bv.files):
            raise ValueError("配列キーが一致しません。")
        for key in av.files:
            x, y = av[key], bv[key]
            if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
                raise ValueError(f"配列が一致しません: {key}")
        count = len(av.files)
    if any(r["trim_result"] is not None for r in a["events"]):
        raise ValueError("controlで回収が記録されています。")
    if any(
        type(r["trim_result"]) is not int or r["trim_result"] not in (0, 1) for r in b["events"]
    ):
        raise ValueError("reclaimの回収記録が不正です。")
    return dict(
        status="passed",
        exact_array_count=count,
        settings_equal=True,
        metrics_equal=True,
        final_parameters_equal=True,
        environment_equal=True,
        arms={
            name: dict(
                **value["time"],
                run_return_peak_mib=value["run_return_peak_mib"],
                synchronization_count=len(value["events"]),
                trim_seconds=sum(r["trim_seconds"] for r in value["events"]),
                maximum_boundary_rss_drop_mib=max(
                    (r["before_rss_bytes"] - r["after_rss_bytes"]) / 2**20 for r in value["events"]
                ),
            )
            for name, value in zip(_ARMS, (a, b))
        },
        peak_reduction_mib=a["time"]["peak_rss_mib"] - b["time"]["peak_rss_mib"],
        note="同期末の差は生存中の状態での回収量。採否は自動決定しない。",
    )


def _pair(args, identity):
    _build_reclaimer()
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError("出力先が既にあります。新しい名前を指定してください。")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "plan.json",
        dict(
            identity=identity,
            sample_count=args.sample_count,
            arms=list(_ARMS),
            purpose=args.purpose,
        ),
    )
    (output / "README.md").write_text(
        "# 生存中の状態を維持した回収の対照\n\n目的: " + args.purpose + "\n\n"
        "control→reclaimを独立processで逐次実行。comparison.jsonがpassedの場合だけ照合成功。\n",
        encoding="utf-8",
    )
    try:
        environment = dict(
            os.environ, **{k: "1" for k in _THREAD_VARIABLES}, PYTHONIOENCODING="utf-8", LC_ALL="C"
        )
        for arm in _ARMS:
            if _source_identity() != identity:
                raise RuntimeError("計画後にコードが変わりました。")
            arm_directory = output / arm
            arm_directory.mkdir()
            command = [
                "/usr/bin/time",
                "-v",
                "-o",
                str(arm_directory / "gnu-time.txt"),
                sys.executable,
                str(Path(__file__).resolve()),
                "--_arm",
                arm,
                "--output-directory",
                str(arm_directory),
                "--sample-count",
                str(args.sample_count),
                "--purpose",
                args.purpose,
            ]
            print(f"{arm}: 開始。出力先 {arm_directory}", flush=True)
            with (arm_directory / "stdout.log").open("x", encoding="utf-8") as stream:
                subprocess.run(
                    command,
                    cwd=_ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if _source_identity() != identity:
            raise RuntimeError("実行中にコードが変わりました。")
        result = _compare(output)
        _write_json(output / "comparison.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        _write_json(
            output / "failure.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}")
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--_arm", choices=_ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        _validate_arguments(args)
        identity = _source_identity()
        if identity["source_sha256"] != _EXPECTED_SOURCE:
            raise ValueError(
                "srcが調査基準と異なります。コードを変更せず担当者へ報告してください。"
            )
        if args.print_plan:
            print(
                json.dumps(
                    dict(
                        arms=list(_ARMS),
                        sample_count=args.sample_count,
                        output_directory=str(args.output_directory.resolve()),
                        identity=identity,
                    ),
                    ensure_ascii=False,
                )
            )
        elif args._arm is not None:
            _arm(args)
        else:
            _pair(args, identity)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
