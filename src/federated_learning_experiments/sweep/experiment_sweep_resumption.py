"""既存の成果物形式を変えずに、掃引の再開用証拠・排他・探索を提供する。"""

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from json import dumps, loads
from os import fsync, walk
from os import replace as replace_file
from pathlib import Path
from platform import machine, python_version, system
from time import sleep
from uuid import uuid4
from zipfile import ZIP_LZMA, ZipFile

from numpy import __version__ as numpy_version
from torch import __version__ as torch_version

from federated_learning_experiments.core.run_provenance import (
    BEHAVIOR_BASELINE_VERSION,
    RunProvenance,
    convert_run_provenance_to_mapping,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import PlannedExperimentRun

_REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[3]
# 全runが済んだ掃引の、runごとの開始の記録をまとめるzip（`.resume/`の下）。
_RUN_START_RECORD_ARCHIVE_NAME = "run_start_records.zip"
_REPLACE_ATTEMPT_COUNT = 8
_REPLACE_RETRY_SECONDS = 0.5


@dataclass(frozen=True)
class SweepResumptionCandidate:
    """過去掃引の保存先と、要求した全runに対する完了・復旧可能・要確認件数。"""

    output_directory: Path
    completed_run_count: int
    total_run_count: int
    recoverable_run_count: int
    conflicting_run_count: int


def describe_planned_run_set(planned_runs: tuple[PlannedExperimentRun, ...]) -> list[dict]:
    """構成名と完全設定の集合を、順序非依存のJSON互換値として返す。"""
    values = [
        {
            "configuration_name": run.configuration_name,
            "run_settings": convert_settings_to_plain_mapping(run.run_settings),
        }
        for run in planned_runs
    ]
    return loads(dumps(sorted(values, key=lambda value: dumps(value, sort_keys=True))))


def describe_resumption_environment() -> dict[str, str]:
    """復旧の証拠を比較する数値環境（既存artifactのenvironmentと同じ粒度）。"""
    return dict(
        system=system(),
        machine=machine(),
        python=python_version(),
        numpy=numpy_version,
        torch=str(torch_version),
    )


def _safe_path(output_directory: Path, *parts: str) -> Path:
    root = output_directory.resolve()
    path = output_directory
    for part in parts:
        if Path(part).name != part or part in ("", ".", ".."):
            raise ValueError("resumption path must stay within output directory")
        path = path / part
        if _is_link(path) or not path.resolve().is_relative_to(root):
            raise ValueError(f"resumption path must not traverse symlinks: {path}")
    return path


def _is_link(path: Path) -> bool:
    # LinuxサーバのPython3.10にはis_junctionがない（junctionはWindows固有）。
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _same_document(left: object, right: object) -> bool:
    return dumps(left, sort_keys=True) == dumps(right, sort_keys=True)


def _write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            fsync(stream.fileno())
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def hold_sweep_output_lock(output_directory: Path) -> Iterator[None]:
    """全workerの終了までOS排他を保持する。lockファイルは残し、OSが解放する。"""
    output_directory.mkdir(parents=True, exist_ok=True)
    path = _safe_path(output_directory, ".sweep.lock")
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                from msvcrt import LK_NBLCK, LK_UNLCK, locking

                locking(stream.fileno(), LK_NBLCK, 1)
            else:
                from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock

                flock(stream.fileno(), LOCK_EX | LOCK_NB)
        except OSError as error:
            raise ValueError(
                f"sweep already running or output lock unavailable: {output_directory}"
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                locking(stream.fileno(), LK_UNLCK, 1)
            else:
                flock(stream.fileno(), LOCK_UN)


def write_sweep_resumption_plan(
    output_directory: Path, planned_runs: tuple[PlannedExperimentRun, ...]
) -> None:
    """全runの補助計画を原子的に保存し、既存との矛盾は拒否する。"""
    path = _safe_path(output_directory, ".resume", "sweep.json")
    data = {
        "version": 1,
        "behavior_baseline_version": BEHAVIOR_BASELINE_VERSION,
        "planned_runs": describe_planned_run_set(planned_runs),
    }
    if path.exists():
        if not _same_document(loads(path.read_text(encoding="utf-8")), data):
            raise ValueError(f"resumption plan differs from requested sweep: {path}")
        return
    _write_atomic(path, data)


def _start_record(output_directory: Path, planned_run: PlannedExperimentRun) -> Path:
    return _safe_path(
        output_directory,
        ".resume",
        planned_run.configuration_name,
        f"{planned_run.run_identifier}.json",
    )


def _read_run_start_record(
    output_directory: Path, planned_run: PlannedExperimentRun
) -> tuple[dict | None, Path | None]:
    """開始の記録を、個別のファイル、無ければ`.resume/run_start_records.zip`の順に読む。

    戻り値: (記録の文書, 記録のあるパス)。どちらにも無ければ(None, None)。zipを読む途中で`OSError`以外の
    例外（壊れた・途中で切れたzip、lzmaの無い環境、jsonの復号の失敗など）が出たら`ValueError`にする。
    """
    path = _start_record(output_directory, planned_run)
    if path.exists():
        return loads(path.read_text(encoding="utf-8")), path
    archive_path = _safe_path(output_directory, ".resume", _RUN_START_RECORD_ARCHIVE_NAME)
    if not archive_path.exists():
        return None, None
    member_name = f"{planned_run.configuration_name}/{planned_run.run_identifier}.json"
    try:
        with ZipFile(archive_path) as archive:
            if member_name not in archive.namelist():
                return None, None
            return loads(archive.read(member_name).decode("utf-8")), archive_path
    except OSError:
        raise
    except Exception as caught_exception:
        raise ValueError(
            f"run start record archive is unreadable: {archive_path}"
        ) from caught_exception


def _describe_start(planned_run: PlannedExperimentRun, run_provenance: RunProvenance) -> dict:
    return loads(
        dumps(
            {
                "version": 1,
                "configuration_name": planned_run.configuration_name,
                "run_identifier": planned_run.run_identifier,
                "run_settings": convert_settings_to_plain_mapping(planned_run.run_settings),
                "run_provenance": convert_run_provenance_to_mapping(run_provenance),
                "environment": describe_resumption_environment(),
                "execution_source_sha256": _execution_source_sha256(),
            }
        )
    )


def _execution_source_sha256() -> str:
    # git追跡の有無にかかわらず実行パッケージとルートのPythonを照合する。
    # 結果・cache・文書は対象外。ソースのsymlinkは保証できないため拒否する。
    digest = sha256()
    for directory, subdirectories, _ in walk(_REPOSITORY_DIRECTORY / "src", followlinks=False):
        if any(_is_link(Path(directory) / name) for name in subdirectories):
            raise ValueError("execution source must not traverse symlink directories")
    files = sorted(
        set(_REPOSITORY_DIRECTORY.glob("*.py")) | set((_REPOSITORY_DIRECTORY / "src").rglob("*.py"))
    )
    for path in files:
        relative = path.relative_to(_REPOSITORY_DIRECTORY)
        safe = _safe_path(_REPOSITORY_DIRECTORY, *relative.parts)
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(safe.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def write_run_start_record(
    output_directory: Path, planned_run: PlannedExperimentRun, run_provenance: RunProvenance
) -> Path:
    """計算前に開始証拠を書く。既存結果があるときは証拠を置き換えない。

    同じ内容の記録が個別のファイルかzipにあれば受け入れて、そのパスを返す（zipなら書かない）。違えば拒否する。
    """
    for suffix in (".json", ".npz"):
        path = _safe_path(
            output_directory,
            planned_run.configuration_name,
            "runs",
            planned_run.run_identifier + suffix,
        )
        if path.exists():
            raise ValueError(
                f"existing artifact must be checked before recording a new start: {path}"
            )
    data = _describe_start(planned_run, run_provenance)
    existing, existing_path = _read_run_start_record(output_directory, planned_run)
    if existing_path is not None:
        if not _same_document(existing, data):
            raise ValueError(f"existing run start record differs: {existing_path}")
        return existing_path
    path = _start_record(output_directory, planned_run)
    _write_atomic(path, data)
    return path


def can_recover_interrupted_run(
    output_directory: Path, planned_run: PlannedExperimentRun, run_provenance: RunProvenance
) -> bool:
    """NPZだけが残り、同一のcleanコード・設定・環境の開始証拠があるかを調べる。"""
    if (
        run_provenance.has_uncommitted_changes is not False
        or run_provenance.code_commit is None
        or run_provenance.behavior_baseline_version != BEHAVIOR_BASELINE_VERSION
    ):
        return False
    try:
        npz = _safe_path(
            output_directory,
            planned_run.configuration_name,
            "runs",
            planned_run.run_identifier + ".npz",
        )
        summary = npz.with_suffix(".json")
        if summary.exists() or summary.is_symlink() or not npz.is_file():
            return False
        proof, _ = _read_run_start_record(output_directory, planned_run)
        if proof is None:
            return False
        return _same_document(proof, _describe_start(planned_run, run_provenance))
    except (OSError, ValueError):
        return False


def _replace_with_retries(source: Path, destination: Path) -> None:
    """置き換える。Windowsで、作ったばかりのファイルをほかのprocess（検査ソフトなど）が一時的に開いていて
    `PermissionError`になったときだけ、少し待って数回やり直す。"""
    for attempt in range(_REPLACE_ATTEMPT_COUNT):
        try:
            replace_file(source, destination)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPT_COUNT - 1:
                raise
            sleep(_REPLACE_RETRY_SECONDS * (attempt + 1))


def _verify_run_start_record_archive(archive_path: Path, expected: dict[str, bytes]) -> None:
    """書いたzipを読み戻し、名前の集合と各記録のバイト列が、まとめる前に読んだものと同じことを確かめる。"""
    with ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError("run start record archive members differ from the records")
        for member_name, data in expected.items():
            if archive.read(member_name) != data:
                raise ValueError(f"run start record differs after packing: {member_name}")


def pack_completed_sweep_run_start_records(output_directory: Path) -> str | None:
    """全runが済んだ掃引の、runごとの開始の記録を`.resume/run_start_records.zip`へまとめる。

    `.resume/`直下の`recovered`以外の各ディレクトリの`*.json`を集め、既にあるzipの記録と合わせて
    LZMAのzipを作り、読み戻して照合してから置き、その後で個別のファイルを消す。`sweep.json`・`recovered/`・
    途中のファイル（`*.partial`）は残す。新しい記録が無ければzipを書き直さない（重複の個別のファイルだけ消す）。
    戻り値: 警告の文字列か`None`。zipを置く前の失敗は、元を残して「まとめなかった」を返す（例外を出さない）。
    """
    resume_directory = output_directory / ".resume"
    if not resume_directory.exists() and not _is_link(resume_directory):
        return None
    partial_path: Path | None = None
    try:
        resume_directory = _safe_path(output_directory, ".resume")
        archive_path = _safe_path(output_directory, ".resume", _RUN_START_RECORD_ARCHIVE_NAME)
        partial_path = _safe_path(
            output_directory, ".resume", _RUN_START_RECORD_ARCHIVE_NAME + ".partial"
        )
        partial_path.unlink(missing_ok=True)
        records: dict[str, tuple[Path, bytes]] = {}
        record_directories = []
        for directory in sorted(resume_directory.iterdir()):
            if directory.name == "recovered" or not (directory.is_dir() or _is_link(directory)):
                continue
            directory = _safe_path(output_directory, ".resume", directory.name)
            record_directories.append(directory)
            for path in sorted(directory.glob("*.json")):
                path = _safe_path(output_directory, ".resume", directory.name, path.name)
                records[f"{directory.name}/{path.name}"] = (path, path.read_bytes())
        if not records:
            return None
        existing: dict[str, bytes] = {}
        if archive_path.exists():
            with ZipFile(archive_path) as archive:
                existing = {name: archive.read(name) for name in archive.namelist()}
        for member_name, (_, data) in records.items():
            if member_name in existing and existing[member_name] != data:
                return (
                    "開始の記録をまとめなかった: zipと個別のファイルで内容が違う記録があります: "
                    f"{member_name}"
                )
        new_records = {
            member_name: data
            for member_name, (_, data) in records.items()
            if member_name not in existing
        }
        if new_records:
            combined = {**existing, **new_records}
            with ZipFile(partial_path, "w", compression=ZIP_LZMA) as archive:
                for member_name, data in combined.items():
                    archive.writestr(member_name, data)
            with partial_path.open("rb+") as stream:
                fsync(stream.fileno())
            _verify_run_start_record_archive(partial_path, combined)
            _replace_with_retries(partial_path, archive_path)
    except Exception as caught_exception:
        if partial_path is not None:
            try:
                partial_path.unlink(missing_ok=True)
            except OSError:
                pass
        return f"開始の記録をまとめなかった: {caught_exception}"
    try:
        for path, _ in records.values():
            path.unlink()
        for directory in record_directories:
            if not any(directory.iterdir()):
                directory.rmdir()
    except OSError as caught_exception:
        return (
            "開始の記録をzipへまとめたが、一部の個別のファイルが残った"
            f"（次の掃引の終わりに重複として消える）: {caught_exception}"
        )
    return None


def recover_interrupted_run(
    output_directory: Path, planned_run: PlannedExperimentRun, run_provenance: RunProvenance
) -> Path:
    """証拠を再確認し、NPZを補助領域へ退避する。呼出し側が出力先の排他を保持する。"""
    if not can_recover_interrupted_run(output_directory, planned_run, run_provenance):
        raise ValueError("interrupted run has no matching clean start record")
    npz = _safe_path(
        output_directory,
        planned_run.configuration_name,
        "runs",
        planned_run.run_identifier + ".npz",
    )
    backup = _safe_path(output_directory, ".resume", "recovered", uuid4().hex, npz.name)
    backup.parent.mkdir(parents=True, exist_ok=True)
    npz.rename(backup)
    return backup


def find_sweep_resumption_candidates(
    results_directory: Path,
    planned_runs: tuple[PlannedExperimentRun, ...],
    run_provenance: RunProvenance,
    read_legacy_plan: Callable[[Path], list[dict] | None],
    find_state: Callable[[PlannedExperimentRun, Path], str],
) -> tuple[SweepResumptionCandidate, ...]:
    """補助計画または旧READMEの全条件を照合し、成果物を変更せず進捗を調べる。"""
    expected = describe_planned_run_set(planned_runs)
    candidates = []
    if _is_link(results_directory):
        return ()
    for current, directories, files in walk(results_directory, followlinks=False):
        root = Path(current)
        directories[:] = [
            name
            for name in directories
            if name not in ("runs", ".resume") and not _is_link(root / name)
        ]
        try:
            auxiliary = _safe_path(root, ".resume", "sweep.json")
            if auxiliary.exists():
                data = loads(auxiliary.read_text(encoding="utf-8"))
                if not _same_document(
                    data,
                    {
                        "version": 1,
                        "behavior_baseline_version": BEHAVIOR_BASELINE_VERSION,
                        "planned_runs": expected,
                    },
                ):
                    continue
                previous = data.get("planned_runs")
            elif "README.md" in files:
                readme = _safe_path(root, "README.md")
                previous = read_legacy_plan(readme)
            else:
                continue
            # 文字列表現でbool/intも区別し、ハッシュだけで同一と扱わない。
            if dumps(previous, sort_keys=True) != dumps(expected, sort_keys=True):
                continue
            completed = recoverable = conflicting = 0
            for run in planned_runs:
                _safe_path(root, run.configuration_name, "runs", run.run_identifier + ".json")
                _safe_path(root, run.configuration_name, "runs", run.run_identifier + ".npz")
                state = find_state(run, root)
                if state == "completed":
                    completed += 1
                elif state != "missing":
                    if can_recover_interrupted_run(root, run, run_provenance):
                        recoverable += 1
                    else:
                        conflicting += 1
            candidates.append(
                SweepResumptionCandidate(
                    root, completed, len(planned_runs), recoverable, conflicting
                )
            )
        except (OSError, ValueError, TypeError):
            # 不明な来歴・壊れた補助計画では旧READMEへfallbackしない。
            continue
    return tuple(sorted(candidates, key=lambda value: str(value.output_directory), reverse=True))
