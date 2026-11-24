"""構成ごとのrunの成果物を、バイト列を保ってまとめ、照合して戻す。"""

from dataclasses import dataclass
from hashlib import sha256
from json import dumps, loads
from os import fsync
from os import replace as replace_file
from pathlib import Path
from time import sleep
from uuid import uuid4
from zipfile import ZIP_LZMA, ZIP_STORED, ZipFile, ZipInfo

RUN_ARTIFACT_ARCHIVE_NAME = "runs.zip"
_RUN_DIRECTORY_NAME = "runs"
_MANIFEST_NAME = "archive_manifest.json"
_CHUNK_BYTE_COUNT = 1024 * 1024


@dataclass(frozen=True)
class ConfigurationRunArtifactPackingResult:
    """zipの配置、まとめなかった理由、配置後の削除の警告。"""

    archive_placed: bool
    not_packed_reason: str | None
    warning_description: str | None


@dataclass(frozen=True)
class RunArtifactPackingResult:
    """まとめた構成と、呼出し側へ知らせる警告。"""

    packed_configuration_names: tuple[str, ...]
    warning_descriptions: tuple[str, ...]


def _is_link(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _check_paths(configuration_directory: Path) -> None:
    root = configuration_directory.resolve()
    for path in (
        configuration_directory,
        configuration_directory / _RUN_DIRECTORY_NAME,
        configuration_directory / RUN_ARTIFACT_ARCHIVE_NAME,
    ):
        if _is_link(path) or not path.resolve().is_relative_to(root):
            raise ValueError(f"run artifact path must not traverse links: {path}")


def _replace_with_retries(source: Path, destination: Path) -> None:
    for attempt in range(8):
        try:
            replace_file(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            sleep(0.5 * (attempt + 1))


def _copy_and_describe(source, destination=None) -> tuple[int, str]:
    digest = sha256()
    byte_count = 0
    while chunk := source.read(_CHUNK_BYTE_COUNT):
        digest.update(chunk)
        byte_count += len(chunk)
        if destination is not None:
            destination.write(chunk)
    return byte_count, digest.hexdigest()


def _check_description(description: tuple[int, str], entry: dict) -> None:
    if description != (entry["byte_count"], entry["sha256"]):
        raise ValueError(f"run artifact differs from manifest: {entry['path']}")


def _check_pairs(names) -> frozenset[str]:
    summaries = set()
    tables = set()
    for name in names:
        if not name or "/" in name or "\\" in name or ".." in name:
            raise ValueError(f"invalid run artifact name: {name}")
        if name.endswith(".json") and name[:-5]:
            summaries.add(name[:-5])
        elif name.endswith(".npz") and name[:-4]:
            tables.add(name[:-4])
        else:
            raise ValueError(f"unexpected run artifact name: {name}")
    if summaries != tables:
        raise ValueError("run artifacts must contain summary and record table pairs")
    return frozenset(summaries)


def _read_manifest(archive: ZipFile) -> list[dict]:
    manifest = loads(archive.read(_MANIFEST_NAME))
    if type(manifest) is not dict or manifest.get("format_version") != 1:
        raise ValueError("unsupported run artifact archive manifest")
    entries = manifest.get("files")
    if type(entries) is not list:
        raise ValueError("invalid run artifact archive manifest files")
    paths = []
    for entry in entries:
        if type(entry) is not dict:
            raise ValueError("invalid run artifact archive manifest entry")
        if "arrays" in entry:
            raise ValueError("研究ツールでまとめたzipなので、unpackしてから流し直してください")
        path = entry.get("path")
        if type(path) is not str or not path.startswith("runs/"):
            raise ValueError(f"invalid run artifact path: {path}")
        if type(entry.get("byte_count")) is not int or entry["byte_count"] < 0:
            raise ValueError("invalid run artifact byte_count")
        digest = entry.get("sha256")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("invalid run artifact sha256")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate run artifact manifest paths")
    _check_pairs(path[5:] for path in paths)
    members = archive.namelist()
    if len(members) != len(set(members)) or set(members) != set(paths) | {_MANIFEST_NAME}:
        raise ValueError("run artifact archive members differ from manifest")
    return entries


def _archive_error(exception: Exception) -> ValueError:
    if isinstance(exception, RuntimeError) and "lzma" in str(exception).lower():
        return ValueError(f"LZMAが使えないので読めない: {exception}")
    return ValueError(
        f"run artifact archive cannot be read: {type(exception).__name__}: {exception}"
    )


def _verify_archive(archive_path: Path, expected: list[dict]) -> None:
    with ZipFile(archive_path) as archive:
        entries = _read_manifest(archive)
        if entries != expected:
            raise ValueError("run artifact manifest differs from source records")
        for entry in entries:
            with archive.open(entry["path"]) as source:
                _check_description(_copy_and_describe(source), entry)


def pack_configuration_run_artifacts(
    configuration_directory: Path,
) -> ConfigurationRunArtifactPackingResult:
    """1つの構成を照合してまとめる。配置前の失敗では元を残す。"""
    archive_path = configuration_directory / RUN_ARTIFACT_ARCHIVE_NAME
    temporary = configuration_directory / (RUN_ARTIFACT_ARCHIVE_NAME + ".partial")
    runs = configuration_directory / _RUN_DIRECTORY_NAME
    placed = False
    temporary_owned = False
    try:
        _check_paths(configuration_directory)
        if not runs.is_dir() or archive_path.exists():
            raise ValueError("runs must be a directory and runs.zip must not already exist")
        files = sorted(runs.iterdir())
        if any(_is_link(path) or not path.is_file() for path in files):
            raise ValueError("runs must contain only regular files")
        _check_pairs(path.name for path in files)
        if not files:
            return ConfigurationRunArtifactPackingResult(False, None, None)
        if _is_link(temporary):
            raise ValueError("partial archive must not be a link")
        temporary.unlink(missing_ok=True)
        temporary_owned = True
        entries = []
        with temporary.open("xb") as stream:
            with ZipFile(stream, "w") as archive:
                for path in files:
                    inner_path = f"{_RUN_DIRECTORY_NAME}/{path.name}"
                    info = ZipInfo(inner_path)
                    info.compress_type = ZIP_LZMA if path.suffix == ".json" else ZIP_STORED
                    with (
                        path.open("rb") as source,
                        archive.open(info, "w", force_zip64=True) as destination,
                    ):
                        byte_count, digest = _copy_and_describe(source, destination)
                    entries.append(dict(path=inner_path, byte_count=byte_count, sha256=digest))
                info = ZipInfo(_MANIFEST_NAME)
                info.compress_type = ZIP_LZMA
                archive.writestr(info, dumps(dict(format_version=1, files=entries)))
            stream.flush()
            fsync(stream.fileno())
        _verify_archive(temporary, entries)
        for path, entry in zip(files, entries, strict=True):
            with path.open("rb") as source:
                _check_description(_copy_and_describe(source), entry)
        _replace_with_retries(temporary, archive_path)
        placed = True
        for path in files:
            path.unlink()
        runs.rmdir()
        return ConfigurationRunArtifactPackingResult(True, None, None)
    except Exception as exception:
        reason = f"{configuration_directory.name}: {type(exception).__name__}: {exception}"
        if placed:
            return ConfigurationRunArtifactPackingResult(True, None, reason)
        if temporary_owned:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_exception:
                reason += f"; partial cleanup failed: {cleanup_exception}"
        return ConfigurationRunArtifactPackingResult(False, reason, None)


def pack_completed_sweep_run_artifacts(
    output_directory: Path,
    configuration_names: tuple[str, ...],
) -> RunArtifactPackingResult:
    """呼出し側が完了を確認し、排他を持つ掃引の成果物をまとめる。"""
    packed = []
    warnings = []
    for name in configuration_names:
        directory = output_directory / name
        runs = directory / _RUN_DIRECTORY_NAME
        if not runs.exists() and not _is_link(runs):
            continue
        result = pack_configuration_run_artifacts(directory)
        if result.archive_placed:
            packed.append(name)
        warnings.extend(
            reason
            for reason in (result.not_packed_reason, result.warning_description)
            if reason is not None
        )
    return RunArtifactPackingResult(tuple(packed), tuple(warnings))


def read_packed_run_summary_and_record_table(
    configuration_directory: Path,
    run_identifier: str,
) -> tuple[bytes, bytes] | None:
    """manifestと照合した1 runのバイト列。無ければNone。"""
    try:
        _check_paths(configuration_directory)
        with ZipFile(configuration_directory / RUN_ARTIFACT_ARCHIVE_NAME) as archive:
            entries = {entry["path"]: entry for entry in _read_manifest(archive)}
            names = [f"runs/{run_identifier}{suffix}" for suffix in (".json", ".npz")]
            if not any(name in entries for name in names):
                return None
            data = []
            for name in names:
                content = archive.read(name)
                _check_description((len(content), sha256(content).hexdigest()), entries[name])
                data.append(content)
            return data[0], data[1]
    except OSError:
        raise
    except Exception as exception:
        raise _archive_error(exception) from exception


def list_packed_run_identifiers(configuration_directory: Path) -> frozenset[str]:
    """manifestに対で記録されたrunの識別子。"""
    try:
        _check_paths(configuration_directory)
        with ZipFile(configuration_directory / RUN_ARTIFACT_ARCHIVE_NAME) as archive:
            entries = _read_manifest(archive)
            return _check_pairs(entry["path"][5:] for entry in entries)
    except OSError:
        raise
    except Exception as exception:
        raise _archive_error(exception) from exception


def restore_packed_run_artifacts(configuration_directory: Path) -> None:
    """全項目を検査した後で戻す。途中のI/O失敗では照合済みのファイルを残す。"""
    archive_path = configuration_directory / RUN_ARTIFACT_ARCHIVE_NAME
    runs = configuration_directory / _RUN_DIRECTORY_NAME
    temporary = None
    try:
        _check_paths(configuration_directory)
        if runs.exists() and not runs.is_dir():
            raise ValueError("runs must be a directory")
        with ZipFile(archive_path) as archive:
            entries = _read_manifest(archive)
            # 段1: 既存の衝突と全memberの破損を、何も書かずに確認する。
            for entry in entries:
                path = configuration_directory / entry["path"]
                if _is_link(path):
                    raise ValueError(f"run artifact must not be a link: {path}")
                if path.exists():
                    if not path.is_file():
                        raise ValueError(f"run artifact must be a regular file: {path}")
                    with path.open("rb") as source:
                        _check_description(_copy_and_describe(source), entry)
                with archive.open(entry["path"]) as source:
                    _check_description(_copy_and_describe(source), entry)
            names = [Path(entry["path"]).name for entry in entries]
            stale_partials = (
                [
                    path
                    for path in runs.iterdir()
                    if path.name.endswith(".partial")
                    and any(path.name.startswith(name + ".") for name in names)
                ]
                if runs.exists()
                else []
            )
            if any(_is_link(path) or not path.is_file() for path in stale_partials):
                raise ValueError("partial run artifact must be a regular file, not a link")
            # 段2: 検査を通った後だけ、前の中断の一時ファイルを消して戻す。
            runs.mkdir(exist_ok=True)
            for path in stale_partials:
                path.unlink()
            for entry in entries:
                destination = configuration_directory / entry["path"]
                if destination.exists():
                    continue
                temporary = destination.with_name(destination.name + f".{uuid4().hex}.partial")
                with archive.open(entry["path"]) as source, temporary.open("xb") as target:
                    _check_description(_copy_and_describe(source, target), entry)
                    target.flush()
                    fsync(target.fileno())
                _replace_with_retries(temporary, destination)
                temporary = None
        for attempt in range(8):
            try:
                archive_path.unlink()
                break
            except PermissionError as exception:
                if attempt == 7:
                    raise ValueError(
                        f"cannot remove restored run archive: {archive_path}"
                    ) from exception
                sleep(0.5 * (attempt + 1))
    except OSError:
        raise
    except Exception as exception:
        raise _archive_error(exception) from exception
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
