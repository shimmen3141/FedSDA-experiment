"""済んだ掃引の結果を、情報を保ったまま、構成ごとに1つのzip（`runs.zip`）へまとめ、元に戻す研究ツール。

「情報を保つ」の範囲: NPZ以外のファイルはバイト列まで元と同じに戻る。NPZは配列（名前・型・形・値）が元と同じに戻るが、
ファイルのバイト列は元と違う（中の圧縮がLZMAになる。元のsha256はmanifestに記録として残す）。

注意: まとめた構成のフォルダには`runs/`が無いので、同じ出力先で掃引を流し直すと、全runを計算し直す。流し直す前に`unpack`する。

まとめる単位: `runs/`を持つ構成のフォルダ（例: `<掃引>/fedsda/`）。`runs/`の全ファイルと、掃引のフォルダの
`.resume/<構成の名前>/`（あれば）を、構成のフォルダの`runs.zip`へ入れる。`run_summary.csv`・README・掃引の記録は残す。
- JSONなどNPZ以外: バイト列をそのまま、zipのLZMAで圧縮して入れる。
- NPZ: 配列（名前・型・形・値）を変えずに、中の各配列をLZMAで圧縮したNPZに書き直し、zipには無圧縮で入れる
  （numpyの`np.load`はLZMAのNPZも読める）。ファイルのバイト列は元と変わるが、読んだ配列は同じ。
- zipの中の`archive_manifest.json`に、元の各ファイルのパス・バイト数・sha256と、NPZなら配列の名前・型・形を記録する。

検査（`pack`）: 書いたzipを読み戻し、NPZ以外はsha256、NPZは配列の名前・型・形・値（浮動小数のNaNは同じとみなす）が
元と一致することを全ファイルで確かめてから、元の`runs/`と`.resume/<構成>/`を消す。一致しなければ、zipを消して止める。
`run_summary.csv`が無い、その行数と`runs/`のrunの数（JSONの数）が違う、または、掃引のフォルダの最新の掃引の記録
（`sweep_record_*.json`）に失敗したrunがある構成は、済んでいないとみなして飛ばす（掃引の記録が無い古い結果は前の2つだけで判断する）。

戻す（`unpack`）: zipから元のパスへ書き戻し、manifestと照合（NPZ以外はsha256、NPZは配列の名前・型・形）してからzipを消す。

使い方（worktreeのルート、プロジェクトのvenv）:
    python -m tools.research_results_archive.archive pack <結果のフォルダ> [--dry-run] [--workers N]
    python -m tools.research_results_archive.archive unpack <結果のフォルダ> [--workers N]
`<結果のフォルダ>`の下を再帰的に探し、見つけた構成のフォルダを処理する。`--workers`が2以上なら、構成のフォルダを
別々のprocessで並列に処理する（1つの構成のフォルダの中は1つのprocess）。LZMAの圧縮は1 processで約0.7 MB/秒
（2026-11-20の手元の実測、1,369 MBで32分）。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

# run_summary.csvは長い欄（列の値を並べた文字列など）を持つことがある。
csv.field_size_limit(10**9)
ARCHIVE_NAME = "runs.zip"
MANIFEST_NAME = "archive_manifest.json"
_RESUME_DIRECTORY = ".resume"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_npz_arrays(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _lzma_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """配列を変えずに、各配列をLZMAで圧縮したNPZのバイト列を作る（`np.savez`と同じ`<名前>.npy`の並び）。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_LZMA) as archive:
        for name, array in arrays.items():
            array_buffer = io.BytesIO()
            np.lib.format.write_array(array_buffer, array, allow_pickle=False)
            archive.writestr(f"{name}.npy", array_buffer.getvalue())
    return buffer.getvalue()


def _array_signature(arrays: dict[str, np.ndarray]) -> list:
    return [[name, array.dtype.str, list(array.shape)] for name, array in arrays.items()]


def _same_arrays(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> bool:
    if list(first) != list(second):
        return False
    for name, array in first.items():
        other = second[name]
        if array.dtype != other.dtype or array.shape != other.shape:
            return False
        equal_nan = array.dtype.kind in "fc"
        if not np.array_equal(array, other, equal_nan=equal_nan):
            return False
    return True


def find_configuration_directories(root: Path) -> list[Path]:
    """`runs/`か`runs.zip`を持つフォルダ（構成のフォルダ）を、パスの順に返す。"""
    found = {path.parent for path in root.rglob("runs") if path.is_dir()}
    found |= {path.parent for path in root.rglob(ARCHIVE_NAME) if path.is_file()}
    return sorted(found)


def _resume_directory(configuration_directory: Path) -> Path:
    return configuration_directory.parent / _RESUME_DIRECTORY / configuration_directory.name


def _incomplete_reason(configuration_directory: Path) -> str | None:
    summary = configuration_directory / "run_summary.csv"
    if not summary.is_file():
        return "run_summary.csvがありません（掃引が済んでいない）"
    with summary.open(encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    json_count = sum(1 for path in (configuration_directory / "runs").glob("*.json"))
    if row_count != json_count:
        return f"run_summary.csvの行数（{row_count}）とrunsのJSONの数（{json_count}）が違います"
    records = sorted(configuration_directory.parent.glob("sweep_record_*.json"))
    if records:
        latest = json.loads(records[-1].read_text(encoding="utf-8"))
        failed = latest.get("run_status_counts", {}).get("failed", 0)
        if failed:
            return f"最新の掃引の記録（{records[-1].name}）に失敗したrunが{failed}件あります"
    return None


def _collect_files(configuration_directory: Path) -> list[tuple[str, Path]]:
    """（zipの中のパス、元のファイル）の組。zipの中のパスは構成のフォルダからの相対（`.resume`は`.resume/`から）。"""
    files = [
        (path.relative_to(configuration_directory).as_posix(), path)
        for path in sorted((configuration_directory / "runs").rglob("*"))
        if path.is_file()
    ]
    resume = _resume_directory(configuration_directory)
    if resume.is_dir():
        files += [
            (f"{_RESUME_DIRECTORY}/{path.relative_to(resume).as_posix()}", path)
            for path in sorted(resume.rglob("*"))
            if path.is_file()
        ]
    return files


def pack_configuration(configuration_directory: Path, *, dry_run: bool) -> dict:
    """1つの構成のフォルダをまとめる。結果の要約（ファイル数・バイト数）を返す。"""
    archive_path = configuration_directory / ARCHIVE_NAME
    if archive_path.exists():
        raise FileExistsError(f"すでにあります: {archive_path}")
    files = _collect_files(configuration_directory)
    original_bytes = sum(path.stat().st_size for _, path in files)
    if dry_run:
        return dict(file_count=len(files), original_bytes=original_bytes, archive_bytes=None)
    manifest = []
    temporary_path = archive_path.with_suffix(".zip.tmp")
    try:
        _write_archive(temporary_path, files, manifest)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    # 読み戻して、全ファイルを元と照合する。
    _verify_and_replace(temporary_path, archive_path, files)
    shutil.rmtree(configuration_directory / "runs")
    resume = _resume_directory(configuration_directory)
    if resume.is_dir():
        shutil.rmtree(resume)
        if not any(resume.parent.iterdir()):
            resume.parent.rmdir()
    return dict(file_count=len(files), original_bytes=original_bytes, archive_bytes=archive_path.stat().st_size)


def _write_archive(temporary_path: Path, files, manifest: list) -> None:
    with zipfile.ZipFile(temporary_path, "w") as archive:
        for inner_path, path in files:
            data = path.read_bytes()
            entry = dict(path=inner_path, byte_count=len(data), sha256=_sha256(data))
            if path.suffix == ".npz":
                arrays = _read_npz_arrays(data)
                entry["arrays"] = _array_signature(arrays)
                archive.writestr(inner_path, _lzma_npz_bytes(arrays), compress_type=zipfile.ZIP_STORED)
            else:
                archive.writestr(inner_path, data, compress_type=zipfile.ZIP_LZMA)
            manifest.append(entry)
        archive.writestr(
            MANIFEST_NAME,
            json.dumps(dict(format_version=1, files=manifest), ensure_ascii=False, indent=1),
            compress_type=zipfile.ZIP_LZMA,
        )


def _verify_and_replace(temporary_path: Path, archive_path: Path, files) -> None:
    """書いたzipを読み戻し、全ファイルを元と照合してから置く。一致しなければ、zipを消して止める。"""
    try:
        with zipfile.ZipFile(temporary_path) as archive:
            names = set(archive.namelist()) - {MANIFEST_NAME}
            if names != {inner_path for inner_path, _ in files}:
                raise ValueError("zipの中のファイルの一覧が元と違います")
            for inner_path, path in files:
                packed = archive.read(inner_path)
                original = path.read_bytes()
                if path.suffix == ".npz":
                    if not _same_arrays(_read_npz_arrays(original), _read_npz_arrays(packed)):
                        raise ValueError(f"NPZの配列が元と違います: {path}")
                elif packed != original:
                    raise ValueError(f"ファイルの内容が元と違います: {path}")
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    temporary_path.replace(archive_path)


def unpack_configuration(configuration_directory: Path) -> dict:
    """`runs.zip`を元のパスへ書き戻し、manifestと照合してからzipを消す。"""
    archive_path = configuration_directory / ARCHIVE_NAME
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(MANIFEST_NAME))["files"]
        targets = []
        for entry in manifest:
            inner_path = entry["path"]
            if inner_path.startswith(f"{_RESUME_DIRECTORY}/"):
                target = _resume_directory(configuration_directory) / inner_path[len(_RESUME_DIRECTORY) + 1 :]
            else:
                target = configuration_directory / inner_path
            if target.exists():
                raise FileExistsError(f"戻す先にすでにあります: {target}")
            targets.append((entry, target))
        for entry, target in targets:
            data = archive.read(entry["path"])
            if "arrays" in entry:
                if _array_signature(_read_npz_arrays(data)) != entry["arrays"]:
                    raise ValueError(f"NPZの配列の名前・型・形がmanifestと違います: {entry['path']}")
            elif _sha256(data) != entry["sha256"]:
                raise ValueError(f"sha256がmanifestと違います: {entry['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    archive_path.unlink()
    return dict(file_count=len(manifest))


def _process(task):
    action, configuration_directory, dry_run = task
    if action == "pack":
        if (configuration_directory / ARCHIVE_NAME).exists():
            return configuration_directory, "skip（まとめ済み）", None
        reason = _incomplete_reason(configuration_directory)
        if reason is not None:
            return configuration_directory, f"skip（{reason}）", None
        return configuration_directory, "pack", pack_configuration(configuration_directory, dry_run=dry_run)
    if not (configuration_directory / ARCHIVE_NAME).exists():
        return configuration_directory, "skip（zipが無い）", None
    return configuration_directory, "unpack", unpack_configuration(configuration_directory)


def main(command_arguments=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("pack", "unpack"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="packで、まとめる対象と元のバイト数だけを表示する")
    parser.add_argument("--workers", type=int, default=1, help="構成のフォルダを並列に処理するprocessの数")
    arguments = parser.parse_args(command_arguments)
    if arguments.workers < 1:
        parser.error("--workersは1以上")
    tasks = [
        (arguments.action, configuration_directory, arguments.dry_run)
        for configuration_directory in find_configuration_directories(arguments.root)
    ]
    if arguments.workers == 1 or len(tasks) <= 1:
        results = map(_process, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=arguments.workers)
        results = executor.map(_process, tasks)
    total_original = total_archive = 0
    try:
        for configuration_directory, status, result in results:
            if result is None:
                print(f"{status} {configuration_directory}", flush=True)
                continue
            if status == "unpack":
                print(f"unpack {configuration_directory}: {result['file_count']} files", flush=True)
                continue
            total_original += result["original_bytes"]
            archive_text = ""
            if result["archive_bytes"] is not None:
                total_archive += result["archive_bytes"]
                archive_text = f" → {result['archive_bytes'] / 1e6:.1f} MB"
            print(
                f"pack {configuration_directory}: {result['file_count']} files, "
                f"{result['original_bytes'] / 1e6:.1f} MB{archive_text}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if arguments.action == "pack":
        print(f"total: {total_original / 1e6:.1f} MB" + ("" if arguments.dry_run else f" → {total_archive / 1e6:.1f} MB"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
