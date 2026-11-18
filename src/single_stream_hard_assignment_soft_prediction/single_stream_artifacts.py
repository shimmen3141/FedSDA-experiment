"""成果物の保存と読込み。JSON（設定・要約・環境・入力のhash）とNPZ（記録の名前付きの配列）。上書きせず、pickleを使わない。"""

import json
import platform
import sys
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from single_stream_hard_assignment_soft_prediction.single_stream_input import SingleStreamInput
from single_stream_hard_assignment_soft_prediction.single_stream_run import SingleStreamRunResult
from single_stream_hard_assignment_soft_prediction.single_stream_settings import (
    SingleStreamHardAssignmentSoftPredictionSettings,
)

RESULT_FORMAT_VERSION = 1
METHOD_NAME = "single_stream_hard_assignment_soft_prediction_hoeffding_tree"


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _source_sha256() -> str:
    """このパッケージのsourceのhash（ファイル名の順に、名前と内容を連結）。"""
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_single_stream_run(
    *,
    result: SingleStreamRunResult,
    settings: SingleStreamHardAssignmentSoftPredictionSettings,
    stream: SingleStreamInput,
    output_directory: Path,
    file_stem: str,
) -> tuple[Path, Path]:
    """JSONとNPZを書く。どちらかが既にあれば書かずに`FileExistsError`。"""
    output_directory = Path(output_directory)
    json_path = output_directory / f"{file_stem}.json"
    npz_path = output_directory / f"{file_stem}.npz"
    for path in (json_path, npz_path):
        if path.exists():
            raise FileExistsError(f"成果物が既にあります: {path}")
    output_directory.mkdir(parents=True, exist_ok=True)
    document = dict(
        result_format_version=RESULT_FORMAT_VERSION,
        method_name=METHOD_NAME,
        settings=asdict(settings),
        summary=_json_safe(result.summary),
        input=dict(
            dataset_name=stream.dataset_name,
            random_seed=stream.random_seed,
            class_count=stream.class_count,
            sample_count=int(len(stream.labels)),
            array_sha256=stream.array_sha256,
        ),
        environment=dict(
            python=sys.version.split()[0],
            platform=platform.platform(),
            river=_package_version("river"),
            numpy=np.__version__,
            source_sha256=_source_sha256(),
        ),
        record_table_file_name=npz_path.name,
    )
    temporary_npz = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_npz, **result.record_tables)
    temporary_npz.replace(npz_path)
    temporary_json = json_path.with_suffix(".tmp.json")
    temporary_json.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(json_path)
    return json_path, npz_path


def load_single_stream_run(*, json_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """JSONと、それが指すNPZを、pickleを使わずに読む。"""
    json_path = Path(json_path)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    if document.get("method_name") != METHOD_NAME:
        raise ValueError(f"この手法の成果物ではありません: {json_path}")
    with np.load(
        json_path.parent / document["record_table_file_name"], allow_pickle=False
    ) as loaded:
        tables = {name: loaded[name] for name in loaded.files}
    return document, tables
