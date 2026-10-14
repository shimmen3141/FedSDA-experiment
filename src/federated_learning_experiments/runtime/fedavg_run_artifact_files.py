"""1 runの結果を、要約（JSON）と記録の表（NPZ）の2つのファイルへ保存し、読み込む。

要約は、形式の版、完全なrun設定の辞書、指標の辞書、実行環境の記録、runの来歴（コードの版と基準版）、
表のファイルの名前を持つ。保存先の名前の決め方、来歴の取り方（git）、重複の確認は、ここでは扱わない。
"""

from dataclasses import dataclass
from json import dumps, loads
from os import replace
from pathlib import Path
from platform import machine, python_version, system
from uuid import uuid4

from numpy import __version__ as numpy_version
from numpy import load, ndarray, savez_compressed
from torch import __version__ as torch_version

from federated_learning_experiments.core.run_provenance import (
    RunProvenance,
    convert_run_provenance_to_mapping,
    read_run_provenance,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.runtime.fedavg_experiment_run import FedavgExperimentRun
from federated_learning_experiments.runtime.fedavg_run_record_tables import (
    build_fedavg_run_record_tables,
)

# 保存の形式の版。要約のキーや、表の名前・列の意味を変えるときに上げる。
# 版2: runの来歴（`run_provenance`）を足した。版1の要約（来歴を持たない）も読む。
RUN_ARTIFACT_FORMAT_VERSION = 2
_READABLE_RUN_ARTIFACT_FORMAT_VERSIONS = (1, 2)
_SUMMARY_FILE_SUFFIX = ".json"
_RECORD_TABLE_FILE_SUFFIX = ".npz"


@dataclass(frozen=True, kw_only=True)
class FedavgRunArtifactPaths:
    """保存した2つのファイルの場所。"""

    summary_file_path: Path
    record_table_file_path: Path


def save_fedavg_experiment_run(
    *,
    experiment_run: FedavgExperimentRun,
    output_directory: Path,
    file_stem: str,
    run_provenance: RunProvenance,
) -> FedavgRunArtifactPaths:
    """1 runの結果を、`<file_stem>.json`と`<file_stem>.npz`へ保存する。

    出力のディレクトリは、作らない。どちらかのファイルがあれば、何も書かずに拒否する。
    一時の名前へ書いてから名前を変え、失敗したら、一時のファイルを消す。
    """
    if type(experiment_run) is not FedavgExperimentRun:
        raise TypeError("experiment_run must be exact FedavgExperimentRun")
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    if type(file_stem) is not str:
        raise TypeError("file_stem must be builtin str")
    if type(run_provenance) is not RunProvenance:
        raise TypeError("run_provenance must be exact RunProvenance")
    if (
        not file_stem
        or file_stem != file_stem.strip()
        or file_stem in (".", "..")
        or any(separator in file_stem for separator in ("/", "\\", ":"))
    ):
        raise ValueError(f"file_stem must be a plain file name without separators: {file_stem!r}")
    if not output_directory.is_dir():
        raise FileNotFoundError(f"output_directory does not exist: {output_directory}")
    artifact_paths = FedavgRunArtifactPaths(
        summary_file_path=output_directory / (file_stem + _SUMMARY_FILE_SUFFIX),
        record_table_file_path=output_directory / (file_stem + _RECORD_TABLE_FILE_SUFFIX),
    )
    for artifact_path in (artifact_paths.summary_file_path, artifact_paths.record_table_file_path):
        if artifact_path.exists():
            raise FileExistsError(f"run artifact already exists: {artifact_path}")
    # 書く内容を、ファイルを作る前に、全部用意する（ここでの失敗は、何も残さない）。
    record_tables = build_fedavg_run_record_tables(experiment_run=experiment_run)
    summary_text = dumps(
        {
            "format_version": RUN_ARTIFACT_FORMAT_VERSION,
            "method_name": "fedavg",
            "run_settings": convert_settings_to_plain_mapping(experiment_run.run_settings),
            # 非有限の値（0で割った値のNaN: 計算量のまとめと、定常の標本がないときの定常精度）は、nullにする。
            "run_metrics": convert_settings_to_plain_mapping(
                experiment_run.run_metrics, non_finite_float_as_none=True
            ),
            "environment": {
                "system": system(),
                "machine": machine(),
                "python": python_version(),
                "numpy": numpy_version,
                "torch": torch_version,
            },
            "run_provenance": convert_run_provenance_to_mapping(run_provenance),
            "record_table_file_name": artifact_paths.record_table_file_path.name,
        },
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    temporary_suffix = f".{uuid4().hex}.partial"
    temporary_summary_file_path = artifact_paths.summary_file_path.with_name(
        artifact_paths.summary_file_path.name + temporary_suffix
    )
    temporary_record_table_file_path = artifact_paths.record_table_file_path.with_name(
        artifact_paths.record_table_file_path.name + temporary_suffix
    )
    try:
        with temporary_record_table_file_path.open("wb") as record_table_stream:
            # 配列の名前（`表の名前.列の名前`）を、そのままキーにする。
            savez_compressed(record_table_stream, **record_tables)  # pyright: ignore[reportArgumentType]
        temporary_summary_file_path.write_text(summary_text + "\n", encoding="utf-8")
        # 表を先に置き、要約を最後に置く（要約があれば、表もある）。
        replace(temporary_record_table_file_path, artifact_paths.record_table_file_path)
        try:
            replace(temporary_summary_file_path, artifact_paths.summary_file_path)
        except BaseException:
            artifact_paths.record_table_file_path.unlink(missing_ok=True)
            raise
    finally:
        temporary_summary_file_path.unlink(missing_ok=True)
        temporary_record_table_file_path.unlink(missing_ok=True)
    return artifact_paths


def load_fedavg_run_artifacts(
    *, summary_file_path: Path
) -> tuple[dict[str, object], dict[str, ndarray]]:
    """保存した要約と、同じディレクトリの記録の表を読む。戻り値: (要約の辞書, `表の名前.列の名前`→配列)。"""
    if not isinstance(summary_file_path, Path):
        raise TypeError("summary_file_path must be pathlib.Path")
    summary = loads(summary_file_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"run summary must be a JSON object: {summary_file_path}")
    format_version = summary.get("format_version")
    if (
        type(format_version) is not int
        or format_version not in _READABLE_RUN_ARTIFACT_FORMAT_VERSIONS
    ):
        raise ValueError(
            f"unsupported run artifact format version in {summary_file_path}: "
            f"{summary.get('format_version')!r} "
            f"(supported: {_READABLE_RUN_ARTIFACT_FORMAT_VERSIONS})"
        )
    if format_version == 1:
        if "run_provenance" in summary:
            raise ValueError(
                f"run summary of format version 1 must not hold run_provenance: {summary_file_path}"
            )
    else:
        try:
            read_run_provenance(summary.get("run_provenance"))
        except (TypeError, ValueError) as caught_exception:
            raise ValueError(
                f"run summary has no valid run_provenance: {summary_file_path} ({caught_exception})"
            ) from caught_exception
    if summary.get("method_name") != "fedavg":
        raise ValueError("run summary must identify fedavg")
    record_table_file_name = summary.get("record_table_file_name")
    if (
        type(record_table_file_name) is not str
        or record_table_file_name in ("", ".", "..")
        or record_table_file_name != record_table_file_name.strip()
        or any(separator in record_table_file_name for separator in ("/", "\\", ":"))
        or Path(record_table_file_name).suffix != _RECORD_TABLE_FILE_SUFFIX
        or Path(record_table_file_name).name != record_table_file_name
    ):
        raise ValueError(f"run summary has no valid record table file name: {summary_file_path}")
    with load(
        summary_file_path.with_name(record_table_file_name), allow_pickle=False
    ) as record_table_file:
        record_tables = {
            table_column_name: record_table_file[table_column_name]
            for table_column_name in record_table_file.files
        }
    return summary, record_tables
