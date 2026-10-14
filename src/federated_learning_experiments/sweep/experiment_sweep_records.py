"""掃引の結果の、一覧の表（構成ごとのCSV）と、実行の記録（実行ごとのJSON）。"""

from csv import writer
from dataclasses import asdict, dataclass
from datetime import datetime
from json import dumps, loads
from os import environ, getpid, replace
from pathlib import Path
from platform import machine, python_version, system
from subprocess import run
from typing import cast

from numpy import __version__ as numpy_version
from torch import __version__ as torch_version
from torch import get_num_threads

from federated_learning_experiments.core.run_provenance import (
    BEHAVIOR_BASELINE_VERSION,
    RunProvenance,
    read_run_provenance,
)
from federated_learning_experiments.sweep.experiment_sweep_execution import (
    RUN_DIRECTORY_NAME,
    THREAD_COUNT_ENVIRONMENT_VARIABLE_NAMES,
    ExperimentRunOutcome,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import ExperimentSweepPlan

RUN_SUMMARY_TABLE_FILE_NAME = "run_summary.csv"
# 版2: コードの版の記録を、runの来歴から作り、期待する動作の基準版を足した。版3: 目的と研究の問いを足した。
SWEEP_RECORD_FORMAT_VERSION = 3
EXPERIMENT_README_FILE_NAME = "README.md"
# 一覧の表の、実行条件の列（要約の`run_settings.execution_settings.experiment_run_conditions`のキー）。
_CONDITION_COLUMN_NAMES = (
    "dataset_name",
    "random_seed",
    "client_count",
    "per_client_sample_count",
    "server_aggregation_interval_per_client_samples",
)
# 一覧の表の、runの来歴の列（実行条件の列の次。要約の形式に依らず、常に置く）。
_RUN_PROVENANCE_COLUMN_NAMES = (
    "code_commit",
    "has_uncommitted_changes",
    "behavior_baseline_version",
)
# 要約の辞書の中の、型の名前のキー（表には入れない）。
_SETTINGS_TYPE_KEY = "settings_type"
_RUN_STATUSES = ("completed", "skipped_existing", "failed")


@dataclass(frozen=True, kw_only=True)
class RunSummaryTableResult:
    """一覧の表を作った結果。読めなかった要約（表から外したもの）の、ファイルと理由を持つ。"""

    table_file_path: Path
    unreadable_run_summary_descriptions: tuple[str, ...]


def _flatten_metric_values(
    *, metric_values: dict[str, object], name_prefix: str
) -> dict[str, object]:
    """入れ子の指標を、`.`でつないだ名前の、平らな辞書にする。"""
    flat_values: dict[str, object] = {}
    for metric_name, metric_value in metric_values.items():
        if metric_name == _SETTINGS_TYPE_KEY:
            continue
        if isinstance(metric_value, dict):
            flat_values.update(
                _flatten_metric_values(
                    metric_values=metric_value, name_prefix=f"{name_prefix}{metric_name}."
                )
            )
        else:
            flat_values[f"{name_prefix}{metric_name}"] = metric_value
    return flat_values


def _format_table_value(table_value: object) -> object:
    """値のない指標（null）は空の欄、列（list）はJSONの文字列にする。"""
    if table_value is None:
        return ""
    if isinstance(table_value, list):
        return dumps(table_value)
    return table_value


def write_run_summary_table(
    *, configuration_directory: Path, configuration_name: str
) -> RunSummaryTableResult:
    """構成のディレクトリの`runs/`にある全部の要約から、1 runを1行とする表（CSV）を作り直す。

    列: 識別子、構成の名前、実行条件、runの来歴（commit、未commitの変更の有無、基準版。来歴を持たない前の形式の
    要約は、空の欄）、指標の全部（入れ子の指標は、`.`でつないだ名前）。
    行の順: dataset、集約間隔、seed、識別子。一時の名前へ書いてから、置き換える。
    読めない要約（JSONでない、要る項目がない、形式の版2なのに来歴が無いか壊れている）は、表から外して、
    ファイルと理由を返す。
    """
    if not isinstance(configuration_directory, Path):
        raise TypeError("configuration_directory must be pathlib.Path")
    if type(configuration_name) is not str:
        raise TypeError("configuration_name must be builtin str")
    run_directory = configuration_directory / RUN_DIRECTORY_NAME
    if not run_directory.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_directory}")
    table_rows: list[dict[str, object]] = []
    unreadable_run_summary_descriptions: list[str] = []
    for summary_file_path in sorted(run_directory.glob("*.json")):
        try:
            summary = loads(summary_file_path.read_text(encoding="utf-8"))
            experiment_run_conditions = summary["run_settings"]["execution_settings"][
                "experiment_run_conditions"
            ]
            for condition_name in _CONDITION_COLUMN_NAMES:
                # 行の並べ替えに使う値の型を確かめる（datasetは文字列、ほかは整数）。
                expected_condition_type = str if condition_name == "dataset_name" else int
                if type(experiment_run_conditions[condition_name]) is not expected_condition_type:
                    raise TypeError(f"{condition_name} must be {expected_condition_type.__name__}")
            if summary.get("format_version") == 1:
                # 読込みと同じく、版1の要約は、来歴を持たない。
                if "run_provenance" in summary:
                    raise ValueError("run summary of format version 1 must not hold run_provenance")
                run_provenance_values: dict[str, object] = {
                    column_name: None for column_name in _RUN_PROVENANCE_COLUMN_NAMES
                }
            else:
                run_provenance = read_run_provenance(summary.get("run_provenance"))
                run_provenance_values = {
                    column_name: getattr(run_provenance, column_name)
                    for column_name in _RUN_PROVENANCE_COLUMN_NAMES
                }
            table_row = {
                "run_identifier": summary_file_path.stem,
                "configuration_name": configuration_name,
                **{
                    condition_name: experiment_run_conditions[condition_name]
                    for condition_name in _CONDITION_COLUMN_NAMES
                },
                **run_provenance_values,
                **_flatten_metric_values(metric_values=summary["run_metrics"], name_prefix=""),
            }
            if configuration_name == "feddrift":
                client_settings = summary["run_settings"]["run_participant_settings"][
                    "run_client_settings"
                ]
                for name in ("detection_batch_sample_count", "loss_increase_threshold"):
                    table_row[name] = client_settings[name]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as caught_exception:
            unreadable_run_summary_descriptions.append(
                f"{summary_file_path}: {type(caught_exception).__name__}: {caught_exception}"
            )
            continue
        table_rows.append(table_row)
    table_rows.sort(
        key=lambda table_row: (
            table_row["dataset_name"],
            table_row["server_aggregation_interval_per_client_samples"],
            table_row["random_seed"],
            table_row["run_identifier"],
        )
    )
    column_names: list[str] = []
    for table_row in table_rows:
        for column_name in table_row:
            if column_name not in column_names:
                column_names.append(column_name)
    table_file_path = configuration_directory / RUN_SUMMARY_TABLE_FILE_NAME
    temporary_file_path = configuration_directory / (
        f".{RUN_SUMMARY_TABLE_FILE_NAME}.{getpid()}.tmp"
    )
    try:
        with temporary_file_path.open("w", encoding="utf-8", newline="") as table_file:
            table_writer = writer(table_file)
            table_writer.writerow(column_names)
            for table_row in table_rows:
                table_writer.writerow(
                    [
                        _format_table_value(table_row.get(column_name))
                        for column_name in column_names
                    ]
                )
        replace(temporary_file_path, table_file_path)
    finally:
        temporary_file_path.unlink(missing_ok=True)
    return RunSummaryTableResult(
        table_file_path=table_file_path,
        unreadable_run_summary_descriptions=tuple(unreadable_run_summary_descriptions),
    )


def describe_code_revision(*, repository_directory: Path) -> dict[str, object]:
    """コードの版（gitのcommitと、未コミットの変更の有無）を返す。取れなければ、理由を返す。

    未コミットの変更は、gitが追跡するファイルだけで判定する（追跡しないファイルは数えない）。
    """
    if not isinstance(repository_directory, Path):
        raise TypeError("repository_directory must be pathlib.Path")

    def describe_unavailable(reason: str) -> dict[str, object]:
        return {"commit": None, "has_uncommitted_changes": None, "unavailable_reason": reason}

    try:
        # 上位のディレクトリのリポジトリを拾わないように、この場所が、作業ツリーの最上位であることを確かめる。
        top_level_process = run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repository_directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if top_level_process.returncode != 0:
            return describe_unavailable(
                f"not a git work tree: {top_level_process.stderr.strip() or repository_directory}"
            )
        if Path(top_level_process.stdout.strip()).resolve() != repository_directory.resolve():
            return describe_unavailable(
                f"not the top level of a git work tree: {repository_directory}"
            )
        commit_process = run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_directory,
            capture_output=True,
            text=True,
            check=False,
        )
        status_process = run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_directory,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as caught_exception:
        return describe_unavailable(
            f"git could not be run: {type(caught_exception).__name__}: {caught_exception}"
        )
    if commit_process.returncode != 0 or status_process.returncode != 0:
        return describe_unavailable(
            "git failed: " + (commit_process.stderr.strip() or status_process.stderr.strip())
        )
    return {
        "commit": commit_process.stdout.strip(),
        "has_uncommitted_changes": bool(status_process.stdout.strip()),
        "unavailable_reason": None,
    }


def build_run_provenance(*, repository_directory: Path) -> RunProvenance:
    """gitの版（`describe_code_revision`）と、今のコードの基準版から、runの来歴を組み立てる。"""
    code_revision = describe_code_revision(repository_directory=repository_directory)
    return RunProvenance(
        code_commit=cast("str | None", code_revision["commit"]),
        has_uncommitted_changes=cast("bool | None", code_revision["has_uncommitted_changes"]),
        code_revision_unavailable_reason=cast("str | None", code_revision["unavailable_reason"]),
        behavior_baseline_version=BEHAVIOR_BASELINE_VERSION,
    )


def describe_run_provenance_for_record(run_provenance: RunProvenance) -> dict[str, object]:
    """実行の記録へ書く、コードの版の辞書（`describe_code_revision`と同じキー）と基準版。"""
    if type(run_provenance) is not RunProvenance:
        raise TypeError("run_provenance must be exact RunProvenance")
    return {
        "code_revision": {
            "commit": run_provenance.code_commit,
            "has_uncommitted_changes": run_provenance.has_uncommitted_changes,
            "unavailable_reason": run_provenance.code_revision_unavailable_reason,
        },
        "behavior_baseline_version": run_provenance.behavior_baseline_version,
    }


def write_experiment_readme(
    *,
    output_directory: Path,
    purpose: str,
    study_name: str | None,
    started_at: str,
    command_arguments: tuple[str, ...],
    run_provenance: RunProvenance,
) -> Path | None:
    """結果のディレクトリに、目的と出どころを書いた`README.md`を作る。既にあれば、変えずにNoneを返す。

    出力のディレクトリは、作らない。結論の欄は、人が書く（未記入の案内だけを置く）。
    """
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    if type(purpose) is not str or not purpose.strip():
        raise ValueError("purpose must be a non-blank builtin str")
    if study_name is not None and type(study_name) is not str:
        raise TypeError("study_name must be builtin str or None")
    datetime.fromisoformat(started_at)
    if type(command_arguments) is not tuple or any(
        type(command_argument) is not str for command_argument in command_arguments
    ):
        raise TypeError("command_arguments must be exact tuple of builtin str")
    if type(run_provenance) is not RunProvenance:
        raise TypeError("run_provenance must be exact RunProvenance")
    if not output_directory.is_dir():
        raise FileNotFoundError(f"output_directory does not exist: {output_directory}")
    code_commit_text = (
        run_provenance.code_commit
        if run_provenance.code_commit is not None
        else f"（取れなかった: {run_provenance.code_revision_unavailable_reason}）"
    )
    readme_text = "\n".join(
        (
            f"# {output_directory.name}",
            "",
            f"- 目的: {purpose}",
            f"- 研究の問い: {study_name if study_name is not None else 'なし'}",
            f"- 作成の日時: {started_at}",
            f"- commit: {code_commit_text}",
            f"- 未commitの変更（gitが追跡するファイル）: {run_provenance.has_uncommitted_changes}",
            f"- 期待する動作の基準版: {run_provenance.behavior_baseline_version}",
            "",
            "## コマンドの引数",
            "",
            "```",
            " ".join(command_arguments),
            "```",
            "",
            "## 結論",
            "",
            "（未記入。結論が出たら、ここと、`docs/experiments/experiment-log.md`の台帳に書く）",
            "",
        )
    )
    readme_file_path = output_directory / EXPERIMENT_README_FILE_NAME
    try:
        # 排他的に作る（既にあれば、人が書いた結論を消さないように、変えない）。
        with readme_file_path.open("x", encoding="utf-8", newline="\n") as readme_file:
            readme_file.write(readme_text)
    except FileExistsError:
        return None
    return readme_file_path


def write_sweep_record(
    *,
    output_directory: Path,
    started_at: str,
    finished_at: str,
    command_arguments: tuple[str, ...],
    sweep_plan: ExperimentSweepPlan,
    worker_count: int,
    outcomes: tuple[ExperimentRunOutcome, ...],
    run_provenance: RunProvenance,
    purpose: str,
    study_name: str | None,
    unreadable_run_summary_descriptions: tuple[str, ...] = (),
) -> Path:
    """1回の実行の記録を、`sweep_record_<開始時刻>.json`へ書く。

    同じ名前があれば、上書きせず、`_2`・`_3`…をつけた名前にする。

    時刻は、ISO 8601の文字列。
    """
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    for time_name, time_text in (("started_at", started_at), ("finished_at", finished_at)):
        if type(time_text) is not str:
            raise TypeError(f"{time_name} must be builtin str")
    started_time = datetime.fromisoformat(started_at)
    datetime.fromisoformat(finished_at)
    if type(command_arguments) is not tuple or any(
        type(command_argument) is not str for command_argument in command_arguments
    ):
        raise TypeError("command_arguments must be exact tuple of builtin str")
    if type(sweep_plan) is not ExperimentSweepPlan:
        raise TypeError("sweep_plan must be exact ExperimentSweepPlan")
    if type(worker_count) is not int:
        raise TypeError("worker_count must be builtin int")
    if type(outcomes) is not tuple or any(
        type(outcome) is not ExperimentRunOutcome for outcome in outcomes
    ):
        raise TypeError("outcomes must be exact tuple of ExperimentRunOutcome")
    if type(run_provenance) is not RunProvenance:
        raise TypeError("run_provenance must be exact RunProvenance")
    if type(purpose) is not str or not purpose.strip():
        raise ValueError("purpose must be a non-blank builtin str")
    if study_name is not None and type(study_name) is not str:
        raise TypeError("study_name must be builtin str or None")
    if type(unreadable_run_summary_descriptions) is not tuple or any(
        type(description) is not str for description in unreadable_run_summary_descriptions
    ):
        raise TypeError("unreadable_run_summary_descriptions must be exact tuple of builtin str")
    record_text = dumps(
        {
            "record_format_version": SWEEP_RECORD_FORMAT_VERSION,
            "started_at": started_at,
            "finished_at": finished_at,
            "purpose": purpose,
            "study_name": study_name,
            "command_arguments": list(command_arguments),
            "sweep_plan": asdict(sweep_plan),
            "worker_count": worker_count,
            **describe_run_provenance_for_record(run_provenance),
            "environment": {
                "system": system(),
                "machine": machine(),
                "python": python_version(),
                "numpy": numpy_version,
                "torch": torch_version,
                # この記録を書いたprocess（コマンドのprocess）の値。runの実行の間は、どのprocessでも、
                # torchのthreadを1つにする（この値には、現れない）。
                "torch_thread_count": get_num_threads(),
                "thread_count_environment_variables": {
                    variable_name: environ.get(variable_name)
                    for variable_name in THREAD_COUNT_ENVIRONMENT_VARIABLE_NAMES
                },
            },
            "run_status_counts": {
                run_status: sum(outcome.run_status == run_status for outcome in outcomes)
                for run_status in _RUN_STATUSES
            },
            "run_outcomes": [asdict(outcome) for outcome in outcomes],
            "unreadable_run_summary_descriptions": list(unreadable_run_summary_descriptions),
        },
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    record_file_stem = f"sweep_record_{started_time.strftime('%Y%m%d_%H%M%S')}"
    record_number = 1
    while True:
        record_file_path = output_directory / (
            record_file_stem + ("" if record_number == 1 else f"_{record_number}") + ".json"
        )
        try:
            # 排他的に作る（同じ名前があれば、次の番号を試す）。
            with record_file_path.open("x", encoding="utf-8", newline="\n") as record_file:
                record_file.write(record_text + "\n")
        except FileExistsError:
            record_number += 1
            continue
        break
    return record_file_path
