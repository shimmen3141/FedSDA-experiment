"""計画のrunを、逐次か、別々のprocessで並列に実行して、保存する。済んだrunは飛ばす。

出力先の形: `<出力先>/<構成の名前>/runs/<識別子>.json`（要約）と`.npz`（記録の表）。
"""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from json import dumps, loads
from multiprocessing import get_context
from os import environ
from pathlib import Path
from time import perf_counter
from traceback import format_exc

from torch import get_num_threads, set_num_threads

from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.runtime.fedavg_experiment_run import (
    execute_fedavg_experiment_run,
)
from federated_learning_experiments.runtime.fedavg_run_artifact_files import (
    load_fedavg_run_artifacts,
    save_fedavg_experiment_run,
)
from federated_learning_experiments.runtime.fedavg_run_settings import FedavgRunSettings
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_artifact_files import (
    load_feddrift_run_artifacts,
    save_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_artifact_files import (
    load_fedsda_run_artifacts,
    save_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
from federated_learning_experiments.sweep.experiment_sweep_plan import PlannedExperimentRun

# 構成のディレクトリの中の、runのファイルを置くディレクトリの名前。
RUN_DIRECTORY_NAME = "runs"
_SUMMARY_FILE_SUFFIX = ".json"
_RECORD_TABLE_FILE_SUFFIX = ".npz"
# 1 runのprocessが使う、計算のthreadの数。
_RUN_THREAD_COUNT = 1
# 数値計算のライブラリの、threadの数を決める環境変数（workerのprocessを作る前に、未設定なら1にする）。
THREAD_COUNT_ENVIRONMENT_VARIABLE_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


@dataclass(frozen=True, kw_only=True)
class ExistingRunState:
    """出力先の、あるrunのファイルの状態。

    `missing`: ファイルがない。`completed`: 要約と表があり、要約のrun設定が、計画と同じ。
    `conflicting`: ファイルが片方しかない、読めない、または、要約のrun設定が計画と違う（`description`に理由）。
    """

    run_state: str
    description: str | None


@dataclass(frozen=True, kw_only=True)
class ExperimentRunOutcome:
    """1 runの結果。`run_status`は、`completed`・`skipped_existing`・`failed`のどれか。"""

    configuration_name: str
    run_identifier: str
    run_status: str
    failure_description: str | None


def _get_run_directory(*, planned_run: PlannedExperimentRun, output_directory: Path) -> Path:
    return output_directory / planned_run.configuration_name / RUN_DIRECTORY_NAME


def _validate_run_settings_type(planned_run: PlannedExperimentRun) -> None:
    if type(planned_run.run_settings) not in (
        FedsdaRunSettings,
        FeddriftRunSettings,
        FedavgRunSettings,
    ):
        raise TypeError(
            "run_settings must be exact FedsdaRunSettings, FeddriftRunSettings or FedavgRunSettings"
        )


def find_existing_run_state(
    *, planned_run: PlannedExperimentRun, output_directory: Path
) -> ExistingRunState:
    """出力先に、そのrunの、済んだ結果があるかを調べる。何も変えない。"""
    if type(planned_run) is not PlannedExperimentRun:
        raise TypeError("planned_run must be exact PlannedExperimentRun")
    _validate_run_settings_type(planned_run)
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    run_directory = _get_run_directory(planned_run=planned_run, output_directory=output_directory)
    summary_file_path = run_directory / (planned_run.run_identifier + _SUMMARY_FILE_SUFFIX)
    record_table_file_path = run_directory / (
        planned_run.run_identifier + _RECORD_TABLE_FILE_SUFFIX
    )
    summary_exists = summary_file_path.exists()
    record_table_exists = record_table_file_path.exists()
    if not summary_exists and not record_table_exists:
        return ExistingRunState(run_state="missing", description=None)
    if not summary_exists or not record_table_exists:
        return ExistingRunState(
            run_state="conflicting",
            description=(
                "only one of the run files exists: "
                f"{summary_file_path if summary_exists else record_table_file_path}"
            ),
        )
    try:
        loader = (
            load_fedavg_run_artifacts
            if type(planned_run.run_settings) is FedavgRunSettings
            else load_feddrift_run_artifacts
            if type(planned_run.run_settings) is FeddriftRunSettings
            else load_fedsda_run_artifacts
        )
        summary, _ = loader(summary_file_path=summary_file_path)
    except Exception as caught_exception:
        # 壊れたファイル（JSONでない、形式の版が違う、表のファイルが読めない、ほか）。
        return ExistingRunState(
            run_state="conflicting",
            description=(
                f"existing run files cannot be read: {summary_file_path} "
                f"({type(caught_exception).__name__}: {caught_exception})"
            ),
        )
    # 保存した辞書と同じ形（JSONを通した形）にして比べる。
    planned_run_settings = loads(dumps(convert_settings_to_plain_mapping(planned_run.run_settings)))
    if summary.get("run_settings") != planned_run_settings:
        return ExistingRunState(
            run_state="conflicting",
            description=f"existing run summary holds different run settings: {summary_file_path}",
        )
    return ExistingRunState(run_state="completed", description=None)


def execute_planned_experiment_run(
    *, planned_run: PlannedExperimentRun, output_directory: Path
) -> ExperimentRunOutcome:
    """1 runを実行して、保存する。runの中の例外は、失敗の結果として返す。

    計算のthreadを1つにして実行し、終わったら、元の数へ戻す。
    """
    if type(planned_run) is not PlannedExperimentRun:
        raise TypeError("planned_run must be exact PlannedExperimentRun")
    _validate_run_settings_type(planned_run)
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    thread_count_before = get_num_threads()
    set_num_threads(_RUN_THREAD_COUNT)
    try:
        run_directory = _get_run_directory(
            planned_run=planned_run, output_directory=output_directory
        )
        run_directory.mkdir(parents=True, exist_ok=True)
        if type(planned_run.run_settings) is FedavgRunSettings:
            fedavg_run = execute_fedavg_experiment_run(run_settings=planned_run.run_settings)
            save_fedavg_experiment_run(
                experiment_run=fedavg_run,
                output_directory=run_directory,
                file_stem=planned_run.run_identifier,
            )
        elif type(planned_run.run_settings) is FeddriftRunSettings:
            feddrift_run = execute_feddrift_experiment_run(run_settings=planned_run.run_settings)
            save_feddrift_experiment_run(
                experiment_run=feddrift_run,
                output_directory=run_directory,
                file_stem=planned_run.run_identifier,
            )
        elif type(planned_run.run_settings) is FedsdaRunSettings:
            experiment_run = execute_fedsda_experiment_run(run_settings=planned_run.run_settings)
            save_fedsda_experiment_run(
                experiment_run=experiment_run,
                output_directory=run_directory,
                file_stem=planned_run.run_identifier,
            )
    except Exception as caught_exception:
        return ExperimentRunOutcome(
            configuration_name=planned_run.configuration_name,
            run_identifier=planned_run.run_identifier,
            run_status="failed",
            failure_description=(
                f"{type(caught_exception).__name__}: {caught_exception}\n{format_exc()}"
            ),
        )
    finally:
        set_num_threads(thread_count_before)
    return ExperimentRunOutcome(
        configuration_name=planned_run.configuration_name,
        run_identifier=planned_run.run_identifier,
        run_status="completed",
        failure_description=None,
    )


def _estimate_run_cost(planned_run: PlannedExperimentRun) -> tuple[int, int, int]:
    """時間のかかる見込みの大きさ（結果には影響しない、投入の順のためだけの値）。"""
    experiment_run_conditions = (
        planned_run.run_settings.execution_settings.experiment_run_conditions
    )
    dataset_cost = {"mnist4": 2, "mnist2": 1}.get(experiment_run_conditions.dataset_name, 0)
    return (
        dataset_cost,
        experiment_run_conditions.client_count * experiment_run_conditions.per_client_sample_count,
        # 集約間隔が短いほど、同期の回数が多い。
        -experiment_run_conditions.server_aggregation_interval_per_client_samples,
    )


def _initialize_worker_process() -> None:
    set_num_threads(_RUN_THREAD_COUNT)


def execute_experiment_sweep(
    *,
    planned_runs: tuple[PlannedExperimentRun, ...],
    output_directory: Path,
    worker_count: int,
    report_progress: Callable[[str], None] | None = None,
) -> tuple[ExperimentRunOutcome, ...]:
    """計画のrunを実行して保存し、runごとの結果を、計画の順で返す。

    出力先に、済んだ結果があるrunは、実行せずに飛ばす。食い違うファイルがあるrunは、上書きせずに、失敗とする。
    `worker_count`が2以上で、残りのrunが2つ以上なら、残りのrunを、別々のprocessで並列に実行する
    （時間のかかる見込みのrunから投入する）。残りが1つなら、呼出し側のprocessで実行する。
    並列のときは、workerのprocessを作る前に、数値計算のライブラリのthreadの数の環境変数を、未設定なら1にする
    （このprocessの環境を変える。workerのprocessは、それを受け継ぐ）。
    途中で例外（中断を含む）が出たら、workerへ渡す前の、待ちのrunを取り消して、例外を伝える。
    """
    if type(planned_runs) is not tuple:
        raise TypeError("planned_runs must be exact tuple")
    for planned_run in planned_runs:
        if type(planned_run) is not PlannedExperimentRun:
            raise TypeError("planned_runs must hold exact PlannedExperimentRun")
    if not planned_runs:
        raise ValueError("planned_runs must not be empty")
    run_keys = [
        (planned_run.configuration_name, planned_run.run_identifier) for planned_run in planned_runs
    ]
    if len(set(run_keys)) != len(run_keys):
        raise ValueError("planned_runs must not hold the same run twice")
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    if type(worker_count) is not int:
        raise TypeError("worker_count must be builtin int")
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if report_progress is not None and not callable(report_progress):
        raise TypeError("report_progress must be callable or None")

    started_time = perf_counter()
    outcomes_by_position: dict[int, ExperimentRunOutcome] = {}

    def record_outcome(run_position: int, outcome: ExperimentRunOutcome) -> None:
        outcomes_by_position[run_position] = outcome
        if report_progress is None:
            return
        progress_message = (
            f"[{len(outcomes_by_position)}/{len(planned_runs)}] "
            f"{outcome.configuration_name}/{outcome.run_identifier}: {outcome.run_status} "
            f"({perf_counter() - started_time:.0f}s)"
        )
        if outcome.failure_description is not None:
            # 理由の1行目（例外の型とメッセージ、または、食い違いの説明）。
            progress_message += f" — {outcome.failure_description.splitlines()[0]}"
        report_progress(progress_message)

    pending_run_positions: list[int] = []
    for run_position, planned_run in enumerate(planned_runs):
        existing_run_state = find_existing_run_state(
            planned_run=planned_run, output_directory=output_directory
        )
        if existing_run_state.run_state == "missing":
            pending_run_positions.append(run_position)
            continue
        record_outcome(
            run_position,
            ExperimentRunOutcome(
                configuration_name=planned_run.configuration_name,
                run_identifier=planned_run.run_identifier,
                run_status=(
                    "skipped_existing" if existing_run_state.run_state == "completed" else "failed"
                ),
                failure_description=existing_run_state.description,
            ),
        )

    if worker_count == 1 or len(pending_run_positions) <= 1:
        for run_position in pending_run_positions:
            record_outcome(
                run_position,
                execute_planned_experiment_run(
                    planned_run=planned_runs[run_position], output_directory=output_directory
                ),
            )
    else:
        for thread_count_variable_name in THREAD_COUNT_ENVIRONMENT_VARIABLE_NAMES:
            environ.setdefault(thread_count_variable_name, str(_RUN_THREAD_COUNT))
        process_pool = ProcessPoolExecutor(
            max_workers=min(worker_count, len(pending_run_positions)),
            mp_context=get_context("spawn"),
            initializer=_initialize_worker_process,
        )
        try:
            run_position_by_future = {
                process_pool.submit(
                    execute_planned_experiment_run,
                    planned_run=planned_runs[run_position],
                    output_directory=output_directory,
                ): run_position
                for run_position in sorted(
                    pending_run_positions,
                    key=lambda run_position: _estimate_run_cost(planned_runs[run_position]),
                    reverse=True,
                )
            }
            for completed_future in as_completed(run_position_by_future):
                run_position = run_position_by_future[completed_future]
                planned_run = planned_runs[run_position]
                try:
                    outcome = completed_future.result()
                except Exception as caught_exception:
                    # workerのprocessそのものの失敗（強制終了、結果を送れない、ほか）。
                    outcome = ExperimentRunOutcome(
                        configuration_name=planned_run.configuration_name,
                        run_identifier=planned_run.run_identifier,
                        run_status="failed",
                        failure_description=(
                            f"{type(caught_exception).__name__}: {caught_exception}"
                        ),
                    )
                record_outcome(run_position, outcome)
        except BaseException:
            # 中断（Ctrl-C）や、進み具合の報告の失敗: workerへ渡す前の、待ちのrunを取り消し、workerのprocessが
            # 終わるのを待ってから、例外を伝える（すでにworkerへ渡したrun——実行中のものと、その次の数件——は、
            # 取り消せない。中断の信号がworkerにも届く環境では、実行中のrunも途中で止まり、一時のファイルや、
            # 表のファイルだけが、残ることがある。保存の済んだrunのファイルは、壊れない）。
            process_pool.shutdown(wait=True, cancel_futures=True)
            raise
        process_pool.shutdown(wait=True)
    return tuple(outcomes_by_position[run_position] for run_position in range(len(planned_runs)))
