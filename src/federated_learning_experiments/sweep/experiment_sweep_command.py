"""実験の掃引のコマンド（リポジトリ直下の`run_experiment_sweep.py`から呼ばれる）。

終了コード: 0＝全部のrunが、済んだか、飛ばされた。1＝失敗したrunがある。2＝引数や計画が不正、または、
実行の前の確認（MNISTのファイル）で止まった。
"""

from argparse import ArgumentParser
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from sys import argv as process_arguments

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.mnist.mnist_training_data import (
    load_mnist_training_data,
    resolve_mnist_data_directory,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
)
from federated_learning_experiments.sweep.experiment_sweep_execution import (
    execute_experiment_sweep,
    find_existing_run_state,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
)
from federated_learning_experiments.sweep.experiment_sweep_records import (
    describe_code_revision,
    write_run_summary_table,
    write_sweep_record,
)

# リポジトリの直下（このファイルは、`src/federated_learning_experiments/sweep/`にある）。
_REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[3]
_DEFAULT_RESULT_DIRECTORY_NAME = "results"
_MNIST_DATASET_NAME_PREFIX = "mnist"


def _build_argument_parser() -> ArgumentParser:
    argument_parser = ArgumentParser(
        prog="run_experiment_sweep.py",
        description=(
            "構成・dataset・seed・集約間隔の全部の組合せを実行し、runごとの結果と、一覧の表を保存する。"
            "出力先に済んだrunがあれば、飛ばして続ける。"
        ),
    )
    argument_parser.add_argument(
        "--configurations",
        nargs="+",
        default=[PROPOSED_FEDSDA_CONFIGURATION_NAME],
        metavar="NAME",
        help=f"実験の構成の名前（使える名前: {', '.join(EXPERIMENT_CONFIGURATION_NAMES)}）",
    )
    argument_parser.add_argument(
        "--datasets", nargs="+", required=True, metavar="NAME", help="datasetの名前"
    )
    argument_parser.add_argument(
        "--seeds", nargs="+", required=True, type=int, metavar="SEED", help="乱数のseed"
    )
    argument_parser.add_argument(
        "--client-count", type=int, default=10, help="clientの数（既定10）"
    )
    argument_parser.add_argument(
        "--per-client-sample-count",
        type=int,
        default=5000,
        help="clientあたりの標本数（既定5000）",
    )
    argument_parser.add_argument(
        "--aggregation-intervals",
        nargs="+",
        required=True,
        type=int,
        metavar="SAMPLES",
        help="サーバの集約の間隔（clientあたりの標本数）",
    )
    argument_parser.add_argument(
        "--workers", type=int, default=1, help="並列に実行するprocessの数（既定1）"
    )
    argument_parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="出力先（省略時: リポジトリ直下のresults/<開始時刻>_<ラベル>）",
    )
    argument_parser.add_argument(
        "--label",
        default="sweep",
        help="出力先を省略したときの、ディレクトリ名のラベル（既定sweep）",
    )
    argument_parser.add_argument(
        "--print-plan",
        action="store_true",
        help="計画のrunと、出力先にある済んだrunを表示するだけで、何も実行せず、何も作らない",
    )
    return argument_parser


def _report_usage_error(*, argument_parser: ArgumentParser, error_message: str) -> int:
    """使い方と、エラーの内容を、標準エラーへ出して、終了コード（2）を返す。"""
    try:
        argument_parser.error(error_message)
    except SystemExit as argument_exit:
        return int(argument_exit.code or 2)


def main(command_arguments: Sequence[str] | None = None) -> int:
    """コマンドを実行して、終了コードを返す。"""
    given_arguments = tuple(
        process_arguments[1:] if command_arguments is None else command_arguments
    )
    argument_parser = _build_argument_parser()
    try:
        parsed_arguments = argument_parser.parse_args(given_arguments)
    except SystemExit as argument_exit:
        # argparseは、不正な引数で終了コード2、`--help`で0を返す。
        return int(argument_exit.code or 0)
    started_time = datetime.now().astimezone().replace(microsecond=0)
    try:
        if parsed_arguments.workers < 1:
            raise ValueError("--workers must be positive")
        sweep_plan = ExperimentSweepPlan(
            configuration_names=tuple(parsed_arguments.configurations),
            dataset_names=tuple(parsed_arguments.datasets),
            random_seeds=tuple(parsed_arguments.seeds),
            client_count=parsed_arguments.client_count,
            per_client_sample_count=parsed_arguments.per_client_sample_count,
            server_aggregation_intervals_per_client_samples=tuple(
                parsed_arguments.aggregation_intervals
            ),
        )
        planned_runs = build_planned_experiment_runs(sweep_plan=sweep_plan)
    except (TypeError, ValueError, RunSettingsValidationError) as caught_exception:
        return _report_usage_error(
            argument_parser=argument_parser, error_message=f"invalid sweep plan: {caught_exception}"
        )
    output_directory = (
        parsed_arguments.output_directory
        if parsed_arguments.output_directory is not None
        else _REPOSITORY_DIRECTORY
        / _DEFAULT_RESULT_DIRECTORY_NAME
        / f"{started_time.strftime('%Y%m%d_%H%M%S')}_{parsed_arguments.label}"
    )

    if parsed_arguments.print_plan:
        print(f"output directory: {output_directory}")
        print(f"planned runs: {len(planned_runs)}")
        for planned_run in planned_runs:
            existing_run_state = find_existing_run_state(
                planned_run=planned_run, output_directory=output_directory
            )
            print(
                f"{planned_run.configuration_name}/{planned_run.run_identifier}: "
                f"{existing_run_state.run_state}"
                + (
                    f" — {existing_run_state.description}"
                    if existing_run_state.description is not None
                    else ""
                )
            )
        return 0

    if any(
        dataset_name.startswith(_MNIST_DATASET_NAME_PREFIX)
        for dataset_name in sweep_plan.dataset_names
    ):
        # MNISTのファイルは、取得しない。置かれていなければ、runを始める前に止まる。
        try:
            load_mnist_training_data(data_directory=resolve_mnist_data_directory())
        except (OSError, ValueError) as caught_exception:
            return _report_usage_error(
                argument_parser=argument_parser,
                error_message=f"MNIST data is not available: {caught_exception}",
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"output directory: {output_directory}")
    print(f"planned runs: {len(planned_runs)}, workers: {parsed_arguments.workers}")
    outcomes = execute_experiment_sweep(
        planned_runs=planned_runs,
        output_directory=output_directory,
        worker_count=parsed_arguments.workers,
        report_progress=lambda progress_message: print(progress_message, flush=True),
    )
    for configuration_name in sweep_plan.configuration_names:
        configuration_directory = output_directory / configuration_name
        if (configuration_directory / "runs").is_dir():
            write_run_summary_table(
                configuration_directory=configuration_directory,
                configuration_name=configuration_name,
            )
    record_file_path = write_sweep_record(
        output_directory=output_directory,
        started_at=started_time.isoformat(),
        finished_at=datetime.now().astimezone().replace(microsecond=0).isoformat(),
        command_arguments=given_arguments,
        sweep_plan=sweep_plan,
        worker_count=parsed_arguments.workers,
        outcomes=outcomes,
        code_revision=describe_code_revision(repository_directory=_REPOSITORY_DIRECTORY),
    )
    failed_run_count = sum(outcome.run_status == "failed" for outcome in outcomes)
    print(
        f"completed: {sum(outcome.run_status == 'completed' for outcome in outcomes)}, "
        f"skipped (existing): {sum(outcome.run_status == 'skipped_existing' for outcome in outcomes)}, "
        f"failed: {failed_run_count}"
    )
    print(f"sweep record: {record_file_path}")
    return 1 if failed_run_count else 0
