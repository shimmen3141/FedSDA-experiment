"""実験の掃引のコマンド（リポジトリ直下の`run_experiment_sweep.py`から呼ばれる）。

終了コード: 0＝全部のrunが、済んだか、飛ばされた。1＝失敗したrunがある、または、出力先に、読めない要約がある。2＝引数や計画が不正、または、
実行の前の確認（MNISTのファイル、出力先の済んだrunの基準版が今のコードと違う）で止まった。

出力先を省略すると、`results/<研究の問い>/<開始日時>_<ラベル>`（研究の問いが無ければ`results/<開始日時>_<ラベル>`）。
実行するときは、目的（`--purpose`）が要り、出力先に、目的と出どころを書いた`README.md`を（無ければ）作る。
"""

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from shlex import split as split_shell_arguments
from sys import argv as process_arguments
from sys import stdin, stdout

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.run_provenance import (
    BEHAVIOR_BASELINE_VERSION,
    RunProvenance,
)
from federated_learning_experiments.data.mnist.mnist_training_data import (
    load_mnist_training_data,
    resolve_mnist_data_directory,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
)
from federated_learning_experiments.sweep.experiment_sweep_execution import (
    ExistingRunBehaviorBaselineMismatchError,
    collect_behavior_baseline_mismatch_descriptions,
    execute_experiment_sweep,
    find_existing_run_state,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    PlannedExperimentRun,
    build_planned_experiment_runs,
)
from federated_learning_experiments.sweep.experiment_sweep_records import (
    build_run_provenance,
    write_experiment_readme,
    write_run_summary_table,
    write_sweep_record,
)
from federated_learning_experiments.sweep.experiment_sweep_resumption import (
    describe_planned_run_set,
    find_sweep_resumption_candidates,
    hold_sweep_output_lock,
    write_sweep_resumption_plan,
)
from federated_learning_experiments.sweep.seed_argument_parsing import SeedListArgumentAction

# リポジトリの直下（このファイルは、`src/federated_learning_experiments/sweep/`にある）。
_REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[3]
_DEFAULT_RESULT_DIRECTORY_NAME = "results"
_MNIST_DATASET_NAME_PREFIX = "mnist"
# ディレクトリの区切りと、Windowsでファイル名に使えない文字。
_DIRECTORY_NAME_SEPARATORS = ("/", "\\", ":", "*", "?", '"', "<", ">", "|")


def validate_plain_directory_name(name: str, *, argument_name: str) -> str:
    """研究の問い・ラベルが、1段のディレクトリ名として使えることを確かめて、そのまま返す。"""
    if type(name) is not str:
        raise TypeError(f"{argument_name} must be builtin str")
    if (
        not name
        or name != name.strip()
        or name in (".", "..")
        or any(separator in name for separator in _DIRECTORY_NAME_SEPARATORS)
    ):
        raise ValueError(
            f"{argument_name} must be a plain directory name without separators: {name!r}"
        )
    return name


def validate_experiment_purpose(purpose: str | None) -> str:
    """目的（`--purpose`）を確かめ、前後の空白を除いて返す。無い・空・改行を含むものは拒否する。"""
    if purpose is None:
        raise ValueError("--purpose is required to run (describe what this experiment is for)")
    if "\n" in purpose or "\r" in purpose:
        raise ValueError("--purpose must be a single line")
    if not purpose.strip():
        raise ValueError("--purpose must not be blank")
    return purpose.strip()


def report_behavior_baseline_mismatch(
    mismatch_error: ExistingRunBehaviorBaselineMismatchError,
) -> None:
    """基準版の違う済んだrunを、標準エラーへ、1件1行の警告で出し、対処を1行出す。"""
    # 呼出しの時点の標準エラーへ出す（読込みの時点で束縛すると、差し替えた標準エラーへ出ない）。
    from sys import stderr

    for mismatch_description in mismatch_error.mismatch_descriptions:
        print(f"warning: {mismatch_description}", file=stderr)
    print(
        f"stopped before running: {mismatch_error}. "
        "Use another output directory for results of the current code.",
        file=stderr,
    )


def report_experiment_readme(*, output_directory: Path, readme_file_path: Path | None) -> None:
    """READMEを作ったか、既にあって変えなかったかを表示する。"""
    if readme_file_path is not None:
        print(f"README: {readme_file_path}")
    else:
        print(f"README: {output_directory / 'README.md'} (already exists; left unchanged)")


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
        "--seeds",
        nargs="+",
        required=True,
        action=SeedListArgumentAction,
        metavar="SEED",
        help="乱数のseed。0-9は両端込み、0-2 5の混在可。重複不可、展開後100000個以内",
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
        "--feddrift-loss-increase-thresholds",
        nargs="+",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="FedDriftの損失増分閾値（feddrift選択時だけ指定可、省略時0.1）。FedSDA・FedAvgはこの軸で増やさない",
    )
    argument_parser.add_argument(
        "--fedsda-pending-assignment-fifo-capacities",
        nargs="+",
        type=int,
        default=None,
        metavar="CAPACITY",
        help="FedSDAの構成の、帰属の保留（FIFO）の容量（標本数）。FedSDAの構成を選んだときだけ",
    )
    argument_parser.add_argument(
        "--fedsda-fixed-share-time-scales",
        nargs="+",
        type=int,
        default=None,
        metavar="TIME_SCALE",
        help=(
            "FedSDAの構成の、Fixed-Shareの時間尺度T（標本数。予測の重みを分け直す速さ。"
            "混ぜる予測の重みの分け合いの割合も1/T）。FedSDAの構成を選んだときだけ"
        ),
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
        "--study",
        default=None,
        help="研究の問いの名前（出力先を省略したとき、results/<研究の問い>/<開始時刻>_<ラベル>へ置く）",
    )
    argument_parser.add_argument(
        "--purpose",
        default=None,
        help="この実験の目的（1行。実行するときは必須。READMEと実行の記録に書く）",
    )
    argument_parser.add_argument(
        "--new-output",
        action="store_true",
        help="同条件の過去掃引を選ばず、新しい出力先で実行する",
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


def _build_sweep_plan(parsed_arguments: Namespace) -> ExperimentSweepPlan:
    return ExperimentSweepPlan(
        configuration_names=tuple(parsed_arguments.configurations),
        dataset_names=tuple(parsed_arguments.datasets),
        random_seeds=tuple(parsed_arguments.seeds),
        client_count=parsed_arguments.client_count,
        per_client_sample_count=parsed_arguments.per_client_sample_count,
        server_aggregation_intervals_per_client_samples=tuple(
            parsed_arguments.aggregation_intervals
        ),
        feddrift_loss_increase_thresholds=(
            tuple(parsed_arguments.feddrift_loss_increase_thresholds)
            if parsed_arguments.feddrift_loss_increase_thresholds is not None
            else None
        ),
        fedsda_fixed_share_time_scales=(
            tuple(parsed_arguments.fedsda_fixed_share_time_scales)
            if parsed_arguments.fedsda_fixed_share_time_scales is not None
            else None
        ),
        fedsda_pending_assignment_fifo_capacities=(
            tuple(parsed_arguments.fedsda_pending_assignment_fifo_capacities)
            if parsed_arguments.fedsda_pending_assignment_fifo_capacities is not None
            else None
        ),
    )


def _is_interactive() -> bool:
    return stdin.isatty() and stdout.isatty()


def _read_legacy_plan(readme_path: Path) -> list[dict] | None:
    # README内の既存writerが生成する引数だけを読む。shellとして実行しない。
    text = readme_path.read_text(encoding="utf-8")
    baseline_line = f"- 期待する動作の基準版: {BEHAVIOR_BASELINE_VERSION}"
    if baseline_line not in text.splitlines():
        return None
    marker = "## コマンドの引数\n\n```\n"
    if marker not in text:
        return None
    arguments = split_shell_arguments(text.split(marker, 1)[1].split("\n```", 1)[0])
    try:
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            parsed = _build_argument_parser().parse_args(arguments)
        return describe_planned_run_set(
            build_planned_experiment_runs(sweep_plan=_build_sweep_plan(parsed))
        )
    except (SystemExit, ValueError, TypeError, RunSettingsValidationError):
        return None


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
    try:
        validate_plain_directory_name(parsed_arguments.label, argument_name="--label")
        if parsed_arguments.study is not None:
            validate_plain_directory_name(parsed_arguments.study, argument_name="--study")
            if parsed_arguments.output_directory is not None:
                raise ValueError(
                    "--study cannot be used with --output-directory "
                    "(the output directory is made from the study name)"
                )
        purpose = (
            None
            if parsed_arguments.print_plan
            else validate_experiment_purpose(parsed_arguments.purpose)
        )
    except ValueError as caught_exception:
        return _report_usage_error(
            argument_parser=argument_parser, error_message=str(caught_exception)
        )
    started_time = datetime.now().astimezone().replace(microsecond=0)
    try:
        if parsed_arguments.workers < 1:
            raise ValueError("--workers must be positive")
        sweep_plan = _build_sweep_plan(parsed_arguments)
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
        / (parsed_arguments.study or "")
        / f"{started_time.strftime('%Y%m%d_%H%M%S')}_{parsed_arguments.label}"
    )

    run_provenance = build_run_provenance(repository_directory=_REPOSITORY_DIRECTORY)
    if parsed_arguments.new_output and parsed_arguments.output_directory is not None:
        return _report_usage_error(
            argument_parser=argument_parser,
            error_message="--new-output cannot be used with --output-directory",
        )
    if parsed_arguments.output_directory is None:
        # 同じ秒に新規実行しても、既存ディレクトリを新規先として使わない。
        initial_output = output_directory
        suffix = 2
        while output_directory.exists():
            output_directory = initial_output.with_name(f"{initial_output.name}_{suffix}")
            suffix += 1
        candidates = find_sweep_resumption_candidates(
            _REPOSITORY_DIRECTORY / _DEFAULT_RESULT_DIRECTORY_NAME,
            planned_runs,
            run_provenance,
            _read_legacy_plan,
            lambda run, directory: (
                find_existing_run_state(planned_run=run, output_directory=directory).run_state
            ),
        )
        if candidates and (parsed_arguments.print_plan or not parsed_arguments.new_output):
            print("同じ条件の掃引が見つかりました。")
            for number, candidate in enumerate(candidates, 1):
                action = (
                    "完了済み・実行せず終了"
                    if candidate.completed_run_count == candidate.total_run_count
                    else "続きから実行"
                )
                print(
                    f"{number}. {candidate.output_directory} "
                    f"({candidate.completed_run_count}/{candidate.total_run_count} run) "
                    f"{action} "
                    f"自動復旧可能{candidate.recoverable_run_count}件・要確認{candidate.conflicting_run_count}件"
                )
            if not parsed_arguments.print_plan:
                if not _is_interactive():
                    return _report_usage_error(
                        argument_parser=argument_parser,
                        error_message="Use --output-directory to resume or --new-output to run anew (non-interactive input).",
                    )
                print(f"{len(candidates) + 1}. 新しいディレクトリで実行\n0. キャンセル")
                while True:
                    try:
                        answer = input("選択: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("キャンセルしました。")
                        return 0
                    if answer.isdecimal() and 0 <= int(answer) <= len(candidates) + 1:
                        choice = int(answer)
                        break
                    print("表示された番号を入力してください。")
                if choice == 0:
                    return 0
                if choice <= len(candidates):
                    selected = candidates[choice - 1]
                    if selected.completed_run_count == selected.total_run_count:
                        print("全run完了済みのため、実行せず終了します。")
                        return 0
                    if selected.conflicting_run_count:
                        return _report_usage_error(
                            argument_parser=argument_parser,
                            error_message="Selected sweep has conflicting artifacts; inspect them before resuming.",
                        )
                    output_directory = selected.output_directory
                    print(
                        "再開: 元READMEは初回の記録として保持し、今回の目的・引数はsweep_recordへ記録します。"
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

    # 要約と実行の記録に、同じ来歴を書く（gitの取り直しで、ずれないように）。
    # 基準版の違う済んだrunがあれば、出力先を作る前に止める（READMEも作らない）。
    mismatch_descriptions = collect_behavior_baseline_mismatch_descriptions(
        planned_runs=planned_runs, output_directory=output_directory
    )
    if mismatch_descriptions:
        report_behavior_baseline_mismatch(
            ExistingRunBehaviorBaselineMismatchError(mismatch_descriptions)
        )
        return 2
    assert purpose is not None
    try:
        with hold_sweep_output_lock(output_directory):
            # 選択後に別processが開始・完了した可能性があるため、排他の下で再確認。
            mismatch_descriptions = collect_behavior_baseline_mismatch_descriptions(
                planned_runs=planned_runs, output_directory=output_directory
            )
            if mismatch_descriptions:
                report_behavior_baseline_mismatch(
                    ExistingRunBehaviorBaselineMismatchError(mismatch_descriptions)
                )
                return 2
            write_sweep_resumption_plan(output_directory, planned_runs)
            return _execute_command_in_output(
                parsed_arguments,
                sweep_plan,
                planned_runs,
                output_directory,
                run_provenance,
                started_time,
                given_arguments,
                purpose,
            )
    except (OSError, ValueError) as error:
        return _report_usage_error(argument_parser=argument_parser, error_message=str(error))


def _execute_command_in_output(
    parsed_arguments: Namespace,
    sweep_plan: ExperimentSweepPlan,
    planned_runs: tuple[PlannedExperimentRun, ...],
    output_directory: Path,
    run_provenance: RunProvenance,
    started_time: datetime,
    given_arguments: tuple[str, ...],
    purpose: str,
) -> int:
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"output directory: {output_directory}")
    report_experiment_readme(
        output_directory=output_directory,
        readme_file_path=write_experiment_readme(
            output_directory=output_directory,
            purpose=purpose,
            study_name=parsed_arguments.study,
            started_at=started_time.isoformat(),
            command_arguments=given_arguments,
            run_provenance=run_provenance,
        ),
    )
    print(f"planned runs: {len(planned_runs)}, workers: {parsed_arguments.workers}")
    try:
        outcomes = execute_experiment_sweep(
            planned_runs=planned_runs,
            output_directory=output_directory,
            worker_count=parsed_arguments.workers,
            run_provenance=run_provenance,
            report_progress=lambda progress_message: print(progress_message, flush=True),
            _output_lock_held=True,
        )
    except ExistingRunBehaviorBaselineMismatchError as mismatch_error:
        report_behavior_baseline_mismatch(mismatch_error)
        return 2
    unreadable_run_summary_descriptions: list[str] = []
    for configuration_name in sweep_plan.configuration_names:
        configuration_directory = output_directory / configuration_name
        if (configuration_directory / "runs").is_dir():
            unreadable_run_summary_descriptions.extend(
                write_run_summary_table(
                    configuration_directory=configuration_directory,
                    configuration_name=configuration_name,
                ).unreadable_run_summary_descriptions
            )
    for unreadable_description in unreadable_run_summary_descriptions:
        print(f"unreadable run summary (left out of the table): {unreadable_description}")
    record_file_path = write_sweep_record(
        output_directory=output_directory,
        started_at=started_time.isoformat(),
        finished_at=datetime.now().astimezone().replace(microsecond=0).isoformat(),
        command_arguments=given_arguments,
        sweep_plan=sweep_plan,
        worker_count=parsed_arguments.workers,
        outcomes=outcomes,
        run_provenance=run_provenance,
        purpose=purpose,
        study_name=parsed_arguments.study,
        unreadable_run_summary_descriptions=tuple(unreadable_run_summary_descriptions),
    )
    failed_run_count = sum(outcome.run_status == "failed" for outcome in outcomes)
    print(
        f"completed: {sum(outcome.run_status == 'completed' for outcome in outcomes)}, "
        f"skipped (existing): {sum(outcome.run_status == 'skipped_existing' for outcome in outcomes)}, "
        f"failed: {failed_run_count}"
    )
    print(f"sweep record: {record_file_path}")
    return 1 if failed_run_count or unreadable_run_summary_descriptions else 0
