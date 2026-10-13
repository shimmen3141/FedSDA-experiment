"""挙動を直した2つの扱い（LEGACY-016・LEGACY-002）の、適用の有無の4通りを、同じ条件で実行する（検証用）。

提案構成（`fedsda`）の設定の、次の2つの扱いだけを替えて、全部の組合せを実行し、組合せごとに、
`run_experiment_sweep.py`と同じ形（`<組合せの名前>/runs/`と`run_summary.csv`）で保存する。

- 登録（LEGACY-016）: `fixed`＝採用したモデルを、それ自身のIDで登録する（既定）。`legacy`＝固定旧と同じ。
- 短い警報（LEGACY-002）: `fixed`＝取り込んだ前区間を、保留から外す（既定）。`legacy`＝固定旧と同じ。

組合せの名前は、`registration-<fixed|legacy>__short-alarm-<fixed|legacy>`。出力先に済んだrunがあれば、飛ばして続ける。
結果の比較は、`analyze_behavior_fix_comparison.py`。旧と同じ扱いは、通常の実験のコマンドからは選べない（この検証と、
旧実装との照合のtestだけが使う）。

使い方（例）:
    python run_behavior_fix_comparison.py --datasets sine2 circle2 sea4 mnist2 --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --aggregation-intervals 50 --workers 14 --output-directory results/behavior-fix-comparison
"""

import os
import sys
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime
from json import dumps
from pathlib import Path

for thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(thread_count_variable_name, "1")
REPOSITORY_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_DIRECTORY / "src"))

REGISTRATION_POLICY_BY_LEVEL = {
    "fixed": "register_each_pending_model_by_its_own_id",
    "legacy": "keep_latest_pending_model_and_reassign_current_model_id",
}
SHORT_ALARM_POLICY_BY_LEVEL = {
    "fixed": "release_earlier_samples_assigned_to_current_model",
    "legacy": "keep_all_pending_samples",
}
LEVELS = ("legacy", "fixed")


def make_combination_name(*, registration_level, short_alarm_level):
    return f"registration-{registration_level}__short-alarm-{short_alarm_level}"


def build_planned_runs(parsed_arguments):
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (
        PendingModelRegistrationSettings,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        build_experiment_run_settings,
    )
    from federated_learning_experiments.sweep.experiment_sweep_plan import (
        PlannedExperimentRun,
        make_run_identifier,
    )

    planned_runs = []
    for registration_level in parsed_arguments.registration_levels:
        for short_alarm_level in parsed_arguments.short_alarm_levels:
            combination_name = make_combination_name(
                registration_level=registration_level, short_alarm_level=short_alarm_level
            )
            for dataset_name in parsed_arguments.datasets:
                for aggregation_interval in parsed_arguments.aggregation_intervals:
                    for random_seed in parsed_arguments.seeds:
                        default_run_settings = build_experiment_run_settings(
                            configuration_name="fedsda",
                            experiment_run_conditions=ExperimentRunConditions(
                                dataset_name=dataset_name,
                                random_seed=random_seed,
                                client_count=parsed_arguments.client_count,
                                per_client_sample_count=parsed_arguments.per_client_sample_count,
                                server_aggregation_interval_per_client_samples=aggregation_interval,
                            ),
                        )
                        run_participant_settings = default_run_settings.run_participant_settings
                        run_client_settings = run_participant_settings.run_client_settings
                        run_settings = replace(
                            default_run_settings,
                            run_participant_settings=replace(
                                run_participant_settings,
                                run_client_settings=replace(
                                    run_client_settings,
                                    pending_model_registration_settings=PendingModelRegistrationSettings(
                                        pending_model_registration_policy=REGISTRATION_POLICY_BY_LEVEL[
                                            registration_level
                                        ]
                                    ),
                                    training_data_assignment_settings=replace(
                                        run_client_settings.training_data_assignment_settings,
                                        pending_sample_retention_after_too_short_change_interval_policy=(
                                            SHORT_ALARM_POLICY_BY_LEVEL[short_alarm_level]
                                        ),
                                    ),
                                ),
                            ),
                        )
                        planned_runs.append(
                            PlannedExperimentRun(
                                configuration_name=combination_name,
                                run_identifier=make_run_identifier(
                                    configuration_name=combination_name, run_settings=run_settings
                                ),
                                run_settings=run_settings,
                            )
                        )
    return tuple(planned_runs)


def main(command_arguments=None):
    argument_parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    argument_parser.add_argument("--datasets", nargs="+", required=True)
    argument_parser.add_argument("--seeds", nargs="+", required=True, type=int)
    argument_parser.add_argument("--aggregation-intervals", nargs="+", required=True, type=int)
    argument_parser.add_argument("--client-count", type=int, default=10)
    argument_parser.add_argument("--per-client-sample-count", type=int, default=5000)
    argument_parser.add_argument(
        "--registration-levels", nargs="+", choices=LEVELS, default=list(LEVELS)
    )
    argument_parser.add_argument(
        "--short-alarm-levels", nargs="+", choices=LEVELS, default=list(LEVELS)
    )
    argument_parser.add_argument("--workers", type=int, default=1)
    argument_parser.add_argument("--output-directory", type=Path, required=True)
    argument_parser.add_argument(
        "--print-plan",
        action="store_true",
        help="計画のrunと、済んだrunを表示するだけで、何も実行せず、何も作らない",
    )
    parsed_arguments = argument_parser.parse_args(command_arguments)

    from federated_learning_experiments.sweep.experiment_sweep_execution import (
        execute_experiment_sweep,
        find_existing_run_state,
    )
    from federated_learning_experiments.sweep.experiment_sweep_records import (
        describe_code_revision,
        write_run_summary_table,
    )

    planned_runs = build_planned_runs(parsed_arguments)
    output_directory = parsed_arguments.output_directory
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
            )
        return 0
    if any(dataset_name.startswith("mnist") for dataset_name in parsed_arguments.datasets):
        from federated_learning_experiments.data.mnist.mnist_training_data import (
            load_mnist_training_data,
            resolve_mnist_data_directory,
        )

        load_mnist_training_data(data_directory=resolve_mnist_data_directory())
    started_time = datetime.now().astimezone().replace(microsecond=0)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"output directory: {output_directory}")
    print(f"planned runs: {len(planned_runs)}, workers: {parsed_arguments.workers}")
    outcomes = execute_experiment_sweep(
        planned_runs=planned_runs,
        output_directory=output_directory,
        worker_count=parsed_arguments.workers,
        report_progress=lambda progress_message: print(progress_message, flush=True),
    )
    combination_names = sorted({planned_run.configuration_name for planned_run in planned_runs})
    for combination_name in combination_names:
        combination_directory = output_directory / combination_name
        if (combination_directory / "runs").is_dir():
            for unreadable_description in write_run_summary_table(
                configuration_directory=combination_directory, configuration_name=combination_name
            ).unreadable_run_summary_descriptions:
                print(f"unreadable run summary (left out of the table): {unreadable_description}")
    record_file_path = (
        output_directory / f"comparison_record_{started_time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    record_file_path.write_text(
        dumps(
            dict(
                started_at=started_time.isoformat(),
                finished_at=datetime.now().astimezone().replace(microsecond=0).isoformat(),
                command_arguments=list(
                    sys.argv[1:] if command_arguments is None else command_arguments
                ),
                combination_names=combination_names,
                worker_count=parsed_arguments.workers,
                code_revision=describe_code_revision(repository_directory=REPOSITORY_DIRECTORY),
                outcomes=[
                    dict(
                        combination_name=outcome.configuration_name,
                        run_identifier=outcome.run_identifier,
                        run_status=outcome.run_status,
                        failure_description=outcome.failure_description,
                    )
                    for outcome in outcomes
                ],
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    failed_run_count = sum(outcome.run_status == "failed" for outcome in outcomes)
    print(
        f"completed: {sum(outcome.run_status == 'completed' for outcome in outcomes)}, "
        f"skipped (existing): {sum(outcome.run_status == 'skipped_existing' for outcome in outcomes)}, "
        f"failed: {failed_run_count}"
    )
    print(f"comparison record: {record_file_path}")
    return 1 if failed_run_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
