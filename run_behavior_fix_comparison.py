"""挙動を直した2つの扱い（LEGACY-016・LEGACY-002）の、適用の有無の組合せを、同じ条件で実行する（検証用）。

提案構成（`fedsda`）の設定の、次の2つの扱いだけを替えて、全部の組合せを実行し、組合せごとに、
`run_experiment_sweep.py`と同じ形（`<組合せの名前>/runs/`と`run_summary.csv`）で保存する。

- 登録（LEGACY-016）: `fixed`＝待ちの済んだ送信保留の全部を、現行かどうかに依らず、それ自身のIDで登録する
  （基準版2・3の扱い）。`waited-as-current`＝送信保留のモデルを、現行モデルとして待った後に、それ自身の
  IDで登録する（LEGACY-016の扱いの見直し。既定。基準版4・5の扱い）。`legacy`＝固定旧と同じ。指定しなければ、`legacy`と`fixed`を
  実行する（2026-10-13の480 runの計画を変えない）。
- 短い警報（LEGACY-002）: `fixed`＝取り込んだ前区間を、保留から外す（既定）。`legacy`＝固定旧と同じ。
- 警報を待つ（IMPROVE-006。`--alarm-wait-levels`）: `off`＝合図があれば、すぐに警報とする。
  `on`＝推定変化区間が最小件数に達するまで、警報を待つ（2026-10-14から既定）。指定しなければ、既定と違う`off`（即時）
  だけを実行する（2026-10-13の480 runの計画を変えない）。

組合せの名前は、`registration-<legacy|fixed|waited-as-current>__short-alarm-<fixed|legacy>`（警報を待つ扱いが`on`の組合せは、
末尾に`__alarm-wait-on`が付く）。出力先に済んだrunがあれば、飛ばして続ける。runの識別子は、run設定から計算するので、
設定の項目が増える前に実行したrunは、済んだrunとして扱われない（集計は、識別子でなく、dataset・集約間隔・seedで
対にするので、前の結果も、そのまま比べられる。組合せを後から足すときは、足す組合せだけを指定する）。
結果の比較は、`analyze_behavior_fix_comparison.py`。旧と同じ扱いは、通常の実験のコマンドからは選べない（この検証と、
旧実装との照合のtestだけが使う）。

出力先を省略すると、`results/behavior-fix-comparison/<開始日時>_<ラベル>`へ置く（研究の問いは`behavior-fix-comparison`に固定）。
再開は、その出力先を`--output-directory`で指定する。実行するときは、目的（`--purpose`）が要り、出力先に、目的と
出どころを書いた`README.md`を（無ければ）作る。1つの出力先は、1つの実験（1つの目的）に使う。

使い方（例）:
    python run_behavior_fix_comparison.py --datasets sine2 circle2 sea4 mnist2 --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --aggregation-intervals 50 --workers 14 --label waited-registration \\
        --purpose "現行として待つ登録の効果を、本規模で確かめる"
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
# 比較scriptの研究の問い（出力先の既定の1段目）。
BEHAVIOR_FIX_COMPARISON_STUDY_NAME = "behavior-fix-comparison"
sys.path.insert(0, str(REPOSITORY_DIRECTORY / "src"))

REGISTRATION_POLICY_BY_LEVEL = {
    "fixed": "register_each_pending_model_by_its_own_id",
    "legacy": "keep_latest_pending_model_and_reassign_current_model_id",
    "waited-as-current": "register_pending_model_after_waiting_as_current_model",
}
SHORT_ALARM_POLICY_BY_LEVEL = {
    "fixed": "release_earlier_samples_assigned_to_current_model",
    "legacy": "keep_all_pending_samples",
}
# 登録の水準と、短い警報の水準（軸ごとに分ける。短い警報の軸に、登録の水準が入らないように）。
REGISTRATION_LEVELS = ("legacy", "fixed", "waited-as-current")
# 登録の水準を指定しないときに実行する水準（2026-10-13の480 runと同じ計画）。
DEFAULT_REGISTRATION_LEVELS = ("legacy", "fixed")
SHORT_ALARM_LEVELS = ("legacy", "fixed")
ALARM_WAIT_POLICY_BY_LEVEL = {
    "off": "raise_alarm_immediately",
    "on": "wait_until_change_interval_reaches_minimum",
}
ALARM_WAIT_LEVELS = ("off", "on")


def make_combination_name(*, registration_level, short_alarm_level, alarm_wait_level="off"):
    combination_name = f"registration-{registration_level}__short-alarm-{short_alarm_level}"
    return combination_name + ("__alarm-wait-on" if alarm_wait_level == "on" else "")


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
            for alarm_wait_level in parsed_arguments.alarm_wait_levels:
                combination_name = make_combination_name(
                    registration_level=registration_level,
                    short_alarm_level=short_alarm_level,
                    alarm_wait_level=alarm_wait_level,
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
                                        loss_change_detection_settings=replace(
                                            run_client_settings.loss_change_detection_settings,
                                            short_change_interval_alarm_policy=ALARM_WAIT_POLICY_BY_LEVEL[
                                                alarm_wait_level
                                            ],
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
                                        configuration_name=combination_name,
                                        run_settings=run_settings,
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
        "--registration-levels",
        nargs="+",
        choices=REGISTRATION_LEVELS,
        default=list(DEFAULT_REGISTRATION_LEVELS),
    )
    argument_parser.add_argument(
        "--short-alarm-levels",
        nargs="+",
        choices=SHORT_ALARM_LEVELS,
        default=list(SHORT_ALARM_LEVELS),
    )
    argument_parser.add_argument(
        "--alarm-wait-levels", nargs="+", choices=ALARM_WAIT_LEVELS, default=["off"]
    )
    argument_parser.add_argument("--workers", type=int, default=1)
    argument_parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="出力先（省略時: results/behavior-fix-comparison/<開始時刻>_<ラベル>。再開は、出力先を指定する）",
    )
    argument_parser.add_argument(
        "--label",
        default="comparison",
        help="出力先を省略したときの、ディレクトリ名のラベル（既定comparison）",
    )
    argument_parser.add_argument(
        "--purpose",
        default=None,
        help="この実験の目的（1行。実行するときは必須。READMEと比較の記録に書く）",
    )
    argument_parser.add_argument(
        "--print-plan",
        action="store_true",
        help="計画のrunと、済んだrunを表示するだけで、何も実行せず、何も作らない",
    )
    given_arguments = tuple(sys.argv[1:] if command_arguments is None else command_arguments)
    parsed_arguments = argument_parser.parse_args(given_arguments)

    from federated_learning_experiments.sweep.experiment_sweep_command import (
        report_behavior_baseline_mismatch,
        report_experiment_readme,
        validate_experiment_purpose,
        validate_plain_directory_name,
    )
    from federated_learning_experiments.sweep.experiment_sweep_execution import (
        ExistingRunBehaviorBaselineMismatchError,
        collect_behavior_baseline_mismatch_descriptions,
        execute_experiment_sweep,
        find_existing_run_state,
    )
    from federated_learning_experiments.sweep.experiment_sweep_records import (
        build_run_provenance,
        describe_run_provenance_for_record,
        write_experiment_readme,
        write_run_summary_table,
    )

    # 引数を読んだ直後に、名前と目的を確かめる（何も作らずに、引数の誤りとして止める）。
    try:
        validate_plain_directory_name(parsed_arguments.label, argument_name="--label")
        purpose = (
            None
            if parsed_arguments.print_plan
            else validate_experiment_purpose(parsed_arguments.purpose)
        )
    except ValueError as caught_exception:
        argument_parser.error(str(caught_exception))
    started_time = datetime.now().astimezone().replace(microsecond=0)
    planned_runs = build_planned_runs(parsed_arguments)
    output_directory = (
        parsed_arguments.output_directory
        if parsed_arguments.output_directory is not None
        else REPOSITORY_DIRECTORY
        / "results"
        / BEHAVIOR_FIX_COMPARISON_STUDY_NAME
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
    if any(dataset_name.startswith("mnist") for dataset_name in parsed_arguments.datasets):
        from federated_learning_experiments.data.mnist.mnist_training_data import (
            load_mnist_training_data,
            resolve_mnist_data_directory,
        )

        load_mnist_training_data(data_directory=resolve_mnist_data_directory())
    # 要約と比較の記録に、同じ来歴を書く。
    run_provenance = build_run_provenance(repository_directory=REPOSITORY_DIRECTORY)
    # 基準版の違う済んだrunがあれば、出力先を作る前に止める（READMEも作らない）。
    mismatch_descriptions = collect_behavior_baseline_mismatch_descriptions(
        planned_runs=planned_runs, output_directory=output_directory
    )
    if mismatch_descriptions:
        report_behavior_baseline_mismatch(
            ExistingRunBehaviorBaselineMismatchError(mismatch_descriptions)
        )
        return 2
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"output directory: {output_directory}")
    report_experiment_readme(
        output_directory=output_directory,
        readme_file_path=write_experiment_readme(
            output_directory=output_directory,
            purpose=purpose,
            study_name=BEHAVIOR_FIX_COMPARISON_STUDY_NAME,
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
        )
    except ExistingRunBehaviorBaselineMismatchError as mismatch_error:
        report_behavior_baseline_mismatch(mismatch_error)
        return 2
    combination_names = sorted({planned_run.configuration_name for planned_run in planned_runs})
    for combination_name in combination_names:
        combination_directory = output_directory / combination_name
        if (combination_directory / "runs").is_dir():
            for unreadable_description in write_run_summary_table(
                configuration_directory=combination_directory, configuration_name=combination_name
            ).unreadable_run_summary_descriptions:
                print(f"unreadable run summary (left out of the table): {unreadable_description}")
    record_text = (
        dumps(
            dict(
                started_at=started_time.isoformat(),
                finished_at=datetime.now().astimezone().replace(microsecond=0).isoformat(),
                purpose=purpose,
                study_name=BEHAVIOR_FIX_COMPARISON_STUDY_NAME,
                command_arguments=list(given_arguments),
                combination_names=combination_names,
                worker_count=parsed_arguments.workers,
                **describe_run_provenance_for_record(run_provenance),
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
        + "\n"
    )
    # 排他的に作る（同じ秒の再実行で、前の記録を上書きしないように。同じ名前があれば、次の番号）。
    record_file_stem = f"comparison_record_{started_time.strftime('%Y%m%d_%H%M%S')}"
    record_number = 1
    while True:
        record_file_path = output_directory / (
            record_file_stem + ("" if record_number == 1 else f"_{record_number}") + ".json"
        )
        try:
            with record_file_path.open("x", encoding="utf-8", newline="\n") as record_file:
                record_file.write(record_text)
        except FileExistsError:
            record_number += 1
            continue
        break
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
