"""FedHASP基準版16の既定のrunを流し、警報ごとの最小記述長の比較の入力を記録する掃引入口（選択は変えない）。

目的（2026-11-21）: agrawal4・randomrbf4で、真の変化（特にそのclientが初めて見る概念）のときに新しいモデルの仮説が
負ける理由を、全仮説の符号長から調べる。

workerのprocessの中で、製品の2つの関数を包む（runの実行の関数を差し替え、そのprocessで包む）:
- `observed_sample_processing.select_alarm_interval_assignment`: client・警報の標本の位置・列の標本の位置・今のモデル。
- `alarm_interval_minimum_description_length_selection.select_assignment_by_minimum_description_length`:
  今のモデルと行き先の保有モデルの標本ごとの対数損失、変化区間の長さごとの新しいモデルの符号長と学習者、
  切り替えの符号長、選んだ結果。
包んだ関数は、元の関数を同じ引数で呼んで、その戻り値を返す。済んだrunごとに
`<出力先>/mdl_hypothesis_trace/<run>.json`へ書く。
"""

import argparse
import json
import os
import sys
from pathlib import Path

for _thread_count_variable_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from federated_learning_experiments.runtime import (  # noqa: E402
    alarm_interval_minimum_description_length_selection as _selection_module,
)
from federated_learning_experiments.runtime import (  # noqa: E402
    observed_sample_processing as _processing_module,
)
from federated_learning_experiments.sweep import (  # noqa: E402
    experiment_sweep_execution,
)
from tools.research_loss_monitor_rebaseline import (  # noqa: E402
    _provenance,
    _read_json,
    _sweep_arguments,
    _write_json,
)

_TRACE_DIRECTORY_NAME = "mdl_hypothesis_trace"
_RECORD_NAME = "mdl_hypothesis_trace_record.json"
_execute_planned_experiment_run = experiment_sweep_execution.execute_planned_experiment_run
_original_select_alarm_interval_assignment = _processing_module.select_alarm_interval_assignment
_original_select_by_description_length = (
    _selection_module.select_assignment_by_minimum_description_length
)


def _traced_run(planned_run, output_directory, run_provenance):
    records = []
    context = {}

    def select_alarm_interval_assignment(**arguments):
        context.clear()
        context.update(
            client_id=arguments["client_id"],
            alarm_sample_index=arguments["alarm_sample_index"],
            sample_indices=[o.sample_index for o in arguments["pending_sample_observations"]],
            current_model_id=arguments[
                "current_training_model_assignment"
            ].current_training_model_id,
        )
        selection = _original_select_alarm_interval_assignment(**arguments)
        if not context.get("recorded"):
            raise RuntimeError("最小記述長の選択が記録されませんでした")
        return selection

    def select_assignment_by_minimum_description_length(**arguments):
        selection = _original_select_by_description_length(**arguments)
        records.append(
            dict(
                **{k: v for k, v in context.items() if k != "recorded"},
                current_model_log_losses=list(arguments["current_model_log_losses"]),
                held_model_log_losses=[
                    [model_id, list(losses)]
                    for model_id, losses in arguments["held_model_log_losses_by_model_id"]
                ],
                new_model_code_lengths={
                    str(k): v
                    for k, v in arguments[
                        "new_model_code_lengths_by_change_interval_sample_count"
                    ].items()
                },
                new_model_learner_names={
                    str(k): v
                    for k, v in arguments[
                        "new_model_learner_names_by_change_interval_sample_count"
                    ].items()
                },
                switching_code_length=arguments["switching_code_length"],
                change_interval_grid_sample_count=arguments["change_interval_grid_sample_count"],
                change_interval_step_sample_count=arguments["change_interval_step_sample_count"],
                selected_hypothesis=selection.selected_hypothesis,
                selected_change_interval_sample_count=selection.change_interval_sample_count,
                selected_destination_model_id=selection.destination_model_id,
                selected_description_length=selection.description_length,
            )
        )
        context["recorded"] = True
        return selection

    _processing_module.select_alarm_interval_assignment = select_alarm_interval_assignment
    _selection_module.select_assignment_by_minimum_description_length = (
        select_assignment_by_minimum_description_length
    )
    try:
        outcome = _execute_planned_experiment_run(
            planned_run=planned_run,
            output_directory=output_directory,
            run_provenance=run_provenance,
        )
    finally:
        _processing_module.select_alarm_interval_assignment = (
            _original_select_alarm_interval_assignment
        )
        _selection_module.select_assignment_by_minimum_description_length = (
            _original_select_by_description_length
        )
    if outcome.run_status == "completed":
        directory = output_directory / _TRACE_DIRECTORY_NAME
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / f"{planned_run.run_identifier}.json", "w", encoding="utf-8") as stream:
            json.dump(dict(run_identifier=planned_run.run_identifier, alarms=records), stream)
    return outcome


def _execute_run_with_trace(*, planned_run, output_directory, run_provenance):
    """workerのprocessでも呼ばれる。"""
    return _traced_run(planned_run, output_directory, run_provenance)


def _run(arguments):
    from federated_learning_experiments.sweep.experiment_sweep_command import (
        main as run_sweep,
    )

    output_directory = arguments.output.resolve()
    record_path = output_directory / _RECORD_NAME
    if not record_path.exists() and output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(f"記録のない出力先は使えません: {output_directory}")
    sweep_arguments = _sweep_arguments(arguments, "product-default-mdl-trace", output_directory)
    if arguments.print_plan:
        return run_sweep([*sweep_arguments, "--print-plan"])
    if record_path.exists():
        _read_json(record_path)
    else:
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_json(record_path, dict(tool="research_mdl_hypothesis_trace", **_provenance()))
    experiment_sweep_execution.execute_planned_experiment_run = _execute_run_with_trace
    try:
        return run_sweep(sweep_arguments)
    finally:
        experiment_sweep_execution.execute_planned_experiment_run = _execute_planned_experiment_run


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--client-count", type=int, default=10)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--purpose", default="FedHASP基準版16の最小記述長の比較の中身を記録する診断")
    parser.add_argument("--print-plan", action="store_true")
    arguments = parser.parse_args(command_arguments)
    try:
        return _run(arguments)
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
