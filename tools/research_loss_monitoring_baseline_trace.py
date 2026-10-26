"""診断用: 標本ごとの、損失の監視の基準（今の学習モデルの損失の平均と件数）を記録して、製品の掃引のコマンドを実行する。

製品の関数を外から包んで値を読むだけで、計算は変えない。若いモデルを学習している間に、基準が新しい概念の損失で
上がって変化を隠すか（LEGACY-021の追記の仮説）を確かめるために使う。

使い方:
    python -m tools.research_loss_monitoring_baseline_trace -- <run_experiment_sweep.pyの引数>

差し替えはこのprocessの中だけに効くので、`--workers 1`を求める。runの識別子・結果は既定と同じになる。
記録は、出力先の`loss_monitoring_baseline_trace/<dataset>_interval<間隔>_seed<seed>.npz`へ、runごとに保存する。
列: client_id, sample_index, current_training_model_id, observed_loss, baseline_observed_loss_count,
baseline_stored_mean_loss（範囲に制限する前の保存平均。件数0ならNaN）, baseline_used_mean_loss, log_e_value, drift_detected。
"""

import os
import sys
from contextvars import ContextVar
from pathlib import Path

for _thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

import numpy as np  # noqa: E402

from federated_learning_experiments.methods.fedsda.loss_change_detection import (  # noqa: E402
    overall_and_true_class_loss_monitoring,
)
from federated_learning_experiments.runtime import (  # noqa: E402
    fedsda_run_client,
    observed_sample_processing,
)
from federated_learning_experiments.sweep.experiment_sweep_command import main as run_sweep  # noqa: E402

_COLUMN_NAMES = (
    "client_id",
    "sample_index",
    "current_training_model_id",
    "observed_loss",
    "baseline_observed_loss_count",
    "baseline_stored_mean_loss",
    "baseline_used_mean_loss",
    "log_e_value",
    "drift_detected",
)
# 実行中の`process_observed_sample`の、client IDと学習モデルの割り当て。外ではNone。
_active_sample_processing: ContextVar[tuple[int, object] | None] = ContextVar(
    "active_sample_processing", default=None
)
_recorded_rows: list[tuple] = []
_last_baseline: list[tuple[int, float] | None] = [None]


def _install_trace() -> None:
    original_process_observed_sample = fedsda_run_client.process_observed_sample
    original_select_baseline = observed_sample_processing.select_loss_monitoring_baseline_mean_loss
    monitor_type = overall_and_true_class_loss_monitoring.OverallAndTrueClassLossMonitor
    original_observe = monitor_type.observe_loss_after_label_observation

    def process_observed_sample_with_trace(**arguments):
        token = _active_sample_processing.set(
            (arguments["client_id"], arguments["current_training_model_assignment"])
        )
        try:
            return original_process_observed_sample(**arguments)
        finally:
            _active_sample_processing.reset(token)

    def select_baseline_with_trace(*, loss_moments):
        if _active_sample_processing.get() is not None:
            if loss_moments is None or loss_moments.observed_loss_count < 1:
                _last_baseline[0] = (0, float("nan"))
            else:
                _last_baseline[0] = (loss_moments.observed_loss_count, loss_moments.mean_loss)
        return original_select_baseline(loss_moments=loss_moments)

    def observe_with_trace(self, **arguments):
        observation = original_observe(self, **arguments)
        active = _active_sample_processing.get()
        if active is not None and _last_baseline[0] is not None:
            client_id, assignment = active
            count, stored_mean = _last_baseline[0]
            _recorded_rows.append(
                (
                    client_id,
                    arguments["sample_index"],
                    assignment.current_training_model_id,
                    arguments["observed_loss"],
                    count,
                    stored_mean,
                    arguments["current_model_baseline_loss_mean"],
                    observation.log_e_value,
                    observation.drift_detected,
                )
            )
            _last_baseline[0] = None
        return observation

    fedsda_run_client.process_observed_sample = process_observed_sample_with_trace
    observed_sample_processing.select_loss_monitoring_baseline_mean_loss = select_baseline_with_trace
    monitor_type.observe_loss_after_label_observation = observe_with_trace


def _argument_value(arguments: list[str], name: str) -> list[str]:
    index = arguments.index(name)
    values = []
    for value in arguments[index + 1 :]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def main(command_arguments: list[str]) -> int:
    if not command_arguments or command_arguments[0] != "--":
        raise SystemExit("使い方: -- <run_experiment_sweep.pyの引数>")
    sweep_arguments = command_arguments[1:]
    if "--workers" not in sweep_arguments or _argument_value(sweep_arguments, "--workers") != ["1"]:
        raise SystemExit("記録はこのprocessの中だけで取れるので、--workers 1を指定してください。")
    if "--output-directory" not in sweep_arguments:
        raise SystemExit("記録の保存先を決めるため、--output-directoryを指定してください。")
    # runごとに記録を分けるため、1回の実行は1 runに限る。
    for name in ("--datasets", "--seeds", "--aggregation-intervals"):
        values = _argument_value(sweep_arguments, name)
        if len(values) != 1 or "-" in values[0]:
            raise SystemExit(f"{name}は1つの値にしてください（1回の実行で1 run）。")
    if "--configurations" in sweep_arguments and _argument_value(sweep_arguments, "--configurations") != ["fedsda"]:
        raise SystemExit("--configurationsはfedsdaだけにしてください。")
    _install_trace()
    exit_code = run_sweep(sweep_arguments)
    dataset = _argument_value(sweep_arguments, "--datasets")[0]
    seed = _argument_value(sweep_arguments, "--seeds")[0]
    interval = _argument_value(sweep_arguments, "--aggregation-intervals")[0]
    trace_directory = Path(_argument_value(sweep_arguments, "--output-directory")[0]) / "loss_monitoring_baseline_trace"
    trace_directory.mkdir(parents=True, exist_ok=True)
    columns = list(zip(*_recorded_rows)) if _recorded_rows else [[] for _ in _COLUMN_NAMES]
    np.savez_compressed(
        trace_directory / f"{dataset}_interval{interval}_seed{seed}.npz",
        **{name: np.asarray(values) for name, values in zip(_COLUMN_NAMES, columns)},
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
