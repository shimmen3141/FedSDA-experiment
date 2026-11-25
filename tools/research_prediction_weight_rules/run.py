"""FedHASP（基準版16の既定）の、モデルの混合の重みの更新の規則だけを変えた変種を流す研究ツール（製品のコードは変えない）。

理論の改良の候補（[付録の下書き](../../docs/research/theory-appendix-draft-20261124.md)、2026-11-25の会話）:
- `log-loss-weights`: 重みの更新の損失を、真のクラスの損失 ℓ = 1 − p_y（2クラスでは |p − y|）から log loss −ln p_y
  （p_y = 1 − ℓ、下限 1e−12）に替え、学習率を1で一定にする（各モデルの重みに p_y を掛けて正規化する、ベイズの更新）。
  共有（一様へ 1/T）、モデルの集合が変わったときのやり直し、集約の後の再生は変えない。log loss は混ぜられる損失なので、
  比べる列に対する後悔が「log N ＋ 切り替えの費用」の形になる。
- `alarm-share-boost`: そのclientで損失の変化の警報が出たら、そのclientのモデルの混合の重みの次の1回の更新だけ、
  共有率を 1/T から 0.25 に上げる（本当の切り替えで重みを速く移し、普段の共有の値段は小さいまま）。集約の後の再生の中の
  更新には使わない。
- `log-loss-weights-alarm-share-boost`: 両方。
- `bounded-loss-unit-learning-rate`（2026-11-25追加、対照）: 損失は製品と同じ ℓ = 1 − p_y のまま、学習率だけを1で一定に
  する。`log-loss-weights` の利得が、log loss（混ぜられる損失）から来るのか、製品の学習率 √(2 ln N / V) が累積の分散 V
  とともに小さくなることを止めた分から来るのかを分ける。
予測（重みで混ぜた確率）・学習・割り当て・通信は変えない。版の混合（同期の版と手元の版）の重みも変えない。

workerのprocessの中で、`FixedShareWeightController` の重みの更新と再生、`FedsdaRunClient.process_observed_sample` を包む。
変種は環境変数で workerへ渡し、変種ごとの出力先（`<出力先>/<変種>/`）で製品の掃引のコマンドを流す。runごとに、
警報の数と共有率を上げた回数を `<出力先>/<変種>/weight_rule_trace/<run>.json` に書き（workerが並列に書くため1 run 1ファイル）、
掃引が失敗なく済んだら `<出力先>/<変種>/weight_rule_trace.json` の1つへまとめて元のファイルを消す（`traces.py`）。
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

for _thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from federated_learning_experiments.methods.fedsda.prediction_combination import (  # noqa: E402
    fixed_share_weights as _weights_module,
)
from federated_learning_experiments.runtime import fedsda_run_client as _client_module  # noqa: E402
from federated_learning_experiments.sweep import experiment_sweep_execution  # noqa: E402
from tools.research_loss_monitor_rebaseline import (  # noqa: E402
    _provenance,
    _read_json,
    _sweep_arguments,
    _write_json,
)
from tools.research_prediction_weight_rules.traces import consolidate_traces  # noqa: E402

VARIANTS = (
    "log-loss-weights",
    "alarm-share-boost",
    "log-loss-weights-alarm-share-boost",
    "product-default",
    "bounded-loss-unit-learning-rate",
)
ALARM_SHARE_RATE = 0.25
LOG_LOSS_PROBABILITY_FLOOR = 1e-12
_VARIANT_ENVIRONMENT_VARIABLE = "FEDHASP_PREDICTION_WEIGHT_RULE_VARIANT"
_VARIANT_RECORD_NAME = "prediction_weight_rule_variant_record.json"
_TRACE_DIRECTORY_NAME = "weight_rule_trace"

_Controller = _weights_module.FixedShareWeightController
_original_update = _Controller.update_weights_after_loss_observation
_original_replay = _Controller.replay_observed_losses
_original_process = _client_module.FedsdaRunClient.process_observed_sample
_execute_planned_experiment_run = experiment_sweep_execution.execute_planned_experiment_run


def _make_update(use_log_loss, use_unit_learning_rate, counters):
    def update_weights_after_loss_observation(self, *, observed_losses_by_model_id, prediction_weights_by_model_id):
        boost = getattr(self, "_research_share_boost_pending", False) and not getattr(self, "_research_in_replay", False)
        if not use_log_loss and not use_unit_learning_rate and not boost:
            return _original_update(
                self,
                observed_losses_by_model_id=observed_losses_by_model_id,
                prediction_weights_by_model_id=prediction_weights_by_model_id,
            )
        # 以下は製品の update_weights_after_loss_observation と同じ検査・順序で、損失・学習率・共有率だけを替える。
        observed_losses_by_model_id = self._validated_observed_losses(observed_losses_by_model_id=observed_losses_by_model_id)
        prediction_weights_by_model_id = self._validated_prediction_weights(
            prediction_weights_by_model_id=prediction_weights_by_model_id
        )
        if set(observed_losses_by_model_id) != set(prediction_weights_by_model_id):
            raise ValueError("observed_losses_by_model_idとprediction_weights_by_model_idは同じID集合にしてください。")
        validated_model_ids = tuple(observed_losses_by_model_id)
        self._synchronize_model_pool(validated_model_ids=validated_model_ids)
        bounded_losses_by_model_id = {
            model_id: min(1.0, max(0.0, observed_losses_by_model_id[model_id])) for model_id in validated_model_ids
        }
        if len(validated_model_ids) == 1:
            return
        if boost:
            self._research_share_boost_pending = False
            counters["share_boosts"] += 1
        previous_leader_model_id = self.select_maximum_weight_model_id(
            prediction_weights_by_model_id=prediction_weights_by_model_id
        )
        expected_observed_loss = sum(
            prediction_weights_by_model_id[model_id] * bounded_losses_by_model_id[model_id]
            for model_id in validated_model_ids
        )
        self._cumulative_observed_loss_variance += sum(
            prediction_weights_by_model_id[model_id] * (bounded_losses_by_model_id[model_id] - expected_observed_loss) ** 2
            for model_id in validated_model_ids
        )
        if use_log_loss:
            learning_rate = 1.0
            update_losses = {
                model_id: -math.log(max(1.0 - bounded_losses_by_model_id[model_id], LOG_LOSS_PROBABILITY_FLOOR))
                for model_id in validated_model_ids
            }
        elif use_unit_learning_rate:
            learning_rate = 1.0
            update_losses = bounded_losses_by_model_id
        else:
            learning_rate = min(
                1.0,
                math.sqrt(
                    2.0 * math.log(len(validated_model_ids)) / max(self._cumulative_observed_loss_variance, 1e-12)
                ),
            )
            update_losses = bounded_losses_by_model_id
        log_terms = {
            model_id: math.log(max(prediction_weights_by_model_id[model_id], 1e-300))
            - learning_rate * update_losses[model_id]
            for model_id in validated_model_ids
        }
        maximum_log_term = max(log_terms.values())
        unnormalized = {model_id: math.exp(value - maximum_log_term) for model_id, value in log_terms.items()}
        total = sum(unnormalized.values())
        share_probability = (
            ALARM_SHARE_RATE
            if boost
            else 1.0 / self._prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        )
        uniform_model_weight = 1.0 / len(validated_model_ids)
        self._weights_by_model_id = {
            model_id: (1.0 - share_probability) * unnormalized[model_id] / total + share_probability * uniform_model_weight
            for model_id in validated_model_ids
        }
        if (
            self.select_maximum_weight_model_id(prediction_weights_by_model_id=self._weights_by_model_id)
            != previous_leader_model_id
        ):
            self._leader_switch_count += 1

    return update_weights_after_loss_observation


def _replay_observed_losses(self, *, observed_loss_sequence):
    self._research_in_replay = True
    try:
        return _original_replay(self, observed_loss_sequence=observed_loss_sequence)
    finally:
        self._research_in_replay = False


def _make_process(counters):
    def process_observed_sample(self, *positional_arguments, **arguments):
        store = self.owners.loss_change_alarm_record_store
        alarm_count_before = len(store._alarm_sample_indices)
        result = _original_process(self, *positional_arguments, **arguments)
        if len(store._alarm_sample_indices) > alarm_count_before:
            counters["alarms"] += 1
            self.owners.fixed_share_weight_controller._research_share_boost_pending = True
        return result

    return process_observed_sample


def _run_with_rule(planned_run, output_directory, run_provenance):
    variant = os.environ.get(_VARIANT_ENVIRONMENT_VARIABLE)
    if variant not in VARIANTS:
        raise RuntimeError(f"変種が環境変数で渡っていません: {variant}")
    counters = {"alarms": 0, "share_boosts": 0}
    use_log_loss = variant.startswith("log-loss-weights")
    use_unit_learning_rate = variant == "bounded-loss-unit-learning-rate"
    use_boost = variant.endswith("alarm-share-boost")
    if variant != "product-default":
        _Controller.update_weights_after_loss_observation = _make_update(use_log_loss, use_unit_learning_rate, counters)
        _Controller.replay_observed_losses = _replay_observed_losses
        if use_boost:
            _client_module.FedsdaRunClient.process_observed_sample = _make_process(counters)
    try:
        outcome = _execute_planned_experiment_run(
            planned_run=planned_run, output_directory=output_directory, run_provenance=run_provenance
        )
    finally:
        _Controller.update_weights_after_loss_observation = _original_update
        _Controller.replay_observed_losses = _original_replay
        _client_module.FedsdaRunClient.process_observed_sample = _original_process
    if outcome.run_status == "completed":
        directory = output_directory / _TRACE_DIRECTORY_NAME
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / f"{planned_run.run_identifier}.json", "w", encoding="utf-8") as stream:
            json.dump(dict(run_identifier=planned_run.run_identifier, variant=variant, **counters), stream)
    return outcome


def _execute_run_with_rule(*, planned_run, output_directory, run_provenance):
    """workerのprocessでも呼ばれる。"""
    return _run_with_rule(planned_run, output_directory, run_provenance)


def _run_variant(arguments, variant):
    from federated_learning_experiments.sweep.experiment_sweep_command import main as run_sweep

    output_directory = arguments.output.resolve() / variant
    record_path = output_directory / _VARIANT_RECORD_NAME
    if record_path.exists():
        if _read_json(record_path).get("variant") != variant:
            raise ValueError(f"出力先に別の変種の記録があります: {record_path}")
    elif output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(f"変種の記録のない出力先は使えません: {output_directory}")
    sweep_arguments = _sweep_arguments(arguments, variant, output_directory)
    if arguments.print_plan:
        return run_sweep([*sweep_arguments, "--print-plan"])
    if not record_path.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            record_path,
            dict(variant=variant, alarm_share_rate=ALARM_SHARE_RATE, log_loss_floor=LOG_LOSS_PROBABILITY_FLOOR, **_provenance()),
        )
    previous_variant = os.environ.get(_VARIANT_ENVIRONMENT_VARIABLE)
    os.environ[_VARIANT_ENVIRONMENT_VARIABLE] = variant
    experiment_sweep_execution.execute_planned_experiment_run = _execute_run_with_rule
    try:
        status = run_sweep(sweep_arguments)
        if status == 0:
            # 失敗の無い掃引の後だけ、runごとのトレースを1つのファイルへまとめる（中断・失敗のときは残して再開に備える）。
            consolidate_traces(output_directory)
        return status
    finally:
        experiment_sweep_execution.execute_planned_experiment_run = _execute_planned_experiment_run
        if previous_variant is None:
            os.environ.pop(_VARIANT_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[_VARIANT_ENVIRONMENT_VARIABLE] = previous_variant


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS[:3]))
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--client-count", type=int, default=10)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--purpose", default="FedHASP基準版16のモデルの混合の重みの更新の規則を変えた変種")
    parser.add_argument("--print-plan", action="store_true")
    arguments = parser.parse_args(command_arguments)
    status = 0
    try:
        for variant in arguments.variants:
            status = max(status, _run_variant(arguments, variant) or 0)
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
