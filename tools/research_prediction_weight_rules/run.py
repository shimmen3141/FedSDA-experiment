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
- `tempered-log-loss-weights-0.5`・`tempered-log-loss-weights-0.25`（2026-11-25追加）: log loss の更新の学習率を
  η = 0.5・0.25 にする（各モデルの重みに p_y^η を掛けて正規化する）。最小形の上で、学習率1の log loss が安定した期間に
  小さく損をした（[研究記録](../../docs/research/fedhasp-prediction-weight-rules-20261125.md)の節7）ので、重みを
  1つのモデルへ寄せすぎることを抑えたときに、その損が消えるかを見る。η ≤ 1 なら後悔の上界は C_α(S)/η。
- `learning-rate-mixture-log-loss-weights`（2026-11-25追加）: 学習率 η ∈ {1, 1/2, 1/4} の log loss の Fixed Share を
  それぞれ持ち、その3つの混合の予測を、上の層の log loss の Fixed Share（学習率1、共有率は同じ 1/T）で混ぜる。
  モデルの重みは Σ_k u_k w_{k,i}（上の層の重み u_k × 学習率 k の重み w_{k,i}）になる。上の層も下の層も、モデルの集合が
  変わったとき・重みの証拠を消すとき（集約の後の再生など）は、製品の重みと同じく一様から作り直す。
- `log-loss-weights-log-loss-version-mixture`（2026-11-25追加）: `log-loss-weights` に加えて、版の混合（同期の版と
  手元の版の2つの候補）の重みの更新も、log loss・学習率1（それぞれの結合予測の真のクラスの確率を掛けて正規化し、
  分け合う）にする。製品の版の混合は、有界の損失 1 − p_y・学習率2。
予測（重みで混ぜた確率）・学習・割り当て・通信は変えない。版の混合（同期の版と手元の版）の重みは、
`log-loss-weights-log-loss-version-mixture` 以外は変えない。

確率の記録（`--record-prediction-probabilities-for <変種>…`、2026-11-25追加）: 指定した変種の run で、clientごとに、
モデルの混合の重みの制御（`FixedShareWeightController`）への呼出しの列（観測前の重み・損失での更新・再生・集約の後の
消去。入れ子の呼出しは記録しない）と、予測ごとの同期の版のモデル別の確率・真のクラス・使った重みを、
`<出力先>/<変種>/prediction_probability_trace/<run>.pkl.xz` に書き、失敗の無い掃引の後に
`<出力先>/<変種>/prediction_probability_trace.zip` へまとめる（`probability_trace.py`）。学習・予測は変えない。
`replay.py` が、この記録から別の重みの規則を再実験なしに計算し直す（学習は重みの規則に依らないことを、どの変種も
対の run で確かめている）。

基点（`--base`）: 既定は製品の `fedsda`。`minimal-soft-prediction` を選ぶと、簡素版の研究ツール
（`tools/research_simplified_fedhasp`）の最小形（独立の神経網、ADWIN＋γ、MDLなし、柔らかい予測あり）の run 設定で流す
（計画を作る関数を、簡素版の研究ツールと同じく差し替える）。変種の記録に基点を書き、別の基点の出力先は使わない。

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
from federated_learning_experiments.methods.fedsda.prediction_combination import (  # noqa: E402
    synchronized_and_local_prediction_mixture as _version_mixture_module,
)
from federated_learning_experiments.runtime import fedsda_run_client as _client_module  # noqa: E402
from federated_learning_experiments.sweep import experiment_sweep_execution  # noqa: E402
from tools.research_loss_monitor_rebaseline import (  # noqa: E402
    _provenance,
    _read_json,
    _sweep_arguments,
    _write_json,
)
from federated_learning_experiments.sweep import experiment_sweep_plan  # noqa: E402
from tools.research_prediction_weight_rules.probability_trace import (  # noqa: E402
    ProbabilityTraceRecorder,
    consolidate_probability_traces,
)
from tools.research_prediction_weight_rules.traces import consolidate_traces  # noqa: E402
from tools.research_simplified_fedhasp.run import make_patched_build as make_simplified_build  # noqa: E402

PRODUCT_BASE = "product-fedsda"
# 簡素版の研究ツール（tools/research_simplified_fedhasp）の変種のうち、柔らかい予測（Fixed-Shareのモデルの混合と版の
# 混合）を持つ最小形。担当教授の指摘を受けて単純な構成で実験し直す方針（2026-11-25ユーザー確認）に合わせる。
BASES = (PRODUCT_BASE, "minimal-soft-prediction")
_original_build = experiment_sweep_plan.build_experiment_run_settings

VARIANTS = (
    "log-loss-weights",
    "alarm-share-boost",
    "log-loss-weights-alarm-share-boost",
    "product-default",
    "bounded-loss-unit-learning-rate",
    "tempered-log-loss-weights-0.5",
    "tempered-log-loss-weights-0.25",
    "learning-rate-mixture-log-loss-weights",
    "log-loss-weights-log-loss-version-mixture",
)
# 警報の後の共有率を上げる変種（clientの標本の処理を包む）。確率の記録とは一緒に使わない。
ALARM_SHARE_BOOST_VARIANTS = ("alarm-share-boost", "log-loss-weights-alarm-share-boost")
TEMPERED_LOG_LOSS_LEARNING_RATES = {"tempered-log-loss-weights-0.5": 0.5, "tempered-log-loss-weights-0.25": 0.25}
LEARNING_RATE_MIXTURE_CANDIDATES = (1.0, 0.5, 0.25)
ALARM_SHARE_RATE = 0.25
LOG_LOSS_PROBABILITY_FLOOR = 1e-12
_VARIANT_ENVIRONMENT_VARIABLE = "FEDHASP_PREDICTION_WEIGHT_RULE_VARIANT"
_RECORD_ENVIRONMENT_VARIABLE = "FEDHASP_PREDICTION_WEIGHT_RULE_RECORD_PROBABILITIES"
_VARIANT_RECORD_NAME = "prediction_weight_rule_variant_record.json"
_TRACE_DIRECTORY_NAME = "weight_rule_trace"

_Controller = _weights_module.FixedShareWeightController
_original_update = _Controller.update_weights_after_loss_observation
_original_replay = _Controller.replay_observed_losses
_original_synchronize = _Controller._synchronize_model_pool
_original_clear = _Controller._clear_weight_evidence
_VersionMixture = _version_mixture_module.SynchronizedAndLocalPredictionMixture
_original_version_mixture_update = _VersionMixture.update_weights_after_loss_observation
_original_process = _client_module.FedsdaRunClient.process_observed_sample
_execute_planned_experiment_run = experiment_sweep_execution.execute_planned_experiment_run


def _make_update(use_log_loss, use_unit_learning_rate, counters, log_loss_learning_rate=1.0):
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
            learning_rate = log_loss_learning_rate
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


def _make_learning_rate_mixture_update(counters):
    def update_weights_after_loss_observation(self, *, observed_losses_by_model_id, prediction_weights_by_model_id):
        # 検査・集合の同期・分散の診断・1位の入替の数え方は、製品の更新（と上の `_make_update`）と同じ。
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
        state = getattr(self, "_research_learning_rate_mixture_state", None)
        if state is None or set(state["model_ids"]) != set(validated_model_ids):
            # 製品の重みが一様から作り直された直後（集合の変化・証拠の消去）と同じく、どの層も一様から始める。
            uniform = {model_id: 1.0 / len(validated_model_ids) for model_id in validated_model_ids}
            state = dict(
                model_ids=validated_model_ids,
                learning_rate_weights=[dict(uniform) for _ in LEARNING_RATE_MIXTURE_CANDIDATES],
                top_weights=[1.0 / len(LEARNING_RATE_MIXTURE_CANDIDATES)] * len(LEARNING_RATE_MIXTURE_CANDIDATES),
            )
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
        true_class_probabilities = {
            model_id: max(1.0 - bounded_losses_by_model_id[model_id], LOG_LOSS_PROBABILITY_FLOOR)
            for model_id in validated_model_ids
        }
        share_probability = 1.0 / self._prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        uniform_model_weight = 1.0 / len(validated_model_ids)
        updated_learning_rate_weights = []
        log_top_terms = []
        for candidate_index, learning_rate in enumerate(LEARNING_RATE_MIXTURE_CANDIDATES):
            weights = state["learning_rate_weights"][candidate_index]
            # 学習率 k の混合の、真のクラスの確率（更新の前の重みで混ぜたもの）。上の層の更新に使う。
            mixture_probability = sum(weights[model_id] * true_class_probabilities[model_id] for model_id in validated_model_ids)
            log_top_terms.append(
                math.log(max(state["top_weights"][candidate_index], 1e-300))
                + math.log(max(mixture_probability, LOG_LOSS_PROBABILITY_FLOOR))
            )
            log_terms = {
                model_id: math.log(max(weights[model_id], 1e-300))
                + learning_rate * math.log(true_class_probabilities[model_id])
                for model_id in validated_model_ids
            }
            maximum_log_term = max(log_terms.values())
            unnormalized = {model_id: math.exp(value - maximum_log_term) for model_id, value in log_terms.items()}
            total = sum(unnormalized.values())
            updated_learning_rate_weights.append(
                {
                    model_id: (1.0 - share_probability) * unnormalized[model_id] / total
                    + share_probability * uniform_model_weight
                    for model_id in validated_model_ids
                }
            )
        maximum_log_top_term = max(log_top_terms)
        unnormalized_top = [math.exp(value - maximum_log_top_term) for value in log_top_terms]
        total_top = sum(unnormalized_top)
        uniform_top_weight = 1.0 / len(LEARNING_RATE_MIXTURE_CANDIDATES)
        updated_top_weights = [
            (1.0 - share_probability) * value / total_top + share_probability * uniform_top_weight
            for value in unnormalized_top
        ]
        state["learning_rate_weights"] = updated_learning_rate_weights
        state["top_weights"] = updated_top_weights
        self._research_learning_rate_mixture_state = state
        self._weights_by_model_id = {
            model_id: sum(
                top_weight * weights[model_id]
                for top_weight, weights in zip(updated_top_weights, updated_learning_rate_weights, strict=True)
            )
            for model_id in validated_model_ids
        }
        if (
            self.select_maximum_weight_model_id(prediction_weights_by_model_id=self._weights_by_model_id)
            != previous_leader_model_id
        ):
            self._leader_switch_count += 1

    return update_weights_after_loss_observation


def _synchronize_model_pool_with_mixture_reset(self, *, validated_model_ids):
    # 製品が重みを一様から作り直すとき（集合が変わったとき）は、学習率の混合の層も作り直す。
    if set(validated_model_ids) != set(self._weights_by_model_id):
        self._research_learning_rate_mixture_state = None
    return _original_synchronize(self, validated_model_ids=validated_model_ids)


def _clear_weight_evidence_with_mixture_reset(self):
    self._research_learning_rate_mixture_state = None
    return _original_clear(self)


def _version_mixture_log_loss_update(self, *, synchronized_loss, local_loss):
    # 製品の版の混合の更新と同じ順（掛けて正規化し、分け合う）で、掛けるものを真のクラスの確率（学習率1の log loss）にする。
    updated_synchronized = self._synchronized_weight * max(1.0 - synchronized_loss, LOG_LOSS_PROBABILITY_FLOOR)
    updated_local = self._local_weight * max(1.0 - local_loss, LOG_LOSS_PROBABILITY_FLOOR)
    total = updated_synchronized + updated_local
    updated_synchronized, updated_local = updated_synchronized / total, updated_local / total
    share_rate = self._share_rate
    self._synchronized_weight = (1 - share_rate) * updated_synchronized + share_rate * updated_local
    self._local_weight = (1 - share_rate) * updated_local + share_rate * updated_synchronized


def install_weight_rule(variant, counters):
    """変種の重みの規則をクラスへ差し込む（`restore_weight_rules` で戻す）。手元で計算し直す `replay.py` も使う。"""
    if variant not in VARIANTS:
        raise ValueError(f"知らない変種です: {variant}")
    if variant == "product-default":
        return
    if variant == "learning-rate-mixture-log-loss-weights":
        _Controller.update_weights_after_loss_observation = _make_learning_rate_mixture_update(counters)
        _Controller._synchronize_model_pool = _synchronize_model_pool_with_mixture_reset
        _Controller._clear_weight_evidence = _clear_weight_evidence_with_mixture_reset
    else:
        use_log_loss = variant.startswith("log-loss-weights") or variant in TEMPERED_LOG_LOSS_LEARNING_RATES
        use_unit_learning_rate = variant == "bounded-loss-unit-learning-rate"
        _Controller.update_weights_after_loss_observation = _make_update(
            use_log_loss, use_unit_learning_rate, counters, TEMPERED_LOG_LOSS_LEARNING_RATES.get(variant, 1.0)
        )
    _Controller.replay_observed_losses = _replay_observed_losses
    if variant == "log-loss-weights-log-loss-version-mixture":
        _VersionMixture.update_weights_after_loss_observation = _version_mixture_log_loss_update
    if variant in ALARM_SHARE_BOOST_VARIANTS:
        _client_module.FedsdaRunClient.process_observed_sample = _make_process(counters)


def restore_weight_rules():
    _Controller.update_weights_after_loss_observation = _original_update
    _Controller.replay_observed_losses = _original_replay
    _Controller._synchronize_model_pool = _original_synchronize
    _Controller._clear_weight_evidence = _original_clear
    _VersionMixture.update_weights_after_loss_observation = _original_version_mixture_update
    _client_module.FedsdaRunClient.process_observed_sample = _original_process


def _run_with_rule(planned_run, output_directory, run_provenance):
    variant = os.environ.get(_VARIANT_ENVIRONMENT_VARIABLE)
    if variant not in VARIANTS:
        raise RuntimeError(f"変種が環境変数で渡っていません: {variant}")
    record_probabilities = os.environ.get(_RECORD_ENVIRONMENT_VARIABLE) == "1"
    if record_probabilities and variant in ALARM_SHARE_BOOST_VARIANTS:
        raise RuntimeError("確率の記録は、警報の後の共有率を上げる変種とは一緒に使えません")
    counters = {"alarms": 0, "share_boosts": 0}
    install_weight_rule(variant, counters)
    recorder = None
    try:
        if record_probabilities:
            # 規則を差し込んだ後のメソッドを包む（記録は規則の上にかぶせる）。
            recorder = ProbabilityTraceRecorder()
            recorder.install()
        try:
            outcome = _execute_planned_experiment_run(
                planned_run=planned_run, output_directory=output_directory, run_provenance=run_provenance
            )
        finally:
            if recorder is not None:
                recorder.uninstall()
    finally:
        restore_weight_rules()
    if outcome.run_status == "completed" and recorder is not None:
        recorder.write(output_directory, planned_run.run_identifier, variant)
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
    record_probabilities = variant in arguments.record_prediction_probabilities_for
    record_path = output_directory / _VARIANT_RECORD_NAME
    if record_path.exists():
        record = _read_json(record_path)
        if record.get("variant") != variant:
            raise ValueError(f"出力先に別の変種の記録があります: {record_path}")
        # 基点の記録が無いのは、基点を選べるようにする前（2026-11-25の最初の5変種）の、製品の既定を基点にした結果。
        if record.get("base", PRODUCT_BASE) != arguments.base:
            raise ValueError(f"出力先は別の基点（{record.get('base', PRODUCT_BASE)}）の結果です: {record_path}")
        # 記録の有無が途中で変わると、記録の無い run が混ざる（記録の指定の無い変種の記録は、記録なし）。
        if record.get("records_prediction_probabilities", False) != record_probabilities:
            raise ValueError(f"出力先は確率の記録の有無が違う結果です: {record_path}")
    elif output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(f"変種の記録のない出力先は使えません: {output_directory}")
    sweep_arguments = _sweep_arguments(arguments, variant, output_directory)
    patched_build = None if arguments.base == PRODUCT_BASE else make_simplified_build(arguments.base)
    if arguments.print_plan:
        if patched_build is not None:
            experiment_sweep_plan.build_experiment_run_settings = patched_build
        try:
            return run_sweep([*sweep_arguments, "--print-plan"])
        finally:
            experiment_sweep_plan.build_experiment_run_settings = _original_build
    if not record_path.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            record_path,
            dict(
                variant=variant,
                base=arguments.base,
                alarm_share_rate=ALARM_SHARE_RATE,
                log_loss_floor=LOG_LOSS_PROBABILITY_FLOOR,
                tempered_log_loss_learning_rates=TEMPERED_LOG_LOSS_LEARNING_RATES,
                learning_rate_mixture_candidates=LEARNING_RATE_MIXTURE_CANDIDATES,
                records_prediction_probabilities=record_probabilities,
                **_provenance(),
            ),
        )
    previous_variant = os.environ.get(_VARIANT_ENVIRONMENT_VARIABLE)
    previous_record = os.environ.get(_RECORD_ENVIRONMENT_VARIABLE)
    os.environ[_VARIANT_ENVIRONMENT_VARIABLE] = variant
    os.environ[_RECORD_ENVIRONMENT_VARIABLE] = "1" if record_probabilities else "0"
    experiment_sweep_execution.execute_planned_experiment_run = _execute_run_with_rule
    if patched_build is not None:
        # 計画（run設定）は親のprocessで作るので、計画の関数の差し替えは親だけでよい（簡素版の研究ツールと同じ）。
        experiment_sweep_plan.build_experiment_run_settings = patched_build
    try:
        status = run_sweep(sweep_arguments)
        if status == 0:
            # 失敗の無い掃引の後だけ、runごとのトレースを1つのファイルへまとめる（中断・失敗のときは残して再開に備える）。
            consolidate_traces(output_directory)
            if record_probabilities:
                consolidate_probability_traces(output_directory)
        return status
    finally:
        experiment_sweep_execution.execute_planned_experiment_run = _execute_planned_experiment_run
        experiment_sweep_plan.build_experiment_run_settings = _original_build
        if previous_variant is None:
            os.environ.pop(_VARIANT_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[_VARIANT_ENVIRONMENT_VARIABLE] = previous_variant
        if previous_record is None:
            os.environ.pop(_RECORD_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[_RECORD_ENVIRONMENT_VARIABLE] = previous_record


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
    parser.add_argument(
        "--base",
        choices=BASES,
        default=PRODUCT_BASE,
        help="重みの規則を差し替える基点の構成。既定は製品の fedsda、minimal-soft-prediction は簡素版の研究ツールの最小形",
    )
    parser.add_argument(
        "--record-prediction-probabilities-for",
        nargs="+",
        choices=[variant for variant in VARIANTS if variant not in ALARM_SHARE_BOOST_VARIANTS],
        default=[],
        help="この変種の run で、重みの制御への呼出しの列と予測ごとのモデル別の確率を記録する（replay.py が使う）",
    )
    parser.add_argument("--purpose", default=None)
    parser.add_argument("--print-plan", action="store_true")
    arguments = parser.parse_args(command_arguments)
    unknown_record_variants = set(arguments.record_prediction_probabilities_for) - set(arguments.variants)
    if unknown_record_variants:
        parser.error(f"確率を記録する変種は --variants にも入れてください: {sorted(unknown_record_variants)}")
    if arguments.purpose is None:
        arguments.purpose = (
            "FedHASP基準版16のモデルの混合の重みの更新の規則を変えた変種"
            if arguments.base == PRODUCT_BASE
            else f"FedHASPの簡素版の最小形（{arguments.base}）のモデルの混合の重みの更新の規則を変えた変種"
        )
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
