"""候補を「仮の専門家」として予測だけに使う案（案A）を調べる、独立した研究用CLI。共通設定への登録はしない。

案A: 候補検証の間は、検証中の候補（固定）を、検証の後に採用されなかった候補は、その写しをclientの中で標本ごとに
学習し続けて、予測に使う。予測は、確定したモデル群の最終予測（混ぜる予測）と仮の専門家の予測を、clientごとの
2者の重みで混ぜる（重みの更新は、同期の時点と局所を混ぜる予測と同じ式: 学習率2、損失は1−真のクラスの確率、
分け合いの割合1/T）。仮の専門家は、次の候補検証が始まると、その候補に置き換わる。採用された候補は、通常の
保有モデルになるので、仮の専門家から外す。

予測の記録の`combined_prediction_is_correct`と`any_model_or_combined_prediction_is_correct`だけを、混ぜた予測で
書き換える。予測の関数の戻り値・モデル・帰属・検出・通信には触れないので、学習の側は、検証あり（`fedsda`）と
同じになる（実行の後に、照合する）。仮の専門家の計算は乱数を使わない。

使い方:
  python tools/research_provisional_candidate_expert/run.py --output <dir> --datasets sea4 mnist2 \
      --intervals 100 500 --seeds 0 1 2 --sample-count 2500 --workers 12
  （--check で、仮の専門家の重みを常に0にした実行が、検証ありと予測の記録まで同じであることを確かめる）
"""

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
# 変種 -> （構成、仮の専門家を使うか、検証の標本数——Noneなら構成の値の10——）。2026-10-20に既定（`fedsda`）が
# 候補検証なし＋候補の共有部の固定になったので、検証ありの変種は、それまでの既定と同じ設定の構成で選ぶ。
# 検証あり（V）、検証なし（O）、検証あり＋仮の専門家（A）。B0: 検証の標本数を30にした検証あり（V30）と、
# それに仮の専門家を足したもの（A30。仮の専門家があれば、検証を延ばしても予測の損が無い）。
# 案B: 検証の標本数の上限を30にし、候補と参照の損失の差のe-過程で、証拠が閾値に達したら早く打ち切る
# （打ち切らずに30標本へ達したら、今の規則で判定する。検証30との違いは、早く打ち切ることだけ）。
_VARIANTS = {
    "validation": ("fedsda-with-candidate-validation", False, None),
    "no-validation": ("fedsda-without-candidate-validation", False, None),
    "provisional-expert": ("fedsda-with-candidate-validation", True, None),
    "validation-forward30": ("fedsda-with-candidate-validation", False, 30),
    "provisional-expert-forward30": ("fedsda-with-candidate-validation", True, 30),
    "sequential-test": ("fedsda-with-candidate-validation", False, 30),
    "provisional-expert-sequential-test": ("fedsda-with-candidate-validation", True, 30),
    # アイデア1: 検証なし（すぐ採用）で、送信までの待ちの同期回数（既定1）を延ばす。既定の登録の扱いは、
    # 待ちの間も現行であり続けたモデルだけを登録するので、短命なモデルはサーバへ共有されない。
    "no-validation-upload-delay2": ("fedsda-without-candidate-validation", False, None),
    "no-validation-upload-delay3": ("fedsda-without-candidate-validation", False, None),
    # アイデア2: 候補の学習で共有部を更新しない（候補のheadとadapterだけを、警報の時点の共有部の上で学習する）。
    # 検証なしでは、候補をその場で採用するので、採用のときの共有部の上書きが実質的に起きない。
    "no-validation-frozen-candidate-backbone": ("fedsda-without-candidate-validation", False, None),
    "validation-frozen-candidate-backbone": ("fedsda-with-candidate-validation", False, None),
    # IMPROVE-022: 既定（2026-10-20から、候補検証なし＋候補の共有部の固定）と、その候補の学習を、早期終了
    # （区間の20%を検証用に取り分け、最大30エポック、改善が3回続けて無ければ止めて最良へ戻す）から、全標本での
    # 固定エポックの学習へ替えたもの。
    "default": ("fedsda", False, None),
    "default-fixed-epochs5": ("fedsda", False, None),
    "default-fixed-epochs10": ("fedsda", False, None),
    "default-fixed-epochs30": ("fedsda", False, None),
}
# IMPROVE-024: 警報の時点で、切り詰めない推定の変化点（警報の位置−検出器が推定した変化の長さ＋1）より後で、
# 保留の窓より前に、現在のモデルへ解放した標本を、そのモデルの学習データ・概念の計数・損失の統計から外す。
_WITHDRAW_RELEASED_SAMPLE_VARIANTS = (
    "default-withdraw-released",
    "default-withdraw-released-into-interval",
)
_VARIANTS["default-withdraw-released"] = ("fedsda", False, None)
# 外した標本を、警報の変化区間の先頭へ足す（再利用の判定・候補の学習・帰属の材料に含める）。
_VARIANTS["default-withdraw-released-into-interval"] = ("fedsda", False, None)
# IMPROVE-021の案A: 警報の区間の再利用の判定を、固定の閾値（区間の平均損失≦履歴の基準＋0.1）から、
# 区間の累積の対数損失の比較へ替える。比べるのは、履歴の基準を持つ各保有モデル（固定）と、「新しいモデル」
# （候補の初期値と同じ、区間の平均損失が最小の保有モデルから始め、区間を1標本ずつ、予測してから学習する逐次の
# 学習者。共有部は固定し、候補の学習と同じ損失とoptimizerの設定で、1標本ごとに1回更新する）。累積の対数損失が
# 最小のものを選び、新しいモデルが最小なら、今と同じ候補の学習・採用へ進む。閾値は使わない。
_PREQUENTIAL_REUSE_VARIANTS = (
    "default-prequential-reuse",
    "default-prequential-reuse-fresh-learner",
    "default-prequential-reuse-fresh-learner-bounded-loss",
    "default-prequential-reuse-fresh-learner-horizon",
    "default-prequential-reuse-fresh-learner-horizon-evaluation-samples",
    "default-prequential-reuse-fresh-learner-horizon-interval",
)
# 先の長さを使う形で、Hを、区間の長さnそのものにする（「この概念は、これまで観測した長さと同じくらい、この先も
# 続く」とみなす）。HにFixed-ShareのT（120）を使うと、新しいモデルを作りすぎた（IMPROVE-021の16・17）。
_VARIANTS["default-prequential-reuse-fresh-learner-horizon-interval"] = ("fedsda", False, None)
# 先の長さを使う形で、ℓ_mを、学習に使っていない標本（警報の前区間から抜き出して保存している評価用の標本、
# `ModelEvaluationSampleStore`）で測る。学習の標本で測ると水準を楽観的に見積もり、新しいモデルを作りすぎる
# （IMPROVE-021の15）。評価用の標本が無いモデルは、学習の標本で測る（その数を記録する）。
_VARIANTS["default-prequential-reuse-fresh-learner-horizon-evaluation-samples"] = ("fedsda", False, None)
# 学習者を初期化し直す形に、保有モデルの「自分の概念での損失の水準」を、先の長さHの期待損失として入れる
# （IMPROVE-021の14、MNIST2の崩れ）。保有モデルmの再利用は、(n＋H)×（区間でのmの1標本あたりの対数損失）、
# 新しいモデルは、（逐次の学習者の区間の累積の対数損失）＋H×ℓ_m（ℓ_mは、mが自分の学習の標本で出す1標本あたりの
# 対数損失。専用のモデルを作れば、mが自分の概念で出す水準まで下がるとみなす）。再利用の方が小さいモデルのうち、
# 区間の損失が最小のものを再利用する。Hは、Fixed-Shareの時間尺度T（既存の設定値）。
_VARIANTS["default-prequential-reuse-fresh-learner-horizon"] = ("fedsda", False, None)
# 学習者を初期化し直す形で、累積を、対数損失ではなく、手法の予測の重みと判定が使う有界の損失（1−真のクラスの確率）で
# 比べる。上限の無い対数損失では、少数の自信を持った誤りで保有モデルの累積が大きくなり、学び始めの学習者
# （2値で1標本あたり約ln2）に負ける（IMPROVE-021の11、CIRCLE2の見逃し）。
_VARIANTS["default-prequential-reuse-fresh-learner-bounded-loss"] = ("fedsda", False, None)
_VARIANTS["default-prequential-reuse"] = ("fedsda", False, None)
# 逐次の学習者を、概念ごとの部分（adapterと出力の層）を初期化し直した状態から始める（共有部はそのまま）。
# 最も合う保有モデルから始めると、「そのモデルを区間で学習し直すと良くなるか」の比較になり、同じ概念でも
# 新しいモデルの側へ偏るため。初期化の乱数は、runの乱数の状態を変えないように、分けて使う。
_VARIANTS["default-prequential-reuse-fresh-learner"] = ("fedsda", False, None)
# 帰属を1つの推論にまとめる案（`docs/research/fedsda-unified-concept-assignment-inference.md`の3）。検出器の合図と、
# 再利用の判定（閾値0.1）を、直近L標本（帰属の保留の容量）の窓の中の区分けの比較（符号長）に置き換える。
_UNIFIED_ASSIGNMENT_INFERENCE_VARIANTS = ("default-unified-assignment-inference",)
_VARIANTS["default-unified-assignment-inference"] = ("fedsda", False, None)
# 同じ設計の7の(a): 「いつ」は今の検出器（e-SR、誤警報の率を抑える）に任せ、警報のときだけ、帰属の保留の標本
# （と警報の標本）の中で区分けの符号長を比べて、変化点と行き先（現在のモデルのまま、別の保有モデル、新しい概念）を
# 決める。再利用の判定の閾値（0.1）と、検出器の推定の変化の長さを置き換える。
_DETECTOR_GATED_SEGMENT_ASSIGNMENT_VARIANTS = ("default-detector-gated-segment-assignment",)
_VARIANTS["default-detector-gated-segment-assignment"] = ("fedsda", False, None)
# 製品の経路の、最小記述長の原理で選ぶ扱い（`alarm_interval_assignment_policy`）を選ぶだけで、研究ツールの包みは
# 使わない（研究ツールの変種`default-detector-gated-segment-assignment`と同じ値になることの照合用。spec
# `fedsda-minimum-description-length-alarm-interval-assignment`のTask 3）。
_VARIANTS["default-minimum-description-length"] = ("fedsda", False, None)
_FIXED_CANDIDATE_EPOCH_COUNT_BY_VARIANT = {
    "default-fixed-epochs5": 5,
    "default-fixed-epochs10": 10,
    "default-fixed-epochs30": 30,
}
_FROZEN_CANDIDATE_BACKBONE_VARIANTS = (
    "no-validation-frozen-candidate-backbone",
    "validation-frozen-candidate-backbone",
)
_UPLOAD_DELAY_ROUND_COUNT_BY_VARIANT = {
    "no-validation-upload-delay2": 2,
    "no-validation-upload-delay3": 3,
}
_SEQUENTIAL_TEST_VARIANTS = ("sequential-test", "provisional-expert-sequential-test")
# 候補の判定のe-過程の有意水準（閾値1/α）と、賭けの割合の混合。αは検出（ClassESR、0.001）とは別に選んだ値。
_SEQUENTIAL_TEST_ALPHA = 0.05
_SEQUENTIAL_TEST_BETTING_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5)
_LEARNING_RATE = 2.0
_INITIAL_PROVISIONAL_WEIGHT = 0.5


class _ProvisionalExpert:
    """1 clientの仮の専門家（分類器、学習の要否、optimizer、混ぜる重み）。"""

    def __init__(self, *, classifier, trains, optimizer_settings, weight):
        import torch

        self.classifier = classifier
        self.trains = trains
        self.weight = weight
        self.optimizer = None
        if trains:
            self.optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=optimizer_settings.learning_rate,
                weight_decay=optimizer_settings.weight_decay,
                amsgrad=optimizer_settings.adam_variant == "amsgrad",
            )


def _true_class_loss(probabilities, class_id, class_count):
    values = probabilities.view(-1)
    if class_count == 2:
        positive = float(values[0])
        return 1.0 - positive if class_id == 1 else positive
    return 1.0 - float(values[class_id])


@contextmanager
def _provisional_expert_policy(*, share_rate, optimizer_settings, check, events):
    import torch
    from torch.nn import BCELoss, CrossEntropyLoss

    import federated_learning_experiments.runtime.alarm_change_interval_resolution as resolution_module
    import federated_learning_experiments.runtime.held_candidate_validation_progress as progress_module
    import federated_learning_experiments.runtime.observed_sample_processing as processing_module
    from federated_learning_experiments.learning.prediction.class_probability_calculations import (
        convert_model_outputs_to_prediction_probabilities,
        predict_class_labels_from_prediction_scores,
    )

    # clientの鍵は、clientごとの保有モデルの登録簿（予測・検証の開始・確定の全部に渡る）。
    experts = {}
    original_start = resolution_module.start_post_alarm_candidate_validation_session
    original_advance = progress_module.advance_post_alarm_candidate_validation
    original_predict = processing_module.predict_observed_sample_and_update_prediction_weights

    def start(**arguments):
        session = original_start(**arguments)
        key = id(arguments["held_model_training_state_registry"])
        previous = experts.get(key)
        experts[key] = _ProvisionalExpert(
            classifier=session.candidate_training_state.candidate_classifier,
            trains=False,
            optimizer_settings=optimizer_settings,
            # 前の仮の専門家の重みを引き継がない（新しい候補）。
            weight=0.0 if check else _INITIAL_PROVISIONAL_WEIGHT,
        )
        events.append(dict(event="session_start", sample_index=arguments["proposal_sample_index"], replaced=previous is not None))
        return session

    def advance(**arguments):
        session = arguments["validation_session"]
        progress = original_advance(**arguments)
        completed = progress.completed_validation
        if completed is None:
            return progress
        key = id(arguments["held_model_training_state_registry"])
        outcome = completed.validation_resolution.resolution_outcome
        events.append(dict(event="session_end", sample_index=arguments["sample_index"], outcome=outcome))
        if outcome == "candidate_adopted_as_new_model":
            experts.pop(key, None)
        else:
            expert = experts.get(key)
            weight = expert.weight if expert is not None else _INITIAL_PROVISIONAL_WEIGHT
            # 採用されなかった候補の写しを、clientの中で学習し続ける（保有モデル・共有部・通信には入れない）。
            experts[key] = _ProvisionalExpert(
                classifier=copy.deepcopy(session.candidate_training_state.candidate_classifier),
                trains=True,
                optimizer_settings=optimizer_settings,
                weight=0.0 if check else weight,
            )
        return progress

    def predict(**arguments):
        prediction = original_predict(**arguments)
        key = id(arguments["held_model_training_state_registry"])
        expert = experts.get(key)
        if expert is None:
            return prediction
        sample = arguments["indexed_observation"].training_sample
        input_features = sample.input_features
        observed_class_labels = sample.observed_class_labels
        main_probabilities = prediction.combined_prediction_probabilities
        class_count = 2 if main_probabilities.shape[1] == 1 else main_probabilities.shape[1]
        expert.classifier.eval()
        with torch.no_grad():
            expert_probabilities = convert_model_outputs_to_prediction_probabilities(
                model_outputs_by_model_id={0: expert.classifier(input_features)},
                class_count=class_count,
            )[0]
        weight = expert.weight
        mixed_probabilities = (1.0 - weight) * main_probabilities + weight * expert_probabilities
        predicted = predict_class_labels_from_prediction_scores(
            prediction_scores=mixed_probabilities, class_count=class_count
        )
        observed_class_id = int(observed_class_labels.view(-1)[0].item())
        mixed_is_correct = bool(int(predicted.view(-1)[0].item()) == observed_class_id)
        store = arguments["sample_prediction_record_store"]
        record = store._sample_prediction_records[-1]
        store._sample_prediction_records[-1] = replace(
            record,
            combined_prediction_is_correct=mixed_is_correct,
            any_model_or_combined_prediction_is_correct=(
                mixed_is_correct or record.any_model_or_combined_prediction_is_correct
            ),
        )
        # ラベル観測後: 2者の重みを更新し（重みを常に0にする照合では更新しない）、学習する仮の専門家なら1歩学習する。
        if not check:
            main_loss = _true_class_loss(main_probabilities, observed_class_id, class_count)
            expert_loss = _true_class_loss(expert_probabilities, observed_class_id, class_count)
            main_weight = (1.0 - weight) * math.exp(-_LEARNING_RATE * main_loss)
            expert_weight = weight * math.exp(-_LEARNING_RATE * expert_loss)
            total = main_weight + expert_weight
            main_weight, expert_weight = main_weight / total, expert_weight / total
            expert.weight = (1.0 - share_rate) * expert_weight + share_rate * main_weight
        if expert.trains:
            expert.classifier.train()
            expert.optimizer.zero_grad()
            outputs = expert.classifier(input_features)
            loss = (
                BCELoss()(outputs, observed_class_labels.to(outputs.dtype))
                if class_count == 2
                else CrossEntropyLoss()(outputs, observed_class_labels.view(-1).long())
            )
            loss.backward()
            expert.optimizer.step()
            expert.classifier.eval()
        return prediction

    with (
        patch.object(resolution_module, "start_post_alarm_candidate_validation_session", start),
        patch.object(progress_module, "advance_post_alarm_candidate_validation", advance),
        patch.object(processing_module, "predict_observed_sample_and_update_prediction_weights", predict),
    ):
        yield experts


@contextmanager
def _sequential_test_policy(*, events):
    """検証の中で、候補と参照の損失の差のe-過程により、早く打ち切る（研究用。実行の間だけ包む）。

    比較する参照は、その標本より前の累積損失が最小の参照（予測可能な選び方）。差d＝参照の損失−候補の損失は
    [−1, 1]なので、賭けの割合λ∈[0.1, 0.5]の積∏(1＋λd)は、候補が良くない（E[d]≦0）ときの検定の
    martingaleになり、その平均もe-過程になる。候補が良い側の証拠と、悪い側（−dで同じ形）の証拠の、どちらかが
    1/αに達したら打ち切る。打ち切ったときは、既存モデルの再適合の判定は今の規則のまま、それまでの標本で行い、
    再適合が無ければ、候補の採否をe-過程の向きで決める。
    """
    import federated_learning_experiments.runtime.post_alarm_candidate_validation_progress as progress_module

    original_observe = progress_module.observe_post_alarm_candidate_validation_sample
    original_evaluate = progress_module.evaluate_candidate_using_post_alarm_losses
    threshold = 1.0 / _SEQUENTIAL_TEST_ALPHA
    # 集めている途中の検証（損失を集める部品のid）→ 打ち切りの向き（"accept"か"reject"）。
    early_decisions = {}

    def sequential_decision(state):
        candidate_losses = state.candidate_losses
        reference_losses_by_model_id = dict(state.reference_losses_by_model_id)
        favorable = [1.0] * len(_SEQUENTIAL_TEST_BETTING_FRACTIONS)
        unfavorable = [1.0] * len(_SEQUENTIAL_TEST_BETTING_FRACTIONS)
        for index, candidate_loss in enumerate(candidate_losses):
            reference_id = min(
                reference_losses_by_model_id,
                key=lambda model_id: (sum(reference_losses_by_model_id[model_id][:index]), model_id),
            )
            difference = reference_losses_by_model_id[reference_id][index] - candidate_loss
            for position, fraction in enumerate(_SEQUENTIAL_TEST_BETTING_FRACTIONS):
                favorable[position] *= 1.0 + fraction * difference
                unfavorable[position] *= 1.0 - fraction * difference
        favorable_evidence = sum(favorable) / len(favorable)
        unfavorable_evidence = sum(unfavorable) / len(unfavorable)
        if favorable_evidence >= threshold:
            return "accept"
        if unfavorable_evidence >= threshold:
            return "reject"
        return None

    def observe(**arguments):
        collection = arguments["post_alarm_candidate_loss_collection"]
        complete = original_observe(**arguments)
        state = collection.get_state_snapshot()
        if complete or len(state.candidate_losses) < 2:
            return complete
        decision = sequential_decision(state)
        if decision is None:
            return False
        early_decisions[id(collection)] = decision
        return True

    def evaluate(**arguments):
        candidate_losses = arguments["candidate_losses"]
        settings = arguments["candidate_model_training_and_acceptance_settings"]
        if len(candidate_losses) >= settings.candidate_post_alarm_validation_sample_count:
            events.append(dict(event="sequential_test", decision="timeout", sample_count=len(candidate_losses)))
            return original_evaluate(**arguments)
        # 打ち切った検証。集めた件数で、今の規則（再適合の判定）を行う。
        decision = None
        for key, value in list(early_decisions.items()):
            decision = value
            del early_decisions[key]
        shortened = replace(settings, candidate_post_alarm_validation_sample_count=len(candidate_losses))
        evaluation = original_evaluate(**{**arguments, "candidate_model_training_and_acceptance_settings": shortened})
        events.append(dict(event="sequential_test", decision=decision, sample_count=len(candidate_losses)))
        if evaluation.reusable_reference_model_id is not None:
            return evaluation
        return replace(
            evaluation,
            candidate_accepted=decision == "accept",
            decision_reason=f"sequential_test_{decision}",
        )

    with (
        patch.object(progress_module, "observe_post_alarm_candidate_validation_sample", observe),
        patch.object(progress_module, "evaluate_candidate_using_post_alarm_losses", evaluate),
    ):
        yield


def _remove_bounded_loss_observation(*, loss_moments, observed_loss):
    """`accumulate_bounded_loss_observation`の逆（1件を外した件数・平均・偏差平方和）。"""
    from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
        BoundedLossMoments,
    )

    count = loss_moments.observed_loss_count - 1
    if count <= 0:
        return BoundedLossMoments(observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0)
    previous_mean = (loss_moments.mean_loss * loss_moments.observed_loss_count - observed_loss) / count
    deviations = loss_moments.sum_squared_loss_deviations - (observed_loss - previous_mean) * (
        observed_loss - loss_moments.mean_loss
    )
    return BoundedLossMoments(
        observed_loss_count=count,
        mean_loss=min(1.0, max(0.0, previous_mean)),
        sum_squared_loss_deviations=max(0.0, deviations),
    )


@contextmanager
def _withdraw_released_samples_policy(*, events, includes_withdrawn_samples_in_change_interval=False):
    """保留から現在のモデルへ解放した標本を記録し、警報の処理の直前に、推定の変化点より後のものを外す（研究用）。

    `includes_withdrawn_samples_in_change_interval`がTrueなら、外した標本を、その警報の変化区間の先頭へ
    （標本の番号の順に）足して、変化区間の解決（再利用の判定、候補の学習、帰属）へ渡す。変化区間の解決を
    呼ばない警報（候補の検証中、短すぎる区間）では、外した標本は、どのモデルにも戻さない（数だけ記録する）。
    """
    import federated_learning_experiments.runtime.alarm_buffer_response as response_module
    import federated_learning_experiments.runtime.alarm_occurrence_handling as alarm_module
    import federated_learning_experiments.runtime.observed_sample_processing as processing_module
    from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
        ModelAndClassLossStatistics,
        ModelAndClassLossStatisticsStore,
    )

    original_release = processing_module.assign_released_pending_samples_to_current_training_model
    original_respond = alarm_module.respond_to_alarm_with_buffered_samples
    original_record_loss = ModelAndClassLossStatisticsStore.record_assigned_loss
    original_resolve = response_module.resolve_alarm_change_interval
    # 保留のownerごとの、解放の記録（標本の番号、標本のオブジェクト、観測概念、損失、クラス）。
    releases = {}
    capturing = []
    # 処理中の警報で外した、変化区間へ足す標本（標本の番号の順の、（標本、観測概念））。
    withdrawn_for_interval = []

    def resolve(**arguments):
        prefix = list(withdrawn_for_interval)
        withdrawn_for_interval.clear()
        if prefix:
            arguments = {
                **arguments,
                "change_interval_training_samples": (
                    *(sample for sample, _ in prefix),
                    *arguments["change_interval_training_samples"],
                ),
                "change_interval_sample_concept_ids": (
                    *(concept for _, concept in prefix),
                    *arguments["change_interval_sample_concept_ids"],
                ),
            }
            events.append(dict(event="withdrawn_into_interval", count=len(prefix)))
        return original_resolve(**arguments)

    def record_loss(self, *, model_id, observed_loss, observed_class_id=None):
        if capturing:
            capturing[-1].append((observed_loss, observed_class_id))
        return original_record_loss(
            self, model_id=model_id, observed_loss=observed_loss, observed_class_id=observed_class_id
        )

    def release(**arguments):
        capturing.append([])
        try:
            released = original_release(**arguments)
        finally:
            losses = capturing.pop()
        history = releases.setdefault(id(arguments["pending_training_assignment_buffer"]), [])
        for observation, (loss, class_id) in zip(released, losses, strict=True):
            history.append((observation.sample_index, observation.training_sample, observation.observed_concept_id, loss, class_id))
        # 古い記録は捨てる（推定の変化の長さの上限——検出器の候補の上限1000——より古いものは使わない）。
        if len(history) > 2000:
            del history[: len(history) - 2000]
        return released

    def respond(**arguments):
        pending = arguments["pending_sample_observations"]
        history = releases.get(id(arguments["pending_training_assignment_buffer"]), [])
        estimate = arguments["proposal_sample_index"] - arguments["estimated_change_span_sample_count"] + 1
        if pending and history and estimate < pending[0].sample_index:
            model_id = arguments["current_training_model_assignment"].current_training_model_id
            store = arguments["training_sample_store"]
            samples = store._training_samples_by_model_id.get(model_id, [])
            candidates = [r for r in history if estimate <= r[0] < pending[0].sample_index]
            candidate_ids = {id(r[1]) for r in candidates}
            present = {id(s) for s in samples} & candidate_ids
            if present:
                store._training_samples_by_model_id[model_id] = [s for s in samples if id(s) not in present]
                counts = arguments["model_training_and_assignment_counts_store"]
                concept_counts = counts._assigned_sample_counts_by_model_and_concept_id.get(model_id, {})
                loss_store = arguments["loss_statistics_store"]
                statistics = loss_store._model_loss_statistics_by_model_id.get(model_id)
                overall = statistics.overall_loss_moments if statistics is not None else None
                classes = dict(statistics.class_loss_moments_by_class_id) if statistics is not None else {}
                new_concept = pending[-1].observed_concept_id
                withdrawn_new = 0
                for _, sample, concept, loss, class_id in candidates:
                    if id(sample) not in present:
                        continue
                    if concept is not None and concept_counts.get(concept, 0) > 0:
                        concept_counts[concept] -= 1
                    if overall is not None and overall.observed_loss_count > 0:
                        overall = _remove_bounded_loss_observation(loss_moments=overall, observed_loss=loss)
                    if class_id in classes and classes[class_id].observed_loss_count > 0:
                        classes[class_id] = _remove_bounded_loss_observation(loss_moments=classes[class_id], observed_loss=loss)
                    withdrawn_new += concept == new_concept
                if statistics is not None:
                    loss_store._model_loss_statistics_by_model_id[model_id] = ModelAndClassLossStatistics(
                        overall_loss_moments=overall, class_loss_moments_by_class_id=tuple(classes.items())
                    )
                events.append(dict(event="withdraw_released", count=len(present), new_concept=withdrawn_new))
                if includes_withdrawn_samples_in_change_interval:
                    withdrawn_for_interval.extend(
                        (sample, concept)
                        for _, sample, concept, _, _ in sorted(candidates, key=lambda r: r[0])
                        if id(sample) in present
                    )
                # 外した標本は、記録からも除く（二重に外さない）。
                releases[id(arguments["pending_training_assignment_buffer"])] = [r for r in history if id(r[1]) not in present]
        try:
            return original_respond(**arguments)
        finally:
            # 変化区間の解決を呼ばなかった警報で、外したまま、どのモデルにも戻さなかった標本の数。
            if withdrawn_for_interval:
                events.append(dict(event="withdrawn_not_reassigned", count=len(withdrawn_for_interval)))
                withdrawn_for_interval.clear()

    with (
        patch.object(processing_module, "assign_released_pending_samples_to_current_training_model", release),
        patch.object(alarm_module, "respond_to_alarm_with_buffered_samples", respond),
        patch.object(ModelAndClassLossStatisticsStore, "record_assigned_loss", record_loss),
        patch.object(response_module, "resolve_alarm_change_interval", resolve),
    ):
        yield


@contextmanager
def _prequential_reuse_policy(
    *,
    optimizer_settings,
    events,
    starts_learner_fresh=False,
    compares_bounded_losses=False,
    horizon_sample_count=None,
    measures_own_loss_on_evaluation_samples=False,
):
    """警報の区間の再利用の判定を、累積の対数損失の比較へ替える（研究用。実行の間だけ包む）。"""
    import torch
    from torch.nn import BCELoss, CrossEntropyLoss

    import federated_learning_experiments.runtime.alarm_change_interval_resolution as resolution_module
    from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
        evaluate_classifier_per_sample_bounded_losses,
    )
    from federated_learning_experiments.learning.training.parameter_optimizer_construction import (
        create_parameter_optimizer,
    )
    from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
        AlarmIntervalModelReuseAssessment,
    )

    import federated_learning_experiments.runtime.alarm_buffer_response as buffer_module

    original_evaluate = resolution_module.evaluate_held_models_for_alarm_interval_reuse
    original_resolve = buffer_module.resolve_alarm_change_interval
    # 解決中の警報の、学習の標本のowner（先の長さを使う形で、保有モデルの自分の損失の水準を計算するため）。
    resolving = {}

    def resolve(**arguments):
        resolving["training_sample_store"] = arguments["training_sample_store"]
        try:
            return original_resolve(**arguments)
        finally:
            resolving.clear()

    import federated_learning_experiments.runtime.alarm_occurrence_handling as alarm_handling_module

    original_respond = alarm_handling_module.respond_to_alarm_with_buffered_samples
    evaluation_sample_stores = []

    def respond(**arguments):
        evaluation_sample_stores.append(arguments["model_evaluation_sample_store"])
        try:
            return original_respond(**arguments)
        finally:
            evaluation_sample_stores.pop()

    def own_mean_log_loss(classifier, model_id):
        if measures_own_loss_on_evaluation_samples and evaluation_sample_stores:
            collections = evaluation_sample_stores[-1].snapshot_ordered_model_evaluation_samples()
            samples = next((c.evaluation_samples for c in collections if c.model_id == model_id), ())
            if samples:
                features = torch.cat([s.input_features for s in samples])
                labels = torch.cat([s.observed_class_labels for s in samples])
                return float(log_losses(classifier, features, labels).mean())
            events.append(dict(event="own_loss_fallback_to_training_samples"))
        collections = resolving["training_sample_store"].snapshot_ordered_model_training_samples()
        samples = next((c.training_samples for c in collections if c.model_id == model_id), ())
        if not samples:
            return None
        features = torch.cat([s.input_features for s in samples])
        labels = torch.cat([s.observed_class_labels for s in samples])
        return float(log_losses(classifier, features, labels).mean())

    def log_losses(classifier, input_features, observed_class_labels):
        # 有界の損失は1−真のクラスの確率なので、対数損失は−log(1−有界の損失)。
        bounded = evaluate_classifier_per_sample_bounded_losses(
            classifier=classifier, input_features=input_features, observed_class_labels=observed_class_labels
        )
        if compares_bounded_losses:
            return bounded
        return -torch.log(torch.clamp(1.0 - bounded, min=1e-12))

    decision_count = [0]

    def prequential_log_loss(classifier, input_features, observed_class_labels):
        learner = copy.deepcopy(classifier)
        if starts_learner_fresh:
            from torch.nn.init import zeros_

            decision_count[0] += 1
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(decision_count[0])
                for module in (learner.model_owned_feature_transform, learner.classification_layer):
                    for submodule in module.modules():
                        if hasattr(submodule, "reset_parameters"):
                            submodule.reset_parameters()
                # 残差のadapterは、広げる側を0で始める（構造の初期化と同じ）。
                expansion = getattr(learner.model_owned_feature_transform, "feature_expansion", None)
                if expansion is not None:
                    zeros_(expansion.weight)
                    zeros_(expansion.bias)
        parameters = tuple(
            parameter
            for name, parameter in learner.named_parameters()
            if not name.startswith("feature_extractor.")
        )
        optimizer = create_parameter_optimizer(parameters=parameters, optimizer_settings=optimizer_settings)
        total = 0.0
        for index in range(input_features.shape[0]):
            x = input_features[index : index + 1]
            y = observed_class_labels[index : index + 1]
            total += float(log_losses(learner, x, y).sum())
            with torch.no_grad():
                shared_features = learner.extract_shared_features(x)
            optimizer.zero_grad()
            predictions = learner.forward_from_shared_features(shared_features)
            if learner.class_count == 2:
                loss = BCELoss()(predictions, y)
            else:
                loss = CrossEntropyLoss()(predictions, y.reshape(-1).long())
            loss.backward()
            optimizer.step()
        return total

    def evaluate(**arguments):
        assessment = original_evaluate(**arguments)
        supported = assessment.baseline_supported_interval_mean_losses_by_model_id
        if not supported:
            return assessment
        registry = arguments["held_model_training_state_registry"]
        x, y = arguments["input_features"], arguments["observed_class_labels"]
        classifiers = {
            model_id: registry.get_held_model_training_state(model_id=model_id).classifier
            for model_id, _ in supported
        }
        held_totals = {model_id: float(log_losses(classifiers[model_id], x, y).sum()) for model_id, _ in supported}
        # 候補の初期値（区間の平均損失が最小の保有モデル。同率は列の先着）から始める逐次の学習者。
        initial_model_id = min(supported, key=lambda evaluated: evaluated[1])[0]
        new_total = prequential_log_loss(classifiers[initial_model_id], x, y)
        if horizon_sample_count is None:
            best_model_id = min(held_totals, key=lambda model_id: held_totals[model_id])
            reuses = held_totals[best_model_id] <= new_total
        else:
            n = int(x.shape[0])
            horizon = n if horizon_sample_count == "interval" else horizon_sample_count
            reusable_ids = []
            for model_id, total in held_totals.items():
                own = own_mean_log_loss(classifiers[model_id], model_id)
                if own is None:
                    continue
                if (n + horizon) * (total / n) <= new_total + horizon * own:
                    reusable_ids.append(model_id)
            reuses = bool(reusable_ids)
            best_model_id = (
                min(reusable_ids, key=lambda model_id: held_totals[model_id])
                if reusable_ids
                else min(held_totals, key=lambda model_id: held_totals[model_id])
            )
        events.append(
            dict(
                event="prequential_reuse_decision",
                reuse=reuses,
                original_reuse=assessment.selected_reuse_model_id is not None,
                same_as_original=(best_model_id if reuses else None) == assessment.selected_reuse_model_id,
                interval_length=int(x.shape[0]),
            )
        )
        reusable = tuple((m, loss) for m, loss in supported if reuses and m == best_model_id)
        return AlarmIntervalModelReuseAssessment(
            baseline_supported_interval_mean_losses_by_model_id=supported,
            reusable_mean_losses_by_model_id=reusable,
        )

    with (
        patch.object(resolution_module, "evaluate_held_models_for_alarm_interval_reuse", evaluate),
        patch.object(buffer_module, "resolve_alarm_change_interval", resolve),
        patch.object(alarm_handling_module, "respond_to_alarm_with_buffered_samples", respond),
    ):
        yield


@contextmanager
def _unified_assignment_inference_policy(*, optimizer_settings, share_rate, window_sample_count, grid_sample_count, events):
    """検出と再利用の判定を、窓の中の区分けの比較へ置き換える（研究用。実行の間だけ包む）。

    標本tの処理の前に、窓W=[t−L+1, t]（警報の後は、警報の次の標本から）について、次の符号長（nats）を比べる。
    - 続く: Σ_W ℓ_c（cは現在のモデル）＋（窓の中の遷移の数）×(−ln(1−share_rate))。
    - 切り替わる（位置τ、行き先s）: Σ_{i<τ} ℓ_c ＋ Σ_{i≥τ} ℓ_s ＋ 遷移の符号長（切り替わり1回は
      −ln(share_rate)＋ln(行き先の数)、ほかは−ln(1−share_rate)）。sは、損失の統計を持つ、c以外の保有モデル、または
      新しい概念（τから始めた、概念ごとの部分を初期化し直した逐次の学習者の損失）。τは、窓の中の、最小の変化区間
      （grid）ごとの位置で、t−τ+1≧grid。
    ℓは対数損失（−log 真のクラスの確率）。保有モデルの損失は、標本tの処理の前の分類器で、窓の標本を計算し直す。
    切り替わる仮説の最小が、続く仮説より短ければ、警報とし、推定の変化の長さをt−τ+1、再利用の行き先をsとする。
    """
    import torch
    from torch.nn import BCELoss, CrossEntropyLoss
    from torch.nn.init import zeros_

    import federated_learning_experiments.runtime.alarm_change_interval_resolution as resolution_module
    import federated_learning_experiments.runtime.fedsda_run_client as client_module
    from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
        evaluate_classifier_per_sample_bounded_losses,
    )
    from federated_learning_experiments.learning.training.parameter_optimizer_construction import (
        create_parameter_optimizer,
    )
    from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
        AlarmIntervalModelReuseAssessment,
    )
    from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
        OverallAndTrueClassLossMonitor,
    )

    original_process = client_module.process_observed_sample
    original_observe = OverallAndTrueClassLossMonitor.observe_loss_after_label_observation
    original_evaluate = resolution_module.evaluate_held_models_for_alarm_interval_reuse
    # clientごとの窓の状態（保有モデルのregistryのidで分ける）。
    states = {}
    # 処理中の標本の判断（監視と再利用の評価が読む）。
    decision = {}
    learner_seed = [0]

    def log_losses(classifier, x, y):
        bounded = evaluate_classifier_per_sample_bounded_losses(classifier=classifier, input_features=x, observed_class_labels=y)
        return -torch.log(torch.clamp(1.0 - bounded, min=1e-12))

    def fresh_learner(classifier):
        learner = copy.deepcopy(classifier)
        learner_seed[0] += 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(learner_seed[0])
            for module in (learner.model_owned_feature_transform, learner.classification_layer):
                for submodule in module.modules():
                    if hasattr(submodule, "reset_parameters"):
                        submodule.reset_parameters()
            expansion = getattr(learner.model_owned_feature_transform, "feature_expansion", None)
            if expansion is not None:
                zeros_(expansion.weight)
                zeros_(expansion.bias)
        parameters = tuple(p for name, p in learner.named_parameters() if not name.startswith("feature_extractor."))
        return dict(classifier=learner, optimizer=create_parameter_optimizer(parameters=parameters, optimizer_settings=optimizer_settings), total=0.0)

    def learn(learner, x, y):
        model = learner["classifier"]
        learner["total"] += float(log_losses(model, x, y).sum())
        with torch.no_grad():
            shared = model.extract_shared_features(x)
        learner["optimizer"].zero_grad()
        predictions = model.forward_from_shared_features(shared)
        loss = BCELoss()(predictions, y) if model.class_count == 2 else CrossEntropyLoss()(predictions, y.reshape(-1).long())
        loss.backward()
        learner["optimizer"].step()

    def process(**arguments):
        registry = arguments["held_model_training_state_registry"]
        loss_store = arguments["loss_statistics_store"]
        observation = arguments["indexed_observation"]
        state = states.setdefault(id(registry), dict(window=[], learners={}, origin=None))
        x = observation.training_sample.input_features
        y = observation.training_sample.observed_class_labels
        t = observation.sample_index
        if state["origin"] is None:
            state["origin"] = t
        window = state["window"]
        window.append((t, x, y))
        del window[: max(len(window) - window_sample_count, 0)]
        # 新しい概念の学習者: gridごとの位置で始め、窓より前に始めたものは捨てる。
        if (t - state["origin"]) % grid_sample_count == 0:
            state["learners"][t] = fresh_learner(registry.get_held_model_training_state(model_id=arguments["current_training_model_assignment"].current_training_model_id).classifier)
        for start in [s for s in state["learners"] if s < window[0][0]]:
            del state["learners"][start]
        for learner in state["learners"].values():
            learn(learner, x, y)
        current = arguments["current_training_model_assignment"].current_training_model_id
        features = torch.cat([w[1] for w in window])
        labels = torch.cat([w[2] for w in window])
        held_ids = [s.model_id for s in registry.snapshot_ordered_held_model_training_states()]
        losses = {m: log_losses(registry.get_held_model_training_state(model_id=m).classifier, features, labels).tolist() for m in held_ids}
        destinations = [m for m in held_ids if m != current and loss_store.get_model_loss_statistics(model_id=m) is not None]
        n = len(window)
        stay_step = -math.log(1.0 - share_rate)
        switch_step = -math.log(share_rate) + math.log(len(destinations) + 1)
        stay = sum(losses[current]) + (n - 1) * stay_step
        best = None
        for position, (index, _, _) in enumerate(window):
            span = t - index + 1
            if span < grid_sample_count:
                continue
            # 遷移はn−1回で、そのうち1回が切り替わり。
            before = sum(losses[current][:position]) + (n - 2) * stay_step + switch_step
            candidates = [(sum(losses[m][position:]), m) for m in destinations]
            if index in state["learners"]:
                candidates.append((state["learners"][index]["total"], None))
            for after, destination in candidates:
                total = before + after
                if best is None or total < best[0]:
                    best = (total, index, destination)
        decision.clear()
        if best is not None and best[0] < stay:
            decision.update(alarm=True, start=best[1], span=t - best[1] + 1, destination=best[2])
            events.append(dict(event="unified_alarm", new=best[2] is None, span=t - best[1] + 1))
        else:
            decision.update(alarm=False)
        try:
            result = original_process(**arguments)
        finally:
            decided = dict(decision)
            decision.clear()
        if decided.get("alarm") and result.alarm_occurrence_handling is not None:
            # 警報の後は、窓と学習者を空にする（警報の次の標本から数える）。
            state["window"] = []
            state["learners"] = {}
            state["origin"] = None
        return result

    def observe(self, **arguments):
        observation = original_observe(self, **arguments)
        if not decision:
            return observation
        events.append(dict(event="detector_signal", detected=bool(observation.drift_detected)))
        if decision["alarm"]:
            return dataclasses.replace(
                observation,
                drift_detected=True,
                detector_candidate_start_sample_index=decision["start"],
                estimated_change_span_sample_count=decision["span"],
            )
        return dataclasses.replace(observation, drift_detected=False)

    def evaluate(**arguments):
        assessment = original_evaluate(**arguments)
        if not decision.get("alarm"):
            return assessment
        supported = assessment.baseline_supported_interval_mean_losses_by_model_id
        destination = decision["destination"]
        reusable = tuple((m, loss) for m, loss in supported if m == destination)
        if destination is not None and not reusable:
            events.append(dict(event="unified_destination_without_baseline"))
        events.append(dict(event="unified_resolution", reuse=bool(reusable)))
        return AlarmIntervalModelReuseAssessment(
            baseline_supported_interval_mean_losses_by_model_id=supported, reusable_mean_losses_by_model_id=reusable
        )

    with (
        patch.object(client_module, "process_observed_sample", process),
        patch.object(OverallAndTrueClassLossMonitor, "observe_loss_after_label_observation", observe),
        patch.object(resolution_module, "evaluate_held_models_for_alarm_interval_reuse", evaluate),
    ):
        yield


@contextmanager
def _detector_gated_segment_assignment_policy(*, optimizer_settings, share_rate, grid_sample_count, events):
    """検出器の警報のときだけ、保留の標本の中の区分けの比較で、変化点と行き先を決める（研究用。実行の間だけ包む）。

    警報の標本tと、保留の標本（t より前、古い順）を合わせた列Wで、次の符号長（nats）を比べる。
    - 現在のモデルのまま: Σ_W ℓ_c。
    - 位置τで行き先sへ切り替わる: Σ_{i<τ} ℓ_c ＋ Σ_{i≥τ} ℓ_s ＋ −ln(share_rate)＋ln(行き先の数)。sは、損失の統計を
      持つ、c以外の保有モデル、または新しい概念（τから始めた、概念ごとの部分を初期化し直した逐次の学習者）。
      τは、列の末尾からgridごと（t−τ+1がgridの倍数）。
    （両方の仮説に共通の、遷移の「続く」の符号長は、差に効かない分を除いて、切り替わり1回ぶんだけを足す。）
    最小の仮説が「現在のモデルのまま」なら、推定の変化の長さを列の長さ（今の既定と同じく、保留の全部を変化区間と
    する上限）にして、現在のモデルを再利用する。切り替わる仮説なら、推定の変化の長さをt−τ+1、再利用の行き先をsとする。
    """
    import torch
    from torch.nn import BCELoss, CrossEntropyLoss
    from torch.nn.init import zeros_

    import federated_learning_experiments.runtime.alarm_change_interval_resolution as resolution_module
    import federated_learning_experiments.runtime.fedsda_run_client as client_module
    from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
        evaluate_classifier_per_sample_bounded_losses,
    )
    from federated_learning_experiments.learning.training.parameter_optimizer_construction import (
        create_parameter_optimizer,
    )
    from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
        AlarmIntervalModelReuseAssessment,
    )
    from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
        OverallAndTrueClassLossMonitor,
    )

    original_process = client_module.process_observed_sample
    original_observe = OverallAndTrueClassLossMonitor.observe_loss_after_label_observation
    original_evaluate = resolution_module.evaluate_held_models_for_alarm_interval_reuse
    from federated_learning_experiments.runtime.alarm_interval_minimum_description_length_selection import (
        derive_new_model_initialization_seed,
    )

    processing = []
    decision = {}

    def log_losses(classifier, x, y):
        bounded = evaluate_classifier_per_sample_bounded_losses(classifier=classifier, input_features=x, observed_class_labels=y)
        return -torch.log(torch.clamp(1.0 - bounded, min=1e-12))

    def prequential_total(classifier, x, y, seed):
        learner = copy.deepcopy(classifier)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for module in (learner.model_owned_feature_transform, learner.classification_layer):
                for submodule in module.modules():
                    if hasattr(submodule, "reset_parameters"):
                        submodule.reset_parameters()
            expansion = getattr(learner.model_owned_feature_transform, "feature_expansion", None)
            if expansion is not None:
                zeros_(expansion.weight)
                zeros_(expansion.bias)
        parameters = tuple(p for name, p in learner.named_parameters() if not name.startswith("feature_extractor."))
        optimizer = create_parameter_optimizer(parameters=parameters, optimizer_settings=optimizer_settings)
        total = 0.0
        for index in range(x.shape[0]):
            xi, yi = x[index : index + 1], y[index : index + 1]
            total += float(log_losses(learner, xi, yi).sum())
            with torch.no_grad():
                shared = learner.extract_shared_features(xi)
            optimizer.zero_grad()
            predictions = learner.forward_from_shared_features(shared)
            loss = BCELoss()(predictions, yi) if learner.class_count == 2 else CrossEntropyLoss()(predictions, yi.reshape(-1).long())
            loss.backward()
            optimizer.step()
        return total

    def decide(arguments, sample_index):
        registry = arguments["held_model_training_state_registry"]
        loss_store = arguments["loss_statistics_store"]
        current = arguments["current_training_model_assignment"].current_training_model_id
        observations = [
            *arguments["pending_sample_observation_store"].snapshot_pending_sample_observations(),
            arguments["indexed_observation"],
        ]
        x = torch.cat([o.training_sample.input_features for o in observations])
        y = torch.cat([o.training_sample.observed_class_labels for o in observations])
        n = len(observations)
        held_ids = [s.model_id for s in registry.snapshot_ordered_held_model_training_states()]
        losses = {m: log_losses(registry.get_held_model_training_state(model_id=m).classifier, x, y).tolist() for m in held_ids}
        destinations = [m for m in held_ids if m != current and loss_store.get_model_loss_statistics(model_id=m) is not None]
        switch = -math.log(share_rate) + math.log(len(destinations) + 1)
        best = (sum(losses[current]), None, current)
        template = registry.get_held_model_training_state(model_id=current).classifier
        for span in range(grid_sample_count, n + 1, grid_sample_count):
            position = n - span
            before = sum(losses[current][:position]) + switch
            options = [(sum(losses[m][position:]), m) for m in destinations]
            seed = derive_new_model_initialization_seed(
                client_id=arguments["client_id"], alarm_sample_index=sample_index, change_interval_sample_count=span
            )
            options.append((prequential_total(template, x[position:], y[position:], seed), None))
            for after, destination in options:
                if before + after < best[0]:
                    best = (before + after, span, destination)
        _, span, destination = best
        if span is None:
            events.append(dict(event="gated_decision", outcome="current", span=n))
            return dict(span=n, destination=current)
        events.append(dict(event="gated_decision", outcome="new" if destination is None else "held", span=span))
        return dict(span=span, destination=destination)

    def process(**arguments):
        processing.append(arguments)
        try:
            return original_process(**arguments)
        finally:
            processing.pop()
            decision.clear()

    def observe(self, **arguments):
        observation = original_observe(self, **arguments)
        if not observation.drift_detected or not processing:
            return observation
        decision.clear()
        decision.update(decide(processing[-1], observation.sample_index))
        return dataclasses.replace(
            observation,
            estimated_change_span_sample_count=decision["span"],
            detector_candidate_start_sample_index=observation.sample_index - decision["span"] + 1,
        )

    def evaluate(**arguments):
        assessment = original_evaluate(**arguments)
        if not decision:
            return assessment
        supported = assessment.baseline_supported_interval_mean_losses_by_model_id
        destination = decision["destination"]
        reusable = tuple((m, loss) for m, loss in supported if m == destination)
        if destination is not None and not reusable:
            events.append(dict(event="gated_destination_without_baseline"))
        return AlarmIntervalModelReuseAssessment(
            baseline_supported_interval_mean_losses_by_model_id=supported, reusable_mean_losses_by_model_id=reusable
        )

    with (
        patch.object(client_module, "process_observed_sample", process),
        patch.object(OverallAndTrueClassLossMonitor, "observe_loss_after_label_observation", observe),
        patch.object(resolution_module, "evaluate_held_models_for_alarm_interval_reuse", evaluate),
    ):
        yield


@contextmanager
def _frozen_candidate_backbone_policy(*, events):
    """候補の学習（`candidate_epoch_training`）の共同更新で、共有部を更新しない（実行の間だけ包む）。"""
    import federated_learning_experiments.learning.training.candidate_epoch_training as training_module

    original_update = training_module.perform_joint_model_parameter_update

    def update(**arguments):
        events.append(dict(event="frozen_candidate_update"))
        return original_update(**{**arguments, "update_shared_features": False})

    with patch.object(training_module, "perform_joint_model_parameter_update", update):
        yield


def _table_row_count(tables, prefix):
    """記録の表の、その表名の最初の列の行数（表が無ければ0）。"""
    columns = sorted(name for name in tables if name.startswith(prefix))
    return int(len(tables[columns[0]])) if columns else 0


def _digest(tensors):
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode() + b"\0")
        digest.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _execute_job(job):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch

    torch.set_num_threads(1)
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        build_experiment_run_settings,
    )

    conditions = ExperimentRunConditions(
        dataset_name=job["dataset"],
        random_seed=job["seed"],
        client_count=10,
        per_client_sample_count=job["sample_count"],
        server_aggregation_interval_per_client_samples=job["interval"],
    )
    configuration_name, uses_provisional_expert, validation_sample_count = _VARIANTS[job["variant"]]
    settings = build_experiment_run_settings(
        configuration_name=configuration_name, experiment_run_conditions=conditions
    )
    if validation_sample_count is not None:
        participant = settings.run_participant_settings
        client = participant.run_client_settings
        settings = replace(
            settings,
            run_participant_settings=replace(
                participant,
                run_client_settings=replace(
                    client,
                    candidate_model_training_and_acceptance_settings=replace(
                        client.candidate_model_training_and_acceptance_settings,
                        candidate_post_alarm_validation_sample_count=validation_sample_count,
                    ),
                ),
            ),
        )
    if job["variant"] in _UPLOAD_DELAY_ROUND_COUNT_BY_VARIANT:
        participant = settings.run_participant_settings
        client = participant.run_client_settings
        settings = replace(
            settings,
            run_participant_settings=replace(
                participant,
                run_client_settings=replace(
                    client,
                    scalar_settings=replace(
                        client.scalar_settings,
                        new_model_upload_delay_round_count=_UPLOAD_DELAY_ROUND_COUNT_BY_VARIANT[
                            job["variant"]
                        ],
                    ),
                ),
            ),
        )
    if job["variant"] == "default-minimum-description-length":
        participant = settings.run_participant_settings
        client = participant.run_client_settings
        settings = replace(
            settings,
            run_participant_settings=replace(
                participant,
                run_client_settings=replace(
                    client,
                    training_data_assignment_settings=replace(
                        client.training_data_assignment_settings,
                        alarm_interval_assignment_policy="minimum_description_length",
                    ),
                ),
            ),
        )
    if job["variant"] in _FIXED_CANDIDATE_EPOCH_COUNT_BY_VARIANT:
        participant = settings.run_participant_settings
        client = participant.run_client_settings
        settings = replace(
            settings,
            run_participant_settings=replace(
                participant,
                run_client_settings=replace(
                    client,
                    candidate_epoch_training_settings=replace(
                        client.candidate_epoch_training_settings,
                        candidate_training_strategy="fixed_epoch_training",
                        maximum_epoch_count=_FIXED_CANDIDATE_EPOCH_COUNT_BY_VARIANT[job["variant"]],
                    ),
                ),
            ),
        )
    client_settings = settings.run_participant_settings.run_client_settings
    started = time.time()
    events = []
    with contextlib.ExitStack() as stack:
        if job["variant"] in _SEQUENTIAL_TEST_VARIANTS:
            stack.enter_context(_sequential_test_policy(events=events))
        if job["variant"] in _FROZEN_CANDIDATE_BACKBONE_VARIANTS:
            stack.enter_context(_frozen_candidate_backbone_policy(events=events))
        if job["variant"] in _DETECTOR_GATED_SEGMENT_ASSIGNMENT_VARIANTS:
            stack.enter_context(
                _detector_gated_segment_assignment_policy(
                    optimizer_settings=client_settings.parameter_optimizer_settings,
                    share_rate=1.0
                    / client_settings.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples,
                    grid_sample_count=client_settings.scalar_settings.minimum_change_interval_sample_count,
                    events=events,
                )
            )
        if job["variant"] in _UNIFIED_ASSIGNMENT_INFERENCE_VARIANTS:
            stack.enter_context(
                _unified_assignment_inference_policy(
                    optimizer_settings=client_settings.parameter_optimizer_settings,
                    share_rate=1.0
                    / client_settings.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples,
                    window_sample_count=client_settings.training_data_assignment_settings.pending_assignment_buffer_capacity_samples,
                    grid_sample_count=client_settings.scalar_settings.minimum_change_interval_sample_count,
                    events=events,
                )
            )
        if job["variant"] in _PREQUENTIAL_REUSE_VARIANTS:
            stack.enter_context(
                _prequential_reuse_policy(
                    optimizer_settings=client_settings.parameter_optimizer_settings,
                    events=events,
                    starts_learner_fresh=job["variant"]
                    in (
                        "default-prequential-reuse-fresh-learner",
                        "default-prequential-reuse-fresh-learner-bounded-loss",
                        "default-prequential-reuse-fresh-learner-horizon",
                        "default-prequential-reuse-fresh-learner-horizon-evaluation-samples",
                        "default-prequential-reuse-fresh-learner-horizon-interval",
                    ),
                    compares_bounded_losses=job["variant"] == "default-prequential-reuse-fresh-learner-bounded-loss",
                    horizon_sample_count="interval"
                    if job["variant"] == "default-prequential-reuse-fresh-learner-horizon-interval"
                    else (
                        client_settings.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
                        if job["variant"]
                        in (
                            "default-prequential-reuse-fresh-learner-horizon",
                            "default-prequential-reuse-fresh-learner-horizon-evaluation-samples",
                        )
                        else None
                    ),
                    measures_own_loss_on_evaluation_samples=job["variant"]
                    == "default-prequential-reuse-fresh-learner-horizon-evaluation-samples",
                )
            )
        if job["variant"] in _WITHDRAW_RELEASED_SAMPLE_VARIANTS:
            stack.enter_context(
                _withdraw_released_samples_policy(
                    events=events,
                    includes_withdrawn_samples_in_change_interval=job["variant"]
                    == "default-withdraw-released-into-interval",
                )
            )
        if uses_provisional_expert:
            stack.enter_context(
                _provisional_expert_policy(
                    share_rate=1.0
                    / client_settings.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples,
                    optimizer_settings=client_settings.parameter_optimizer_settings,
                    check=job["check"],
                    events=events,
                )
            )
        run = execute_fedsda_experiment_run(run_settings=settings)
    tests = [e for e in events if e["event"] == "sequential_test"]
    clients = run.measured_run.participants.client_operations
    records = [
        record
        for client in clients
        for record in client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
    ]
    metrics = run.run_metrics
    from federated_learning_experiments.runtime.fedsda_run_record_tables import (
        build_fedsda_run_record_tables,
    )

    tables = build_fedsda_run_record_tables(experiment_run=run)
    repository = run.measured_run.participants.server_operations.owners.global_model_repository
    volume = metrics.communication_volume
    return dict(
        **{key: job[key] for key in ("variant", "dataset", "interval", "seed", "sample_count", "check")},
        accuracy=metrics.prediction_accuracy,
        record_accuracy=sum(r.combined_prediction_is_correct for r in records) / len(records),
        any_correct_rate=sum(r.any_model_or_combined_prediction_is_correct for r in records) / len(records),
        prediction_record_digest=hashlib.sha256(
            json.dumps([r.combined_prediction_is_correct for r in records]).encode()
        ).hexdigest(),
        uploaded_bytes=volume.uploaded_byte_count,
        downloaded_bytes=volume.downloaded_byte_count,
        # 通信の内訳（モデルの転送回数、通知の回数）と、登録・クロス評価・統合の件数。
        communication_volume=dataclasses.asdict(volume),
        registered_model_count=_table_row_count(tables, "model_registration."),
        client_cross_evaluation_count=_table_row_count(tables, "client_cross_evaluation."),
        merged_model_count=int(tables["model_clustering.merged_with_other_models"].sum())
        if "model_clustering.merged_with_other_models" in tables
        else 0,
        final_global_model_count=metrics.final_global_model_count,
        upload_delay_round_count=client_settings.scalar_settings.new_model_upload_delay_round_count,
        candidate_training_strategy=client_settings.candidate_epoch_training_settings.candidate_training_strategy,
        candidate_maximum_epoch_count=client_settings.candidate_epoch_training_settings.maximum_epoch_count,
        model_computation_counts=dataclasses.asdict(metrics.model_computation_counts),
        validation_sample_count=client_settings.candidate_model_training_and_acceptance_settings.candidate_post_alarm_validation_sample_count,
        candidate_model_acceptance_policy=client_settings.candidate_model_training_and_acceptance_settings.candidate_model_acceptance_policy,
        detection_count=metrics.training_model_switch_detection_metrics.detection_count,
        # 検出・変化点の推定・切替えの遅れの指標（掃引の`run_summary.csv`と同じ量。既定と比べるため）。
        training_model_switch_detection_metrics=dataclasses.asdict(metrics.training_model_switch_detection_metrics),
        loss_change_alarm_detection_metrics=dataclasses.asdict(
            metrics.change_response_metrics.loss_change_alarm_detection_metrics
        ),
        change_point_estimation_metrics=dataclasses.asdict(metrics.change_response_metrics.change_point_estimation_metrics),
        training_model_switch_timing_metrics=dataclasses.asdict(
            metrics.change_response_metrics.training_model_switch_timing_metrics
        ),
        server_parameter_digest=_digest(
            {
                f"{model_id}.{name}": value
                for model_id in repository.global_model_ids
                for name, value in repository.get_global_model_parameters(model_id=model_id).items()
            }
        ),
        session_starts=sum(e["event"] == "session_start" for e in events),
        frozen_candidate_update_count=sum(e["event"] == "frozen_candidate_update" for e in events),
        withdrawn_released_sample_count=sum(e["count"] for e in events if e["event"] == "withdraw_released"),
        withdrawn_new_concept_sample_count=sum(e["new_concept"] for e in events if e["event"] == "withdraw_released"),
        prequential_reuse_decisions={
            key: sum(
                bool(e[field]) == value
                for e in events
                if e["event"] == "prequential_reuse_decision"
            )
            for key, field, value in (
                ("reuse", "reuse", True),
                ("new", "reuse", False),
                ("original_rule_reuse", "original_reuse", True),
                ("same_as_original_rule", "same_as_original", True),
            )
        },
        detector_gated_segment_assignment={
            outcome: sum(e["event"] == "gated_decision" and e["outcome"] == outcome for e in events)
            for outcome in ("current", "held", "new")
        }
        | {"destinations_without_baseline": sum(e["event"] == "gated_destination_without_baseline" for e in events)},
        unified_assignment_inference={
            "alarms": sum(e["event"] == "unified_alarm" for e in events),
            "alarms_to_new": sum(e["event"] == "unified_alarm" and e["new"] for e in events),
            "resolutions_reusing": sum(e["event"] == "unified_resolution" and e["reuse"] for e in events),
            "resolutions_creating": sum(e["event"] == "unified_resolution" and not e["reuse"] for e in events),
            "destinations_without_baseline": sum(e["event"] == "unified_destination_without_baseline" for e in events),
            "detector_signals": sum(e["event"] == "detector_signal" and e["detected"] for e in events),
        },
        own_loss_fallback_count=sum(e["event"] == "own_loss_fallback_to_training_samples" for e in events),
        withdrawn_into_interval_sample_count=sum(
            e["count"] for e in events if e["event"] == "withdrawn_into_interval"
        ),
        withdrawn_not_reassigned_sample_count=sum(
            e["count"] for e in events if e["event"] == "withdrawn_not_reassigned"
        ),
        sequential_test_decisions={
            decision: sum(e["decision"] == decision for e in tests)
            for decision in ("accept", "reject", "timeout")
        },
        sequential_test_mean_sample_count=(
            sum(e["sample_count"] for e in tests) / len(tests) if tests else None
        ),
        session_outcomes={
            outcome: sum(e.get("outcome") == outcome for e in events)
            for outcome in sorted({e["outcome"] for e in events if "outcome" in e})
        },
        seconds=time.time() - started,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=2500)
    # 既定は最初の3変種（uraniumの本規模の確認と同じ。B0の2変種は、指定したときだけ）。
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["validation", "no-validation", "provisional-expert"],
        choices=list(_VARIANTS),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--check", action="store_true", help="仮の専門家の重みを常に0にする（検証ありとの一致の照合）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="既存の出力先の済んだrunを飛ばして、残りを実行する（同じ引数で）",
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=arguments.resume)
    results_path = arguments.output / "results.json"
    jobs = [
        dict(
            variant=v,
            dataset=d,
            interval=i,
            seed=s,
            sample_count=arguments.sample_count,
            check=arguments.check and _VARIANTS[v][1],
        )
        for d in arguments.datasets
        for i in arguments.intervals
        for s in arguments.seeds
        for v in arguments.variants
    ]

    def job_key(job):
        return tuple(job[k] for k in ("variant", "dataset", "interval", "seed", "sample_count", "check"))

    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else []
    done = {job_key(result) for result in results}
    jobs = [job for job in jobs if job_key(job) not in done]
    print(f"skipped {len(done)} completed runs, running {len(jobs)}", flush=True)
    # 重いMNISTから先に投げる。
    jobs.sort(key=lambda job: not job["dataset"].startswith("mnist"))
    failures = []
    with ProcessPoolExecutor(max_workers=arguments.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_execute_job, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                # 1 runの失敗で、ほかのrunを止めない（--resumeで、もう一度実行される）。
                failures.append(dict(job=futures[future], error=repr(error)))
                print("FAILED", futures[future], repr(error), flush=True)
                (arguments.output / "failures.json").write_text(
                    json.dumps(failures, indent=1), encoding="utf-8"
                )
                continue
            results.append(result)
            print(
                result["variant"],
                result["dataset"],
                result["interval"],
                result["seed"],
                round(result["accuracy"], 4),
                int(result["seconds"]),
                flush=True,
            )
            temporary_path = results_path.with_name("results.json.tmp")
            temporary_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
            temporary_path.replace(results_path)
    print(f"completed {len(results)}, failed {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
