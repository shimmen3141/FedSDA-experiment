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
# 変種 -> （構成、仮の専門家を使うか、検証の標本数——Noneなら構成の値の10——）。
# 検証あり（V）、検証なし（O）、検証あり＋仮の専門家（A）。B0: 検証の標本数を30にした検証あり（V30）と、
# それに仮の専門家を足したもの（A30。仮の専門家があれば、検証を延ばしても予測の損が無い）。
# 案B: 検証の標本数の上限を30にし、候補と参照の損失の差のe-過程で、証拠が閾値に達したら早く打ち切る
# （打ち切らずに30標本へ達したら、今の規則で判定する。検証30との違いは、早く打ち切ることだけ）。
_VARIANTS = {
    "validation": ("fedsda", False, None),
    "no-validation": ("fedsda-without-candidate-validation", False, None),
    "provisional-expert": ("fedsda", True, None),
    "validation-forward30": ("fedsda", False, 30),
    "provisional-expert-forward30": ("fedsda", True, 30),
    "sequential-test": ("fedsda", False, 30),
    "provisional-expert-sequential-test": ("fedsda", True, 30),
    # アイデア1: 検証なし（すぐ採用）で、送信までの待ちの同期回数（既定1）を延ばす。既定の登録の扱いは、
    # 待ちの間も現行であり続けたモデルだけを登録するので、短命なモデルはサーバへ共有されない。
    "no-validation-upload-delay2": ("fedsda-without-candidate-validation", False, None),
    "no-validation-upload-delay3": ("fedsda-without-candidate-validation", False, None),
}
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
    client_settings = settings.run_participant_settings.run_client_settings
    started = time.time()
    events = []
    with contextlib.ExitStack() as stack:
        if job["variant"] in _SEQUENTIAL_TEST_VARIANTS:
            stack.enter_context(_sequential_test_policy(events=events))
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
        validation_sample_count=client_settings.candidate_model_training_and_acceptance_settings.candidate_post_alarm_validation_sample_count,
        candidate_model_acceptance_policy=client_settings.candidate_model_training_and_acceptance_settings.candidate_model_acceptance_policy,
        detection_count=metrics.training_model_switch_detection_metrics.detection_count,
        server_parameter_digest=_digest(
            {
                f"{model_id}.{name}": value
                for model_id in repository.global_model_ids
                for name, value in repository.get_global_model_parameters(model_id=model_id).items()
            }
        ),
        session_starts=sum(e["event"] == "session_start" for e in events),
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
