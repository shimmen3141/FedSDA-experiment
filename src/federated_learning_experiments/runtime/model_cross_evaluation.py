"""サーバのクロス評価: グローバルモデルの全部の組を、対象のモデルを保有するclientへ評価させて集計する。"""

from dataclasses import dataclass
from random import Random

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
)
from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    ClientCrossEvaluationRecord,
    CrossEvaluationRecordStore,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.client_model_cross_evaluation import (
    ClientModelCrossEvaluation,
    ModelPairCorrectnessCounts,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    split_shared_and_concept_specific_parameters,
)


@dataclass(frozen=True, kw_only=True)
class CrossEvaluationLossSums:
    """（評価する側のモデル、対象のモデル）の組の、全clientぶんの有界損失の件数・和・2乗和。"""

    evaluated_sample_count: int
    bounded_loss_sum: float
    squared_bounded_loss_sum: float


@dataclass(frozen=True, kw_only=True)
class ModelPairUniqueCorrectnessCounts:
    """モデルの対（小さいID、大きいID）の、評価した標本数と、片方だけが正解した数。両方向の評価を足し合わせる。

    クラス別は（クラス、件数、小さいIDのモデルだけ正解、大きいIDのモデルだけ正解）の列で、初めて現れた順。
    """

    evaluated_sample_count: int
    lower_id_model_only_correct_count: int
    higher_id_model_only_correct_count: int
    class_counts: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True, kw_only=True)
class ModelCrossEvaluation:
    """クロス評価1回の結果。

    損失の表は、評価したモデルの全部の組（評価する側、対象）を持つ（対象を評価するclientがいなければ件数0）。
    正誤の集計は、正誤の比較が1回以上返った対（小さいID、大きいID）だけを持つ。
    """

    cross_evaluated_model_ids: tuple[int, ...]
    loss_sums_by_candidate_and_target_model_id: dict[tuple[int, int], CrossEvaluationLossSums]
    unique_correctness_counts_by_model_pair: dict[tuple[int, int], ModelPairUniqueCorrectnessCounts]


def _validate_cross_evaluation_inputs(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    global_model_repository: GlobalModelRepository,
    communication_volume_record_store: CommunicationVolumeRecordStore,
    cross_evaluation_record_store: CrossEvaluationRecordStore,
    cross_evaluated_model_ids: tuple[int, ...],
    round_index: int,
    maximum_evaluating_client_count_per_model: int,
    python_random_generator: Random,
) -> None:
    if type(run_clients) is not tuple:
        raise TypeError("run_clients must be builtin tuple")
    for run_client in run_clients:
        if type(run_client) is not FedsdaRunClient:
            raise TypeError("run_clients must hold exact FedsdaRunClient")
    client_ids = [run_client.client_id for run_client in run_clients]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("run_clients must not hold duplicate client IDs")
    for owner_name, owner, required_type in (
        ("global_model_repository", global_model_repository, GlobalModelRepository),
        (
            "communication_volume_record_store",
            communication_volume_record_store,
            CommunicationVolumeRecordStore,
        ),
        (
            "cross_evaluation_record_store",
            cross_evaluation_record_store,
            CrossEvaluationRecordStore,
        ),
        ("python_random_generator", python_random_generator, Random),
    ):
        if type(owner) is not required_type:
            raise TypeError(f"{owner_name} must be exact {required_type.__name__}")
    if type(cross_evaluated_model_ids) is not tuple:
        raise TypeError("cross_evaluated_model_ids must be builtin tuple")
    held_global_model_ids = global_model_repository.global_model_ids
    for model_id in cross_evaluated_model_ids:
        if type(model_id) is not int:
            raise TypeError("cross_evaluated_model_ids must hold builtin int")
        if model_id not in held_global_model_ids:
            raise ValueError("cross_evaluated_model_ids must hold global model IDs with parameters")
    if len(set(cross_evaluated_model_ids)) != len(cross_evaluated_model_ids):
        raise ValueError("cross_evaluated_model_ids must not hold duplicates")
    if type(round_index) is not int:
        raise TypeError("round_index must be builtin int")
    if round_index < 0:
        raise ValueError("round_index must be nonnegative")
    if type(maximum_evaluating_client_count_per_model) is not int:
        raise TypeError("maximum_evaluating_client_count_per_model must be builtin int")
    if maximum_evaluating_client_count_per_model < 1:
        raise ValueError("maximum_evaluating_client_count_per_model must be positive")


def _correctness_counts_as_record_values(
    correctness_counts: ModelPairCorrectnessCounts,
) -> tuple[int, int, int, int]:
    return (
        correctness_counts.candidate_only_correct_count,
        correctness_counts.target_only_correct_count,
        correctness_counts.both_correct_count,
        correctness_counts.both_wrong_count,
    )


def _make_client_cross_evaluation_record(
    *,
    round_index: int,
    client_id: int,
    candidate_model_id: int,
    target_model_id: int,
    client_cross_evaluation: ClientModelCrossEvaluation,
) -> ClientCrossEvaluationRecord:
    correctness_counts = client_cross_evaluation.correctness_counts
    return ClientCrossEvaluationRecord(
        round_index=round_index,
        client_id=client_id,
        candidate_model_id=candidate_model_id,
        target_model_id=target_model_id,
        evaluated_sample_count=client_cross_evaluation.evaluated_sample_count,
        bounded_loss_sum=client_cross_evaluation.bounded_loss_sum,
        squared_bounded_loss_sum=client_cross_evaluation.squared_bounded_loss_sum,
        correctness_counts=(
            None
            if correctness_counts is None
            else _correctness_counts_as_record_values(correctness_counts)
        ),
        class_correctness_counts=tuple(
            (
                class_id,
                class_correctness_counts.evaluated_sample_count,
                *_correctness_counts_as_record_values(class_correctness_counts),
            )
            for class_id, class_correctness_counts in client_cross_evaluation.class_correctness_counts
        ),
    )


class _ModelPairUniqueCorrectnessAccumulator:
    """1つの対（小さいID、大きいID）の、片方だけが正解した数を、全体とクラス別に足していく。"""

    def __init__(self) -> None:
        self.overall_counts = [0, 0, 0]
        self.class_counts_by_class_id: dict[int, list[int]] = {}

    def add_client_cross_evaluation(
        self,
        *,
        client_cross_evaluation: ClientModelCrossEvaluation,
        candidate_is_lower_id_model: bool,
    ) -> None:
        correctness_counts = client_cross_evaluation.correctness_counts
        if correctness_counts is None:
            return
        self._add(self.overall_counts, correctness_counts, candidate_is_lower_id_model)
        for class_id, class_correctness_counts in client_cross_evaluation.class_correctness_counts:
            self._add(
                self.class_counts_by_class_id.setdefault(class_id, [0, 0, 0]),
                class_correctness_counts,
                candidate_is_lower_id_model,
            )

    @staticmethod
    def _add(
        accumulated_counts: list[int],
        correctness_counts: ModelPairCorrectnessCounts,
        candidate_is_lower_id_model: bool,
    ) -> None:
        accumulated_counts[0] += correctness_counts.evaluated_sample_count
        if candidate_is_lower_id_model:
            accumulated_counts[1] += correctness_counts.candidate_only_correct_count
            accumulated_counts[2] += correctness_counts.target_only_correct_count
        else:
            accumulated_counts[1] += correctness_counts.target_only_correct_count
            accumulated_counts[2] += correctness_counts.candidate_only_correct_count

    def to_unique_correctness_counts(self) -> ModelPairUniqueCorrectnessCounts:
        return ModelPairUniqueCorrectnessCounts(
            evaluated_sample_count=self.overall_counts[0],
            lower_id_model_only_correct_count=self.overall_counts[1],
            higher_id_model_only_correct_count=self.overall_counts[2],
            class_counts=tuple(
                (class_id, *class_counts)
                for class_id, class_counts in self.class_counts_by_class_id.items()
            ),
        )


def cross_evaluate_global_models(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    global_model_repository: GlobalModelRepository,
    communication_volume_record_store: CommunicationVolumeRecordStore,
    cross_evaluation_record_store: CrossEvaluationRecordStore,
    cross_evaluated_model_ids: tuple[int, ...],
    round_index: int,
    maximum_evaluating_client_count_per_model: int,
    python_random_generator: Random,
) -> ModelCrossEvaluation:
    """モデルの全部の組（評価する側、対象）を、渡されたIDの順に、対象を保有するclientへ評価させる。

    評価する側と対象が違う組では、clientが保有する対象のモデルとの正誤の比較も求める。
    組ごとに、軽量メッセージと、評価する側のモデルの転送（1回のクロス評価の中で、共有部はclientごとに1回、
    概念固有部は（モデル、client）ごとに1回）を通信量へ足し、clientの評価ごとに診断の記録を足す。
    グローバルモデルとclientの状態は変えない（乱数は進む）。途中で失敗したら、済んだ分は残る。
    """
    _validate_cross_evaluation_inputs(
        run_clients=run_clients,
        global_model_repository=global_model_repository,
        communication_volume_record_store=communication_volume_record_store,
        cross_evaluation_record_store=cross_evaluation_record_store,
        cross_evaluated_model_ids=cross_evaluated_model_ids,
        round_index=round_index,
        maximum_evaluating_client_count_per_model=maximum_evaluating_client_count_per_model,
        python_random_generator=python_random_generator,
    )
    candidate_parameter_snapshots = {
        model_id: global_model_repository.get_global_model_parameters(model_id=model_id)
        for model_id in cross_evaluated_model_ids
    }
    split_candidate_parameter_snapshots = {
        model_id: split_shared_and_concept_specific_parameters(
            parameter_snapshot=parameter_snapshot
        )
        for model_id, parameter_snapshot in candidate_parameter_snapshots.items()
    }
    # モデルIDごとの、保有するclientの列（渡されたclientの順）。最初に1回だけ作る。
    holding_clients_by_model_id: dict[int, list[FedsdaRunClient]] = {}
    for run_client in run_clients:
        for held_model_id in run_client.get_cross_evaluation_held_model_ids():
            holding_clients_by_model_id.setdefault(held_model_id, []).append(run_client)

    client_ids_sent_shared_parameters: set[int] = set()
    candidate_and_client_ids_sent_concept_specific_parameters: set[tuple[int, int]] = set()
    loss_sums_by_candidate_and_target_model_id: dict[tuple[int, int], CrossEvaluationLossSums] = {}
    unique_correctness_accumulators: dict[
        tuple[int, int], _ModelPairUniqueCorrectnessAccumulator
    ] = {}
    for candidate_model_id in cross_evaluated_model_ids:
        shared_parameters, concept_specific_parameters = split_candidate_parameter_snapshots[
            candidate_model_id
        ]
        for target_model_id in cross_evaluated_model_ids:
            evaluating_clients = holding_clients_by_model_id.get(target_model_id, [])
            if len(evaluating_clients) > maximum_evaluating_client_count_per_model:
                evaluating_clients = python_random_generator.sample(
                    evaluating_clients, maximum_evaluating_client_count_per_model
                )
            # 評価の依頼と、評価の統計の返信。
            for transfer_direction in ("download", "upload"):
                communication_volume_record_store.record_messages(
                    transfer_direction=transfer_direction, message_count=len(evaluating_clients)
                )
            for evaluating_client in evaluating_clients:
                client_id = evaluating_client.client_id
                if client_id not in client_ids_sent_shared_parameters:
                    communication_volume_record_store.record_parameter_transfer(
                        transfer_direction="download", parameter_snapshot=shared_parameters
                    )
                    client_ids_sent_shared_parameters.add(client_id)
                if (
                    candidate_model_id,
                    client_id,
                ) not in candidate_and_client_ids_sent_concept_specific_parameters:
                    communication_volume_record_store.record_model_transfers(
                        transfer_direction="download", model_count=1
                    )
                    communication_volume_record_store.record_parameter_transfer(
                        transfer_direction="download",
                        parameter_snapshot=concept_specific_parameters,
                    )
                    candidate_and_client_ids_sent_concept_specific_parameters.add(
                        (candidate_model_id, client_id)
                    )
            evaluated_sample_count = 0
            bounded_loss_sum = 0.0
            squared_bounded_loss_sum = 0.0
            for evaluating_client in evaluating_clients:
                client_cross_evaluation = (
                    evaluating_client.evaluate_candidate_model_on_target_model_samples(
                        candidate_parameter_snapshot=candidate_parameter_snapshots[
                            candidate_model_id
                        ],
                        target_model_id=target_model_id,
                        compare_correctness_with_held_target_model=(
                            candidate_model_id != target_model_id
                        ),
                    )
                )
                if client_cross_evaluation.correctness_counts is not None:
                    unique_correctness_accumulators.setdefault(
                        (
                            min(candidate_model_id, target_model_id),
                            max(candidate_model_id, target_model_id),
                        ),
                        _ModelPairUniqueCorrectnessAccumulator(),
                    ).add_client_cross_evaluation(
                        client_cross_evaluation=client_cross_evaluation,
                        candidate_is_lower_id_model=candidate_model_id < target_model_id,
                    )
                cross_evaluation_record_store.append_client_cross_evaluation_record(
                    client_cross_evaluation_record=_make_client_cross_evaluation_record(
                        round_index=round_index,
                        client_id=evaluating_client.client_id,
                        candidate_model_id=candidate_model_id,
                        target_model_id=target_model_id,
                        client_cross_evaluation=client_cross_evaluation,
                    )
                )
                evaluated_sample_count += client_cross_evaluation.evaluated_sample_count
                bounded_loss_sum += client_cross_evaluation.bounded_loss_sum
                squared_bounded_loss_sum += client_cross_evaluation.squared_bounded_loss_sum
            loss_sums_by_candidate_and_target_model_id[(candidate_model_id, target_model_id)] = (
                CrossEvaluationLossSums(
                    evaluated_sample_count=evaluated_sample_count,
                    bounded_loss_sum=bounded_loss_sum,
                    squared_bounded_loss_sum=squared_bounded_loss_sum,
                )
            )
    return ModelCrossEvaluation(
        cross_evaluated_model_ids=cross_evaluated_model_ids,
        loss_sums_by_candidate_and_target_model_id=loss_sums_by_candidate_and_target_model_id,
        unique_correctness_counts_by_model_pair={
            model_pair: unique_correctness_accumulator.to_unique_correctness_counts()
            for model_pair, unique_correctness_accumulator in unique_correctness_accumulators.items()
        },
    )
