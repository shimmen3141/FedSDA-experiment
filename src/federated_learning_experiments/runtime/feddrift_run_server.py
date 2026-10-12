"""FedDriftの時刻準備とR回の学習同期。"""

from random import Random

from torch import Tensor, cat, isfinite

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
)
from federated_learning_experiments.evaluation.computation_cost_summary import (
    ServerComputationCounts,
)
from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    CrossEvaluationRecordStore,
)
from federated_learning_experiments.evaluation.feddrift_model_diagnostic_records import (
    FeddriftSynchronizedTimestepModelCounts,
)
from federated_learning_experiments.evaluation.model_clustering_record_store import (
    ModelClusteringRecordStore,
    ModelPairClusteringObservation,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    _copy_model_and_class_loss_statistics,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    validate_classifier_bounded_loss_inputs,
)
from federated_learning_experiments.learning.training.held_model_training_batch_sampling import (
    _validate_observed_training_sample,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    _validate_held_model_training_state_inputs,
)
from federated_learning_experiments.learning.training.independent_model_parameter_update import (
    _validate_independent_model_structure,
    validate_independent_model_parameter_update_inputs,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.methods.feddrift.cached_model_cluster_consolidation import (
    average_feddrift_cached_cluster_parameters,
    consolidate_feddrift_cached_model_clusters,
)
from federated_learning_experiments.methods.feddrift.loss_distance_model_clustering import (
    _loss_increase_distances,
    cluster_feddrift_model_ids,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.client_model_cross_evaluation import (
    _validate_candidate_parameter_snapshot,
    evaluate_candidate_model_on_target_model_samples,
)
from federated_learning_experiments.runtime.feddrift_run_client import FeddriftRunClient
from federated_learning_experiments.runtime.feddrift_server_settings import FeddriftServerSettings
from federated_learning_experiments.runtime.held_model_registration_confirmation import (
    _validate_confirmation_inputs,
    confirm_held_model_registration,
)
from federated_learning_experiments.runtime.model_clustering_and_consolidation import (
    _compute_concept_specific_parameter_distance,
    _make_model_clustering_observations,
)
from federated_learning_experiments.runtime.model_cross_evaluation import (
    _make_client_cross_evaluation_record,
)


def _validate_timestep(timestep_index: int) -> None:
    if type(timestep_index) is not int:
        raise TypeError("時刻はbuiltin intが必要です。")
    if timestep_index < 0:
        raise ValueError("時刻は非負が必要です。")


def _client_parameters(client: FeddriftRunClient) -> dict[int, dict[str, Tensor]]:
    return {
        state.model_id: snapshot_classifier_parameters(classifier=state.classifier)
        for state in client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    }


class FeddriftRunServer:
    """状態の唯一のownerとサーバ進行の入口。"""

    def __init__(
        self,
        *,
        run_clients: tuple[FeddriftRunClient, ...],
        server_settings: FeddriftServerSettings,
        initial_parameter_snapshot: dict[str, Tensor],
        initial_loss_statistics: ModelAndClassLossStatistics,
        python_random_generator: Random,
    ) -> None:
        if (
            type(server_settings) is not FeddriftServerSettings
            or type(python_random_generator) is not Random
        ):
            raise TypeError("設定とRandomの型が不正です。")
        server_settings.__post_init__()
        self.run_clients = run_clients
        self._settings = server_settings
        self._random = python_random_generator
        self._validate_clients()
        reference = next(iter(_client_parameters(run_clients[0]).values()))
        _validate_candidate_parameter_snapshot(
            candidate_parameter_snapshot=initial_parameter_snapshot,
            reference_parameter_snapshot=reference,
        )
        if any(not isfinite(value).all().item() for value in initial_parameter_snapshot.values()):
            raise ValueError("初期parameterは有限値が必要です。")
        statistics = _copy_model_and_class_loss_statistics(loss_statistics=initial_loss_statistics)
        self.global_model_repository = GlobalModelRepository(
            initial_model_id=0,
            initial_parameter_snapshot=initial_parameter_snapshot,
            initial_loss_statistics=statistics,
        )
        self.communication_volume_record_store = CommunicationVolumeRecordStore()
        self.cross_evaluation_record_store = CrossEvaluationRecordStore()
        self.model_clustering_record_store = ModelClusteringRecordStore()
        self._synchronized_timestep_model_counts: list[FeddriftSynchronizedTimestepModelCounts] = []
        self._model_weights = {0: server_settings.initial_model_aggregation_weight}
        self._isolation_until: dict[int, int] = {}
        self._aggregation_parameter_multiply_accumulate_count = 0
        self._consolidation_parameter_multiply_accumulate_count = 0
        self._diagnostic_parameter_distance_multiply_accumulate_count = 0
        self._client_consolidation_parameter_multiply_accumulate_count = 0

    def finalize_observed_stream(self, *, timestep_index: int) -> None:
        """観測済みbufferを確定し、readyがある場合だけ最終回収・集約・配布する。"""
        self._validate_operation(timestep_index, validate_training_inputs=False)
        next_id = self.global_model_repository.next_global_model_id
        potential_registration_count = 0
        reference = next(iter(_client_parameters(self.run_clients[0]).values()))
        for client in self.run_clients:
            stream = client.stream_processor
            count = stream.processed_sample_count
            if type(count) is not int or count < 0:
                raise ValueError("観測件数は非負builtin intが必要です。")
            buffer = stream.buffered_training_samples
            client.detection_batch_processor._validate_stream_finalization_inputs()
            if buffer:
                client.detection_batch_processor.validate_detection_batch_inputs(
                    training_samples=buffer, processed_sample_count=count
                )
            owners = client.owners
            delay = owners.pending_model_upload_state.remaining_upload_delay_round_count
            if type(delay) is not int or delay < 0:
                raise ValueError("保留の待機回数は非負builtin intが必要です。")
            pending = owners.pending_model_upload_state.get_pending_model_upload()
            if pending is not None:
                _validate_candidate_parameter_snapshot(
                    candidate_parameter_snapshot=pending.parameter_snapshot,
                    reference_parameter_snapshot=reference,
                )
                statistics = owners.loss_statistics_store.get_model_loss_statistics(
                    model_id=pending.model_id
                )
                if statistics is None:
                    raise ValueError("保留モデルの損失統計が必要です。")
                _copy_model_and_class_loss_statistics(loss_statistics=statistics)
            current = owners.current_training_model_assignment.current_training_model_id
            if current >= 0 and current not in self.global_model_repository.global_model_ids:
                raise ValueError("現在の正式モデルがrepositoryに存在しません。")
            potential_registration_count += (
                bool(buffer) or client.has_model_ready_for_server_registration()
            )
        planned_ids = range(next_id, next_id + potential_registration_count)
        if any(
            mid in client.get_held_model_ids() for mid in planned_ids for client in self.run_clients
        ) or any(mid in self.global_model_repository.global_model_ids for mid in planned_ids):
            raise ValueError("割当予定の正式IDが既存モデルと衝突しています。")
        for client in self.run_clients:
            client.flush_detection_buffer()
        if not any(client.has_model_ready_for_server_registration() for client in self.run_clients):
            return
        registrations = []
        for client in self.run_clients:
            if not client.has_model_ready_for_server_registration():
                continue
            pending = client.owners.pending_model_upload_state.get_pending_model_upload()
            if pending is None:
                raise ValueError("readyモデルのsnapshotが必要です。")
            statistics = client.owners.loss_statistics_store.get_model_loss_statistics(
                model_id=pending.model_id
            )
            if statistics is None:
                raise ValueError("readyモデルの損失統計が必要です。")
            registrations.append((client, pending, statistics))
        for client, pending, statistics in registrations:
            owners = client.owners
            self._record_parameter_transfer("upload", pending.parameter_snapshot, 1)
            self.communication_volume_record_store.record_messages(
                transfer_direction="upload", message_count=1
            )
            mid = self.global_model_repository.allocate_global_model_id()
            self.global_model_repository.set_global_model_parameters(
                model_id=mid, parameter_snapshot=pending.parameter_snapshot
            )
            self.global_model_repository.set_global_model_loss_statistics(
                model_id=mid, loss_statistics=statistics
            )
            self._model_weights.setdefault(mid, self._settings.initial_model_aggregation_weight)
            self.global_model_repository.record_model_registration(
                model_id=mid,
                registered_round_index=timestep_index,
                registering_client_id=client.client_id,
            )
            confirm_held_model_registration(
                registered_global_model_id=mid,
                held_model_training_state_registry=owners.held_model_training_state_registry,
                loss_statistics_store=owners.loss_statistics_store,
                training_sample_store=owners.training_sample_store,
                evaluation_sample_store=owners.evaluation_sample_store,
                model_training_and_assignment_counts_store=owners.model_training_and_assignment_counts_store,
                current_training_model_assignment=owners.current_training_model_assignment,
                pending_model_upload_state=owners.pending_model_upload_state,
            )
            self.communication_volume_record_store.record_messages(
                transfer_direction="download", message_count=1
            )
        self._aggregate_and_distribute_models(
            active_ids=tuple(sorted(self.global_model_repository.global_model_ids)),
            update_model_weights=False,
        )
        for client in self.run_clients:
            state = client.owners.pending_model_upload_state
            pending = state.get_pending_model_upload()
            if pending is not None and not state.has_ready_model_upload():
                state.queue_ready_model_upload(
                    model_id=pending.model_id, parameter_snapshot=pending.parameter_snapshot
                )

    def snapshot_model_weights(self) -> dict[int, int]:
        return dict(self._model_weights)

    def snapshot_server_computation_counts(self) -> ServerComputationCounts:
        """サーバの成功操作に対応するパラメータMAC。"""
        return ServerComputationCounts(
            aggregation_parameter_multiply_accumulate_count=self._aggregation_parameter_multiply_accumulate_count,
            consolidation_parameter_multiply_accumulate_count=self._consolidation_parameter_multiply_accumulate_count,
            diagnostic_parameter_distance_multiply_accumulate_count=self._diagnostic_parameter_distance_multiply_accumulate_count,
        )

    @property
    def client_consolidation_parameter_multiply_accumulate_count(self) -> int:
        """client自身のパラメータ統合MACの合計。"""
        count = self._client_consolidation_parameter_multiply_accumulate_count
        if type(count) is not int:
            raise TypeError("client統合MACはbuiltin intが必要です。")
        if count < 0:
            raise ValueError("client統合MACは非負が必要です。")
        return count

    def snapshot_synchronized_timestep_model_counts(
        self,
    ) -> tuple[FeddriftSynchronizedTimestepModelCounts, ...]:
        """完了した通常同期のモデル数を記録順で返す。"""
        if type(self._synchronized_timestep_model_counts) is not list:
            raise TypeError("同期モデル数のownerはlistが必要です。")
        for record in self._synchronized_timestep_model_counts:
            if type(record) is not FeddriftSynchronizedTimestepModelCounts:
                raise TypeError("exact FeddriftSynchronizedTimestepModelCountsが必要です。")
            record.__post_init__()
            if len(record.client_held_model_counts) != len(self.run_clients):
                raise ValueError("同期保有数とclient件数が対応していません。")
        return tuple(self._synchronized_timestep_model_counts)

    def snapshot_isolation_until(self) -> dict[int, int]:
        return dict(self._isolation_until)

    def aggregate_and_distribute_training_round(self, *, timestep_index: int) -> None:
        """学習済みclientをID順・client順に集約し、全モデルを配布する。"""
        self._validate_operation(timestep_index)
        active_ids = tuple(
            sorted(
                {
                    mid
                    for client in self.run_clients
                    for mid in client.get_held_model_ids()
                    if mid >= 0
                }
            )
        )
        self._aggregate_and_distribute_models(active_ids=active_ids, update_model_weights=True)

    def _aggregate_and_distribute_models(
        self, *, active_ids: tuple[int, ...], update_model_weights: bool
    ) -> None:
        client_parameters = tuple(_client_parameters(client) for client in self.run_clients)
        planned_parameters = {
            mid: self.global_model_repository.get_global_model_parameters(model_id=mid)
            for mid in self.global_model_repository.global_model_ids
        }
        planned_statistics = dict(
            self.global_model_repository.snapshot_global_model_loss_statistics()
        )
        planned_weights = dict(self._model_weights)
        uploads: list[dict[str, Tensor]] = []
        for mid in active_ids:
            total_weight, total_count, weighted_mean = 0, 0, 0.0
            averaged = None
            for client, parameters in zip(self.run_clients, client_parameters, strict=True):
                if mid not in parameters:
                    continue
                samples = next(
                    (
                        collection.training_samples
                        for collection in client.owners.training_sample_store.snapshot_ordered_model_training_samples()
                        if collection.model_id == mid
                    ),
                    (),
                )
                weight = len(samples)
                if not weight:
                    continue
                values = parameters[mid]
                uploads.append(values)
                if averaged is None:
                    averaged = {name: value * weight for name, value in values.items()}
                else:
                    for name in averaged:
                        averaged[name] = averaged[name] + values[name] * weight
                total_weight += weight
                statistics = client.owners.loss_statistics_store.get_model_loss_statistics(
                    model_id=mid
                )
                if statistics is not None:
                    moments = statistics.overall_loss_moments
                    weighted_mean += moments.mean_loss * moments.observed_loss_count
                    total_count += moments.observed_loss_count
            if total_weight and averaged is not None:
                planned_parameters[mid] = {
                    name: value / total_weight for name, value in averaged.items()
                }
                if update_model_weights:
                    planned_weights[mid] = total_weight
            if total_count:
                planned_statistics[mid] = ModelAndClassLossStatistics(
                    overall_loss_moments=BoundedLossMoments(
                        observed_loss_count=total_count,
                        mean_loss=weighted_mean / total_count,
                        sum_squared_loss_deviations=0.0,
                    )
                )
        reference = next(iter(client_parameters[0].values()))
        for parameters in planned_parameters.values():
            _validate_candidate_parameter_snapshot(
                candidate_parameter_snapshot=parameters, reference_parameter_snapshot=reference
            )
            if any(not isfinite(value).all().item() for value in parameters.values()):
                raise ValueError("集約parameterは有限値が必要です。")
        for client in self.run_clients:
            current = client.owners.current_training_model_assignment.current_training_model_id
            if current >= 0 and current not in planned_parameters:
                raise ValueError("配布後の現在モデルが存在しません。")
        for parameters in uploads:
            self._record_parameter_transfer("upload", parameters, 1)
        for mid, parameters in planned_parameters.items():
            self.global_model_repository.set_global_model_parameters(
                model_id=mid, parameter_snapshot=parameters
            )
        for mid, statistics in planned_statistics.items():
            self.global_model_repository.set_global_model_loss_statistics(
                model_id=mid, loss_statistics=statistics
            )
        self._model_weights = planned_weights
        for parameters in planned_parameters.values():
            self._record_parameter_transfer("download", parameters, len(self.run_clients))
        for client in self.run_clients:
            client.apply_server_model_distribution(
                model_id_mapping={},
                model_parameters_by_model_id=planned_parameters,
                server_loss_statistics=tuple(planned_statistics.items()),
                rebuilt_model_optimizer_settings=self._settings.rebuilt_model_optimizer_settings,
            )

        self._aggregation_parameter_multiply_accumulate_count += sum(
            value.numel() for parameters in uploads for value in parameters.values()
        )

    def _record_parameter_transfer(
        self, direction: str, parameters: dict[str, Tensor], count: int
    ) -> None:
        self.communication_volume_record_store.record_model_transfers(
            transfer_direction=direction, model_count=count
        )
        self.communication_volume_record_store.record_parameter_transfer(
            transfer_direction=direction, parameter_snapshot=parameters, transfer_count=count
        )

    def synchronize_timestep(self, *, timestep_index: int) -> None:
        """観測を追加せず、準備後に設定された回数だけ学習・集約・配布する。"""
        self._validate_operation(timestep_index)
        self.prepare_timestep(timestep_index=timestep_index)
        for _ in range(self._settings.local_training_round_count):
            for client in self.run_clients:
                client.train_local_round()
            self.aggregate_and_distribute_training_round(timestep_index=timestep_index)
        self._synchronized_timestep_model_counts.append(
            FeddriftSynchronizedTimestepModelCounts(
                timestep_index=timestep_index,
                global_model_count=len(self.global_model_repository.global_model_ids),
                client_held_model_counts=tuple(
                    len(
                        client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
                    )
                    for client in self.run_clients
                ),
            )
        )

    def _validate_clients(self, *, validate_training_inputs: bool = True) -> None:
        if type(self.run_clients) is not tuple or not self.run_clients:
            raise TypeError("client列は非空builtin tupleが必要です。")
        ids = []
        reference = None
        for client in self.run_clients:
            if type(client) is not FeddriftRunClient:
                raise TypeError("clientはexact FeddriftRunClientが必要です。")
            if type(client.client_id) is not int or client.client_id < 0 or client.client_id in ids:
                raise ValueError("client IDは非負builtin intで一意が必要です。")
            ids.append(client.client_id)
            if client.python_random_generator is not self._random:
                raise ValueError("全clientとserverは同じrun Randomを借りることが必要です。")
            owners = client.owners
            client._settings.__post_init__()
            _validate_confirmation_inputs(
                registered_global_model_id=0,
                held_model_training_state_registry=owners.held_model_training_state_registry,
                loss_statistics_store=owners.loss_statistics_store,
                training_sample_store=owners.training_sample_store,
                evaluation_sample_store=owners.evaluation_sample_store,
                model_training_and_assignment_counts_store=owners.model_training_and_assignment_counts_store,
                current_training_model_assignment=owners.current_training_model_assignment,
                pending_model_upload_state=owners.pending_model_upload_state,
            )
            states = owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
            for state in states:
                _validate_held_model_training_state_inputs(
                    model_id=state.model_id,
                    classifier=state.classifier,
                    concept_specific_parameter_optimizer_state=state.concept_specific_parameter_optimizer_state,
                )
                _validate_independent_model_structure(state.classifier)
                parameters = snapshot_classifier_parameters(classifier=state.classifier)
                if reference is None:
                    reference = parameters
                _validate_candidate_parameter_snapshot(
                    candidate_parameter_snapshot=parameters, reference_parameter_snapshot=reference
                )
            owners.held_model_training_state_registry.get_held_model_training_state(
                model_id=owners.current_training_model_assignment.current_training_model_id
            )
            owners.loss_statistics_store.get_state_snapshot()
            pending = owners.pending_model_upload_state.get_pending_model_upload()
            if pending is not None:
                pending.__post_init__()
                if (
                    pending.model_id
                    != owners.current_training_model_assignment.current_training_model_id
                    or pending.model_id >= 0
                ):
                    raise ValueError("登録保留は現行の一時モデルであることが必要です。")
            states_by_id = {state.model_id: state for state in states}
            for (
                collection
            ) in owners.evaluation_sample_store.snapshot_ordered_model_evaluation_samples():
                for sample in collection.evaluation_samples:
                    _validate_observed_training_sample(
                        training_sample=ObservedTrainingSample(
                            input_features=sample.input_features,
                            observed_class_labels=sample.observed_class_labels,
                        ),
                        expected_input_feature_count=states[0].classifier.input_feature_count,
                        tensor_name="evaluation_sample",
                    )
                    validate_classifier_bounded_loss_inputs(
                        classifier=states[0].classifier,
                        input_features=sample.input_features,
                        observed_class_labels=sample.observed_class_labels,
                    )
            for (
                collection
            ) in owners.training_sample_store.snapshot_ordered_model_training_samples():
                if collection.model_id not in states_by_id:
                    continue
                state = states_by_id[collection.model_id]
                for sample in collection.training_samples:
                    _validate_observed_training_sample(
                        training_sample=sample,
                        expected_input_feature_count=state.classifier.input_feature_count,
                        tensor_name="training_sample",
                    )
                    validate_classifier_bounded_loss_inputs(
                        classifier=state.classifier,
                        input_features=sample.input_features,
                        observed_class_labels=sample.observed_class_labels,
                    )
                if collection.training_samples and validate_training_inputs:
                    validate_independent_model_parameter_update_inputs(
                        classifier=state.classifier,
                        parameter_optimizer=state.concept_specific_parameter_optimizer_state.parameter_optimizer,
                        input_features=cat([s.input_features for s in collection.training_samples]),
                        observed_class_labels=cat(
                            [s.observed_class_labels for s in collection.training_samples]
                        ),
                    )

    def _validate_operation(
        self, timestep_index: int, *, validate_training_inputs: bool = True
    ) -> None:
        _validate_timestep(timestep_index)
        self.snapshot_synchronized_timestep_model_counts()
        self.snapshot_server_computation_counts()
        _ = self.client_consolidation_parameter_multiply_accumulate_count
        self._settings.__post_init__()
        self._validate_clients(validate_training_inputs=validate_training_inputs)
        next_id = self.global_model_repository.next_global_model_id
        if type(next_id) is not int or next_id < 0:
            raise ValueError("次の正式IDは非負builtin intが必要です。")
        if any(next_id in client.get_held_model_ids() for client in self.run_clients):
            raise ValueError("次の正式IDは保有モデルと衝突できません。")
        for mapping in (self._model_weights, self._isolation_until):
            if type(mapping) is not dict:
                raise TypeError("重みと隔離期限はbuiltin dictが必要です。")
            if any(
                type(mid) is not int or mid < 0 or type(value) is not int or value < 0
                for mid, value in mapping.items()
            ):
                raise ValueError("重みと隔離期限は非負builtin intが必要です。")
        reference = next(iter(_client_parameters(self.run_clients[0]).values()))
        for model_id in self.global_model_repository.global_model_ids:
            parameters = self.global_model_repository.get_global_model_parameters(model_id=model_id)
            _validate_candidate_parameter_snapshot(
                candidate_parameter_snapshot=parameters, reference_parameter_snapshot=reference
            )
            if any(not isfinite(value).all().item() for value in parameters.values()):
                raise ValueError("server parameterは有限値が必要です。")
        for _, statistics in self.global_model_repository.snapshot_global_model_loss_statistics():
            _copy_model_and_class_loss_statistics(loss_statistics=statistics)

    def _cross_evaluate(
        self, timestep_index: int, model_ids: tuple[int, ...]
    ) -> dict[tuple[int, int], tuple[int, float, float]]:
        statistics = {}
        for candidate in model_ids:
            parameters = self.global_model_repository.get_global_model_parameters(
                model_id=candidate
            )
            for target in model_ids:
                holders = [
                    client for client in self.run_clients if target in client.get_held_model_ids()
                ]
                if len(holders) > self._settings.maximum_cross_evaluation_client_count:
                    holders = self._random.sample(
                        holders, self._settings.maximum_cross_evaluation_client_count
                    )
                total_count, total_loss, total_squares = 0, 0.0, 0.0
                for client in holders:
                    owners = client.owners
                    evaluation = evaluate_candidate_model_on_target_model_samples(
                        candidate_parameter_snapshot=parameters,
                        target_model_id=target,
                        compare_correctness_with_held_target_model=False,
                        held_model_training_state_registry=owners.held_model_training_state_registry,
                        model_evaluation_sample_store=owners.evaluation_sample_store,
                        training_sample_store=owners.training_sample_store,
                        current_training_model_assignment=owners.current_training_model_assignment,
                        maximum_evaluation_sample_count=self._settings.maximum_client_evaluation_sample_count,
                        python_random_generator=self._random,
                    )
                    total_count += evaluation.evaluated_sample_count
                    total_loss += evaluation.bounded_loss_sum
                    total_squares += evaluation.squared_bounded_loss_sum
                    self.cross_evaluation_record_store.append_client_cross_evaluation_record(
                        client_cross_evaluation_record=_make_client_cross_evaluation_record(
                            round_index=timestep_index,
                            client_id=client.client_id,
                            candidate_model_id=candidate,
                            target_model_id=target,
                            client_cross_evaluation=evaluation,
                        )
                    )
                for direction in ("upload", "download"):
                    self.communication_volume_record_store.record_messages(
                        transfer_direction=direction, message_count=len(holders)
                    )
                statistics[(candidate, target)] = (total_count, total_loss, total_squares)
        return statistics

    def _record_clustering(
        self,
        timestep_index: int,
        ids: tuple[int, ...],
        clusters: tuple[tuple[int, ...], ...],
        distances: dict[tuple[int, int], float],
    ) -> None:
        by_id = {mid: cluster for cluster in clusters for mid in cluster}
        labels = {}
        for mid in ids:
            counts: dict[int, int] = {}
            for client in self.run_clients:
                assigned = client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts().assigned_sample_counts_by_model_and_concept_id.get(
                    mid, {}
                )
                for concept, count in assigned.items():
                    counts[concept] = counts.get(concept, 0) + count
            if counts:
                winners = [
                    concept for concept, count in counts.items() if count == max(counts.values())
                ]
                if len(winners) == 1:
                    labels[mid] = winners[0]
        pairs = tuple(
            ModelPairClusteringObservation(
                round_index=timestep_index,
                lower_model_id=left,
                higher_model_id=right,
                loss_increase_distance=distance,
                decision_score=distance,
                assigned_to_same_cluster=by_id[left] == by_id[right],
                true_concepts_match=(
                    labels[left] == labels[right] if left in labels and right in labels else None
                ),
                concept_specific_parameter_distance=_compute_concept_specific_parameter_distance(
                    left_parameters=self.global_model_repository.get_global_model_parameters(
                        model_id=left
                    ),
                    right_parameters=self.global_model_repository.get_global_model_parameters(
                        model_id=right
                    ),
                ),
            )
            for (left, right), distance in distances.items()
        )
        models = _make_model_clustering_observations(
            round_index=timestep_index,
            clustered_model_ids=ids,
            loss_increase_distances_by_model_pair=distances,
            model_cluster_by_model_id=by_id,
        )
        self.model_clustering_record_store.append_clustering_observations(
            pair_observations=pairs, model_observations=models
        )

        if pairs:
            parameter_count = sum(
                value.numel()
                for value in self.global_model_repository.get_global_model_parameters(
                    model_id=ids[0]
                ).values()
            )
            self._diagnostic_parameter_distance_multiply_accumulate_count += (
                3 * parameter_count * len(pairs)
            )

    def prepare_timestep(self, *, timestep_index: int) -> None:
        """成熟モデルの統合の後、新モデルを隔離登録する。"""
        self._validate_operation(timestep_index)
        ids = tuple(sorted(self.global_model_repository.global_model_ids))
        # 成熟モデルの平均は全clientが持つsnapshotから作る。欠落を通信の前に拒否する。
        for client in self.run_clients:
            if any(mid not in _client_parameters(client) for mid in ids):
                raise ValueError("全clientがグローバルモデルを保有することが必要です。")
        self.communication_volume_record_store.record_messages(
            transfer_direction="upload", message_count=len(self.run_clients)
        )
        mature = tuple(mid for mid in ids if timestep_index >= self._isolation_until.get(mid, 0))
        if len(mature) > 1:
            statistics = self._cross_evaluate(timestep_index, mature)
            clusters = cluster_feddrift_model_ids(
                model_ids=mature,
                loss_sums_by_model_pair=statistics,
                minimum_evaluated_sample_count=self._settings.minimum_cross_evaluation_sample_count,
                maximum_same_cluster_loss_increase=self._settings.maximum_same_cluster_loss_increase,
                cluster_linkage=self._settings.cluster_linkage,
            )
            distances = _loss_increase_distances(
                model_ids=mature,
                loss_sums_by_model_pair=statistics,
                minimum_evaluated_sample_count=self._settings.minimum_cross_evaluation_sample_count,
            )
            self._record_clustering(timestep_index, mature, clusters, distances)
            if len(clusters) < len(mature):
                all_clusters = clusters + tuple((mid,) for mid in ids if mid not in mature)
                parameters = {
                    mid: self.global_model_repository.get_global_model_parameters(model_id=mid)
                    for mid in ids
                }
                consolidation = consolidate_feddrift_cached_model_clusters(
                    timestep_index=timestep_index,
                    model_clusters=all_clusters,
                    model_parameters_by_model_id=parameters,
                    historical_model_weights_by_model_id={
                        mid: weight for mid, weight in self._model_weights.items() if mid in ids
                    },
                    loss_moments_by_model_id={
                        mid: stat.overall_loss_moments
                        for mid, stat in self.global_model_repository.snapshot_global_model_loss_statistics()
                    },
                    isolation_until_by_model_id={
                        mid: expiry for mid, expiry in self._isolation_until.items() if mid in ids
                    },
                )
                server_statistics = tuple(
                    (mid, ModelAndClassLossStatistics(overall_loss_moments=stat))
                    for mid, stat in consolidation.loss_moments_by_representative_id.items()
                )
                own_inputs = tuple(
                    {mid: value for mid, value in _client_parameters(client).items() if mid >= 0}
                    for client in self.run_clients
                )
                own_parameters = tuple(
                    average_feddrift_cached_cluster_parameters(
                        model_clusters=all_clusters,
                        model_parameters_by_model_id=values,
                        cluster_weights_by_representative_id=consolidation.cluster_weights_by_representative_id,
                    )
                    for values in own_inputs
                )
                for client, own in zip(self.run_clients, own_parameters, strict=True):
                    client.apply_server_model_distribution(
                        model_id_mapping=consolidation.model_id_mapping,
                        model_parameters_by_model_id=own,
                        server_loss_statistics=server_statistics,
                        rebuilt_model_optimizer_settings=self._settings.rebuilt_model_optimizer_settings,
                    )
                self.communication_volume_record_store.record_messages(
                    transfer_direction="download", message_count=len(self.run_clients)
                )
                for mid in ids:
                    self.global_model_repository.remove_global_model(model_id=mid)
                for mid, value in consolidation.model_parameters_by_representative_id.items():
                    self.global_model_repository.set_global_model_parameters(
                        model_id=mid, parameter_snapshot=value
                    )
                for mid, value in server_statistics:
                    self.global_model_repository.set_global_model_loss_statistics(
                        model_id=mid, loss_statistics=value
                    )
                self._model_weights = consolidation.model_weights_by_representative_id
                self._isolation_until = consolidation.isolation_until_by_representative_id
                self._consolidation_parameter_multiply_accumulate_count += sum(
                    value.numel() for values in parameters.values() for value in values.values()
                )
                self._client_consolidation_parameter_multiply_accumulate_count += sum(
                    value.numel()
                    for own in own_inputs
                    for values in own.values()
                    for value in values.values()
                )
        for client in self.run_clients:
            if not client.has_model_ready_for_server_registration():
                continue
            mid = self.global_model_repository.allocate_global_model_id()
            self.global_model_repository.record_model_registration(
                model_id=mid,
                registered_round_index=timestep_index,
                registering_client_id=client.client_id,
            )
            owners = client.owners
            confirm_held_model_registration(
                registered_global_model_id=mid,
                held_model_training_state_registry=owners.held_model_training_state_registry,
                loss_statistics_store=owners.loss_statistics_store,
                training_sample_store=owners.training_sample_store,
                evaluation_sample_store=owners.evaluation_sample_store,
                model_training_and_assignment_counts_store=owners.model_training_and_assignment_counts_store,
                current_training_model_assignment=owners.current_training_model_assignment,
                pending_model_upload_state=owners.pending_model_upload_state,
            )
            self._isolation_until[mid] = (
                timestep_index + self._settings.new_model_isolation_timestep_count
            )
            self.communication_volume_record_store.record_messages(
                transfer_direction="download", message_count=1
            )
