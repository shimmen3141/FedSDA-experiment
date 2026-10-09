"""最終構成のFedSDAのclient 1つを組み立て、単一runの実行の枠が求める操作を提供する。"""

from copy import deepcopy
from dataclasses import dataclass
from random import Random

from torch import float32, tensor

from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import (
    AdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.evaluation.adaptation_record_store import (
    AdaptationRecordStore,
)
from federated_learning_experiments.evaluation.loss_change_alarm_record_store import (
    LossChangeAlarmRecordStore,
)
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.evaluation.sample_prediction_record_store import (
    SamplePredictionRecordStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.local_training_request_schedule import (
    LocalTrainingRequestSchedule,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
    select_loss_monitoring_baseline_mean_loss,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights import (
    FixedSharePredictionWeightController,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store import (
    PendingSampleObservationStore,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.runtime.candidate_validation_session_holder import (
    CandidateValidationSessionHolder,
)
from federated_learning_experiments.runtime.fedsda_run_client_settings import (
    FedsdaRunClientSettings,
)
from federated_learning_experiments.runtime.held_candidate_validation_progress import (
    HeldIncompleteCandidateValidationFinalization,
    finalize_held_incomplete_candidate_validation,
)
from federated_learning_experiments.runtime.held_model_training_request_handling import (
    train_held_models_for_pending_training_requests,
)
from federated_learning_experiments.runtime.observed_sample_processing import (
    ObservedSampleProcessing,
    process_observed_sample,
)

# 観測標本（ObservedSample）が持つ特徴の数。
_OBSERVED_SAMPLE_FEATURE_COUNT = 2


@dataclass(frozen=True, kw_only=True)
class FedsdaRunClientOwners:
    """clientが持つ全ownerへの参照。状態はowner自身が持つ。field名は、標本1件の処理の引数名と同じ。"""

    fixed_share_prediction_weight_controller: FixedSharePredictionWeightController
    sample_prediction_record_store: SamplePredictionRecordStore
    validation_session_holder: CandidateValidationSessionHolder
    adaptation_record_store: AdaptationRecordStore
    diagnostic_evidence_collection: AdaHedgeDiagnosticEvidenceCollection
    loss_change_monitor: OverallAndTrueClassLossMonitor
    loss_change_alarm_record_store: LossChangeAlarmRecordStore
    pending_training_assignment_buffer: PendingTrainingAssignmentBuffer
    pending_sample_observation_store: PendingSampleObservationStore
    held_model_training_state_registry: HeldModelTrainingStateRegistry
    loss_statistics_store: ModelAndClassLossStatisticsStore
    training_sample_store: ModelTrainingSampleStore
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore
    current_training_model_assignment: CurrentTrainingModelAssignment
    model_evaluation_sample_store: ModelEvaluationSampleStore
    temporary_model_id_allocator: TemporaryModelIdAllocator
    pending_model_upload_state: PendingModelUploadState
    local_training_request_schedule: LocalTrainingRequestSchedule
    shared_parameter_optimizer_state: ParameterOptimizerState


def _validate_round_index(*, round_index: int) -> None:
    if type(round_index) is not int:
        raise TypeError("round_index must be builtin int")
    if round_index < 0:
        raise ValueError("round_index must be nonnegative")


class FedsdaRunClient:
    """最終構成のFedSDAのclient。ownerと設定を持ち、操作を既存の部品へ渡す（判断は持たない）。

    生成は`assemble_fedsda_run_client`が行う。`__init__`は、受け取ったものを検査せずに持つ。
    """

    def __init__(
        self,
        *,
        client_id: int,
        owners: FedsdaRunClientOwners,
        run_client_settings: FedsdaRunClientSettings,
        architecture_reference_classifier: ResidualAdapterClassifier,
        python_random_generator: Random,
    ) -> None:
        self._client_id = client_id
        self._owners = owners
        self._run_client_settings = run_client_settings
        self._architecture_reference_classifier = architecture_reference_classifier
        self._python_random_generator = python_random_generator

    @property
    def client_id(self) -> int:
        return self._client_id

    @property
    def owners(self) -> FedsdaRunClientOwners:
        return self._owners

    def process_observed_sample(
        self,
        *,
        observed_sample: ObservedSample,
        sample_index: int,
        evaluation_concept_id: int | None = None,
    ) -> ObservedSampleProcessing:
        """観測標本を1行のtensorへ変換して、標本1件の処理（予測を含む）を行う。

        真の概念IDは診断にだけ使う。渡されなければ、概念別の診断と割当概念の計数は行われない。
        位置と概念IDの型、位置の連続は、標本1件の処理が、どの更新より前に確かめる。
        """
        if type(observed_sample) is not ObservedSample:
            raise TypeError("observed_sample must be exact ObservedSample")
        owners = self._owners
        settings = self._run_client_settings
        scalar_settings = settings.scalar_settings
        return process_observed_sample(
            indexed_observation=IndexedObservedTrainingSample(
                sample_index=sample_index,
                training_sample=ObservedTrainingSample(
                    input_features=tensor([list(observed_sample.feature_values)], dtype=float32),
                    observed_class_labels=tensor(
                        [[float(observed_sample.class_label)]], dtype=float32
                    ),
                ),
                observed_concept_id=evaluation_concept_id,
            ),
            fixed_share_prediction_weight_controller=owners.fixed_share_prediction_weight_controller,
            sample_prediction_record_store=owners.sample_prediction_record_store,
            validation_session_holder=owners.validation_session_holder,
            adaptation_record_store=owners.adaptation_record_store,
            diagnostic_evidence_collection=owners.diagnostic_evidence_collection,
            loss_change_monitor=owners.loss_change_monitor,
            loss_change_alarm_record_store=owners.loss_change_alarm_record_store,
            pending_training_assignment_buffer=owners.pending_training_assignment_buffer,
            pending_sample_observation_store=owners.pending_sample_observation_store,
            held_model_training_state_registry=owners.held_model_training_state_registry,
            loss_statistics_store=owners.loss_statistics_store,
            training_sample_store=owners.training_sample_store,
            model_training_and_assignment_counts_store=owners.model_training_and_assignment_counts_store,
            current_training_model_assignment=owners.current_training_model_assignment,
            model_evaluation_sample_store=owners.model_evaluation_sample_store,
            temporary_model_id_allocator=owners.temporary_model_id_allocator,
            pending_model_upload_state=owners.pending_model_upload_state,
            local_training_request_schedule=owners.local_training_request_schedule,
            candidate_model_training_and_acceptance_settings=settings.candidate_model_training_and_acceptance_settings,
            maximum_reference_mean_loss_increase=scalar_settings.maximum_tolerated_mean_loss_increase,
            minimum_candidate_mean_loss_improvement=scalar_settings.minimum_candidate_mean_loss_improvement,
            upload_delay_round_count=scalar_settings.new_model_upload_delay_round_count,
            minimum_change_interval_sample_count=scalar_settings.minimum_change_interval_sample_count,
            maximum_alarm_interval_mean_loss_increase=scalar_settings.maximum_tolerated_mean_loss_increase,
            candidate_parameter_initialization_settings=settings.candidate_parameter_initialization_settings,
            architecture_reference_classifier=self._architecture_reference_classifier,
            parameter_optimizer_settings=settings.parameter_optimizer_settings,
            candidate_epoch_training_settings=settings.candidate_epoch_training_settings,
            detector_name=scalar_settings.detector_name,
            batch_sample_count=scalar_settings.local_training_batch_sample_count,
            python_random_generator=self._python_random_generator,
            local_training_settings=settings.local_training_settings,
            shared_feature_extractor=self._architecture_reference_classifier.feature_extractor,
            shared_parameter_optimizer=owners.shared_parameter_optimizer_state.parameter_optimizer,
        )

    def flush_pending_local_updates(self, *, round_index: int) -> tuple[float, ...]:
        """ラウンド境界で、保留中の学習要求があれば学習する。完了した共同更新の損失を返す。"""
        _validate_round_index(round_index=round_index)
        owners = self._owners
        return train_held_models_for_pending_training_requests(
            local_training_request_schedule=owners.local_training_request_schedule,
            held_model_training_state_registry=owners.held_model_training_state_registry,
            training_sample_store=owners.training_sample_store,
            model_training_and_assignment_counts_store=owners.model_training_and_assignment_counts_store,
            batch_sample_count=self._run_client_settings.scalar_settings.local_training_batch_sample_count,
            python_random_generator=self._python_random_generator,
            local_training_settings=self._run_client_settings.local_training_settings,
            shared_feature_extractor=self._architecture_reference_classifier.feature_extractor,
            shared_parameter_optimizer=owners.shared_parameter_optimizer_state.parameter_optimizer,
        )

    def has_model_ready_for_server_registration(self) -> bool:
        """送信保留のモデルがあり、送信までの待ちが済んでいるときだけ真。状態は変えない。"""
        return self._owners.pending_model_upload_state.has_ready_model_upload()

    def advance_new_model_upload_wait_after_synchronization(self, *, round_index: int) -> None:
        """同期の後で、送信保留のモデルの待ちを1ラウンド進める。"""
        _validate_round_index(round_index=round_index)
        self._owners.pending_model_upload_state.advance_upload_readiness_at_round_boundary()

    def finalize_incomplete_candidate_validation(
        self,
    ) -> HeldIncompleteCandidateValidationFinalization | None:
        """終端で、保持中の未完了の候補検証を回収し、候補検証へ渡した標本の概念IDの保持を外す。"""
        owners = self._owners
        if owners.validation_session_holder.held_validation_session is None:
            return None
        validation_assignment_sample_concept_ids = (
            owners.pending_sample_observation_store.validation_assignment_sample_concept_ids
        )
        if validation_assignment_sample_concept_ids is None:
            raise ValueError(
                "validation assignment sample concept IDs must be held while a session is held"
            )
        last_observed_sample_index = owners.pending_training_assignment_buffer.get_state_snapshot().last_observed_sample_index
        incomplete_validation_finalization = finalize_held_incomplete_candidate_validation(
            validation_session_holder=owners.validation_session_holder,
            adaptation_record_store=owners.adaptation_record_store,
            processed_sample_count=(
                0 if last_observed_sample_index is None else last_observed_sample_index + 1
            ),
            pending_assignment_sample_concept_ids=validation_assignment_sample_concept_ids,
            held_model_training_state_registry=owners.held_model_training_state_registry,
            loss_statistics_store=owners.loss_statistics_store,
            training_sample_store=owners.training_sample_store,
            model_training_and_assignment_counts_store=owners.model_training_and_assignment_counts_store,
            current_training_model_assignment=owners.current_training_model_assignment,
        )
        owners.pending_sample_observation_store.release_validation_assignment_sample_concept_ids()
        return incomplete_validation_finalization


def _validate_optimizer_state_parameters(
    *,
    parameter_optimizer_state: ParameterOptimizerState,
    expected_parameters: tuple[object, ...],
    optimizer_state_argument_name: str,
) -> None:
    optimizer_parameters = tuple(
        parameter
        for parameter_group in parameter_optimizer_state.parameter_optimizer.param_groups
        for parameter in parameter_group["params"]
    )
    if len(optimizer_parameters) != len(expected_parameters) or any(
        parameter is not expected_parameter
        for parameter, expected_parameter in zip(optimizer_parameters, expected_parameters)
    ):
        raise ValueError(
            f"{optimizer_state_argument_name} must optimize exactly the matching parameters of "
            "initial_classifier in the same order"
        )


def _validate_fedsda_run_client_assembly_inputs(
    *,
    client_id: int,
    initial_model_id: int,
    initial_classifier: ResidualAdapterClassifier,
    initial_concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    initial_shared_parameter_optimizer_state: ParameterOptimizerState,
    initial_loss_statistics: ModelAndClassLossStatistics,
    run_client_settings: FedsdaRunClientSettings,
    python_random_generator: Random,
) -> None:
    if type(run_client_settings) is not FedsdaRunClientSettings:
        raise TypeError("run_client_settings must be exact FedsdaRunClientSettings")
    # frozenを回避して組み立てた束も拒否できるよう、束の検査をもう一度行う。
    run_client_settings.__post_init__()
    for identifier_name, identifier in (
        ("client_id", client_id),
        ("initial_model_id", initial_model_id),
    ):
        if type(identifier) is not int:
            raise TypeError(f"{identifier_name} must be builtin int")
    if client_id < 0:
        raise ValueError("client_id must be nonnegative")
    # 負のIDは、clientが採番する一時IDの領域（評価標本の保持など、負のIDを対象外にする部品がある）。
    if initial_model_id < 0:
        raise ValueError("initial_model_id must be nonnegative")
    if type(python_random_generator) is not Random:
        raise TypeError("python_random_generator must be exact random.Random")
    if type(initial_classifier) is not ResidualAdapterClassifier:
        raise TypeError("initial_classifier must be exact ResidualAdapterClassifier")
    for optimizer_state_argument_name, parameter_optimizer_state in (
        (
            "initial_concept_specific_parameter_optimizer_state",
            initial_concept_specific_parameter_optimizer_state,
        ),
        ("initial_shared_parameter_optimizer_state", initial_shared_parameter_optimizer_state),
    ):
        if type(parameter_optimizer_state) is not ParameterOptimizerState:
            raise TypeError(
                f"{optimizer_state_argument_name} must be exact ParameterOptimizerState"
            )
    if type(initial_loss_statistics) is not ModelAndClassLossStatistics:
        raise TypeError("initial_loss_statistics must be exact ModelAndClassLossStatistics")
    if initial_classifier.feature_extractor.input_feature_count != _OBSERVED_SAMPLE_FEATURE_COUNT:
        raise ValueError(
            "initial_classifier must take the feature count of observed samples "
            f"({_OBSERVED_SAMPLE_FEATURE_COUNT})"
        )
    _validate_optimizer_state_parameters(
        parameter_optimizer_state=initial_concept_specific_parameter_optimizer_state,
        expected_parameters=tuple(initial_classifier.residual_adapter.parameters())
        + tuple(initial_classifier.classification_layer.parameters()),
        optimizer_state_argument_name="initial_concept_specific_parameter_optimizer_state",
    )
    _validate_optimizer_state_parameters(
        parameter_optimizer_state=initial_shared_parameter_optimizer_state,
        expected_parameters=tuple(initial_classifier.feature_extractor.parameters()),
        optimizer_state_argument_name="initial_shared_parameter_optimizer_state",
    )


def assemble_fedsda_run_client(
    *,
    client_id: int,
    initial_model_id: int,
    initial_classifier: ResidualAdapterClassifier,
    initial_concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    initial_shared_parameter_optimizer_state: ParameterOptimizerState,
    initial_loss_statistics: ModelAndClassLossStatistics,
    run_client_settings: FedsdaRunClientSettings,
    python_random_generator: Random,
) -> FedsdaRunClient:
    """引数の検査の後、初期モデルを写し、ownerを作って、clientを返す。

    渡された初期モデル（分類器と2つのoptimizerの状態）は変えず、値と蓄積した状態が同じ写しを保有させる。
    乱数生成器は借りる（写さない）。組立ての間、乱数は使わない。分類器のパラメータの置き場所と型
    （CPUのfloat32）は、保有モデルの登録が、写しに対して確かめる（拒否のとき、渡されたものは変わらない）。
    """
    _validate_fedsda_run_client_assembly_inputs(
        client_id=client_id,
        initial_model_id=initial_model_id,
        initial_classifier=initial_classifier,
        initial_concept_specific_parameter_optimizer_state=initial_concept_specific_parameter_optimizer_state,
        initial_shared_parameter_optimizer_state=initial_shared_parameter_optimizer_state,
        initial_loss_statistics=initial_loss_statistics,
        run_client_settings=run_client_settings,
        python_random_generator=python_random_generator,
    )
    # 3つを同時に写すので、写しの中でも、optimizerは写しの分類器のパラメータを指す。
    (
        held_classifier,
        concept_specific_parameter_optimizer_state,
        shared_parameter_optimizer_state,
    ) = deepcopy(
        (
            initial_classifier,
            initial_concept_specific_parameter_optimizer_state,
            initial_shared_parameter_optimizer_state,
        )
    )
    scalar_settings = run_client_settings.scalar_settings
    held_model_training_state_registry = HeldModelTrainingStateRegistry()
    held_model_training_state_registry.register_held_model_training_state(
        model_id=initial_model_id,
        classifier=held_classifier,
        concept_specific_parameter_optimizer_state=concept_specific_parameter_optimizer_state,
    )
    owners = FedsdaRunClientOwners(
        fixed_share_prediction_weight_controller=FixedSharePredictionWeightController(
            prediction_combination_settings=run_client_settings.prediction_combination_settings
        ),
        sample_prediction_record_store=SamplePredictionRecordStore(),
        validation_session_holder=CandidateValidationSessionHolder(),
        adaptation_record_store=AdaptationRecordStore(),
        diagnostic_evidence_collection=AdaHedgeDiagnosticEvidenceCollection(),
        loss_change_monitor=OverallAndTrueClassLossMonitor(
            loss_change_detection_settings=run_client_settings.loss_change_detection_settings,
            class_count=held_classifier.class_count,
            initial_baseline_loss_mean=select_loss_monitoring_baseline_mean_loss(
                loss_moments=initial_loss_statistics.overall_loss_moments
            ),
            maximum_retained_candidate_count=scalar_settings.loss_monitor_maximum_retained_candidate_count,
            betting_fractions=run_client_settings.loss_monitor_betting_fractions,
        ),
        loss_change_alarm_record_store=LossChangeAlarmRecordStore(),
        pending_training_assignment_buffer=PendingTrainingAssignmentBuffer(
            training_data_assignment_settings=run_client_settings.training_data_assignment_settings
        ),
        pending_sample_observation_store=PendingSampleObservationStore(),
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=ModelAndClassLossStatisticsStore(
            initial_loss_statistics_by_model_id={initial_model_id: initial_loss_statistics}
        ),
        training_sample_store=ModelTrainingSampleStore(),
        model_training_and_assignment_counts_store=ModelTrainingAndAssignmentCountsStore(),
        current_training_model_assignment=CurrentTrainingModelAssignment(
            initial_model_id=initial_model_id
        ),
        model_evaluation_sample_store=ModelEvaluationSampleStore(
            maximum_stored_sample_count_per_model=scalar_settings.maximum_stored_evaluation_sample_count_per_model,
            added_batch_sample_count=scalar_settings.added_evaluation_batch_sample_count,
        ),
        temporary_model_id_allocator=TemporaryModelIdAllocator(client_id=client_id),
        pending_model_upload_state=PendingModelUploadState(),
        local_training_request_schedule=LocalTrainingRequestSchedule(
            local_training_schedule_settings=run_client_settings.local_training_schedule_settings
        ),
        shared_parameter_optimizer_state=shared_parameter_optimizer_state,
    )
    return FedsdaRunClient(
        client_id=client_id,
        owners=owners,
        run_client_settings=run_client_settings,
        architecture_reference_classifier=held_classifier,
        python_random_generator=python_random_generator,
    )
