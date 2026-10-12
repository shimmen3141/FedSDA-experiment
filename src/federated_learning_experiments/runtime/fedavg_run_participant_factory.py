"""FedAvgの初期準備と単一モデル参加者の組立て。"""

from copy import deepcopy
from random import Random

from numpy.random import RandomState

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.dataset_definitions import get_dataset_definition
from federated_learning_experiments.data.observed_sample_generation import (
    ObservedSampleGenerator,
    is_observed_sample_generator_of_dataset,
)
from federated_learning_experiments.execution.run_participant_contracts import RunParticipants
from federated_learning_experiments.execution.run_random_sources import RunRandomSources
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.methods.fedavg.fedavg_client import FedavgClient
from federated_learning_experiments.runtime.fedavg_run_server import FedavgRunServer
from federated_learning_experiments.runtime.fedavg_run_settings import FedavgRunParticipantSettings
from federated_learning_experiments.runtime.initial_model_pretraining import pretrain_initial_model


class FedavgRunParticipantFactory:
    """検査済み設定からrunごとの独立した状態を準備する。"""

    def __init__(self, *, run_participant_settings: FedavgRunParticipantSettings) -> None:
        self._settings = run_participant_settings
        self._prepared: RunParticipants | None = None
        self._sources: RunRandomSources | None = None

    @property
    def prepared_run_participants(self) -> RunParticipants | None:
        return self._prepared

    def validate_configuration(self) -> None:
        if type(self._settings) is not FedavgRunParticipantSettings:
            raise TypeError("exact FedAvg参加者設定が必要です。")
        self._settings.__post_init__()

    def prepare_run(
        self,
        *,
        experiment_run_conditions: ExperimentRunConditions,
        run_random_sources: RunRandomSources,
        sample_generator: ObservedSampleGenerator,
    ) -> RunParticipants:
        self.validate_configuration()
        if type(experiment_run_conditions) is not ExperimentRunConditions:
            raise TypeError("exact実行条件が必要です。")
        experiment_run_conditions.__post_init__()
        if (
            type(run_random_sources) is not RunRandomSources
            or type(run_random_sources.python_random_generator) is not Random
            or type(run_random_sources.numpy_random_generator) is not RandomState
        ):
            raise TypeError("exact run乱数源が必要です。")
        if (
            not is_observed_sample_generator_of_dataset(
                sample_generator=sample_generator,
                dataset_name=experiment_run_conditions.dataset_name,
            )
            or getattr(sample_generator, "numpy_random_generator", None)
            is not run_random_sources.numpy_random_generator
        ):
            raise ValueError("datasetとrunのNumPy乱数を借用する生成器が必要です。")
        definition = get_dataset_definition(dataset_name=experiment_run_conditions.dataset_name)
        settings = self._settings
        initial = pretrain_initial_model(
            initial_model_pretraining_settings=settings.initial_model_pretraining_settings,
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="independent_models_without_shared_backbone",
                residual_adapter_requested_rank=1,
            ),
            input_feature_count=definition.input_feature_count,
            hidden_layer_widths=settings.hidden_layer_widths,
            class_count=definition.class_count,
            parameter_optimizer_settings=settings.parameter_optimizer_settings,
            sample_generator=sample_generator,
            python_random_generator=run_random_sources.python_random_generator,
        )
        clients = tuple(
            FedavgClient(
                client_id=i,
                initial_classifier=initial.classifier,
                initial_parameter_optimizer_state=initial.concept_specific_parameter_optimizer_state,
                initial_loss_statistics=initial.loss_statistics,
                local_training_batch_sample_count=settings.local_training_batch_sample_count,
                updates_per_sample=settings.updates_per_sample,
                local_update_interval_per_client_samples=settings.local_update_interval_per_client_samples,
                python_random_generator=run_random_sources.python_random_generator,
            )
            for i in range(experiment_run_conditions.client_count)
        )
        server = FedavgRunServer(
            run_clients=clients,
            initial_parameter_snapshot=snapshot_classifier_parameters(
                classifier=initial.classifier
            ),
            initial_loss_statistics=deepcopy(initial.loss_statistics),
        )
        participants = RunParticipants(client_operations=clients, server_operations=server)
        self._prepared = participants
        self._sources = run_random_sources
        return participants
