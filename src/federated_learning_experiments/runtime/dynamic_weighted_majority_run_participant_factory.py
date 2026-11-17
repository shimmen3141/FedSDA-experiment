"""DWMの初期準備（事前学習1回）と、clientごとに独立なDWMとサーバの組立て。"""

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
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.methods.dynamic_weighted_majority.dynamic_weighted_majority_client import (
    DynamicWeightedMajorityClient,
)
from federated_learning_experiments.runtime.dynamic_weighted_majority_run_server import (
    DynamicWeightedMajorityRunServer,
)
from federated_learning_experiments.runtime.dynamic_weighted_majority_run_settings import (
    DynamicWeightedMajorityRunParticipantSettings,
)
from federated_learning_experiments.runtime.initial_model_pretraining import pretrain_initial_model


class DynamicWeightedMajorityRunParticipantFactory:
    """検査済み設定から、FedAvgと同じ事前学習を1回行い、各clientへ同じ初期状態を渡す。"""

    def __init__(
        self, *, run_participant_settings: DynamicWeightedMajorityRunParticipantSettings
    ) -> None:
        self._settings = run_participant_settings
        self._prepared: RunParticipants | None = None

    @property
    def prepared_run_participants(self) -> RunParticipants | None:
        return self._prepared

    def validate_configuration(self) -> None:
        if type(self._settings) is not DynamicWeightedMajorityRunParticipantSettings:
            raise TypeError("exact DWM参加者設定が必要です。")
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
        # FedAvgのfactoryと同じ構造・同じ引数で事前学習する（同じ乱数の消費）。
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
            DynamicWeightedMajorityClient(
                client_id=i,
                initial_classifier=initial.classifier,
                initial_parameter_optimizer_state=initial.concept_specific_parameter_optimizer_state,
                local_training_batch_sample_count=settings.local_training_batch_sample_count,
                updates_per_sample=settings.updates_per_sample,
                weight_decrease_factor=settings.weight_decrease_factor,
                removal_weight_threshold=settings.removal_weight_threshold,
                period_sample_count=settings.period_sample_count,
                python_random_generator=run_random_sources.python_random_generator,
            )
            for i in range(experiment_run_conditions.client_count)
        )
        server = DynamicWeightedMajorityRunServer(run_clients=clients)
        participants = RunParticipants(client_operations=clients, server_operations=server)
        self._prepared = participants
        return participants
