"""借りたPython乱数でclient順に評価用の二概念系列を生成する。"""

from random import Random

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.data.observed_streams import ClientConceptTrace


def generate_random_client_concept_traces(
    *,
    experiment_run_conditions: ExperimentRunConditions,
    concept_schedule_settings: RandomConceptScheduleSettings,
    python_random_generator: Random,
) -> tuple[ClientConceptTrace, ...]:
    """全clientの概念列を生成し、変更試行と概念選択の乱数順を維持する。"""
    evaluation_concept_traces: list[ClientConceptTrace] = []
    for client_id in range(experiment_run_conditions.client_count):
        concept_ids_by_sample_index: list[int] = []
        current_concept_id = 0
        last_concept_change_sample_index = 0
        for sample_index in range(experiment_run_conditions.per_client_sample_count):
            if (
                sample_index - last_concept_change_sample_index
                > concept_schedule_settings.minimum_sample_index_gap_before_change_trial
                and python_random_generator.random()
                < concept_schedule_settings.per_eligible_sample_concept_change_probability
            ):
                alternative_concept_ids = [
                    concept_id for concept_id in range(2) if concept_id != current_concept_id
                ]
                # 候補が一つでもchoiceの乱数消費を省略しない。
                current_concept_id = python_random_generator.choice(alternative_concept_ids)
                last_concept_change_sample_index = sample_index
            concept_ids_by_sample_index.append(current_concept_id)
        evaluation_concept_traces.append(
            ClientConceptTrace(
                client_id=client_id,
                concept_ids_by_sample_index=tuple(concept_ids_by_sample_index),
            )
        )
    return tuple(evaluation_concept_traces)
