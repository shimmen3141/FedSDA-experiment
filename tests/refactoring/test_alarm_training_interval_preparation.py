"""位置付き入力と準備結果の構造不変・借用・keyword専用契約。"""

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import (
    PreparedAlarmTrainingIntervals,
)


def test_alarm_preparation_records_are_immutable_and_borrow_payloads():
    training_sample = ObservedTrainingSample(
        input_features=torch.ones(1, 2), observed_class_labels=torch.zeros(1, 1)
    )
    indexed_observation = IndexedObservedTrainingSample(
        sample_index=7, training_sample=training_sample, observed_concept_id=None
    )
    earlier_observations = (indexed_observation,)
    prepared_intervals = PreparedAlarmTrainingIntervals(
        earlier_observations=earlier_observations,
        change_interval_observations=(),
        change_interval_start_sample_index=None,
        earlier_interval_absorbed_model_id=-3,
    )
    assert indexed_observation.training_sample is training_sample
    assert prepared_intervals.earlier_observations is earlier_observations
    assert type(prepared_intervals.earlier_observations) is tuple
    for record, expected_field_names in (
        (indexed_observation, ("sample_index", "training_sample", "observed_concept_id")),
        (
            prepared_intervals,
            (
                "earlier_observations",
                "change_interval_observations",
                "change_interval_start_sample_index",
                "earlier_interval_absorbed_model_id",
            ),
        ),
    ):
        assert tuple(field.name for field in fields(record)) == expected_field_names
        assert all(field.kw_only for field in fields(record))
        for field_name in expected_field_names:
            with pytest.raises(FrozenInstanceError):
                setattr(record, field_name, None)
    with pytest.raises(TypeError):
        IndexedObservedTrainingSample(7, training_sample, None)
    with pytest.raises(TypeError):
        PreparedAlarmTrainingIntervals(earlier_observations, (), None, -3)
    # payload自体は凍結しない。record constructorは境界検査を代行しない。
    training_sample.input_features.fill_(2)
    assert indexed_observation.training_sample.input_features[0, 0].item() == 2
    assert (
        IndexedObservedTrainingSample(
            sample_index=True, training_sample=None, observed_concept_id=False
        ).sample_index
        is True
    )
