"""クラスタリングの診断の記録のowner: 追加の順、観測の検査、不正な入力で変わらないこと。"""

import math
from dataclasses import replace

import pytest
from test_held_candidate_validation_progress import make_subclass_copy

from federated_learning_experiments.evaluation.model_clustering_record_store import (
    ModelClusteringObservation,
    ModelClusteringRecordStore,
    ModelPairClusteringObservation,
)

VALID_PAIR_OBSERVATION_FIELDS = dict(
    round_index=9,
    lower_model_id=1,
    higher_model_id=3,
    loss_increase_distance=0.001,
    decision_score=0.0,
    assigned_to_same_cluster=True,
    true_concepts_match=False,
    concept_specific_parameter_distance=0.25,
)
VALID_MODEL_OBSERVATION_FIELDS = dict(
    round_index=9,
    model_id=3,
    nearest_model_id=1,
    nearest_loss_increase_distance=0.001,
    representative_model_id=1,
    cluster_model_count=2,
    maximum_within_cluster_distance=0.001,
    evaluated_within_cluster_pair_count=1,
    possible_within_cluster_pair_count=1,
    merged_with_other_models=True,
    absorbed_into_representative=True,
)
# 1モデルのクラスタの、距離を持つ対がないモデル。
SINGLETON_MODEL_OBSERVATION_FIELDS = dict(
    round_index=9,
    model_id=0,
    nearest_model_id=None,
    nearest_loss_increase_distance=None,
    representative_model_id=0,
    cluster_model_count=1,
    maximum_within_cluster_distance=None,
    evaluated_within_cluster_pair_count=0,
    possible_within_cluster_pair_count=0,
    merged_with_other_models=False,
    absorbed_into_representative=False,
)


def make_pair_observation(**replaced_fields):
    return ModelPairClusteringObservation(**VALID_PAIR_OBSERVATION_FIELDS | replaced_fields)


def make_model_observation(**replaced_fields):
    return ModelClusteringObservation(**VALID_MODEL_OBSERVATION_FIELDS | replaced_fields)


def test_record_store_keeps_observations_in_appended_order():
    record_store = ModelClusteringRecordStore()
    assert record_store.snapshot_pair_clustering_observations() == ()
    assert record_store.snapshot_model_clustering_observations() == ()
    first_pair_observations = (
        make_pair_observation(),
        # 診断値がない・無限大の対、負の距離。
        make_pair_observation(
            higher_model_id=2,
            loss_increase_distance=-0.01,
            decision_score=0.874,
            assigned_to_same_cluster=False,
            true_concepts_match=None,
            concept_specific_parameter_distance=math.inf,
        ),
    )
    first_model_observations = (
        ModelClusteringObservation(**SINGLETON_MODEL_OBSERVATION_FIELDS),
        make_model_observation(),
    )
    record_store.append_clustering_observations(
        pair_observations=first_pair_observations, model_observations=first_model_observations
    )
    # 対の観測がないクラスタリング（どの対も距離を持たない）も足せる。
    second_model_observations = (
        ModelClusteringObservation(**SINGLETON_MODEL_OBSERVATION_FIELDS | dict(round_index=12)),
    )
    pair_snapshot = record_store.snapshot_pair_clustering_observations()
    record_store.append_clustering_observations(
        pair_observations=(), model_observations=second_model_observations
    )
    assert record_store.snapshot_pair_clustering_observations() == first_pair_observations
    assert record_store.snapshot_model_clustering_observations() == (
        *first_model_observations,
        *second_model_observations,
    )
    assert pair_snapshot == first_pair_observations


@pytest.mark.parametrize(
    "invalid_fields,expected_exception",
    [
        (dict(round_index=True), TypeError),
        (dict(round_index=-1), ValueError),
        (dict(lower_model_id=1.0), TypeError),
        (dict(lower_model_id=-101), ValueError),
        (dict(higher_model_id=1), ValueError),
        (dict(higher_model_id=0), ValueError),
        (dict(loss_increase_distance=0), TypeError),
        (dict(loss_increase_distance=math.nan), ValueError),
        (dict(decision_score=None), TypeError),
        (dict(decision_score=math.nan), ValueError),
        (dict(assigned_to_same_cluster=1), TypeError),
        (dict(true_concepts_match=0), TypeError),
        (dict(concept_specific_parameter_distance="0.25"), TypeError),
        (dict(concept_specific_parameter_distance=math.nan), ValueError),
    ],
)
def test_pair_observation_rejects_invalid_fields(invalid_fields, expected_exception):
    with pytest.raises(expected_exception):
        make_pair_observation(**invalid_fields)


@pytest.mark.parametrize(
    "invalid_fields,expected_exception",
    [
        (dict(round_index=9.0), TypeError),
        (dict(model_id=-1), ValueError),
        (dict(nearest_model_id=None), ValueError),
        (dict(nearest_loss_increase_distance=None), ValueError),
        (dict(nearest_model_id=3), ValueError),
        (dict(nearest_model_id=True), TypeError),
        (dict(nearest_loss_increase_distance=math.nan), ValueError),
        (dict(representative_model_id=4), ValueError),
        (dict(cluster_model_count=0), ValueError),
        (dict(cluster_model_count=2.0), TypeError),
        (dict(maximum_within_cluster_distance=None), ValueError),
        (dict(maximum_within_cluster_distance=1), TypeError),
        (dict(evaluated_within_cluster_pair_count=2), ValueError),
        (dict(possible_within_cluster_pair_count=3), ValueError),
        (dict(merged_with_other_models=False), ValueError),
        (dict(merged_with_other_models=1), TypeError),
        (dict(absorbed_into_representative=False), ValueError),
        # 距離を持つ対がないのに、クラスタ内の最大の距離がある。
        (dict(evaluated_within_cluster_pair_count=0), ValueError),
    ],
)
def test_model_observation_rejects_invalid_fields(invalid_fields, expected_exception):
    with pytest.raises(expected_exception):
        make_model_observation(**invalid_fields)


def test_record_store_rejects_invalid_observations_without_change():
    record_store = ModelClusteringRecordStore()
    pair_observation = make_pair_observation()
    model_observation = make_model_observation()
    record_store.append_clustering_observations(
        pair_observations=(pair_observation,), model_observations=(model_observation,)
    )
    mutated_model_observation = replace(model_observation)
    object.__setattr__(mutated_model_observation, "cluster_model_count", 3)
    for invalid_arguments in (
        dict(pair_observations=[pair_observation]),
        dict(pair_observations=(pair_observation, None)),
        dict(pair_observations=(make_subclass_copy(pair_observation),)),
        dict(pair_observations=(model_observation,)),
        dict(model_observations=[model_observation]),
        dict(model_observations=(pair_observation,)),
        dict(model_observations=(make_subclass_copy(model_observation),)),
        # 対の観測は正しいが、モデルの観測が不正（frozenを回避して書き換えた記録）。
        dict(model_observations=(model_observation, mutated_model_observation)),
    ):
        with pytest.raises((TypeError, ValueError)):
            record_store.append_clustering_observations(
                **dict(
                    pair_observations=(pair_observation,), model_observations=(model_observation,)
                )
                | invalid_arguments
            )
        assert record_store.snapshot_pair_clustering_observations() == (pair_observation,)
        assert record_store.snapshot_model_clustering_observations() == (model_observation,)
