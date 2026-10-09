"""標本1件の予測を、実旧の最終構成のclientの予測（_record_prediction）と、標本ごとに照合する。"""

import random
from collections import defaultdict
from dataclasses import replace
from functools import reduce
from operator import add

import pytest
import torch
from test_alarm_occurrence_handling import build_alarm_occurrence_oracle
from test_fixed_share_prediction_weights import (
    assert_fixed_share_state_matches_reference,
    capture_fixed_share_controller_state,
)
from test_held_adahedge_diagnostic_notification import (
    assert_diagnostic_collection_matches_legacy,
    get_diagnostic_collection_snapshot,
)
from test_held_candidate_validation_progress import make_subclass_copy
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

import federated_learning_experiments.runtime.observed_sample_prediction as sample_prediction_module
from federated_drift_experiment import config
from federated_drift_experiment.clients.shared_backbone import (
    ResidualAdapterRestartingSoftRoutingFedSDAClient,
)
from federated_drift_experiment.diagnostics.routing import RoutingLeaveOneOutDiagnostics
from federated_drift_experiment.expert_routing import (
    AdaHedgeRouter,
    SoftRoutingActivationController,
    SwitchingExpertRouter,
)
from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence import (
    AdaHedgeDiagnosticEvidence,
)
from federated_learning_experiments.evaluation.sample_prediction_record_store import (
    SamplePredictionRecord,
    SamplePredictionRecordStore,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
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
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights import (
    FixedSharePredictionWeightController,
)
from federated_learning_experiments.runtime.observed_sample_prediction import (
    ObservedSamplePrediction,
    predict_observed_sample_and_update_prediction_weights,
)
from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import (
    notify_diagnostics_of_training_assignment_change,
)

FIXED_SHARE_TIME_SCALE_SAMPLE_COUNT = 6
LEGACY_PREDICTION_HISTORY_NAMES = (
    "history_accuracy",
    "history_concept",
    "history_model_id",
    "history_routing_soft_active",
    "history_routing_gate_open",
    "history_routing_effective_experts",
    "history_routing_max_weight",
    "history_routing_oracle_correct",
    "history_routing_leader_correct",
    "history_routing_oracle_concept_correct",
    "history_routing_switching_correct",
    "history_routing_switching_leader_id",
    "history_routing_switching_effective_experts",
)


def enable_real_legacy_final_configuration_prediction(
    *, legacy_client, monkeypatch, fixed_share_time_scale_sample_count
):
    """実旧clientを最終構成のクラスにし、実旧の予測（_record_prediction）が読む属性と設定を与える。

    上流のoracleは、現行モデルだけで予測するクラスのclientを作り、予測を空の関数へ差し替えている。
    クラスの切替えで加わるのは、予測、警報側の2つの通知（常時有効では状態を変えない）、切替の通知
    （上流のoracleが同じ実メソッドを呼んでいる）だけ。
    """
    monkeypatch.setattr(config, "SOFT_ROUTING_CONTEXT", "switching")
    monkeypatch.setattr(config, "SOFT_ROUTING_ACTIVATION_POLICY", "always")
    monkeypatch.setattr(config, "ROUTING_ARCHIVE_SHADOW_DIAGNOSTICS", False)
    legacy_client.__class__ = ResidualAdapterRestartingSoftRoutingFedSDAClient
    legacy_client.__dict__.pop("_record_prediction", None)
    legacy_client.soft_routing_activation = SoftRoutingActivationController("always")
    legacy_client._soft_routing_activation_replay_samples = ()
    legacy_client._soft_routing_activation_replay_model_ids = ()
    legacy_client.routing_active_set = None
    legacy_client.switching_expert_router = SwitchingExpertRouter(
        fixed_share_time_scale_sample_count
    )
    legacy_client.oracle_concept_expert_routers = defaultdict(AdaHedgeRouter)
    legacy_client.routing_leave_one_out_diagnostics = RoutingLeaveOneOutDiagnostics()
    for history_name in LEGACY_PREDICTION_HISTORY_NAMES:
        setattr(legacy_client, history_name, [])
    legacy_client.routing_diagnostics = defaultdict(int)
    legacy_client.routing_oracle_concept_diagnostics = defaultdict(int)
    legacy_client.routing_switching_diagnostics = defaultdict(float)
    legacy_client.routing_class_diagnostics = defaultdict(lambda: defaultdict(int))


def count_records(sample_prediction_records, record_condition):
    return sum(
        1
        for sample_prediction_record in sample_prediction_records
        if record_condition(sample_prediction_record)
    )


def derive_legacy_overall_counts(sample_prediction_records):
    """記録から、旧の集計の計数（全体、または1つの観測クラス）を導く。"""
    return {
        "sample_count": len(sample_prediction_records),
        "oracle_correct_count": count_records(
            sample_prediction_records,
            lambda record: record.any_model_or_combined_prediction_is_correct,
        ),
        "mixture_correct_count": count_records(
            sample_prediction_records, lambda record: record.combined_prediction_is_correct
        ),
        "leader_correct_count": count_records(
            sample_prediction_records,
            lambda record: record.maximum_weight_model_prediction_is_correct,
        ),
        "confidence_leader_correct_count": count_records(
            sample_prediction_records,
            lambda record: record.highest_confidence_model_prediction_is_correct,
        ),
        "missed_oracle_count": count_records(
            sample_prediction_records,
            lambda record: (
                record.any_model_or_combined_prediction_is_correct
                and not record.combined_prediction_is_correct
            ),
        ),
        "confidence_leader_missed_oracle_count": count_records(
            sample_prediction_records,
            lambda record: (
                record.any_model_or_combined_prediction_is_correct
                and not record.highest_confidence_model_prediction_is_correct
            ),
        ),
    }


def assert_prediction_state_matches_legacy(
    *,
    fixed_share_prediction_weight_controller,
    diagnostic_evidence_collection,
    sample_prediction_record_store,
    legacy_client,
):
    """Fixed-Shareの重み、診断証拠、記録を、実旧clientの対応する属性（標本ごとの列と集計の計数）と照合する。"""
    assert_fixed_share_state_matches_reference(
        controller=fixed_share_prediction_weight_controller,
        reference_router=legacy_client.switching_expert_router,
    )
    assert_diagnostic_collection_matches_legacy(diagnostic_evidence_collection, legacy_client)
    records = sample_prediction_record_store.snapshot_sample_prediction_records()
    assert all(type(record) is SamplePredictionRecord for record in records)
    combined_correct = [record.combined_prediction_is_correct for record in records]
    effective_model_counts = [record.effective_model_count for record in records]
    maximum_weight_model_ids = [record.maximum_weight_model_id for record in records]
    # 標本ごとの列。最終構成では、旧の「実際の予測」と「Fixed-Shareの予測」は同じ値。
    assert legacy_client.history_accuracy == [float(is_correct) for is_correct in combined_correct]
    assert legacy_client.history_routing_switching_correct == [
        int(is_correct) for is_correct in combined_correct
    ]
    assert legacy_client.history_concept == [record.observed_concept_id for record in records]
    assert legacy_client.history_model_id == maximum_weight_model_ids
    assert legacy_client.history_routing_switching_leader_id == maximum_weight_model_ids
    assert legacy_client.history_routing_max_weight == [
        record.maximum_prediction_weight for record in records
    ]
    assert legacy_client.history_routing_effective_experts == effective_model_counts
    assert legacy_client.history_routing_switching_effective_experts == effective_model_counts
    assert legacy_client.history_routing_oracle_correct == [
        int(record.any_model_or_combined_prediction_is_correct) for record in records
    ]
    assert legacy_client.history_routing_leader_correct == [
        int(record.maximum_weight_model_prediction_is_correct) for record in records
    ]
    assert legacy_client.history_routing_oracle_concept_correct == [
        int(record.true_concept_diagnostic_prediction_is_correct) for record in records
    ]
    # 常時有効: 全標本が混合予測で、累積損失による採用の列は増えない。
    assert legacy_client.history_routing_soft_active == [True] * len(records)
    assert legacy_client.soft_routing_activation.soft_sample_count == len(records)
    assert legacy_client.soft_routing_activation.hard_sample_count == 0
    assert legacy_client.history_routing_gate_open == []
    if not records:
        return
    # 集計の計数。
    assert dict(legacy_client.routing_diagnostics) == derive_legacy_overall_counts(records)
    assert {
        class_id: dict(class_counts)
        for class_id, class_counts in legacy_client.routing_class_diagnostics.items()
    } == {
        class_id: derive_legacy_overall_counts(
            [record for record in records if record.observed_class_id == class_id]
        )
        for class_id in dict.fromkeys(record.observed_class_id for record in records)
    }
    combined_correct_count = sum(combined_correct)
    assert dict(legacy_client.routing_switching_diagnostics) == {
        "sample_count": len(records),
        "correct_count": combined_correct_count,
        "actual_correct_count": combined_correct_count,
        "global_correct_count": count_records(
            records, lambda record: record.global_diagnostic_prediction_is_correct
        ),
        # 旧は標本ごとに足す。組み込みのsumは浮動小数を補正つきで足すので、逐次加算で導く。
        "effective_experts_sum": reduce(add, effective_model_counts, 0.0),
    }
    assert dict(legacy_client.routing_oracle_concept_diagnostics) == {
        "sample_count": len(records),
        "correct_count": count_records(
            records, lambda record: record.true_concept_diagnostic_prediction_is_correct
        ),
    }


def build_sample_prediction_oracle(*, monkeypatch, valid_run_settings_mapping, class_count):
    """警報1回ぶんの処理のoracleの状態（保有モデル2つ、共有部は1つ）へ、予測のownerを足す。

    実旧clientを最終構成のクラスにして、実旧の予測を実行できるようにする。警報は処理しない。
    """
    # モデルの初期値を固定する（組立てのたびに同じモデルになり、呼出し側の乱数を進めない）。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        oracle_parts = build_alarm_occurrence_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=class_count,
        )
    handling_arguments, legacy_client = oracle_parts[0], oracle_parts[-1]
    enable_real_legacy_final_configuration_prediction(
        legacy_client=legacy_client,
        monkeypatch=monkeypatch,
        fixed_share_time_scale_sample_count=FIXED_SHARE_TIME_SCALE_SAMPLE_COUNT,
    )
    prediction_arguments = dict(
        fixed_share_prediction_weight_controller=FixedSharePredictionWeightController(
            prediction_combination_settings=replace(
                valid_run_settings_mapping["prediction_combination_settings"],
                fixed_share_weight_redistribution_time_scale_samples=FIXED_SHARE_TIME_SCALE_SAMPLE_COUNT,
            )
        ),
        diagnostic_evidence_collection=handling_arguments["diagnostic_evidence_collection"],
        sample_prediction_record_store=SamplePredictionRecordStore(),
        held_model_training_state_registry=handling_arguments["held_model_training_state_registry"],
        current_training_model_assignment=handling_arguments["current_training_model_assignment"],
    )
    return prediction_arguments, legacy_client


def make_prediction_observation(*, sample_index, class_count, observed_concept_id=0):
    """決まった規則の標本。特徴とラベルを位置ごとに変えて、モデルごとの正否と損失を変える。"""
    return IndexedObservedTrainingSample(
        sample_index=sample_index,
        training_sample=ObservedTrainingSample(
            input_features=torch.tensor(
                [[((sample_index * 5) % 11) / 5.0 - 1.0, ((sample_index * 3) % 7) / 3.0 - 1.0]]
            ),
            observed_class_labels=torch.tensor([[float((sample_index // 3) % class_count)]]),
        ),
        observed_concept_id=observed_concept_id,
    )


def run_legacy_prediction(*, legacy_client, indexed_observation):
    legacy_client.processed_samples = indexed_observation.sample_index + 1
    legacy_client._record_prediction(
        indexed_observation.training_sample.input_features,
        indexed_observation.training_sample.observed_class_labels,
        indexed_observation.observed_concept_id,
    )


def reassign_model_id_in_both(
    *, prediction_arguments, legacy_client, original_model_id, reassigned_model_id
):
    """モデル集合を変える: 現在の学習帰属でないモデルのIDを、両実装で付け替える。"""
    assert original_model_id != legacy_client.current_model_id
    prediction_arguments[
        "held_model_training_state_registry"
    ].reassign_held_model_training_state_id(
        original_model_id=original_model_id, reassigned_model_id=reassigned_model_id
    )
    legacy_client.models[reassigned_model_id] = legacy_client.models.pop(original_model_id)


def change_training_assignment_in_both(*, prediction_arguments, legacy_client, model_id):
    """学習帰属を変える: 新は帰属の変更と診断への通知、実旧は現行モデルの変更（切替の通知を含む）。"""
    notify_diagnostics_of_training_assignment_change(
        assignment_change=prediction_arguments[
            "current_training_model_assignment"
        ].assign_model_for_training(model_id=model_id),
        diagnostic_evidence_collection=prediction_arguments["diagnostic_evidence_collection"],
    )
    legacy_client._set_local_current_model(model_id)


@pytest.mark.parametrize("class_count", (2, 4))
def test_prediction_matches_real_legacy_prediction_for_each_sample(
    class_count, monkeypatch, valid_run_settings_mapping
):
    prediction_arguments, legacy_client = build_sample_prediction_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=class_count,
    )
    first_model_id, second_model_id = sorted(legacy_client.models)
    assert legacy_client.current_model_id == second_model_id
    observed_maximum_weight_model_ids = set()
    observed_correctness = set()
    for sample_index in range(48):
        if sample_index == 12:
            # 一時IDへ付け替える（モデル集合が変わり、重みと証拠が初期化される）。
            reassign_model_id_in_both(
                prediction_arguments=prediction_arguments,
                legacy_client=legacy_client,
                original_model_id=first_model_id,
                reassigned_model_id=-7,
            )
        if sample_index == 24:
            # 学習帰属の変更: globalの証拠だけが再始動し、Fixed-Shareの重みは保たれる。
            fixed_share_state = capture_fixed_share_controller_state(
                controller=prediction_arguments["fixed_share_prediction_weight_controller"]
            )
            change_training_assignment_in_both(
                prediction_arguments=prediction_arguments,
                legacy_client=legacy_client,
                model_id=-7,
            )
            assert (
                capture_fixed_share_controller_state(
                    controller=prediction_arguments["fixed_share_prediction_weight_controller"]
                )
                == fixed_share_state
            )
        if sample_index == 36:
            reassign_model_id_in_both(
                prediction_arguments=prediction_arguments,
                legacy_client=legacy_client,
                original_model_id=second_model_id,
                reassigned_model_id=second_model_id + 20,
            )
        indexed_observation = make_prediction_observation(
            sample_index=sample_index,
            class_count=class_count,
            observed_concept_id=(sample_index // 5) % 3,
        )
        run_legacy_prediction(legacy_client=legacy_client, indexed_observation=indexed_observation)
        sample_prediction = predict_observed_sample_and_update_prediction_weights(
            indexed_observation=indexed_observation, **prediction_arguments
        )
        assert type(sample_prediction) is ObservedSamplePrediction
        assert_prediction_state_matches_legacy(
            legacy_client=legacy_client,
            **{
                argument_name: prediction_arguments[argument_name]
                for argument_name in (
                    "fixed_share_prediction_weight_controller",
                    "diagnostic_evidence_collection",
                    "sample_prediction_record_store",
                )
            },
        )
        # 結果: 記録は足したものと同じ。予測クラスの正否は記録と一致し、重みは総和1でID昇順。
        sample_prediction_record = sample_prediction.sample_prediction_record
        assert (
            prediction_arguments[
                "sample_prediction_record_store"
            ].snapshot_sample_prediction_records()[-1]
            is sample_prediction_record
        )
        assert sample_prediction.predicted_class_labels.shape == (1, 1)
        assert (
            sample_prediction.predicted_class_labels.item()
            == indexed_observation.training_sample.observed_class_labels.item()
        ) == sample_prediction_record.combined_prediction_is_correct
        held_model_ids = sorted(legacy_client.models)
        for mapping_by_model_id in (
            sample_prediction.prediction_probabilities_by_model_id,
            sample_prediction.prediction_weights_by_model_id,
            sample_prediction.global_diagnostic_weights_by_model_id,
            sample_prediction.observed_losses_by_model_id,
        ):
            assert list(mapping_by_model_id) == held_model_ids
        assert sum(sample_prediction.prediction_weights_by_model_id.values()) == pytest.approx(
            1.0, abs=1e-12
        )
        assert sample_prediction.prediction_weights_by_model_id[
            sample_prediction_record.maximum_weight_model_id
        ] == max(sample_prediction.prediction_weights_by_model_id.values())
        observed_maximum_weight_model_ids.add(sample_prediction_record.maximum_weight_model_id)
        observed_correctness.add(
            (
                sample_prediction_record.combined_prediction_is_correct,
                sample_prediction_record.any_model_or_combined_prediction_is_correct,
                sample_prediction_record.maximum_weight_model_prediction_is_correct,
            )
        )
    # この標本列が、最大重みのモデルの交代と、正否の組合せの違いを通っていること。
    assert len(observed_maximum_weight_model_ids) >= 3
    assert len(observed_correctness) >= 3
    controller = prediction_arguments["fixed_share_prediction_weight_controller"]
    assert controller.model_pool_reset_count == 2
    assert (
        prediction_arguments[
            "diagnostic_evidence_collection"
        ].global_diagnostic_evidence.concept_operation_restart_count
        >= 1
    )


@pytest.mark.parametrize("class_count", (2, 4))
def test_prediction_with_single_held_model_matches_real_legacy_prediction(
    class_count, monkeypatch, valid_run_settings_mapping
):
    prediction_arguments, legacy_client = build_sample_prediction_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=class_count,
    )
    current_model_id = legacy_client.current_model_id
    current_held_state = prediction_arguments[
        "held_model_training_state_registry"
    ].get_held_model_training_state(model_id=current_model_id)
    single_model_registry = HeldModelTrainingStateRegistry()
    single_model_registry.register_held_model_training_state(
        model_id=current_model_id,
        classifier=current_held_state.classifier,
        concept_specific_parameter_optimizer_state=current_held_state.concept_specific_parameter_optimizer_state,
    )
    prediction_arguments["held_model_training_state_registry"] = single_model_registry
    legacy_client.models = {current_model_id: legacy_client.models[current_model_id]}
    for sample_index in range(8):
        indexed_observation = make_prediction_observation(
            sample_index=sample_index, class_count=class_count
        )
        run_legacy_prediction(legacy_client=legacy_client, indexed_observation=indexed_observation)
        sample_prediction = predict_observed_sample_and_update_prediction_weights(
            indexed_observation=indexed_observation, **prediction_arguments
        )
        assert sample_prediction.prediction_weights_by_model_id == {current_model_id: 1.0}
        assert sample_prediction.sample_prediction_record.effective_model_count == 1.0
        assert_prediction_state_matches_legacy(
            legacy_client=legacy_client,
            fixed_share_prediction_weight_controller=prediction_arguments[
                "fixed_share_prediction_weight_controller"
            ],
            diagnostic_evidence_collection=prediction_arguments["diagnostic_evidence_collection"],
            sample_prediction_record_store=prediction_arguments["sample_prediction_record_store"],
        )


def test_first_prediction_prefers_current_training_model_among_equal_weights(
    monkeypatch, valid_run_settings_mapping
):
    """最初の標本では重みが同率。現在の学習帰属のモデルを選び、帰属を変えれば選ぶモデルも変わる。"""
    for preferred_position in (0, 1):
        prediction_arguments, legacy_client = build_sample_prediction_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=2,
        )
        preferred_model_id = sorted(legacy_client.models)[preferred_position]
        change_training_assignment_in_both(
            prediction_arguments=prediction_arguments,
            legacy_client=legacy_client,
            model_id=preferred_model_id,
        )
        indexed_observation = make_prediction_observation(sample_index=0, class_count=2)
        run_legacy_prediction(legacy_client=legacy_client, indexed_observation=indexed_observation)
        sample_prediction = predict_observed_sample_and_update_prediction_weights(
            indexed_observation=indexed_observation, **prediction_arguments
        )
        assert sample_prediction.prediction_weights_by_model_id == dict.fromkeys(
            sorted(legacy_client.models), 0.5
        )
        assert sample_prediction.sample_prediction_record.maximum_weight_model_id == (
            preferred_model_id
        )
        assert legacy_client.history_model_id == [preferred_model_id]


def snapshot_prediction_state(*, prediction_arguments):
    """予測が触れうる全状態: Fixed-Shareの重み、診断証拠、記録、帰属、全モデルのパラメータと訓練の別。"""
    held_states = prediction_arguments[
        "held_model_training_state_registry"
    ].snapshot_ordered_held_model_training_states()
    return dict(
        fixed_share_state=capture_fixed_share_controller_state(
            controller=prediction_arguments["fixed_share_prediction_weight_controller"]
        ),
        diagnostic_state=get_diagnostic_collection_snapshot(
            prediction_arguments["diagnostic_evidence_collection"]
        ),
        records=prediction_arguments[
            "sample_prediction_record_store"
        ].snapshot_sample_prediction_records(),
        current_training_model_id=prediction_arguments[
            "current_training_model_assignment"
        ].current_training_model_id,
        held_model_ids=tuple(held_state.model_id for held_state in held_states),
        training_modes=tuple(held_state.classifier.training for held_state in held_states),
        parameters=tuple(
            {
                parameter_name: parameter.detach().clone()
                for parameter_name, parameter in held_state.classifier.named_parameters()
            }
            for held_state in held_states
        ),
        torch_random_state=torch.get_rng_state().clone(),
        python_random_state=random.getstate(),
    )


def assert_prediction_state_unchanged(*, state_snapshot, prediction_arguments):
    current_snapshot = snapshot_prediction_state(prediction_arguments=prediction_arguments)
    for state_name in state_snapshot:
        if state_name == "parameters":
            for parameters, current_parameters in zip(
                state_snapshot["parameters"], current_snapshot["parameters"], strict=True
            ):
                assert parameters.keys() == current_parameters.keys()
                for parameter_name, parameter in parameters.items():
                    assert torch.equal(parameter, current_parameters[parameter_name])
        elif state_name == "torch_random_state":
            assert torch.equal(state_snapshot[state_name], current_snapshot[state_name])
        else:
            assert state_snapshot[state_name] == current_snapshot[state_name], state_name


def build_predicted_sample_prediction_oracle(*, monkeypatch, valid_run_settings_mapping):
    """標本を4件予測した後の状態（重みと証拠が均等でなく、記録の最後の位置が3）。"""
    prediction_arguments, _ = build_sample_prediction_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    for sample_index in range(4):
        predict_observed_sample_and_update_prediction_weights(
            indexed_observation=make_prediction_observation(
                sample_index=sample_index, class_count=2
            ),
            **prediction_arguments,
        )
    return prediction_arguments


def replace_training_sample(indexed_observation, **replaced_fields):
    return replace(
        indexed_observation,
        training_sample=replace(indexed_observation.training_sample, **replaced_fields),
    )


class ObservationSubclass(IndexedObservedTrainingSample):
    pass


class TrainingSampleSubclass(ObservedTrainingSample):
    pass


class TensorSubclass(torch.Tensor):
    pass


INVALID_OBSERVATION_CASES = {
    "observation_not_record": (lambda observation: vars(observation), TypeError),
    "observation_subclass": (
        lambda observation: ObservationSubclass(**vars(observation)),
        TypeError,
    ),
    "sample_index_bool": (lambda observation: replace(observation, sample_index=True), TypeError),
    "sample_index_float": (lambda observation: replace(observation, sample_index=4.0), TypeError),
    "sample_index_negative": (
        lambda observation: replace(observation, sample_index=-1),
        ValueError,
    ),
    # 記録の最後の位置（3）の次でない位置。
    "sample_index_repeated": (lambda observation: replace(observation, sample_index=3), ValueError),
    "sample_index_skipped": (lambda observation: replace(observation, sample_index=5), ValueError),
    "concept_id_bool": (
        lambda observation: replace(observation, observed_concept_id=True),
        TypeError,
    ),
    "concept_id_float": (
        lambda observation: replace(observation, observed_concept_id=1.0),
        TypeError,
    ),
    "training_sample_subclass": (
        lambda observation: replace(
            observation, training_sample=TrainingSampleSubclass(**vars(observation.training_sample))
        ),
        TypeError,
    ),
    "features_not_tensor": (
        lambda observation: replace_training_sample(observation, input_features=[[0.1, 0.2]]),
        TypeError,
    ),
    "features_tensor_subclass": (
        lambda observation: replace_training_sample(
            observation,
            input_features=observation.training_sample.input_features.as_subclass(TensorSubclass),
        ),
        TypeError,
    ),
    "labels_not_tensor": (
        lambda observation: replace_training_sample(observation, observed_class_labels=[[1.0]]),
        TypeError,
    ),
    "features_one_dimensional": (
        lambda observation: replace_training_sample(
            observation, input_features=torch.tensor([0.1, 0.2])
        ),
        ValueError,
    ),
    "features_two_samples": (
        lambda observation: replace_training_sample(
            observation, input_features=torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        ),
        ValueError,
    ),
    "labels_one_dimensional": (
        lambda observation: replace_training_sample(
            observation, observed_class_labels=torch.tensor([1.0])
        ),
        ValueError,
    ),
    # 標本の中身。
    "features_wrong_count": (
        lambda observation: replace_training_sample(
            observation, input_features=torch.tensor([[0.1, 0.2, 0.3]])
        ),
        ValueError,
    ),
    "features_float64": (
        lambda observation: replace_training_sample(
            observation, input_features=torch.tensor([[0.1, 0.2]], dtype=torch.float64)
        ),
        ValueError,
    ),
    "features_nan": (
        lambda observation: replace_training_sample(
            observation, input_features=torch.tensor([[0.1, float("nan")]])
        ),
        ValueError,
    ),
    # 負の無限大の特徴は、ReLUの後で有限の出力になりうるので、特徴そのものの有限性で拒否する。
    "features_negative_infinity": (
        lambda observation: replace_training_sample(
            observation, input_features=torch.tensor([[float("-inf"), 0.2]])
        ),
        ValueError,
    ),
    "labels_int64": (
        lambda observation: replace_training_sample(
            observation, observed_class_labels=torch.tensor([[1]])
        ),
        ValueError,
    ),
    "labels_out_of_range": (
        lambda observation: replace_training_sample(
            observation, observed_class_labels=torch.tensor([[2.0]])
        ),
        ValueError,
    ),
    "labels_negative": (
        lambda observation: replace_training_sample(
            observation, observed_class_labels=torch.tensor([[-1.0]])
        ),
        ValueError,
    ),
    "labels_fractional": (
        lambda observation: replace_training_sample(
            observation, observed_class_labels=torch.tensor([[0.5]])
        ),
        ValueError,
    ),
    "labels_nan": (
        lambda observation: replace_training_sample(
            observation, observed_class_labels=torch.tensor([[float("nan")]])
        ),
        ValueError,
    ),
}


@pytest.mark.parametrize("invalid_case_name", INVALID_OBSERVATION_CASES)
def test_prediction_rejects_invalid_observation_before_any_update(
    invalid_case_name, monkeypatch, valid_run_settings_mapping
):
    make_invalid_observation, expected_exception = INVALID_OBSERVATION_CASES[invalid_case_name]
    prediction_arguments = build_predicted_sample_prediction_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    valid_observation = make_prediction_observation(sample_index=4, class_count=2)
    state_snapshot = snapshot_prediction_state(prediction_arguments=prediction_arguments)
    with pytest.raises(expected_exception):
        predict_observed_sample_and_update_prediction_weights(
            indexed_observation=make_invalid_observation(valid_observation), **prediction_arguments
        )
    assert_prediction_state_unchanged(
        state_snapshot=state_snapshot, prediction_arguments=prediction_arguments
    )
    # 拒否の後に、正しい標本を予測できる。
    predict_observed_sample_and_update_prediction_weights(
        indexed_observation=valid_observation, **prediction_arguments
    )
    assert prediction_arguments["sample_prediction_record_store"].last_recorded_sample_index == 4


@pytest.mark.parametrize(
    "owner_argument_name",
    [
        "fixed_share_prediction_weight_controller",
        "diagnostic_evidence_collection",
        "sample_prediction_record_store",
        "held_model_training_state_registry",
        "current_training_model_assignment",
    ],
)
@pytest.mark.parametrize("invalid_owner_kind", ["subclass", "other_type"])
def test_prediction_rejects_invalid_owner_before_any_update(
    owner_argument_name, invalid_owner_kind, monkeypatch, valid_run_settings_mapping
):
    prediction_arguments = build_predicted_sample_prediction_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    state_snapshot = snapshot_prediction_state(prediction_arguments=prediction_arguments)
    invalid_owner = (
        make_subclass_copy(prediction_arguments[owner_argument_name])
        if invalid_owner_kind == "subclass"
        else object()
    )
    with pytest.raises(TypeError, match=owner_argument_name):
        predict_observed_sample_and_update_prediction_weights(
            indexed_observation=make_prediction_observation(sample_index=4, class_count=2),
            **prediction_arguments | {owner_argument_name: invalid_owner},
        )
    assert_prediction_state_unchanged(
        state_snapshot=state_snapshot, prediction_arguments=prediction_arguments
    )


def test_prediction_rejects_missing_held_models_before_any_update(
    monkeypatch, valid_run_settings_mapping
):
    prediction_arguments = build_predicted_sample_prediction_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    valid_observation = make_prediction_observation(sample_index=4, class_count=2)
    state_snapshot = snapshot_prediction_state(prediction_arguments=prediction_arguments)
    # 保有モデルが1つもない。
    with pytest.raises(ValueError, match="at least one held model"):
        predict_observed_sample_and_update_prediction_weights(
            indexed_observation=valid_observation,
            **prediction_arguments
            | dict(held_model_training_state_registry=HeldModelTrainingStateRegistry()),
        )
    # 現在の学習帰属のモデルを保有していない。
    with pytest.raises(ValueError, match="current training model"):
        predict_observed_sample_and_update_prediction_weights(
            indexed_observation=valid_observation,
            **prediction_arguments
            | dict(
                current_training_model_assignment=CurrentTrainingModelAssignment(
                    initial_model_id=12345
                )
            ),
        )
    assert_prediction_state_unchanged(
        state_snapshot=state_snapshot, prediction_arguments=prediction_arguments
    )


def test_prediction_rejects_non_finite_model_output_before_any_update(
    monkeypatch, valid_run_settings_mapping
):
    prediction_arguments = build_predicted_sample_prediction_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    state_snapshot = snapshot_prediction_state(prediction_arguments=prediction_arguments)
    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            ResidualAdapterClassifier,
            "forward_from_shared_features",
            lambda self, shared_features: torch.full((1, 1), float("nan")),
        )
        with pytest.raises(ValueError):
            predict_observed_sample_and_update_prediction_weights(
                indexed_observation=make_prediction_observation(sample_index=4, class_count=2),
                **prediction_arguments,
            )
    assert_prediction_state_unchanged(
        state_snapshot=state_snapshot, prediction_arguments=prediction_arguments
    )
    predict_observed_sample_and_update_prediction_weights(
        indexed_observation=make_prediction_observation(sample_index=4, class_count=2),
        **prediction_arguments,
    )


@pytest.mark.parametrize("class_count", (2, 4))
def test_prediction_does_not_depend_on_observed_label_or_concept_id(
    class_count, monkeypatch, valid_run_settings_mapping
):
    """同じ状態から、ラベルと概念IDだけが違う標本を予測すると、確率・重み・結合・予測クラスが同じになる。"""
    predictions = []
    for observed_class_id, observed_concept_id in ((0, 0), (1, 2), (class_count - 1, None)):
        prediction_arguments, _ = build_sample_prediction_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=class_count,
        )
        for sample_index in range(6):
            predict_observed_sample_and_update_prediction_weights(
                indexed_observation=make_prediction_observation(
                    sample_index=sample_index, class_count=class_count
                ),
                **prediction_arguments,
            )
        predictions.append(
            predict_observed_sample_and_update_prediction_weights(
                indexed_observation=replace_training_sample(
                    replace(
                        make_prediction_observation(sample_index=6, class_count=class_count),
                        observed_concept_id=observed_concept_id,
                    ),
                    observed_class_labels=torch.tensor([[float(observed_class_id)]]),
                ),
                **prediction_arguments,
            )
        )
    reference_prediction = predictions[0]
    for other_prediction in predictions[1:]:
        assert torch.equal(
            other_prediction.predicted_class_labels, reference_prediction.predicted_class_labels
        )
        assert torch.equal(
            other_prediction.combined_prediction_probabilities,
            reference_prediction.combined_prediction_probabilities,
        )
        assert (
            other_prediction.prediction_weights_by_model_id
            == reference_prediction.prediction_weights_by_model_id
        )
        assert (
            other_prediction.global_diagnostic_weights_by_model_id
            == reference_prediction.global_diagnostic_weights_by_model_id
        )
        for (
            model_id,
            probabilities,
        ) in reference_prediction.prediction_probabilities_by_model_id.items():
            assert torch.equal(
                other_prediction.prediction_probabilities_by_model_id[model_id], probabilities
            )
        assert (
            other_prediction.sample_prediction_record.maximum_weight_model_id
            == reference_prediction.sample_prediction_record.maximum_weight_model_id
        )
    # ラベルが違えば、損失は違う（ラベルを見た後の値だけが変わる）。
    assert (
        predictions[1].observed_losses_by_model_id
        != reference_prediction.observed_losses_by_model_id
    )


def test_prediction_without_concept_id_skips_only_true_concept_diagnostics(
    monkeypatch, valid_run_settings_mapping
):
    """概念IDのない標本では、概念別の証拠を作らず、記録の概念の2項目だけがNoneになる。"""
    results = {}
    for observed_concept_id in (1, None):
        prediction_arguments = build_predicted_sample_prediction_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )
        diagnostic_evidence_collection = prediction_arguments["diagnostic_evidence_collection"]
        created_true_concept_ids = diagnostic_evidence_collection.created_true_concept_ids
        assert 1 not in created_true_concept_ids
        sample_prediction = predict_observed_sample_and_update_prediction_weights(
            indexed_observation=make_prediction_observation(
                sample_index=4, class_count=2, observed_concept_id=observed_concept_id
            ),
            **prediction_arguments,
        )
        results[observed_concept_id] = (
            sample_prediction,
            snapshot_prediction_state(prediction_arguments=prediction_arguments),
            created_true_concept_ids,
            diagnostic_evidence_collection.created_true_concept_ids,
        )
    prediction_with_concept, state_with_concept, _, concept_ids_with_concept = results[1]
    prediction_without_concept, state_without_concept, concept_ids_before, concept_ids_after = (
        results[None]
    )
    assert concept_ids_with_concept == (*concept_ids_before, 1)
    assert concept_ids_after == concept_ids_before
    record_without_concept = prediction_without_concept.sample_prediction_record
    assert record_without_concept.observed_concept_id is None
    assert record_without_concept.true_concept_diagnostic_prediction_is_correct is None
    assert (
        type(
            prediction_with_concept.sample_prediction_record.true_concept_diagnostic_prediction_is_correct
        )
        is bool
    )
    assert record_without_concept == replace(
        prediction_with_concept.sample_prediction_record,
        observed_concept_id=None,
        true_concept_diagnostic_prediction_is_correct=None,
    )
    # Fixed-Shareの重みと、globalの証拠は、概念IDの有無で変わらない。
    assert state_without_concept["fixed_share_state"] == state_with_concept["fixed_share_state"]
    assert state_without_concept["diagnostic_state"][0] == state_with_concept["diagnostic_state"][0]


def test_prediction_appends_record_before_updating_weights_and_evidence(
    monkeypatch, valid_run_settings_mapping
):
    """記録の追加が失敗すると、重みと証拠は更新されない（同期だけが済む）。"""
    prediction_arguments = build_predicted_sample_prediction_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    state_snapshot = snapshot_prediction_state(prediction_arguments=prediction_arguments)

    def fail_to_append(self, *, sample_prediction_record):
        raise RuntimeError("append failed")

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            SamplePredictionRecordStore, "append_sample_prediction_record", fail_to_append
        )
        with pytest.raises(RuntimeError, match="append failed"):
            predict_observed_sample_and_update_prediction_weights(
                indexed_observation=make_prediction_observation(
                    sample_index=4, class_count=2, observed_concept_id=0
                ),
                **prediction_arguments,
            )
    # 概念0の証拠は既にあり、モデル集合も同じなので、同期は何も変えない。
    assert_prediction_state_unchanged(
        state_snapshot=state_snapshot, prediction_arguments=prediction_arguments
    )


def test_prediction_stops_after_failed_update_without_rollback(
    monkeypatch, valid_run_settings_mapping
):
    """globalの証拠の更新が失敗すると、記録は残り、後の更新（概念別、Fixed-Share）へ進まない。"""
    prediction_arguments = build_predicted_sample_prediction_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    state_snapshot = snapshot_prediction_state(prediction_arguments=prediction_arguments)

    def fail_to_update(self, *, observed_losses_by_model_id, diagnostic_weights_by_model_id):
        raise RuntimeError("update failed")

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            AdaHedgeDiagnosticEvidence, "update_evidence_after_loss_observation", fail_to_update
        )
        with pytest.raises(RuntimeError, match="update failed"):
            predict_observed_sample_and_update_prediction_weights(
                indexed_observation=make_prediction_observation(
                    sample_index=4, class_count=2, observed_concept_id=0
                ),
                **prediction_arguments,
            )
    current_snapshot = snapshot_prediction_state(prediction_arguments=prediction_arguments)
    assert len(current_snapshot["records"]) == len(state_snapshot["records"]) + 1
    assert current_snapshot["fixed_share_state"] == state_snapshot["fixed_share_state"]
    assert current_snapshot["diagnostic_state"] == state_snapshot["diagnostic_state"]


def test_prediction_runs_steps_in_design_order(monkeypatch, valid_run_settings_mapping):
    """全モデルの評価→重みの取得（global、Fixed-Share、概念別）→記録→更新（global、概念別、Fixed-Share）。"""
    prediction_arguments = build_predicted_sample_prediction_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    diagnostic_evidence_collection = prediction_arguments["diagnostic_evidence_collection"]
    global_diagnostic_evidence = diagnostic_evidence_collection.global_diagnostic_evidence
    step_names = []

    def record_step_then_call(owner_type, method_name, make_step_name):
        original_method = getattr(owner_type, method_name)

        def recording_method(self, *positional_arguments, **keyword_arguments):
            step_names.append(make_step_name(self))
            return original_method(self, *positional_arguments, **keyword_arguments)

        monkeypatch.setattr(owner_type, method_name, recording_method)

    def name_evidence(evidence):
        return "global" if evidence is global_diagnostic_evidence else "true_concept"

    record_step_then_call(SharedFeatureExtractor, "forward", lambda extractor: "shared_features")
    record_step_then_call(
        AdaHedgeDiagnosticEvidence,
        "get_diagnostic_weights_before_loss_observation",
        lambda evidence: f"get_{name_evidence(evidence)}_weights",
    )
    record_step_then_call(
        FixedSharePredictionWeightController,
        "get_prediction_weights_before_label_observation",
        lambda controller: "get_fixed_share_weights",
    )
    record_step_then_call(
        SamplePredictionRecordStore,
        "append_sample_prediction_record",
        lambda store: "append_record",
    )
    record_step_then_call(
        AdaHedgeDiagnosticEvidence,
        "update_evidence_after_loss_observation",
        lambda evidence: f"update_{name_evidence(evidence)}_evidence",
    )
    record_step_then_call(
        FixedSharePredictionWeightController,
        "update_weights_after_loss_observation",
        lambda controller: "update_fixed_share_weights",
    )
    predict_observed_sample_and_update_prediction_weights(
        indexed_observation=make_prediction_observation(
            sample_index=4, class_count=2, observed_concept_id=0
        ),
        **prediction_arguments,
    )
    # 共有特徴の計算は、保有モデルが2つでも1回だけ。
    assert step_names == [
        "shared_features",
        "get_global_weights",
        "get_fixed_share_weights",
        "get_true_concept_weights",
        "append_record",
        "update_global_evidence",
        "update_true_concept_evidence",
        "update_fixed_share_weights",
    ]


def test_prediction_updates_true_concept_evidence_with_unnormalized_weights(
    monkeypatch, valid_run_settings_mapping
):
    """更新へ渡す重み: globalとFixed-Shareは結合に使った正規化後、概念別は取得したままの値（旧と同じ）。"""
    prediction_arguments = build_predicted_sample_prediction_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    global_diagnostic_evidence = prediction_arguments[
        "diagnostic_evidence_collection"
    ].global_diagnostic_evidence
    # 正規化で値が変わる重み（総和が1との差1e-12以内で、1でない）を、両方の証拠の取得が返すようにする。
    unnormalized_weights = {4: 0.25 + 4e-13, 9: 0.75}
    normalized_weights = sample_prediction_module.normalize_model_prediction_weights(
        prediction_weights_by_model_id=unnormalized_weights
    )
    assert normalized_weights != unnormalized_weights
    updated_weights = {}
    original_update = AdaHedgeDiagnosticEvidence.update_evidence_after_loss_observation

    def record_update(self, *, observed_losses_by_model_id, diagnostic_weights_by_model_id):
        updated_weights[self is global_diagnostic_evidence] = dict(diagnostic_weights_by_model_id)
        return original_update(
            self,
            observed_losses_by_model_id=observed_losses_by_model_id,
            diagnostic_weights_by_model_id=diagnostic_weights_by_model_id,
        )

    monkeypatch.setattr(
        AdaHedgeDiagnosticEvidence,
        "get_diagnostic_weights_before_loss_observation",
        lambda self, *, model_ids: dict(unnormalized_weights),
    )
    monkeypatch.setattr(
        AdaHedgeDiagnosticEvidence, "update_evidence_after_loss_observation", record_update
    )
    sample_prediction = predict_observed_sample_and_update_prediction_weights(
        indexed_observation=make_prediction_observation(
            sample_index=4, class_count=2, observed_concept_id=0
        ),
        **prediction_arguments,
    )
    assert updated_weights == {True: normalized_weights, False: unnormalized_weights}
    assert sample_prediction.global_diagnostic_weights_by_model_id == normalized_weights


def test_prediction_does_not_change_models_or_consume_random_numbers(
    monkeypatch, valid_run_settings_mapping
):
    prediction_arguments = build_predicted_sample_prediction_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    state_snapshot = snapshot_prediction_state(prediction_arguments=prediction_arguments)
    sample_prediction = predict_observed_sample_and_update_prediction_weights(
        indexed_observation=make_prediction_observation(sample_index=4, class_count=2),
        **prediction_arguments,
    )
    current_snapshot = snapshot_prediction_state(prediction_arguments=prediction_arguments)
    for unchanged_state_name in (
        "current_training_model_id",
        "held_model_ids",
        "training_modes",
        "python_random_state",
    ):
        assert current_snapshot[unchanged_state_name] == state_snapshot[unchanged_state_name]
    assert torch.equal(current_snapshot["torch_random_state"], state_snapshot["torch_random_state"])
    for parameters, current_parameters in zip(
        state_snapshot["parameters"], current_snapshot["parameters"], strict=True
    ):
        for parameter_name, parameter in parameters.items():
            assert torch.equal(parameter, current_parameters[parameter_name])
    # 結果のtensorは、勾配の計算に結びつかない。
    assert not sample_prediction.combined_prediction_probabilities.requires_grad
    for probabilities in sample_prediction.prediction_probabilities_by_model_id.values():
        assert not probabilities.requires_grad
    # 重みと証拠は更新されている。
    assert current_snapshot["fixed_share_state"] != state_snapshot["fixed_share_state"]
    assert current_snapshot["diagnostic_state"] != state_snapshot["diagnostic_state"]
