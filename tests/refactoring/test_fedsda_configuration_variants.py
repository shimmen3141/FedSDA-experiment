"""提案構成以外の構成（ablationの構成）の全体runを、小さい条件で、実旧の、同じ指定の全体runと照合する。

照合の内容（指標の全部と、31の離散列）は、提案構成の照合（test_fedsda_run_metric_derivation.pyの
`compare_small_run_with_real_legacy`）と同じ。構成ごとに、その構成に固有の経路を通ったことも確かめる。
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_fedsda_run_metric_derivation import (
    PROPOSED_CONFIGURATION_VARIANT,
    SMALL_RUN_CONDITIONS,
    compare_small_run_with_real_legacy,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import (
    RunSettingsValidationError,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME,
    CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME,
    EXPERIMENT_CONFIGURATION_NAMES,
    LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME,
    OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME,
    PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME,
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
    SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME,
    build_experiment_run_settings,
)

# 指標の設定（許容する検出の遅れ、回復の窓）。
MAXIMUM_DELAY, RECOVERY_WINDOW = 100, 50


def replace_prediction_combination_settings(run_participant_settings, **changed_fields):
    """参加者の設定の束の、予測の結合の設定の一部を、置き換える。"""
    run_client_settings = run_participant_settings.run_client_settings
    return replace(
        run_participant_settings,
        run_client_settings=replace(
            run_client_settings,
            prediction_combination_settings=replace(
                run_client_settings.prediction_combination_settings, **changed_fields
            ),
        ),
    )


def replace_candidate_model_training_and_acceptance_settings(
    run_participant_settings, **changed_fields
):
    """参加者の設定の束の、候補モデルの学習と採否の設定の一部を、置き換える。"""
    run_client_settings = run_participant_settings.run_client_settings
    return replace(
        run_participant_settings,
        run_client_settings=replace(
            run_client_settings,
            candidate_model_training_and_acceptance_settings=replace(
                run_client_settings.candidate_model_training_and_acceptance_settings,
                **changed_fields,
            ),
        ),
    )


def replace_loss_monitoring_scope(
    run_participant_settings, *, loss_monitoring_scope, detector_name
):
    """参加者の設定の束の、損失の監視の対象と、それに対応する検出器の表示名を、置き換える。"""
    run_client_settings = run_participant_settings.run_client_settings
    return replace(
        run_participant_settings,
        run_client_settings=replace(
            run_client_settings,
            loss_change_detection_settings=replace(
                run_client_settings.loss_change_detection_settings,
                loss_monitoring_scope=loss_monitoring_scope,
            ),
            scalar_settings=replace(
                run_client_settings.scalar_settings, detector_name=detector_name
            ),
        ),
    )


# 構成: 名前（掃引のコマンドでの名前）、旧の構成名、旧の設定の上書き、新の参加者の設定の変更。
CONFIGURATION_VARIANTS = {
    PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME: SimpleNamespace(
        name=PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME,
        legacy_mode_name=PROPOSED_CONFIGURATION_VARIANT.legacy_mode_name,
        # 旧の`tools/run_main_ablation_suite.sh`の`no-recalibration`。
        legacy_setting_overrides={"SHARED_BACKBONE_ROUTING_RECALIBRATION": "none"},
        modify_run_participant_settings=lambda run_participant_settings: (
            replace_prediction_combination_settings(
                run_participant_settings,
                prediction_weight_recalibration_after_aggregation_policy=(
                    "keep_weights_without_recalibration"
                ),
            )
        ),
    ),
    CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME: SimpleNamespace(
        name=CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME,
        legacy_mode_name=PROPOSED_CONFIGURATION_VARIANT.legacy_mode_name,
        # 旧の`tools/run_main_ablation_suite.sh`の`immediate-creation`。
        legacy_setting_overrides={"NEW_MODEL_CREATION_POLICY": "immediate"},
        modify_run_participant_settings=lambda run_participant_settings: (
            replace_candidate_model_training_and_acceptance_settings(
                run_participant_settings,
                candidate_model_acceptance_policy=(
                    "current_model_first_reuse_then_candidate_adoption_without_validation"
                ),
            )
        ),
    ),
    LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME: SimpleNamespace(
        name=LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME,
        legacy_mode_name=PROPOSED_CONFIGURATION_VARIANT.legacy_mode_name,
        # 旧の`tools/run_main_ablation_suite.sh`の`distance-average`（linkageは、提案構成と同じaverage）。
        legacy_setting_overrides={"FEDSDA_CLUSTERING_DECISION": "distance"},
        modify_run_participant_settings=lambda run_participant_settings: replace(
            run_participant_settings,
            model_clustering_criteria=replace(
                run_participant_settings.model_clustering_criteria,
                model_pair_comparison_strategy="mean_loss_increase",
            ),
        ),
        # 統合の方式の宣言（条件の保存用）も、同じ値にする。
        modify_model_consolidation_settings=lambda model_consolidation_settings: replace(
            model_consolidation_settings, model_pair_comparison_strategy="mean_loss_increase"
        ),
    ),
    OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME: SimpleNamespace(
        name=OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME,
        # 旧の`tools/run_main_ablation_suite.sh`の`overall-esr`（構成名だけが違う）。
        legacy_mode_name="FedSDA_NoCached_ResidualAdapter_ESR_RestartingSoftRouting",
        legacy_setting_overrides={},
        modify_run_participant_settings=lambda run_participant_settings: (
            replace_loss_monitoring_scope(
                run_participant_settings,
                loss_monitoring_scope="overall_loss",
                detector_name="e-SR",
            )
        ),
    ),
    ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME: SimpleNamespace(
        name=ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME,
        legacy_mode_name=PROPOSED_CONFIGURATION_VARIANT.legacy_mode_name,
        # 旧の`tools/run_main_ablation_suite.sh`の`global-routing`。
        legacy_setting_overrides={"SOFT_ROUTING_CONTEXT": "global"},
        # 旧の、Fixed-Shareの結合と実際の予測の差の指標が、0とは限らない構成。
        fixed_share_weights_are_used_for_prediction=False,
        modify_run_participant_settings=lambda run_participant_settings: (
            replace_prediction_combination_settings(
                run_participant_settings,
                prediction_combination_strategy="adahedge_weighted_prediction",
            )
        ),
    ),
    SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME: SimpleNamespace(
        name=SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME,
        # 旧の`tools/run_main_ablation_suite.sh`の`hard-routing`（重みを持たないclient。再較正は、既定の「しない」）。
        legacy_mode_name="FedSDA_NoCached_ResidualAdapter_ClassESR",
        legacy_setting_overrides={"SHARED_BACKBONE_ROUTING_RECALIBRATION": "none"},
        modify_run_participant_settings=lambda run_participant_settings: (
            replace_prediction_combination_settings(
                run_participant_settings,
                prediction_combination_strategy="current_training_model_prediction",
                prediction_weight_recalibration_after_aggregation_policy=(
                    "keep_weights_without_recalibration"
                ),
            )
        ),
        # 実旧のclientは、重みと、予測の内訳の記録を、持たない。
        legacy_records_prediction_weight_diagnostics=False,
    ),
}
# 構成と条件ごとの、照合の結果（構成の結果と、同じ条件の提案構成の結果）。
VARIANT_COMPARISONS = {}


@pytest.mark.parametrize("small_run_condition", SMALL_RUN_CONDITIONS)
@pytest.mark.parametrize("configuration_name", sorted(CONFIGURATION_VARIANTS))
def test_configuration_variant_matches_real_legacy_whole_run(
    configuration_name, small_run_condition, monkeypatch, tmp_path, valid_run_settings_mapping
):
    """構成の全体runの、指標と離散列が、実旧の、同じ指定の全体runと一致する。"""
    (
        random_seed,
        client_count,
        per_client_sample_count,
        aggregation_interval,
        minimum_change_gap,
        concept_change_probability,
        update_interval,
    ) = small_run_condition
    comparisons = {}
    for variant_role, configuration_variant in (
        ("variant", CONFIGURATION_VARIANTS[configuration_name]),
        ("proposed", PROPOSED_CONFIGURATION_VARIANT),
    ):
        with pytest.MonkeyPatch.context() as variant_monkeypatch:
            comparisons[variant_role] = compare_small_run_with_real_legacy(
                monkeypatch=variant_monkeypatch,
                tmp_path=tmp_path,
                valid_run_settings_mapping=valid_run_settings_mapping,
                random_seed=random_seed,
                client_count=client_count,
                per_client_sample_count=per_client_sample_count,
                aggregation_interval=aggregation_interval,
                minimum_change_gap=minimum_change_gap,
                concept_change_probability=concept_change_probability,
                update_interval=update_interval,
                maximum_delay=MAXIMUM_DELAY,
                recovery_window=RECOVERY_WINDOW,
                configuration_variant=configuration_variant,
            )
    VARIANT_COMPARISONS[(configuration_name, small_run_condition)] = comparisons


def collect_variant_comparisons(configuration_name):
    comparisons = [
        VARIANT_COMPARISONS[(configuration_name, small_run_condition)]
        for small_run_condition in SMALL_RUN_CONDITIONS
        if (configuration_name, small_run_condition) in VARIANT_COMPARISONS
    ]
    if len(comparisons) < len(SMALL_RUN_CONDITIONS):
        pytest.skip("構成の、小さい条件の全部を実行したときだけ確かめる")
    return comparisons


def test_without_prediction_weight_recalibration_takes_its_own_path():
    """再較正をしない構成は、再較正の回数と、再生した標本数が0で、同じ条件の提案構成では0でない。結果も違う。"""
    comparisons = collect_variant_comparisons(
        PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME
    )
    for comparison in comparisons:
        variant_metrics = comparison["variant"].run_metrics
        assert variant_metrics.fixed_share_weight_recalibration_replayed_sample_count == 0
        assert variant_metrics.adahedge_weight_recalibration_replayed_sample_count == 0
        assert (
            variant_metrics.prediction_diagnostic_metrics.adahedge_weight_aggregation_recalibration_count
            == 0
        )
        # サーバの同期の結果は、clientごとに、再較正なし（None）。
        synchronizations = comparison[
            "variant"
        ].participants.server_operations.snapshot_server_round_synchronizations()
        assert synchronizations
        assert all(
            prediction_recalibration is None
            for synchronization in synchronizations
            for prediction_recalibration in synchronization.prediction_recalibrations
        )
        # 実旧のclientも、再較正を、1回も行っていない。
        for legacy_client in comparison["variant"].legacy_run["legacy_clients"]:
            assert legacy_client.switching_expert_router.aggregation_recalibration_count == 0
            assert legacy_client.expert_router.aggregation_recalibration_count == 0
    # 同じ条件の提案構成は、再較正で標本を再生していて、予測の結果が、この構成と違う条件がある。
    assert any(
        comparison["proposed"].run_metrics.fixed_share_weight_recalibration_replayed_sample_count
        > 0
        for comparison in comparisons
    )
    assert any(
        comparison["variant"].run_metrics.prediction_diagnostic_metrics
        != comparison["proposed"].run_metrics.prediction_diagnostic_metrics
        for comparison in comparisons
    )
    assert any(
        comparison["variant"].run_metrics.prediction_accuracy
        != comparison["proposed"].run_metrics.prediction_accuracy
        for comparison in comparisons
    )


def test_without_candidate_validation_takes_its_own_path():
    """候補検証をしない構成は、警報の位置で候補を採用し、候補検証の開始・判定が、新旧とも無い。結果も違う。"""
    comparisons = collect_variant_comparisons(CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME)
    adopted_at_alarm_count = 0
    for comparison in comparisons:
        variant_comparison = comparison["variant"]
        for run_client, legacy_client in zip(
            variant_comparison.participants.client_operations,
            variant_comparison.legacy_run["legacy_clients"],
            strict=True,
        ):
            adaptation_outcomes = [
                adaptation_record.adaptation_outcome
                for adaptation_record in run_client.owners.adaptation_record_store.get_state_snapshot().adaptation_records
            ]
            # 候補検証に関わる結果（開始、検証中の警報、検証の確定）は、1つも無い。
            assert not any(
                adaptation_outcome.startswith("post_alarm_validation_")
                or adaptation_outcome
                in (
                    "alarm_interval_candidate_validation_started",
                    "alarm_during_candidate_validation",
                )
                for adaptation_outcome in adaptation_outcomes
            )
            assert run_client.owners.validation_session_holder.held_validation_session is None
            assert (
                run_client.owners.candidate_validation_decision_record_store.snapshot_candidate_validation_decision_records()
                == ()
            )
            client_adopted_at_alarm_count = adaptation_outcomes.count(
                "alarm_interval_candidate_adopted_as_new_model"
            )
            adopted_at_alarm_count += client_adopted_at_alarm_count
            # 実旧も、候補の判定の記録が無く、警報の位置での作成（`create`）の数が、新と同じ。
            assert legacy_client.provisional_model_decisions == []
            legacy_actions = [
                legacy_event.action for legacy_event in legacy_client.adaptation_events
            ]
            assert legacy_actions.count("create") == client_adopted_at_alarm_count
            assert not {
                "create_pending",
                "create_rejected",
                "forward_validation_pending",
            } & set(legacy_actions)
    assert adopted_at_alarm_count > 0
    # 同じ条件の提案構成は、候補検証を開始していて、結果が、この構成と違う条件がある。
    assert any(
        comparison["proposed"].run_metrics.change_response_metrics
        != comparison["variant"].run_metrics.change_response_metrics
        for comparison in comparisons
    )
    assert any(
        comparison["variant"].run_metrics.prediction_accuracy
        != comparison["proposed"].run_metrics.prediction_accuracy
        for comparison in comparisons
    )


def collect_pair_clustering_observations(comparison):
    return comparison.participants.server_operations.owners.model_clustering_record_store.snapshot_pair_clustering_observations()


def test_loss_distance_clustering_takes_its_own_path():
    """損失の増加量で判定する構成は、対の判定の値が、損失の増加量そのもの。提案構成では、別の値。結果も違う。"""
    comparisons = collect_variant_comparisons(LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME)
    variant_pair_observations = [
        pair_observation
        for comparison in comparisons
        for pair_observation in collect_pair_clustering_observations(comparison["variant"])
    ]
    assert variant_pair_observations
    assert all(
        pair_observation.decision_score == pair_observation.loss_increase_distance
        for pair_observation in variant_pair_observations
    )
    # 同じクラスタになった対と、ならなかった対の、両方を通っている。
    assert {
        pair_observation.assigned_to_same_cluster for pair_observation in variant_pair_observations
    } == {True, False}
    # 実旧も、判定の値として、損失の距離を使っている。
    for comparison in comparisons:
        legacy_server = comparison["variant"].legacy_run["legacy_server"]
        assert legacy_server.clustering_decision == "distance"
        assert legacy_server.linkage == "average"
    # 同じ条件の提案構成は、判定の値が、損失の増加量と違う対があり、結果が、この構成と違う条件がある。
    assert any(
        pair_observation.decision_score != pair_observation.loss_increase_distance
        for comparison in comparisons
        for pair_observation in collect_pair_clustering_observations(comparison["proposed"])
    )
    assert any(
        comparison["variant"].run_metrics.model_diagnostic_metrics
        != comparison["proposed"].run_metrics.model_diagnostic_metrics
        for comparison in comparisons
    )


def test_overall_loss_esr_detection_takes_its_own_path():
    """全体の損失だけを監視する構成は、クラス別の検出器を持たず、1標本につき、検出器の更新が1回。結果も違う。"""
    comparisons = collect_variant_comparisons(OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME)
    alarm_count = 0
    for comparison in comparisons:
        variant_comparison = comparison["variant"]
        for run_client, legacy_client in zip(
            variant_comparison.participants.client_operations,
            variant_comparison.legacy_run["legacy_clients"],
            strict=True,
        ):
            owners = run_client.owners
            assert (
                owners.loss_change_monitor.get_state_snapshot().class_esr_states_by_class_id == ()
            )
            adaptation_records = (
                owners.adaptation_record_store.get_state_snapshot().adaptation_records
            )
            alarm_count += len(adaptation_records)
            assert all(
                adaptation_record.detector_name == "e-SR"
                for adaptation_record in adaptation_records
                if adaptation_record.adaptation_outcome
                != "server_consolidation_training_model_remapped"
            )
            # 実旧のclientは、全体の損失のe-SRのclientで、クラス別の検出器を持たない。
            assert legacy_client._detector_label() == "e-SR"
            assert not hasattr(legacy_client, "class_e_detectors")
            # 検出器の更新は、処理した標本1件につき1回（提案構成は2回）。
            assert legacy_client.compute_counters["drift_detector_updates"] == (
                legacy_client.processed_samples
            )
        for proposed_client in comparison["proposed"].legacy_run["legacy_clients"]:
            assert proposed_client.compute_counters["drift_detector_updates"] == (
                2 * proposed_client.processed_samples
            )
    assert alarm_count > 0
    # 同じ条件の提案構成と、結果が違う条件がある。
    assert any(
        comparison["variant"].run_metrics.change_response_metrics
        != comparison["proposed"].run_metrics.change_response_metrics
        for comparison in comparisons
    )
    assert any(
        comparison["variant"].run_metrics.prediction_accuracy
        != comparison["proposed"].run_metrics.prediction_accuracy
        for comparison in comparisons
    )


def collect_sample_prediction_records(comparison):
    return [
        prediction_record
        for run_client in comparison.participants.client_operations
        for prediction_record in run_client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
    ]


def test_adahedge_prediction_weights_take_their_own_path():
    """AdaHedgeの重みで予測する構成は、実際の予測が、AdaHedgeの重みでの結合と同じ。Fixed-Share側は、比較用に残る。"""
    comparisons = collect_variant_comparisons(ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME)
    variant_records = [
        prediction_record
        for comparison in comparisons
        for prediction_record in collect_sample_prediction_records(comparison["variant"])
    ]
    proposed_records = [
        prediction_record
        for comparison in comparisons
        for prediction_record in collect_sample_prediction_records(comparison["proposed"])
    ]
    # この構成: 実際の予測の正否は、全標本で、AdaHedgeの重みでの結合の正否と同じ。
    assert all(
        prediction_record.combined_prediction_is_correct
        == prediction_record.adahedge_weighted_prediction_is_correct
        for prediction_record in variant_records
    )
    # Fixed-Shareの重みでの結合と、正否・最大重みのモデル・実効モデル数が違う標本がある。
    assert any(
        prediction_record.combined_prediction_is_correct
        != prediction_record.fixed_share_weighted_prediction_is_correct
        for prediction_record in variant_records
    )
    assert any(
        prediction_record.maximum_weight_model_id
        != prediction_record.fixed_share_maximum_weight_model_id
        for prediction_record in variant_records
    )
    assert any(
        prediction_record.effective_model_count
        != prediction_record.fixed_share_effective_model_count
        for prediction_record in variant_records
    )
    # 提案構成: 実際の予測は、全標本で、Fixed-Share側の項目と同じ。AdaHedgeの重みでの結合と違う標本がある。
    assert all(
        (
            prediction_record.combined_prediction_is_correct,
            prediction_record.maximum_weight_model_id,
            prediction_record.effective_model_count,
        )
        == (
            prediction_record.fixed_share_weighted_prediction_is_correct,
            prediction_record.fixed_share_maximum_weight_model_id,
            prediction_record.fixed_share_effective_model_count,
        )
        for prediction_record in proposed_records
    )
    assert any(
        prediction_record.combined_prediction_is_correct
        != prediction_record.adahedge_weighted_prediction_is_correct
        for prediction_record in proposed_records
    )
    # 実旧も、AdaHedgeの重みで予測している（Fixed-Share側の診断の「実際の正解数」が、AdaHedge側の正解数と同じ）。
    for comparison in comparisons:
        for legacy_client in comparison["variant"].legacy_run["legacy_clients"]:
            switching_diagnostics = legacy_client.routing_switching_diagnostics
            assert (
                switching_diagnostics["actual_correct_count"]
                == switching_diagnostics["global_correct_count"]
            )
    # 同じ条件の提案構成と、予測の結果が違う条件がある。
    assert any(
        comparison["variant"].run_metrics.prediction_accuracy
        != comparison["proposed"].run_metrics.prediction_accuracy
        for comparison in comparisons
    )


def test_single_model_prediction_takes_its_own_path():
    """現在のモデルだけで予測する構成は、重みを使わず、記録の重みの項目が、現在のモデルに重み1の値になる。"""
    comparisons = collect_variant_comparisons(SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME)
    multiple_held_model_sample_count = 0
    for comparison in comparisons:
        variant_comparison = comparison["variant"]
        for run_client, legacy_client in zip(
            variant_comparison.participants.client_operations,
            variant_comparison.legacy_run["legacy_clients"],
            strict=True,
        ):
            owners = run_client.owners
            prediction_records = (
                owners.sample_prediction_record_store.snapshot_sample_prediction_records()
            )
            held_model_counts = owners.held_model_count_record_store.snapshot_held_model_counts()
            assert len(prediction_records) == len(held_model_counts) > 0
            multiple_held_model_sample_count += sum(
                held_model_count > 1 for held_model_count in held_model_counts
            )
            for prediction_record in prediction_records:
                # 重みは、現在のモデルに1。どの重みでの結合も、現在のモデルの予測。
                assert prediction_record.maximum_prediction_weight == 1.0
                assert prediction_record.effective_model_count == 1.0
                assert prediction_record.fixed_share_effective_model_count == 1.0
                assert (
                    prediction_record.fixed_share_maximum_weight_model_id
                    == prediction_record.maximum_weight_model_id
                )
                assert (
                    len(
                        {
                            prediction_record.combined_prediction_is_correct,
                            prediction_record.any_model_or_combined_prediction_is_correct,
                            prediction_record.maximum_weight_model_prediction_is_correct,
                            prediction_record.highest_confidence_model_prediction_is_correct,
                            prediction_record.fixed_share_weighted_prediction_is_correct,
                            prediction_record.adahedge_weighted_prediction_is_correct,
                        }
                    )
                    == 1
                )
            # Fixed-Shareの重みと、AdaHedgeの証拠は、1回も使われていない。
            assert owners.fixed_share_weight_controller.weights_by_model_id == {}
            assert owners.fixed_share_weight_controller.model_pool_reset_count == 0
            all_sample_weight_evidence = (
                owners.adahedge_weight_evidence_collection.all_sample_weight_evidence
            )
            assert all_sample_weight_evidence.cumulative_losses_by_model_id == {}
            assert all_sample_weight_evidence.concept_operation_restart_count == 0
            assert owners.adahedge_weight_evidence_collection.created_true_concept_ids == ()
            # 実旧のclientは、重みを持たない。予測したモデルは、新の記録と同じ。
            assert not hasattr(legacy_client, "expert_router")
            assert not hasattr(legacy_client, "switching_expert_router")
            assert legacy_client.history_model_id == [
                prediction_record.maximum_weight_model_id
                for prediction_record in prediction_records
            ]
    # 複数のモデルを保有している標本でも、現在のモデルだけで予測している。
    assert multiple_held_model_sample_count > 0
    # 学習帰属の変更が起きていて（この構成でも、同じ条件の提案構成でも）、提案構成では、AdaHedgeの証拠が始め直されている。
    # この構成で始め直しが0回なのは、変更が無かったからではない。
    for variant_role in ("variant", "proposed"):
        assert any(
            run_client.owners.adaptation_record_store.get_state_snapshot().training_model_switch_sample_indices
            for comparison in comparisons
            for run_client in comparison[variant_role].participants.client_operations
        )
    assert any(
        run_client.owners.adahedge_weight_evidence_collection.all_sample_weight_evidence.concept_operation_restart_count
        > 0
        for comparison in comparisons
        for run_client in comparison["proposed"].participants.client_operations
    )
    # 指標でも、予測の内訳は、全部、現在のモデルの予測の値になる。結合して予測した標本は、無い。
    for comparison in comparisons:
        run_metrics = comparison["variant"].run_metrics
        diagnostic_metrics = run_metrics.prediction_diagnostic_metrics
        correct_counts = diagnostic_metrics.prediction_correct_counts
        assert (
            len(
                {
                    correct_counts.combined_prediction_correct_count,
                    correct_counts.any_model_or_combined_prediction_correct_count,
                    correct_counts.maximum_weight_model_prediction_correct_count,
                    correct_counts.highest_confidence_model_prediction_correct_count,
                    diagnostic_metrics.fixed_share_weighted_prediction_correct_count,
                    diagnostic_metrics.adahedge_weighted_prediction_correct_count,
                }
            )
            == 1
        )
        assert diagnostic_metrics.mean_effective_model_count == 1.0
        assert diagnostic_metrics.mean_fixed_share_effective_model_count == 1.0
        assert diagnostic_metrics.fixed_share_weight_leader_switch_count == 0
        assert run_metrics.mixed_prediction_sample_count == 0
    # 同じ条件の提案構成と、予測の結果が違う条件がある。
    assert any(
        comparison["variant"].run_metrics.prediction_accuracy
        != comparison["proposed"].run_metrics.prediction_accuracy
        for comparison in comparisons
    )


def test_configuration_names_build_run_settings_that_differ_only_in_the_changed_element():
    """掃引のコマンドの構成の名前から作るrun設定は、提案構成と、変える要素だけが違う。"""
    assert EXPERIMENT_CONFIGURATION_NAMES == (
        PROPOSED_FEDSDA_CONFIGURATION_NAME,
        PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME,
        CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME,
        LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME,
        OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME,
        ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME,
        SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME,
    )
    assert SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME == "fedsda-single-model-prediction"
    assert ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME == ("fedsda-adahedge-prediction-weights")
    assert OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME == ("fedsda-overall-loss-esr-detection")
    assert LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME == "fedsda-loss-distance-clustering"
    assert PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME == (
        "fedsda-without-prediction-weight-recalibration"
    )
    assert CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME == (
        "fedsda-without-candidate-validation"
    )
    assert set(CONFIGURATION_VARIANTS) == set(EXPERIMENT_CONFIGURATION_NAMES) - {
        PROPOSED_FEDSDA_CONFIGURATION_NAME
    }
    experiment_run_conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=3,
        per_client_sample_count=200,
        server_aggregation_interval_per_client_samples=50,
    )
    proposed_run_settings = build_experiment_run_settings(
        configuration_name=PROPOSED_FEDSDA_CONFIGURATION_NAME,
        experiment_run_conditions=experiment_run_conditions,
    )
    for configuration_name, configuration_variant in CONFIGURATION_VARIANTS.items():
        run_settings = build_experiment_run_settings(
            configuration_name=configuration_name,
            experiment_run_conditions=experiment_run_conditions,
        )
        assert run_settings != proposed_run_settings
        # サーバが読む判定の基準と、統合の方式の宣言とで、モデル対の比べ方が違う設定は、作れない。
        if (
            run_settings.model_consolidation_settings
            != proposed_run_settings.model_consolidation_settings
        ):
            # どちらの向きのずれも、拒否する。
            for mismatched_settings_fields in (
                dict(
                    model_consolidation_settings=proposed_run_settings.model_consolidation_settings
                ),
                dict(run_participant_settings=proposed_run_settings.run_participant_settings),
            ):
                with pytest.raises(RunSettingsValidationError) as error_information:
                    replace(run_settings, **mismatched_settings_fields)
                assert (
                    error_information.value.configuration_parameter_name
                    == "model_pair_comparison_strategy"
                )
        # testの側の変更（実旧と照合したもの）を、提案構成の設定へ当てると、同じ設定になる。
        assert run_settings == replace(
            proposed_run_settings,
            run_participant_settings=configuration_variant.modify_run_participant_settings(
                proposed_run_settings.run_participant_settings
            ),
            model_consolidation_settings=getattr(
                configuration_variant,
                "modify_model_consolidation_settings",
                lambda model_consolidation_settings: model_consolidation_settings,
            )(proposed_run_settings.model_consolidation_settings),
        )


def test_overall_loss_scope_requires_its_own_detector_display_name():
    """監視の対象が全体の損失だけの設定は、検出器の表示名が`e-SR`のときだけ作れる（混合の表示名は、拒否する）。"""
    run_settings = build_experiment_run_settings(
        configuration_name=OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=0,
            client_count=3,
            per_client_sample_count=200,
            server_aggregation_interval_per_client_samples=50,
        ),
    )
    run_client_settings = run_settings.run_participant_settings.run_client_settings
    assert run_client_settings.loss_change_detection_settings.loss_monitoring_scope == (
        "overall_loss"
    )
    assert run_client_settings.scalar_settings.detector_name == "e-SR"
    with pytest.raises(RunSettingsValidationError) as error_information:
        replace(
            run_client_settings,
            scalar_settings=replace(
                run_client_settings.scalar_settings,
                detector_name="overall + class-conditional e-SR mixture",
            ),
        )
    assert error_information.value.configuration_parameter_name == "detector_name"


def test_single_model_prediction_requires_no_weight_recalibration():
    """現在のモデルだけで予測する設定は、集約後の、重みの再較正をしない設定と組のときだけ作れる。"""
    run_settings = build_experiment_run_settings(
        configuration_name=SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=0,
            client_count=3,
            per_client_sample_count=200,
            server_aggregation_interval_per_client_samples=50,
        ),
    )
    run_client_settings = run_settings.run_participant_settings.run_client_settings
    prediction_combination_settings = run_client_settings.prediction_combination_settings
    assert prediction_combination_settings.prediction_combination_strategy == (
        "current_training_model_prediction"
    )
    assert (
        prediction_combination_settings.prediction_weight_recalibration_after_aggregation_policy
        == "keep_weights_without_recalibration"
    )
    with pytest.raises(RunSettingsValidationError) as error_information:
        replace(
            run_client_settings,
            prediction_combination_settings=replace(
                prediction_combination_settings,
                prediction_weight_recalibration_after_aggregation_policy=(
                    "recompute_buffer_losses_and_replay_weight_updates"
                ),
            ),
        )
    assert error_information.value.configuration_parameter_name == (
        "prediction_weight_recalibration_after_aggregation_policy"
    )
