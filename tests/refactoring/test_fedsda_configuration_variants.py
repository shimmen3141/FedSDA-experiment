"""提案構成以外の構成（ablationの構成）の全体runを、小さい条件で、実旧の、同じ指定の全体runと照合する。

照合の内容（指標の全部と、31の離散列）は、提案構成の照合（test_fedsda_run_metric_derivation.pyの
`compare_small_run_with_real_legacy`）と同じ。構成ごとに、その構成に固有の経路を通ったことも確かめる。
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_fedsda_run_client as run_client_test_module
import test_fedsda_stream_protocol_run as whole_run_test_module
from test_cross_evaluation_download_deduplication import observe_legacy_duplicate_downloads
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
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.runtime import (
    observed_sample_processing as observed_sample_processing_module,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME,
    CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME,
    CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME,
    EXPERIMENT_CONFIGURATION_NAMES,
    FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME,
    INDEPENDENT_MODELS_CONFIGURATION_NAME,
    LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME,
    LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME,
    OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME,
    OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME,
    PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME,
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
    SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME,
    SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME,
    SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME,
    WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME,
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


def replace_candidate_acceptance_and_shared_feature_update(
    run_participant_settings,
    *,
    candidate_model_acceptance_policy,
    candidate_shared_feature_update_policy,
):
    """参加者の設定の束の、候補の採否の方針と、候補の学習での共有部の更新を、一緒に置き換える。

    2つは組合せの制約（候補検証と共有部の固定は組み合わせられない）があるので、1回の置き換えで替える。
    """
    run_client_settings = run_participant_settings.run_client_settings
    return replace(
        run_participant_settings,
        run_client_settings=replace(
            run_client_settings,
            candidate_model_training_and_acceptance_settings=replace(
                run_client_settings.candidate_model_training_and_acceptance_settings,
                candidate_model_acceptance_policy=candidate_model_acceptance_policy,
            ),
            candidate_epoch_training_settings=replace(
                run_client_settings.candidate_epoch_training_settings,
                candidate_shared_feature_update_policy=candidate_shared_feature_update_policy,
            ),
        ),
    )


def replace_loss_monitoring_scope(
    run_participant_settings, *, detector_name, **changed_detection_settings_fields
):
    """参加者の設定の束の、損失の監視の設定（対象、検出器）と、それに対応する検出器の表示名を、置き換える。"""
    run_client_settings = run_participant_settings.run_client_settings
    if changed_detection_settings_fields.get("drift_detector_name") == "adwin":
        changed_detection_settings_fields["loss_monitoring_baseline_update_policy"] = (
            "keep_baseline_until_alarm"
        )
    return replace(
        run_participant_settings,
        run_client_settings=replace(
            run_client_settings,
            loss_change_detection_settings=replace(
                run_client_settings.loss_change_detection_settings,
                **changed_detection_settings_fields,
            ),
            scalar_settings=replace(
                run_client_settings.scalar_settings, detector_name=detector_name
            ),
        ),
    )


def replace_loss_monitor_maximum_window(run_participant_settings):
    """参加者の設定の束の、監視の窓（候補）の上限を、提案構成のrun設定と同じ値（旧の既定の1000）にする。"""
    run_client_settings = run_participant_settings.run_client_settings
    return replace(
        run_participant_settings,
        run_client_settings=replace(
            run_client_settings,
            scalar_settings=replace(
                run_client_settings.scalar_settings,
                loss_monitor_maximum_retained_candidate_count=1000,
            ),
        ),
    )


def replace_with_independent_models(run_participant_settings):
    """参加者の設定の束を、モデルが共有部を持たない構成のものへ、置き換える。

    モデル構造、ローカル学習の方式（モデルごとに順に学習する）、集約後の重みの再較正をしない方針の、3つ。
    """
    settings_without_recalibration = replace_prediction_combination_settings(
        run_participant_settings,
        prediction_weight_recalibration_after_aggregation_policy="keep_weights_without_recalibration",
    )
    run_client_settings = settings_without_recalibration.run_client_settings
    return replace(
        settings_without_recalibration,
        model_architecture_settings=replace(
            run_participant_settings.model_architecture_settings,
            model_architecture_name="independent_models_without_shared_backbone",
        ),
        run_client_settings=replace(
            run_client_settings,
            local_training_settings=replace(
                run_client_settings.local_training_settings,
                local_model_parameter_update_strategy="sequential_per_model_training",
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
        # 2026-10-20からの既定は、候補の学習で共有部を固定するので、一緒に更新する値も明示する（旧には、
        # 固定する扱いが無い）。
        modify_run_participant_settings=lambda run_participant_settings: (
            replace_candidate_acceptance_and_shared_feature_update(
                run_participant_settings,
                candidate_model_acceptance_policy=(
                    "current_model_first_reuse_then_candidate_adoption_without_validation"
                ),
                candidate_shared_feature_update_policy="update_shared_features_with_candidate",
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
                # 混ぜる予測（2026-10-16からの既定）とは組み合わせられないので、局所を明示する。
                prediction_model_parameter_source="current_local_parameters",
            )
        ),
        # 実旧のclientは、重みと、予測の内訳の記録を、持たない。
        legacy_records_prediction_weight_diagnostics=False,
    ),
    CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME: SimpleNamespace(
        name=CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME,
        # 旧の`tools/run_main_ablation_suite.sh`の`class-adwin`（`--fixed-adwin-delta 0.05`）。
        legacy_mode_name="FedSDA_NoCached_ResidualAdapter_ClassADWIN_RestartingSoftRouting",
        # 窓の上限は、旧の既定（1000）にする（小さい上限では、ADWIN自身が検知しない）。
        legacy_setting_overrides={"ADWIN_DELTA": 0.05, "ADWIN_MAX_WINDOW": 1000},
        modify_run_participant_settings=lambda run_participant_settings: (
            replace_loss_monitoring_scope(
                replace_loss_monitor_maximum_window(run_participant_settings),
                drift_detector_name="adwin",
                short_change_interval_alarm_policy=RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
                detector_name="ADWIN",
            )
        ),
        # 実旧のADWINのclientは、監視の値（e値）を持たない。
        legacy_records_monitored_log_e_values=False,
    ),
    OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME: SimpleNamespace(
        name=OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME,
        # 旧の`tools/run_main_ablation_suite.sh`の`overall-adwin`。
        legacy_mode_name="FedSDA_NoCached_ResidualAdapter_ADWIN_RestartingSoftRouting",
        # 窓の上限は、旧の既定（1000）にする（小さい上限では、ADWIN自身が検知しない）。
        legacy_setting_overrides={"ADWIN_DELTA": 0.05, "ADWIN_MAX_WINDOW": 1000},
        modify_run_participant_settings=lambda run_participant_settings: (
            replace_loss_monitoring_scope(
                replace_loss_monitor_maximum_window(run_participant_settings),
                drift_detector_name="adwin",
                loss_monitoring_scope="overall_loss",
                short_change_interval_alarm_policy=RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
                detector_name="ADWIN",
            )
        ),
        legacy_records_monitored_log_e_values=False,
    ),
    WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME: SimpleNamespace(
        name=WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME,
        # 旧の`tools/run_main_ablation_suite.sh`の`shared-backbone`（モデルが、共有部と出力層だけ）。
        legacy_mode_name="FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting",
        legacy_setting_overrides={},
        modify_run_participant_settings=lambda run_participant_settings: replace(
            run_participant_settings,
            model_architecture_settings=replace(
                run_participant_settings.model_architecture_settings,
                model_architecture_name="shared_backbone_without_residual_adapter",
            ),
        ),
    ),
    INDEPENDENT_MODELS_CONFIGURATION_NAME: SimpleNamespace(
        name=INDEPENDENT_MODELS_CONFIGURATION_NAME,
        # 旧の`tools/run_main_ablation_suite.sh`の`independent`（モデルは、共有部を持たないMLP。
        # サーバは、集約後の再較正を呼ばない）。
        legacy_mode_name="FedSDA_NoCached_ClassESR_RestartingSoftRouting",
        legacy_setting_overrides={"SHARED_BACKBONE_ROUTING_RECALIBRATION": "none"},
        modify_run_participant_settings=lambda run_participant_settings: (
            replace_with_independent_models(run_participant_settings)
        ),
        # 実旧のモデルは、共有部を持たない（計算量の、共有部の計数が、概念固有部と同じ値になる）。
        legacy_models_share_feature_extractor=False,
    ),
}
# 構成と条件ごとの、照合の結果（構成の結果と、同じ条件の提案構成の結果）。
VARIANT_COMPARISONS = {}
# ADWINで監視する構成の照合で通った、警報の経路の回数（構成の名前→経路→回数）。
ADWIN_ALARM_PATH_COUNTS = {}


def count_adwin_alarm_paths(*, monkeypatch, configuration_name):
    """ADWINの監視と、補助の警報判定を包んで、通った経路を数える（動きは変えない）。"""
    path_counts = ADWIN_ALARM_PATH_COUNTS.setdefault(
        configuration_name,
        dict(adwin_detection=0, window_loss_check_alarm=0, window_loss_check_without_alarm=0),
    )
    observe_loss_with_adwin = OverallAndTrueClassLossMonitor._observe_loss_with_adwin
    window_loss_check = (
        observed_sample_processing_module._current_model_loss_worsened_over_adwin_window
    )

    def counting_observe_loss_with_adwin(loss_change_monitor, **observation_arguments):
        monitoring_observation = observe_loss_with_adwin(
            loss_change_monitor, **observation_arguments
        )
        path_counts["adwin_detection"] += monitoring_observation.drift_detected
        return monitoring_observation

    def counting_window_loss_check(**check_arguments):
        loss_worsened = window_loss_check(**check_arguments)
        path_counts[
            "window_loss_check_alarm" if loss_worsened else "window_loss_check_without_alarm"
        ] += 1
        return loss_worsened

    monkeypatch.setattr(
        OverallAndTrueClassLossMonitor, "_observe_loss_with_adwin", counting_observe_loss_with_adwin
    )
    monkeypatch.setattr(
        observed_sample_processing_module,
        "_current_model_loss_worsened_over_adwin_window",
        counting_window_loss_check,
    )


@pytest.mark.parametrize("small_run_condition", SMALL_RUN_CONDITIONS)
@pytest.mark.parametrize("configuration_name", sorted(CONFIGURATION_VARIANTS))
@pytest.mark.slow
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
            if variant_role == "variant" and configuration_name in (
                CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME,
                OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME,
            ):
                count_adwin_alarm_paths(
                    monkeypatch=variant_monkeypatch, configuration_name=configuration_name
                )
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


# 下の「固有の経路」のtestは、同じ構成の、上の照合の結果を読む。並列の実行では、構成の名前で、別のprocessへ
# 配る（tests/conftest.py）ので、読む構成の名前を、parametrizeの値として持つ（testのidの先頭が、構成の名前）。
def collect_variant_comparisons(configuration_name):
    comparisons = [
        VARIANT_COMPARISONS[(configuration_name, small_run_condition)]
        for small_run_condition in SMALL_RUN_CONDITIONS
        if (configuration_name, small_run_condition) in VARIANT_COMPARISONS
    ]
    if len(comparisons) < len(SMALL_RUN_CONDITIONS):
        pytest.skip("構成の、小さい条件の全部を実行したときだけ確かめる")
    return comparisons


@pytest.mark.parametrize(
    "configuration_name", [PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME]
)
def test_without_prediction_weight_recalibration_takes_its_own_path(configuration_name):
    """再較正をしない構成は、再較正の回数と、再生した標本数が0で、同じ条件の提案構成では0でない。結果も違う。"""
    comparisons = collect_variant_comparisons(configuration_name)
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


@pytest.mark.parametrize("configuration_name", [CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME])
def test_without_candidate_validation_takes_its_own_path(configuration_name):
    """候補検証をしない構成は、警報の位置で候補を採用し、候補検証の開始・判定が、新旧とも無い。結果も違う。"""
    comparisons = collect_variant_comparisons(configuration_name)
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
    # 検証なしへ揃えた旧との照合の基点と、同じ数値になる。
    assert all(
        comparison["proposed"].run_metrics == comparison["variant"].run_metrics
        for comparison in comparisons
    )


def collect_pair_clustering_observations(comparison):
    return comparison.participants.server_operations.owners.model_clustering_record_store.snapshot_pair_clustering_observations()


@pytest.mark.parametrize("configuration_name", [LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME])
def test_loss_distance_clustering_takes_its_own_path(configuration_name):
    """損失の増加量で判定する構成は、対の判定の値が、損失の増加量そのもの。提案構成では、別の値。結果も違う。"""
    comparisons = collect_variant_comparisons(configuration_name)
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


@pytest.mark.parametrize("configuration_name", [OVERALL_LOSS_ESR_DETECTION_CONFIGURATION_NAME])
def test_overall_loss_esr_detection_takes_its_own_path(configuration_name):
    """全体の損失だけを監視する構成は、クラス別の検出器を持たず、1標本につき、検出器の更新が1回。結果も違う。"""
    comparisons = collect_variant_comparisons(configuration_name)
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


@pytest.mark.parametrize("configuration_name", [ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME])
def test_adahedge_prediction_weights_take_their_own_path(configuration_name):
    """AdaHedgeの重みで予測する構成は、実際の予測が、AdaHedgeの重みでの結合と同じ。Fixed-Share側は、比較用に残る。"""
    comparisons = collect_variant_comparisons(configuration_name)
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


@pytest.mark.parametrize("configuration_name", [SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME])
def test_single_model_prediction_takes_its_own_path(configuration_name):
    """現在のモデルだけで予測する構成は、重みを使わず、記録の重みの項目が、現在のモデルに重み1の値になる。"""
    comparisons = collect_variant_comparisons(configuration_name)
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
                owners.all_sample_adahedge_weight_evidence_holder.all_sample_weight_evidence
            )
            assert all_sample_weight_evidence.cumulative_losses_by_model_id == {}
            assert all_sample_weight_evidence.concept_operation_restart_count == 0
            assert (
                owners.true_concept_adahedge_diagnostic_evidence_collection.created_true_concept_ids
                == ()
            )
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
        run_client.owners.all_sample_adahedge_weight_evidence_holder.all_sample_weight_evidence.concept_operation_restart_count
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


@pytest.mark.parametrize(
    ("configuration_name", "class_detectors_are_used"),
    (
        (CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME, True),
        (OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME, False),
    ),
)
def test_adwin_detection_takes_its_own_path(configuration_name, class_detectors_are_used):
    """ADWINで監視する構成は、e値を持たず、ADWINの窓で警報を出す。クラス別の窓は、クラス別の構成だけが持つ。"""
    comparisons = collect_variant_comparisons(configuration_name)
    alarm_count = 0
    class_window_was_held = False
    for comparison in comparisons:
        variant_comparison = comparison["variant"]
        # 監視の値（e値）は、無い（負の無限大）。
        assert (
            variant_comparison.run_metrics.change_response_metrics.maximum_monitored_log_e_value
            == float("-inf")
        )
        for run_client, legacy_client in zip(
            variant_comparison.participants.client_operations,
            variant_comparison.legacy_run["legacy_clients"],
            strict=True,
        ):
            owners = run_client.owners
            monitoring_state = owners.loss_change_monitor.get_state_snapshot()
            assert monitoring_state.overall_esr_state is None
            assert monitoring_state.class_esr_states_by_class_id == ()
            # 実旧のclientの、ADWINの窓と、run終了時の状態が同じ。
            assert monitoring_state.overall_adwin_state.window_losses == tuple(
                legacy_client.adwin.window
            )
            assert monitoring_state.overall_adwin_state.loss_sum == legacy_client.adwin.total
            assert hasattr(legacy_client, "class_adwins") == class_detectors_are_used
            if class_detectors_are_used:
                assert {
                    class_id: class_adwin_state.window_losses
                    for class_id, class_adwin_state in monitoring_state.class_adwin_states_by_class_id
                } == {
                    class_id: tuple(class_adwin.window)
                    for class_id, class_adwin in legacy_client.class_adwins.items()
                    if class_adwin.width > 0
                }
                class_window_was_held |= bool(monitoring_state.class_adwin_states_by_class_id)
            else:
                assert monitoring_state.class_adwin_states_by_class_id == ()
            assert legacy_client._detector_label() == "ADWIN"
            assert not hasattr(legacy_client, "e_detector")
            adaptation_records = (
                owners.adaptation_record_store.get_state_snapshot().adaptation_records
            )
            alarm_count += len(
                owners.loss_change_alarm_record_store.get_state_snapshot().alarm_sample_indices
            )
            assert all(
                adaptation_record.detector_name == "ADWIN"
                for adaptation_record in adaptation_records
                if adaptation_record.adaptation_outcome
                != "server_consolidation_training_model_remapped"
            )
    assert alarm_count > 0
    assert class_window_was_held == class_detectors_are_used
    # ADWINの検知による警報と、補助の判定（直近の損失の悪化）による警報の、両方を通っている。
    path_counts = ADWIN_ALARM_PATH_COUNTS[configuration_name]
    assert path_counts["adwin_detection"] > 0
    assert path_counts["window_loss_check_alarm"] > 0
    assert path_counts["window_loss_check_without_alarm"] > 0
    # 同じ条件の提案構成と、結果が違う条件がある。
    assert any(
        comparison["variant"].run_metrics.change_response_metrics
        != comparison["proposed"].run_metrics.change_response_metrics
        for comparison in comparisons
    )


@pytest.mark.parametrize(
    "configuration_name", [WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME]
)
def test_without_concept_specific_adapter_takes_its_own_path(configuration_name):
    """adapterを持たない構成は、保有モデルの概念固有部が、出力層だけ。実旧のモデルも、adapterを持たない。"""
    comparisons = collect_variant_comparisons(configuration_name)
    multiple_held_model_client_count = 0
    for comparison in comparisons:
        for variant_role, adapter_is_expected in (("variant", False), ("proposed", True)):
            role_comparison = comparison[variant_role]
            for run_client, legacy_client in zip(
                role_comparison.participants.client_operations,
                role_comparison.legacy_run["legacy_clients"],
                strict=True,
            ):
                held_states = run_client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
                multiple_held_model_client_count += (
                    variant_role == "variant" and len(held_states) > 1
                )
                for held_state in held_states:
                    parameter_names = set(held_state.classifier.state_dict())
                    assert (
                        any(
                            parameter_name.startswith("model_owned_feature_transform.")
                            for parameter_name in parameter_names
                        )
                        == adapter_is_expected
                    )
                    # 概念固有部のパラメータの数が、実旧のモデルの、共有部以外のパラメータの数と同じ。
                    legacy_model = legacy_client.models[held_state.model_id]
                    assert sum(
                        parameter.numel()
                        for parameter_name, parameter in held_state.classifier.state_dict().items()
                        if not parameter_name.startswith("feature_extractor.")
                    ) == sum(
                        parameter.numel() for parameter in legacy_model.personalized_parameters()
                    )
                    assert hasattr(legacy_model, "adapter") == adapter_is_expected
    assert multiple_held_model_client_count > 0
    # 同じ条件の提案構成と、結果と、通信量（パラメータの数）が違う。
    assert any(
        comparison["variant"].run_metrics.prediction_accuracy
        != comparison["proposed"].run_metrics.prediction_accuracy
        for comparison in comparisons
    )
    assert all(
        comparison["variant"].run_metrics.final_parameter_value_count
        < comparison["proposed"].run_metrics.final_parameter_value_count
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
        CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME,
        OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME,
        WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME,
        INDEPENDENT_MODELS_CONFIGURATION_NAME,
        "feddrift",
        "fedavg",
        FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME,
        SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME,
        SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME,
        LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME,
    )
    assert INDEPENDENT_MODELS_CONFIGURATION_NAME == "fedsda-independent-models"
    assert WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME == (
        "fedsda-without-concept-specific-adapter"
    )
    assert CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME == "fedsda-classwise-adwin-detection"
    assert OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME == (
        "fedsda-overall-loss-adwin-detection"
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
        PROPOSED_FEDSDA_CONFIGURATION_NAME,
        "feddrift",
        "fedavg",
        # 旧実装に対応する構成が無いので、実旧とは照合しない。
        FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME,
        SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME,
        SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME,
        LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME,
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


# モデル構造を変える構成は、mnist2（784特徴・10クラス）でも、全体runを実旧と照合する（了承済みの照合の基準）。
# 条件は、提案構成のmnist2の照合（test_fedsda_stream_protocol_run.py）と同じ2つ。
MODEL_ARCHITECTURE_VARIANT_NAMES = (
    WITHOUT_CONCEPT_SPECIFIC_ADAPTER_CONFIGURATION_NAME,
    INDEPENDENT_MODELS_CONFIGURATION_NAME,
)
# 構成と条件ごとの、新しいモデルの登録があったか。
MNIST2_NEW_MODEL_REGISTRATION_BY_CONDITION = {}
# 条件の最後の要素（提案構成で通る経路の集合）は、この構成には当てはまらないので、外す。
MNIST2_WHOLE_RUN_CONDITIONS = tuple(
    whole_run_condition[:-1]
    for whole_run_condition in whole_run_test_module.OTHER_DATASET_WHOLE_RUN_CONDITIONS
    if whole_run_condition[0] == "mnist2"
)


@pytest.mark.parametrize("whole_run_condition", MNIST2_WHOLE_RUN_CONDITIONS)
@pytest.mark.parametrize("configuration_name", MODEL_ARCHITECTURE_VARIANT_NAMES)
@pytest.mark.slow
def test_model_architecture_variant_matches_real_legacy_whole_run_on_mnist2(
    configuration_name, whole_run_condition, monkeypatch, valid_run_settings_mapping
):
    """モデル構造を変える構成の、mnist2での全体runの全状態が、実旧の、同じ指定の全体runと一致する。"""
    assert len(MNIST2_WHOLE_RUN_CONDITIONS) == 2
    (
        dataset_name,
        random_seed,
        client_count,
        per_client_sample_count,
        aggregation_interval,
        minimum_change_gap,
        concept_change_probability,
        update_interval,
    ) = whole_run_condition
    configuration_variant = CONFIGURATION_VARIANTS[configuration_name]
    # 新旧の組立てが読む、testの定数（MNISTの小さい条件の、隠れ層の幅）。
    for test_module in (run_client_test_module, whole_run_test_module):
        monkeypatch.setattr(
            test_module, "HIDDEN_LAYER_WIDTHS", whole_run_test_module.MNIST_HIDDEN_LAYER_WIDTHS
        )
    execution_settings = whole_run_test_module.make_execution_settings(
        random_seed=random_seed,
        client_count=client_count,
        per_client_sample_count=per_client_sample_count,
        aggregation_interval=aggregation_interval,
        minimum_change_gap=minimum_change_gap,
        concept_change_probability=concept_change_probability,
        dataset_name=dataset_name,
    )
    if not getattr(configuration_variant, "legacy_models_share_feature_extractor", True):
        observe_legacy_duplicate_downloads(monkeypatch)
    legacy_run = whole_run_test_module.run_real_legacy_whole_run(
        monkeypatch=monkeypatch,
        execution_settings=execution_settings,
        update_interval=update_interval,
        legacy_mode_name=configuration_variant.legacy_mode_name,
        legacy_setting_overrides=configuration_variant.legacy_setting_overrides,
    )
    run_result, participants, run_random_sources = (
        whole_run_test_module.execute_stream_protocol_run_with_factory(
            execution_settings=execution_settings,
            run_participant_settings=configuration_variant.modify_run_participant_settings(
                whole_run_test_module.make_run_participant_settings(
                    valid_run_settings_mapping, update_interval=update_interval
                )
            ),
        )
    )
    whole_run_test_module.assert_whole_run_matches_legacy(
        run_result=run_result,
        participants=participants,
        run_random_sources=run_random_sources,
        legacy_run=legacy_run,
    )
    # 784特徴で実行している。
    assert all(
        len(observed_sample.feature_values) == 784
        for observed_client_stream in run_result.observed_client_streams
        for observed_sample in observed_client_stream.observed_samples
    )
    MNIST2_NEW_MODEL_REGISTRATION_BY_CONDITION[(configuration_name, whole_run_condition)] = any(
        synchronization.registered_client_models
        for synchronization in participants.server_operations.snapshot_server_round_synchronizations()
    )


@pytest.mark.parametrize("configuration_name", MODEL_ARCHITECTURE_VARIANT_NAMES)
def test_model_architecture_variant_registers_new_models_on_mnist2(configuration_name):
    """mnist2の照合が、新しいモデルの登録（集約と配布に、複数のモデルが関わる）まで進む条件を含む。"""
    registrations = [
        MNIST2_NEW_MODEL_REGISTRATION_BY_CONDITION[(configuration_name, whole_run_condition)]
        for whole_run_condition in MNIST2_WHOLE_RUN_CONDITIONS
        if (configuration_name, whole_run_condition) in MNIST2_NEW_MODEL_REGISTRATION_BY_CONDITION
    ]
    if len(registrations) < len(MNIST2_WHOLE_RUN_CONDITIONS):
        pytest.skip("mnist2の条件の全部を実行したときだけ確かめる")
    assert any(registrations)


@pytest.mark.parametrize("configuration_name", [INDEPENDENT_MODELS_CONFIGURATION_NAME])
def test_independent_models_take_their_own_path(configuration_name):
    """共有部を持たない構成は、保有モデルが、パラメータを共有せず、共有部の計算も無い。実旧のモデルも、共有部を持たない。"""
    comparisons = collect_variant_comparisons(configuration_name)
    multiple_held_model_client_count = 0
    for comparison in comparisons:
        variant_comparison = comparison["variant"]
        run_metrics = variant_comparison.run_metrics
        # 共有部の計算は、無い（学習・推論の標本数、optimizerの更新、積和演算）。
        model_computation_counts = run_metrics.model_computation_counts
        assert model_computation_counts.shared_part_training_example_count == 0
        assert model_computation_counts.shared_part_inference_example_count == 0
        assert model_computation_counts.shared_parameter_optimizer_step_count == 0
        assert model_computation_counts.concept_specific_parameter_optimizer_step_count > 0
        # 集約後の、予測の重みの再較正は、行わない。
        assert run_metrics.fixed_share_weight_recalibration_replayed_sample_count == 0
        assert run_metrics.adahedge_weight_recalibration_replayed_sample_count == 0
        for run_client, legacy_client in zip(
            variant_comparison.participants.client_operations,
            variant_comparison.legacy_run["legacy_clients"],
            strict=True,
        ):
            owners = run_client.owners
            held_states = owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
            multiple_held_model_client_count += len(held_states) > 1
            # 共有部のoptimizerは、無い。
            assert (
                owners.shared_parameter_optimizer_state_holder.held_shared_parameter_optimizer_state.parameter_optimizer_if_any
                is None
            )
            parameter_ids = set()
            for held_state in held_states:
                classifier = held_state.classifier
                assert list(classifier.feature_extractor.parameters()) == []
                # 全パラメータが、このモデルだけのもの（ほかの保有モデルと、同じParameterを持たない）。
                classifier_parameter_ids = {id(parameter) for parameter in classifier.parameters()}
                assert classifier_parameter_ids
                assert not classifier_parameter_ids & parameter_ids
                parameter_ids |= classifier_parameter_ids
                # 実旧のモデルは、層を1列に持ち、パラメータの数が同じ。
                legacy_model = legacy_client.models[held_state.model_id]
                assert not hasattr(legacy_model, "backbone")
                assert sum(parameter.numel() for parameter in classifier.parameters()) == sum(
                    parameter.numel() for parameter in legacy_model.parameters()
                )
            assert not hasattr(legacy_client, "_shared_backbone")
    duplicate_counts = [
        getattr(
            comparison["variant"].legacy_run["legacy_server"],
            "_observed_duplicate_download_counts",
            [0, 0, 0],
        )
        for comparison in comparisons
    ]
    assert any(counts[0] > 0 for counts in duplicate_counts)
    assert any(counts == [0, 0, 0] for counts in duplicate_counts)
    assert multiple_held_model_client_count > 0
    # 同じ条件の提案構成は、共有部の計算があり、結果が違う。最終のパラメータの数は、複数のグローバルモデルが残る条件で、
    # 提案構成（共有部を1回だけ数える）より多い。
    assert all(
        comparison[
            "proposed"
        ].run_metrics.model_computation_counts.shared_part_training_example_count
        > 0
        for comparison in comparisons
    )
    assert any(
        comparison["variant"].run_metrics.prediction_accuracy
        != comparison["proposed"].run_metrics.prediction_accuracy
        for comparison in comparisons
    )
    assert any(
        comparison["variant"].run_metrics.final_global_model_count > 1
        and comparison["variant"].run_metrics.final_parameter_value_count
        > comparison["proposed"].run_metrics.final_parameter_value_count
        for comparison in comparisons
    )


def test_independent_models_require_sequential_training_and_no_weight_recalibration():
    """共有部を持たないモデル構造は、モデルごとに順に学習する方式・再較正をしない方針との組でだけ、設定を作れる。"""
    run_settings = build_experiment_run_settings(
        configuration_name=INDEPENDENT_MODELS_CONFIGURATION_NAME,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=0,
            client_count=3,
            per_client_sample_count=200,
            server_aggregation_interval_per_client_samples=50,
        ),
    )
    run_participant_settings = run_settings.run_participant_settings
    run_client_settings = run_participant_settings.run_client_settings
    assert run_participant_settings.model_architecture_settings.model_architecture_name == (
        "independent_models_without_shared_backbone"
    )
    assert run_client_settings.local_training_settings.local_model_parameter_update_strategy == (
        "sequential_per_model_training"
    )
    for mismatched_settings_fields, rejected_parameter_name in (
        (
            dict(
                run_client_settings=replace(
                    run_client_settings,
                    local_training_settings=replace(
                        run_client_settings.local_training_settings,
                        local_model_parameter_update_strategy=(
                            "joint_backbone_adapter_and_head_training"
                        ),
                    ),
                )
            ),
            "local_model_parameter_update_strategy",
        ),
        (
            dict(
                model_architecture_settings=replace(
                    run_participant_settings.model_architecture_settings,
                    model_architecture_name="shared_backbone_residual_adapter",
                )
            ),
            "local_model_parameter_update_strategy",
        ),
        (
            dict(
                run_client_settings=replace(
                    run_client_settings,
                    prediction_combination_settings=replace(
                        run_client_settings.prediction_combination_settings,
                        prediction_weight_recalibration_after_aggregation_policy=(
                            "recompute_buffer_losses_and_replay_weight_updates"
                        ),
                    ),
                )
            ),
            "prediction_weight_recalibration_after_aggregation_policy",
        ),
    ):
        with pytest.raises(RunSettingsValidationError) as error_information:
            replace(run_participant_settings, **mismatched_settings_fields)
        assert error_information.value.configuration_parameter_name == rejected_parameter_name


def test_parallel_scope_keeps_tests_of_one_configuration_in_one_process(request):
    """並列の実行での組が、このファイルの、同じ構成の名前を持つtestの全部で、同じ組になる。"""
    from conftest import select_parallel_test_scope

    scopes_by_configuration_name = {}
    for test_item in request.session.items:
        if test_item.path.name != "test_fedsda_configuration_variants.py":
            continue
        call_specification = getattr(test_item, "callspec", None)
        test_scope = select_parallel_test_scope(
            test_node_id=test_item.nodeid, configuration_names=EXPERIMENT_CONFIGURATION_NAMES
        )
        if call_specification is None or "configuration_name" not in call_specification.params:
            assert "::" not in test_scope, test_item.nodeid
            continue
        configuration_name = call_specification.params["configuration_name"]
        assert "::configuration-group-" in test_scope, test_item.nodeid
        scopes_by_configuration_name.setdefault(configuration_name, set()).add(test_scope)
    assert all(len(scopes) == 1 for scopes in scopes_by_configuration_name.values())
