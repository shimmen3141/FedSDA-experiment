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
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
    PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME,
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
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
        assert variant_metrics.prediction_weight_recalibration_replayed_sample_count == 0
        assert variant_metrics.global_diagnostic_recalibration_replayed_sample_count == 0
        assert (
            variant_metrics.prediction_diagnostic_metrics.global_diagnostic_aggregation_recalibration_count
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
        comparison["proposed"].run_metrics.prediction_weight_recalibration_replayed_sample_count > 0
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


def test_configuration_names_build_run_settings_that_differ_only_in_the_changed_element():
    """掃引のコマンドの構成の名前から作るrun設定は、提案構成と、変える要素だけが違う。"""
    assert EXPERIMENT_CONFIGURATION_NAMES == (
        PROPOSED_FEDSDA_CONFIGURATION_NAME,
        PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME,
    )
    assert PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME == (
        "fedsda-without-prediction-weight-recalibration"
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
        # testの側の変更（実旧と照合したもの）を、提案構成の設定へ当てると、同じ設定になる。
        assert run_settings == replace(
            proposed_run_settings,
            run_participant_settings=configuration_variant.modify_run_participant_settings(
                proposed_run_settings.run_participant_settings
            ),
        )
