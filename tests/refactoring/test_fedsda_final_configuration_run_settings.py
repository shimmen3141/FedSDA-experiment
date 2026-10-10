"""最終構成の既定値を持つ完全なrun設定を、旧の設定の値と照合する。"""

import json
from dataclasses import replace

import pytest
from test_fedsda_run_metric_derivation import legacy_regression, make_golden_condition_settings
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_drift_experiment import config
from federated_drift_experiment.data.specs import DATASET_SPECS
from federated_drift_experiment.drift_detectors.e_detector import BoundedMeanEDetector
from federated_drift_experiment.experiment_spec.configuration import (
    ExperimentConfiguration,
    temporary_config,
)
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.data.dataset_definitions import DEFINED_DATASET_NAMES
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
    build_final_configuration_fedsda_run_settings,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings

GOLDEN_DATASET_NAMES = tuple(legacy_regression.CASES)
# 旧の回帰testの条件が、最終構成の既定値から変えている、旧の設定の項目。
REDUCED_LEGACY_SETTING_NAMES_BY_DATASET_NAME = {
    "sine2": {
        "N_CLIENTS",
        "TOTAL_DATA_POINTS",
        "PRETRAIN_SAMPLES",
        "MIN_STABLE_PERIOD",
        "DRIFT_PROB",
        "STABLE_WINDOW",
    },
    "sea2": {
        "N_CLIENTS",
        "TOTAL_DATA_POINTS",
        "PRETRAIN_SAMPLES",
        "MIN_STABLE_PERIOD",
        "DRIFT_PROB",
        "STABLE_WINDOW",
    },
    "mnist2": {
        "N_CLIENTS",
        "TOTAL_DATA_POINTS",
        "PRETRAIN_SAMPLES",
        "PRETRAIN_EPOCHS",
        "MIN_STABLE_PERIOD",
        "DRIFT_PROB",
        "STABLE_WINDOW",
    },
}


def read_legacy_final_configuration_values(dataset_name, legacy_overrides=None):
    """旧の設定を、最終構成の固定設定（と、渡した上書き）で有効化して、全部の値を読む。

    戻り値: (旧の設定の値の辞書, 旧のdatasetの定義の隠れ層の幅)。旧のdatasetの定義の学習率は、旧と同じく、
    通常の学習率と新規モデルの学習率の両方へ優先させる。
    """
    configuration = ExperimentConfiguration(
        mode=legacy_regression.MODE,
        dataset=dataset_name,
        seed=0,
        concept_schedule="random",
        series="final_configuration_settings",
        sweep_parameter=None,
        sweep_value=None,
        parameters=(),
        algorithm=legacy_regression.ALGORITHM,
    )
    with temporary_config(legacy_overrides or {}), configuration.activated():
        legacy_values = {
            setting_name: getattr(config, setting_name)
            for setting_name in dir(config)
            if setting_name.isupper()
        }
        legacy_dataset_spec = config.dataset_spec()
    if legacy_dataset_spec.learning_rate is not None:
        legacy_values = legacy_values | dict(
            BASE_LR=legacy_dataset_spec.learning_rate,
            NEW_MODEL_LR=legacy_dataset_spec.learning_rate,
        )
    return legacy_values, tuple(legacy_dataset_spec.hidden_dims)


def make_run_settings_from_legacy_values(
    *, dataset_name, legacy_values, hidden_layer_widths, model_consolidation_settings
):
    """旧の設定の値を、既存のtestの対応（goldenを再現する設定を作るもの）で、完全なrun設定へ写す。"""
    with pytest.MonkeyPatch.context() as monkeypatch:
        execution_settings, run_participant_settings, run_metric_settings = (
            make_golden_condition_settings(
                dataset_name=dataset_name,
                legacy_values=legacy_values,
                hidden_layer_widths=hidden_layer_widths,
                monkeypatch=monkeypatch,
            )
        )
    return FedsdaRunSettings(
        execution_settings=execution_settings,
        run_participant_settings=run_participant_settings,
        model_consolidation_settings=model_consolidation_settings,
        run_metric_settings=run_metric_settings,
    )


def make_experiment_run_conditions(dataset_name, legacy_values):
    return ExperimentRunConditions(
        dataset_name=dataset_name,
        random_seed=0,
        client_count=legacy_values["N_CLIENTS"],
        per_client_sample_count=legacy_values["TOTAL_DATA_POINTS"],
        server_aggregation_interval_per_client_samples=legacy_values["AGGREGATION_INTERVAL"],
    )


@pytest.mark.parametrize("dataset_name", DEFINED_DATASET_NAMES)
def test_final_configuration_defaults_match_legacy_configuration(
    dataset_name, valid_run_settings_mapping
):
    """既定値が、旧の設定の初期値に最終構成の固定設定を重ねた値と、全部の項目で一致する（全dataset）。"""
    legacy_values, hidden_layer_widths = read_legacy_final_configuration_values(dataset_name)
    expected_run_settings = make_run_settings_from_legacy_values(
        dataset_name=dataset_name,
        legacy_values=legacy_values,
        hidden_layer_widths=hidden_layer_widths,
        model_consolidation_settings=valid_run_settings_mapping["model_consolidation_settings"],
    )
    run_settings = build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=make_experiment_run_conditions(dataset_name, legacy_values)
    )
    assert type(run_settings) is FedsdaRunSettings
    assert run_settings == expected_run_settings
    # 項目ごとの違いが分かるように、保存用の辞書でも比べる。
    assert convert_settings_to_plain_mapping(run_settings) == convert_settings_to_plain_mapping(
        expected_run_settings
    )
    # 既存のtestの対応が、旧の設定からではなく、決めた値で渡している項目を、旧の値と照合する。
    run_client_settings = run_settings.run_participant_settings.run_client_settings
    assert (
        run_client_settings.loss_monitor_betting_fractions == BoundedMeanEDetector.DEFAULT_LAMBDAS
    )
    assert (
        run_client_settings.local_training_schedule_settings.joint_update_iterations_per_training_request
        == legacy_values["UPDATES_PER_SAMPLE"]
    )
    assert (legacy_values["OPTIMIZER"], legacy_values["AMSGRAD"]) == ("adam", True)
    assert run_client_settings.parameter_optimizer_settings.adam_variant == "amsgrad"
    assert run_client_settings.rebuilt_model_parameter_optimizer_settings.adam_variant == "amsgrad"
    assert (legacy_values["NEW_MODEL_TRAINING"], legacy_values["NEW_MODEL_INITIALIZATION"]) == (
        "early_stopping",
        "best_candidate",
    )
    assert legacy_values["CONCEPT_SCHEDULE"] == "random"
    # datasetごとのモデルの既定は、旧のdatasetの定義。
    legacy_dataset_spec = DATASET_SPECS[dataset_name]
    assert run_settings.run_participant_settings.hidden_layer_widths == tuple(
        legacy_dataset_spec.hidden_dims
    )
    expected_learning_rate = (
        0.01 if legacy_dataset_spec.learning_rate is None else legacy_dataset_spec.learning_rate
    )
    assert run_client_settings.parameter_optimizer_settings.learning_rate == expected_learning_rate
    assert (
        run_client_settings.rebuilt_model_parameter_optimizer_settings.learning_rate
        == expected_learning_rate
    )


def test_mnist_and_synthetic_defaults_differ_only_in_model_defaults():
    """datasetで変わる既定は、隠れ層の幅と学習率だけ（合成データどうし、MNISTどうしは、同じ）。"""

    def build(dataset_name):
        run_settings = build_final_configuration_fedsda_run_settings(
            experiment_run_conditions=ExperimentRunConditions(
                dataset_name=dataset_name,
                random_seed=0,
                client_count=10,
                per_client_sample_count=5000,
                server_aggregation_interval_per_client_samples=50,
            )
        )
        # dataset名だけを、同じ値へそろえて比べる。
        return replace(
            run_settings,
            execution_settings=replace(
                run_settings.execution_settings,
                experiment_run_conditions=replace(
                    run_settings.execution_settings.experiment_run_conditions, dataset_name="sine2"
                ),
            ),
        )

    assert build("sea2") == build("sine2") == build("sea4") == build("circle2")
    assert build("mnist2") == build("mnist4")
    mnist_run_settings = build("mnist2")
    synthetic_run_settings = build("sine2")
    assert mnist_run_settings != synthetic_run_settings
    mnist_client_settings = mnist_run_settings.run_participant_settings.run_client_settings
    assert (
        replace(
            mnist_run_settings,
            run_participant_settings=replace(
                mnist_run_settings.run_participant_settings,
                hidden_layer_widths=(32, 32),
                run_client_settings=replace(
                    mnist_client_settings,
                    parameter_optimizer_settings=replace(
                        mnist_client_settings.parameter_optimizer_settings, learning_rate=0.01
                    ),
                    rebuilt_model_parameter_optimizer_settings=replace(
                        mnist_client_settings.rebuilt_model_parameter_optimizer_settings,
                        learning_rate=0.01,
                    ),
                ),
            ),
        )
        == synthetic_run_settings
    )


@pytest.mark.parametrize("dataset_name", GOLDEN_DATASET_NAMES)
def test_golden_conditions_are_the_defaults_with_only_the_reduced_items_replaced(
    dataset_name, valid_run_settings_mapping
):
    """goldenの条件は、既定値から、旧の回帰testが縮小した項目だけを置き換えた設定と、一致する。"""
    golden_overrides = {**legacy_regression.COMMON, **legacy_regression.CASES[dataset_name]}
    default_legacy_values, _ = read_legacy_final_configuration_values(dataset_name)
    golden_legacy_values, hidden_layer_widths = read_legacy_final_configuration_values(
        dataset_name, golden_overrides
    )
    # 旧の回帰testの条件が、既定値から変えているのは、決めた項目だけ。
    assert {
        setting_name
        for setting_name in default_legacy_values
        if default_legacy_values[setting_name] != golden_legacy_values[setting_name]
    } == REDUCED_LEGACY_SETTING_NAMES_BY_DATASET_NAME[dataset_name]
    # 既存の（goldenを再現する）設定。
    golden_run_settings = make_run_settings_from_legacy_values(
        dataset_name=dataset_name,
        legacy_values=golden_legacy_values,
        hidden_layer_widths=hidden_layer_widths,
        model_consolidation_settings=valid_run_settings_mapping["model_consolidation_settings"],
    )
    # 既定値から、縮小した項目だけを置き換える。
    default_run_settings = build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=make_experiment_run_conditions(dataset_name, golden_legacy_values)
    )
    reduced_run_settings = replace(
        default_run_settings,
        execution_settings=replace(
            default_run_settings.execution_settings,
            concept_schedule_settings=replace(
                default_run_settings.execution_settings.concept_schedule_settings,
                minimum_sample_index_gap_before_change_trial=golden_overrides["MIN_STABLE_PERIOD"],
                per_eligible_sample_concept_change_probability=golden_overrides["DRIFT_PROB"],
            ),
        ),
        run_participant_settings=replace(
            default_run_settings.run_participant_settings,
            initial_model_pretraining_settings=replace(
                default_run_settings.run_participant_settings.initial_model_pretraining_settings,
                pretraining_sample_count=golden_overrides["PRETRAIN_SAMPLES"],
                pretraining_epoch_count=golden_overrides["PRETRAIN_EPOCHS"],
            ),
        ),
        run_metric_settings=replace(
            default_run_settings.run_metric_settings,
            post_change_recovery_window_sample_count=golden_overrides["STABLE_WINDOW"],
        ),
    )
    assert reduced_run_settings == golden_run_settings
    assert reduced_run_settings != default_run_settings


def test_defaults_can_be_replaced_and_are_validated_again():
    """既定値の一部を置き換えた値を作れる。置き換えた後も、検証が行われる。"""
    run_settings = build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sea2",
            random_seed=3,
            client_count=4,
            per_client_sample_count=200,
            server_aggregation_interval_per_client_samples=20,
        )
    )
    conditions = run_settings.execution_settings.experiment_run_conditions
    assert (conditions.dataset_name, conditions.random_seed, conditions.client_count) == (
        "sea2",
        3,
        4,
    )
    replaced_run_settings = replace(
        run_settings,
        run_metric_settings=replace(
            run_settings.run_metric_settings, post_change_recovery_window_sample_count=7
        ),
    )
    assert replaced_run_settings.run_metric_settings.post_change_recovery_window_sample_count == 7
    assert replaced_run_settings.run_participant_settings is run_settings.run_participant_settings
    with pytest.raises(RunSettingsValidationError):
        replace(run_settings, run_metric_settings=None)
    # 呼出しごとに、等しい値を返す。
    assert run_settings == build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=conditions
    )


@pytest.mark.parametrize("invalid_conditions", [None, "sine2", {"dataset_name": "sine2"}])
def test_invalid_run_conditions_are_rejected(invalid_conditions):
    with pytest.raises((TypeError, RunSettingsValidationError)):
        build_final_configuration_fedsda_run_settings(experiment_run_conditions=invalid_conditions)
    with pytest.raises(TypeError):
        build_final_configuration_fedsda_run_settings(invalid_conditions)  # type: ignore[misc]


def test_final_configuration_settings_are_saved_as_a_complete_plain_mapping():
    """保存用の辞書が、全部の項目（実行条件、機能別の設定、束、統合、指標）を持ち、JSONにできる。"""
    run_settings = build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="mnist2",
            random_seed=1,
            client_count=10,
            per_client_sample_count=5000,
            server_aggregation_interval_per_client_samples=50,
        )
    )
    plain_mapping = convert_settings_to_plain_mapping(run_settings)
    assert json.loads(json.dumps(plain_mapping)) == plain_mapping
    assert plain_mapping["settings_type"] == "FedsdaRunSettings"
    assert list(plain_mapping) == [
        "settings_type",
        "execution_settings",
        "run_participant_settings",
        "model_consolidation_settings",
        "run_metric_settings",
    ]
    assert plain_mapping["execution_settings"]["experiment_run_conditions"] == {
        "settings_type": "ExperimentRunConditions",
        "dataset_name": "mnist2",
        "random_seed": 1,
        "client_count": 10,
        "per_client_sample_count": 5000,
        "server_aggregation_interval_per_client_samples": 50,
    }
    participant_mapping = plain_mapping["run_participant_settings"]
    assert participant_mapping["hidden_layer_widths"] == [1568]
    assert participant_mapping["run_client_settings"]["parameter_optimizer_settings"] == {
        "settings_type": "AdamParameterOptimizerSettings",
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "adam_variant": "amsgrad",
    }
    assert participant_mapping["run_client_settings"]["loss_monitor_betting_fractions"] == [
        0.05,
        0.1,
        0.2,
        0.4,
        0.8,
    ]
    # どれか1つの値が違えば、辞書も違う。
    other_run_settings = replace(
        run_settings,
        run_metric_settings=replace(
            run_settings.run_metric_settings, maximum_detection_delay_sample_count=99
        ),
    )
    assert convert_settings_to_plain_mapping(other_run_settings) != plain_mapping
