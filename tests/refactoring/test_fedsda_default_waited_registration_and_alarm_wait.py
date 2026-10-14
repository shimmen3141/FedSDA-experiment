"""FedSDAの既定を案Y（現行モデルとして待った後に登録する＋最小件数まで警報を待つ）にする（2026-10-14、ユーザー判断）。

ADWINの検出器の2構成は、待つ扱いと組み合わせられない（設定の検査で拒否される）ので、警報を即時に明示する。
"""

import behavior_golden_cases as golden

from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
    WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (
    REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
from federated_learning_experiments.sweep.experiment_configurations import (
    CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME,
    EXPERIMENT_CONFIGURATION_NAMES,
    OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME,
    build_experiment_run_settings,
)
from federated_learning_experiments.sweep.experiment_sweep_command import _build_argument_parser

ADWIN_CONFIGURATION_NAMES = (
    CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME,
    OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME,
)


def test_every_fedsda_configuration_uses_waited_registration_and_alarm_wait_by_default():
    """全部のFedSDAの構成の組立てが通り、登録は待った後に登録、警報は待つ（ADWINの2構成だけ即時）。"""
    fedsda_configuration_names = []
    for configuration_name in EXPERIMENT_CONFIGURATION_NAMES:
        case_run_settings = golden.build_case_settings(configuration_name, 3)
        default_run_settings = build_experiment_run_settings(
            configuration_name=configuration_name,
            experiment_run_conditions=case_run_settings.execution_settings.experiment_run_conditions,
        )
        if type(default_run_settings) is not FedsdaRunSettings:
            continue
        fedsda_configuration_names.append(configuration_name)
        client_settings = default_run_settings.run_participant_settings.run_client_settings
        assert (
            client_settings.pending_model_registration_settings.pending_model_registration_policy
            == REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME
        ), configuration_name
        expected_alarm_policy = (
            RAISE_ALARM_IMMEDIATELY_POLICY_NAME
            if configuration_name in ADWIN_CONFIGURATION_NAMES
            else WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME
        )
        assert (
            client_settings.loss_change_detection_settings.short_change_interval_alarm_policy
            == expected_alarm_policy
        ), configuration_name
    assert len(fedsda_configuration_names) == 11
    assert set(ADWIN_CONFIGURATION_NAMES) <= set(fedsda_configuration_names)


def test_sweep_command_arguments_are_only_extended_by_the_artifact_organization():
    """既定の変更で、実験のコマンドの引数は変えない（要求1.4）。後のspec（experiment-artifact-organization）で、
    `--study`・`--purpose`が、held-model-training-sample-recencyで`--fedsda-training-sample-selections`が増えた。"""
    argument_parser = _build_argument_parser()
    assert sorted(
        option_string
        for action in argument_parser._actions
        for option_string in action.option_strings
    ) == [
        "--aggregation-intervals",
        "--client-count",
        "--configurations",
        "--datasets",
        "--feddrift-loss-increase-thresholds",
        "--fedsda-training-sample-selections",
        "--help",
        "--label",
        "--output-directory",
        "--per-client-sample-count",
        "--print-plan",
        "--purpose",
        "--seeds",
        "--study",
        "--workers",
        "-h",
    ]
