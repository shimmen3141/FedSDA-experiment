"""クロス評価で、配布するモデルを使う扱い（`cross_evaluation_model_transfer_policy`）の通信量と、状態の不変。

新しい扱いは、状態の更新の順を変えず、通信量の記録だけを、次のプロトコルに合わせる: 集約の後、クロス評価より前に、
全clientへ全グローバルモデルを配布し（clientは手元の保有モデルをまだ置き換えない）、クロス評価ではメッセージだけを
送り、統合が起きたら、ID対応と、値が変わった代表のモデルの概念固有部（と、変わったなら共有部）を送り直す。
"""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from behavior_golden_cases import build_case_settings

import federated_learning_experiments.runtime.server_round_synchronization as round_module
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_record_tables import (
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.runtime.global_model_distribution import (
    record_global_model_distribution_volume,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    split_shared_and_concept_specific_parameters,
)
from federated_learning_experiments.runtime.server_round_synchronization import (
    _record_consolidated_model_resend_volume,
)

SEND = "send_candidate_models_for_cross_evaluation"
REUSE = "reuse_distributed_models_for_cross_evaluation"
# クロス評価と統合が起きる、goldenの大きい条件のケース（版7。通信の扱いは、旧の扱いに明示される）。
_CASE_NAME = "fedsda-sine2-adoption-without-validation"


def with_policy(run_settings, policy):
    return replace(
        run_settings,
        run_participant_settings=replace(
            run_settings.run_participant_settings, cross_evaluation_model_transfer_policy=policy
        ),
    )


def byte_size(parameter_snapshot):
    return sum(value.numel() * value.element_size() for value in parameter_snapshot.values())


def run_with_old_policy_and_expected_new_volume(run_settings):
    """旧の扱いで実行し、ラウンドごとの記録から、新しい扱いで記録されるはずの通信量を計算する。"""
    expected = dict(downloaded_byte_count=0, downloaded_model_count=0)
    observed = dict(cross_evaluation_rounds=0, consolidation_rounds=0, shared_resend_rounds=0)
    state = {}
    original_cross = round_module.cross_evaluate_global_models
    original_distribute = round_module.distribute_global_models_to_clients

    def cross(**arguments):
        repository = arguments["global_model_repository"]
        client_count = len(arguments["run_clients"])
        # 先に配布するもの（集約の後の、その時点の全グローバルモデル）。
        early = CommunicationVolumeRecordStore()
        record_global_model_distribution_volume(
            communication_volume_record_store=early,
            global_model_repository=repository,
            client_count=client_count,
        )
        early_snapshot = early.get_state_snapshot()
        first_model_id = repository.global_model_ids[0]
        state["early_shared"] = {
            name: value.clone()
            for name, value in split_shared_and_concept_specific_parameters(
                parameter_snapshot=repository.get_global_model_parameters(model_id=first_model_id)
            )[0].items()
        }
        store = arguments["communication_volume_record_store"]
        before = store.get_state_snapshot()
        result = original_cross(**arguments)
        after = store.get_state_snapshot()
        # 旧の扱いのクロス評価の、モデルとパラメータの転送を引き、先の配布を足す（メッセージは同じ）。
        expected["downloaded_byte_count"] += early_snapshot.downloaded_byte_count - (
            after.downloaded_byte_count - before.downloaded_byte_count
        )
        expected["downloaded_model_count"] += early_snapshot.downloaded_model_count - (
            after.downloaded_model_count - before.downloaded_model_count
        )
        state["cross_evaluated"] = True
        observed["cross_evaluation_rounds"] += 1
        return result

    def distribute(**arguments):
        store = arguments["communication_volume_record_store"]
        before = store.get_state_snapshot()
        result = original_distribute(**arguments)
        after = store.get_state_snapshot()
        if state.pop("cross_evaluated", False):
            # 旧の扱いの、統合の後の配布を引き、送り直しを足す（ID対応のメッセージは、どちらも統合のときだけ）。
            expected["downloaded_byte_count"] -= (
                after.downloaded_byte_count - before.downloaded_byte_count
            )
            expected["downloaded_model_count"] -= (
                after.downloaded_model_count - before.downloaded_model_count
            )
            mapping = arguments["model_id_mapping"]
            if mapping:
                observed["consolidation_rounds"] += 1
                repository = arguments["global_model_repository"]
                client_count = len(arguments["run_clients"])
                representatives = {d for s, d in mapping.items() if s != d}
                for model_id in representatives:
                    expected["downloaded_byte_count"] += client_count * byte_size(
                        split_shared_and_concept_specific_parameters(
                            parameter_snapshot=repository.get_global_model_parameters(
                                model_id=model_id
                            )
                        )[1]
                    )
                expected["downloaded_model_count"] += client_count * len(representatives)
                shared_after = split_shared_and_concept_specific_parameters(
                    parameter_snapshot=repository.get_global_model_parameters(
                        model_id=repository.global_model_ids[0]
                    )
                )[0]
                if any(
                    not torch.equal(shared_after[name], state["early_shared"][name])
                    for name in shared_after
                ):
                    observed["shared_resend_rounds"] += 1
                    expected["downloaded_byte_count"] += client_count * byte_size(shared_after)
        return result

    with (
        patch.object(round_module, "cross_evaluate_global_models", cross),
        patch.object(round_module, "distribute_global_models_to_clients", distribute),
    ):
        run = execute_fedsda_experiment_run(run_settings=with_policy(run_settings, SEND))
    volume = run.run_metrics.communication_volume
    expected["downloaded_byte_count"] += volume.downloaded_byte_count
    expected["downloaded_model_count"] += volume.downloaded_model_count
    return run, expected, observed


def assert_runs_match_except_communication_volume(first_run, second_run):
    first_tables = build_fedsda_run_record_tables(experiment_run=first_run)
    second_tables = build_fedsda_run_record_tables(experiment_run=second_run)
    assert first_tables.keys() == second_tables.keys()
    for name, column in first_tables.items():
        assert np.array_equal(column, second_tables[name], equal_nan=column.dtype.kind == "f"), name
    assert replace(first_run.run_metrics, communication_volume=None) == replace(
        second_run.run_metrics, communication_volume=None
    )
    first_repository = (
        first_run.measured_run.participants.server_operations.owners.global_model_repository
    )
    second_repository = (
        second_run.measured_run.participants.server_operations.owners.global_model_repository
    )
    assert first_repository.global_model_ids == second_repository.global_model_ids
    for model_id in first_repository.global_model_ids:
        first_parameters = first_repository.get_global_model_parameters(model_id=model_id)
        second_parameters = second_repository.get_global_model_parameters(model_id=model_id)
        assert first_parameters.keys() == second_parameters.keys()
        for name in first_parameters:
            assert torch.equal(first_parameters[name], second_parameters[name]), (model_id, name)


@pytest.mark.slow
def test_reusing_distributed_models_changes_only_the_communication_volume():
    """2つの扱いで、通信量以外の全部が同じ。新しい扱いの下りは、ラウンドごとの記録から計算した値と一致する。"""
    run_settings = build_case_settings(_CASE_NAME, 7)
    torch.manual_seed(0)
    old_run, expected, observed = run_with_old_policy_and_expected_new_volume(run_settings)
    torch.manual_seed(0)
    new_run = execute_fedsda_experiment_run(run_settings=with_policy(run_settings, REUSE))
    # 比較が空にならない（クロス評価と統合が起きる条件）。
    assert observed["cross_evaluation_rounds"] > 0 and observed["consolidation_rounds"] > 0
    assert_runs_match_except_communication_volume(old_run, new_run)
    old_volume = old_run.run_metrics.communication_volume
    new_volume = new_run.run_metrics.communication_volume
    assert new_volume.downloaded_byte_count == expected["downloaded_byte_count"]
    assert new_volume.downloaded_model_count == expected["downloaded_model_count"]
    assert new_volume.downloaded_byte_count < old_volume.downloaded_byte_count
    # 上りと、メッセージの数は変わらない。
    for field_name in (
        "uploaded_model_count",
        "uploaded_message_count",
        "uploaded_parameter_value_count",
        "uploaded_byte_count",
        "downloaded_message_count",
    ):
        assert getattr(new_volume, field_name) == getattr(old_volume, field_name), field_name


@pytest.mark.parametrize("policy", [SEND, REUSE])
def test_rounds_without_cross_evaluation_record_the_same_volume(policy):
    """クロス評価が起きない条件では、2つの扱いの通信量が同じ。"""
    run_settings = build_case_settings("fedsda", 7)
    send_run = execute_fedsda_experiment_run(run_settings=with_policy(run_settings, SEND))
    tables = build_fedsda_run_record_tables(experiment_run=send_run)
    assert all(
        len(column) == 0
        for name, column in tables.items()
        if name.startswith("client_cross_evaluation.")
    ) or not any(name.startswith("client_cross_evaluation.") for name in tables)
    run = execute_fedsda_experiment_run(run_settings=with_policy(run_settings, policy))
    assert run.run_metrics.communication_volume == send_run.run_metrics.communication_volume


class _Repository:
    def __init__(self, parameters_by_model_id):
        self._parameters_by_model_id = parameters_by_model_id
        self.global_model_ids = tuple(parameters_by_model_id)

    def get_global_model_parameters(self, *, model_id):
        return self._parameters_by_model_id[model_id]


def _parameters(shared_value, concept_value):
    return {
        "feature_extractor.hidden_layers.0.weight": torch.full((2, 3), shared_value),
        "classification_layer.weight": torch.full((1, 2), concept_value),
    }


@pytest.mark.parametrize("shared_changed", [False, True])
def test_resend_counts_representatives_and_the_shared_part_only_when_it_changed(shared_changed):
    """送り直しは、代表の概念固有部×client数。共有部は、先に送った値とビット単位で違うときだけ数える。"""
    repository = _Repository({0: _parameters(1.0 + shared_changed, 5.0), 3: _parameters(1.0, 7.0)})
    store = CommunicationVolumeRecordStore()
    _record_consolidated_model_resend_volume(
        communication_volume_record_store=store,
        global_model_repository=repository,
        model_id_mapping={0: 0, 2: 0, 3: 3},
        early_distributed_shared_parameters={
            "feature_extractor.hidden_layers.0.weight": torch.full((2, 3), 1.0)
        },
        client_count=4,
    )
    snapshot = store.get_state_snapshot()
    concept_bytes = 2 * 4
    shared_bytes = 6 * 4
    assert snapshot.downloaded_byte_count == 4 * (concept_bytes + shared_changed * shared_bytes)
    assert snapshot.downloaded_model_count == 4
    assert snapshot.downloaded_message_count == 4
    assert snapshot.uploaded_byte_count == 0


def test_resend_records_nothing_without_consolidation():
    store = CommunicationVolumeRecordStore()
    _record_consolidated_model_resend_volume(
        communication_volume_record_store=store,
        global_model_repository=_Repository({0: _parameters(1.0, 5.0)}),
        model_id_mapping={},
        early_distributed_shared_parameters={
            "feature_extractor.hidden_layers.0.weight": torch.full((2, 3), 1.0)
        },
        client_count=4,
    )
    assert store.get_state_snapshot() == CommunicationVolumeRecordStore().get_state_snapshot()


@pytest.mark.parametrize("invalid_value", ["reuse", None, True, 1])
def test_invalid_cross_evaluation_model_transfer_policy_is_rejected(invalid_value):
    run_settings = build_case_settings("fedsda", 7)
    with pytest.raises(RunSettingsValidationError) as error_information:
        with_policy(run_settings, invalid_value)
    assert (
        error_information.value.configuration_parameter_name
        == "cross_evaluation_model_transfer_policy"
    )


def test_default_and_all_fedsda_configurations_reuse_distributed_models():
    """最終構成の既定と、FedSDAの全部の構成は、配布するモデルを使う扱い。"""
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
        build_experiment_run_settings,
    )

    conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=0,
        client_count=2,
        per_client_sample_count=40,
        server_aggregation_interval_per_client_samples=10,
    )
    fedsda_configuration_count = 0
    for configuration_name in EXPERIMENT_CONFIGURATION_NAMES:
        run_settings = build_experiment_run_settings(
            configuration_name=configuration_name, experiment_run_conditions=conditions
        )
        if type(run_settings) is not FedsdaRunSettings:
            continue
        fedsda_configuration_count += 1
        assert (
            run_settings.run_participant_settings.cross_evaluation_model_transfer_policy == REUSE
        ), configuration_name
    assert fedsda_configuration_count > 1
