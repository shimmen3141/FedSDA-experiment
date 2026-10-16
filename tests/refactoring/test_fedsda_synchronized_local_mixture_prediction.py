"""同期の時点のモデルの結合予測と、局所更新したモデルの結合予測を混ぜる設定
（`mixture_of_synchronized_and_local_parameters`）を確認する。

混ぜる重みのowner、設定、予測の関数と標本の処理への接続、手元の診断（実行時の差し替え）との一致、構成名を、
ここで確かめる。この構成はgoldenの事例にしない（`behavior_golden_cases.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`）。
"""

import math
from dataclasses import replace

import pytest
import torch

import federated_learning_experiments.runtime.observed_sample_processing as observed_sample_processing
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    combine_model_prediction_probabilities,
    convert_model_outputs_to_prediction_probabilities,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_and_local_prediction_mixture import (
    SynchronizedAndLocalPredictionMixture,
    true_class_loss_of_prediction_probabilities,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME,
    LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME,
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
    SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME,
    SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME,
    build_experiment_run_settings,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
)

# ---- 混ぜる重みのowner ----


def test_mixture_starts_from_equal_weights_and_mixes_in_order():
    mixture = SynchronizedAndLocalPredictionMixture(share_rate=1 / 30)
    assert mixture.snapshot_weights() == (0.5, 0.5)
    synchronized = torch.tensor([[0.8]], dtype=torch.float32)
    local = torch.tensor([[0.2]], dtype=torch.float32)
    mixed = mixture.mix_prediction_probabilities(
        synchronized_probabilities=synchronized, local_probabilities=local
    )
    assert torch.equal(mixed, synchronized * 0.5 + local * 0.5)


def test_mixture_update_follows_exponential_weights_and_shares():
    share = 1 / 30
    mixture = SynchronizedAndLocalPredictionMixture(share_rate=share)
    mixture.update_weights_after_loss_observation(synchronized_loss=0.1, local_loss=0.7)
    raw = (0.5 * math.exp(-2.0 * 0.1), 0.5 * math.exp(-2.0 * 0.7))
    normalized = (raw[0] / sum(raw), raw[1] / sum(raw))
    expected = (
        (1 - share) * normalized[0] + share * normalized[1],
        (1 - share) * normalized[1] + share * normalized[0],
    )
    assert mixture.snapshot_weights() == expected
    assert SynchronizedAndLocalPredictionMixture.LEARNING_RATE == 2.0
    # 分け合いで、片方が0にならない。
    for _ in range(200):
        mixture.update_weights_after_loss_observation(synchronized_loss=0.0, local_loss=1.0)
    weights = mixture.snapshot_weights()
    assert weights[1] > 0 and math.isclose(sum(weights), 1.0)


@pytest.mark.parametrize("share_rate", [0.0, 1.0, -0.1, float("nan"), float("inf"), 1])
def test_mixture_rejects_share_rates_out_of_range(share_rate):
    with pytest.raises((TypeError, ValueError)):
        SynchronizedAndLocalPredictionMixture(share_rate=share_rate)


@pytest.mark.parametrize(
    ("probabilities", "observed_class_id", "class_count", "expected"),
    [
        # 2値は、クラス1の確率だけを持つ。真のクラスが0なら、確率は1−v。損失は、確率をいったん作ってから1から引く。
        ([[0.3]], 1, 2, 1.0 - float(torch.tensor(0.3, dtype=torch.float32))),
        ([[0.3]], 0, 2, 1.0 - (1.0 - float(torch.tensor(0.3, dtype=torch.float32)))),
        ([[0.2, 0.5, 0.3]], 2, 3, 1.0 - float(torch.tensor(0.3, dtype=torch.float32))),
    ],
)
def test_true_class_loss_matches_hand_computed_values(
    probabilities, observed_class_id, class_count, expected
):
    loss = true_class_loss_of_prediction_probabilities(
        prediction_probabilities=torch.tensor(probabilities, dtype=torch.float32),
        observed_class_id=observed_class_id,
        class_count=class_count,
    )
    assert type(loss) is float
    assert loss == expected


# ---- runでの確認（診断との一致、検出力、設定、構成名） ----

MIXTURE_VALUE = "mixture_of_synchronized_and_local_parameters"
_CONDITIONS = ExperimentRunConditions(
    dataset_name="sine2",
    random_seed=1,
    client_count=3,
    per_client_sample_count=400,
    server_aggregation_interval_per_client_samples=50,
)


def _with_source_and_capacity(run_settings, parameter_source, capacity=None):
    participant = run_settings.run_participant_settings
    client = participant.run_client_settings
    prediction = replace(
        client.prediction_combination_settings, prediction_model_parameter_source=parameter_source
    )
    assignment = client.training_data_assignment_settings
    if capacity is not None:
        prediction = replace(
            prediction, fixed_share_weight_redistribution_time_scale_samples=capacity
        )
        assignment = replace(assignment, pending_assignment_buffer_capacity_samples=capacity)
    return replace(
        run_settings,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                client,
                prediction_combination_settings=prediction,
                training_data_assignment_settings=assignment,
            ),
        ),
    )


def _records(experiment_run):
    return tuple(
        client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        for client in experiment_run.measured_run.participants.client_operations
    )


def _diagnostic_true_class_loss(probabilities, class_id, class_count):
    """診断（local_sync_mixture_probe_2.py）の式の書き写し。実装の関数は使わない。"""
    values = probabilities.reshape(-1)
    if class_count == 2:
        probability = float(values[0]) if class_id == 1 else 1.0 - float(values[0])
    else:
        probability = float(values[class_id])
    return 1.0 - probability


def _diagnostic_predicted_class(probabilities, class_count):
    values = probabilities.reshape(-1)
    if class_count == 2:
        return int(float(values[0]) > 0.5)
    return int(values.argmax())


def _run_with_diagnostic_mixture(monkeypatch, run_settings, share_rate):
    """同期の時点の予測の構成で、予測の関数を包み、診断と同じ混ぜ方の予測の正否と重みを、clientごとに記録する。

    記録する正否は、(混ぜた予測の正否, 混ぜた予測かモデルごとの予測のどれかの正否)。
    """
    states_by_registry = {}
    original_predict = (
        observed_sample_processing.predict_observed_sample_and_update_prediction_weights
    )

    def predict_and_mix(**arguments):
        registry = arguments["held_model_training_state_registry"]
        observation = arguments["indexed_observation"]
        # 局所の確率は、元の予測（写しを使う）より前に、現在の分類器で計算する（診断と同じ順）。
        classifiers = {
            state.model_id: state.classifier
            for state in registry.snapshot_ordered_held_model_training_states()
        }
        model_ids = tuple(sorted(classifiers))
        class_count = classifiers[model_ids[0]].class_count
        with torch.no_grad():
            shared = classifiers[model_ids[0]].extract_shared_features(
                observation.training_sample.input_features
            )
            local_probabilities = convert_model_outputs_to_prediction_probabilities(
                model_outputs_by_model_id={
                    model_id: classifiers[model_id].forward_from_shared_features(shared)
                    for model_id in model_ids
                },
                class_count=class_count,
            )
        result = original_predict(**arguments)
        weights = result.prediction_weights_by_model_id
        local_combined = combine_model_prediction_probabilities(
            prediction_probabilities_by_model_id={m: local_probabilities[m] for m in weights},
            prediction_weights_by_model_id=weights,
            class_count=class_count,
        )
        sync_combined = result.combined_prediction_probabilities
        class_id = int(observation.training_sample.observed_class_labels.reshape(-1)[0].item())
        state = states_by_registry.setdefault(id(registry), {"rows": [], "weights": [0.5, 0.5]})
        w_sync, w_local = state["weights"]
        mixed = sync_combined * w_sync + local_combined * w_local
        mixed_correct = _diagnostic_predicted_class(mixed, class_count) == class_id
        any_model_correct = any(
            _diagnostic_predicted_class(probabilities, class_count) == class_id
            for probabilities in result.prediction_probabilities_by_model_id.values()
        )
        state["rows"].append((mixed_correct, mixed_correct or any_model_correct))
        losses = (
            _diagnostic_true_class_loss(sync_combined, class_id, class_count),
            _diagnostic_true_class_loss(local_combined, class_id, class_count),
        )
        updated = (w_sync * math.exp(-2.0 * losses[0]), w_local * math.exp(-2.0 * losses[1]))
        total = sum(updated)
        updated = tuple(value / total for value in updated)
        state["weights"] = [
            (1 - share_rate) * updated[0] + share_rate * updated[1],
            (1 - share_rate) * updated[1] + share_rate * updated[0],
        ]
        return result

    with monkeypatch.context() as patch:
        patch.setattr(
            observed_sample_processing,
            "predict_observed_sample_and_update_prediction_weights",
            predict_and_mix,
        )
        experiment_run = execute_fedsda_experiment_run(
            run_settings=_with_source_and_capacity(
                run_settings, "parameters_at_last_synchronization"
            )
        )
    return experiment_run, states_by_registry


_SYNC_SIDE_FIELDS = (
    "sample_index",
    "observed_concept_id",
    "observed_class_id",
    "maximum_weight_model_id",
    "maximum_prediction_weight",
    "effective_model_count",
    "maximum_weight_model_prediction_is_correct",
    "adahedge_weighted_prediction_is_correct",
    "true_concept_diagnostic_prediction_is_correct",
    "highest_confidence_model_prediction_is_correct",
    "fixed_share_weighted_prediction_is_correct",
    "fixed_share_maximum_weight_model_id",
    "fixed_share_effective_model_count",
)


@pytest.mark.parametrize(
    ("configuration_name", "capacity"),
    [
        (PROPOSED_FEDSDA_CONFIGURATION_NAME, None),
        (PROPOSED_FEDSDA_CONFIGURATION_NAME, 60),
        (ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME, None),
    ],
)
def test_mixture_prediction_matches_diagnostic_substitution(
    monkeypatch, configuration_name, capacity
):
    run_settings = _with_source_and_capacity(
        build_experiment_run_settings(
            configuration_name=configuration_name, experiment_run_conditions=_CONDITIONS
        ),
        MIXTURE_VALUE,
        capacity,
    )
    time_scale = run_settings.run_participant_settings.run_client_settings.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
    experiment_run = execute_fedsda_experiment_run(run_settings=run_settings)
    oracle_run, oracle_states = _run_with_diagnostic_mixture(
        monkeypatch, run_settings, 1.0 / time_scale
    )
    clients = experiment_run.measured_run.participants.client_operations
    oracle_clients = oracle_run.measured_run.participants.client_operations
    for client, oracle_client in zip(clients, oracle_clients, strict=True):
        records = client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        oracle_records = (
            oracle_client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        )
        state = oracle_states[id(oracle_client.owners.held_model_training_state_registry)]
        assert [
            (r.combined_prediction_is_correct, r.any_model_or_combined_prediction_is_correct)
            for r in records
        ] == state["rows"]
        # 同期の側の項目は、同期の時点の予測の構成（オラクルのrun）と同じ。
        assert [tuple(getattr(r, f) for f in _SYNC_SIDE_FIELDS) for r in records] == [
            tuple(getattr(r, f) for f in _SYNC_SIDE_FIELDS) for r in oracle_records
        ]
        # 混ぜる重みも、最後まで同じ。
        assert client.synchronized_and_local_prediction_mixture.snapshot_weights() == tuple(
            state["weights"]
        )
    # 検出力: 同期の時点の予測の構成とは、混ぜた予測の正否が少なくとも1標本で違う。
    assert [
        r.combined_prediction_is_correct
        for client_records in _records(experiment_run)
        for r in client_records
    ] != [
        r.combined_prediction_is_correct
        for client_records in _records(oracle_run)
        for r in client_records
    ]


def test_mixture_value_is_accepted_and_rejected_with_single_model_prediction():
    settings = build_experiment_run_settings(
        configuration_name=SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME,
        experiment_run_conditions=_CONDITIONS,
    )
    assert (
        settings.run_participant_settings.run_client_settings.prediction_combination_settings.prediction_model_parameter_source
        == MIXTURE_VALUE
    )
    with pytest.raises(RunSettingsValidationError):
        _with_source_and_capacity(
            build_experiment_run_settings(
                configuration_name=SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME,
                experiment_run_conditions=_CONDITIONS,
            ),
            MIXTURE_VALUE,
        )


def test_configuration_has_the_same_settings_as_proposed():
    """混ぜる予測は、2026-10-16に提案構成の既定になった。この構成は、提案構成と同じ設定（名前だけが違う）。

    局所の予測の構成（2026-10-15までの既定の予測）とは、予測に使うパラメータだけが違う。
    """
    proposed, mixture, local = (
        convert_settings_to_plain_mapping(
            build_experiment_run_settings(
                configuration_name=configuration_name, experiment_run_conditions=_CONDITIONS
            )
        )
        for configuration_name in (
            PROPOSED_FEDSDA_CONFIGURATION_NAME,
            SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME,
            LOCAL_PARAMETER_PREDICTION_CONFIGURATION_NAME,
        )
    )
    assert mixture == proposed
    holder = local["run_participant_settings"]["run_client_settings"][
        "prediction_combination_settings"
    ]
    assert holder["prediction_model_parameter_source"] == "current_local_parameters"
    holder["prediction_model_parameter_source"] = MIXTURE_VALUE
    assert local == mixture


def test_configuration_combines_with_the_pending_fifo_capacity_axis():
    (planned_run,) = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=(SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME,),
            dataset_names=("sine2",),
            random_seeds=(0,),
            client_count=2,
            per_client_sample_count=50,
            server_aggregation_intervals_per_client_samples=(10,),
            fedsda_pending_assignment_fifo_capacities=(60,),
        )
    )
    client = planned_run.run_settings.run_participant_settings.run_client_settings
    assert client.prediction_combination_settings.prediction_model_parameter_source == MIXTURE_VALUE
    assert client.training_data_assignment_settings.pending_assignment_buffer_capacity_samples == 60
    # 容量の軸は、Fixed-Shareの時間尺度を替えない。
    assert (
        client.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        == 120
    )


def _none_arguments(function):
    import inspect

    return {
        name: None
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }


@pytest.mark.parametrize(
    "function",
    [
        observed_sample_processing.predict_observed_sample_and_update_prediction_weights,
        observed_sample_processing.process_observed_sample,
    ],
)
def test_mixture_owner_is_checked_before_processing(function):
    with pytest.raises(TypeError, match="synchronized_and_local_prediction_mixture"):
        function(**_none_arguments(function), synchronized_and_local_prediction_mixture=object())
    # 混ぜる重みのownerは、写しのownerと一緒にだけ渡せる。
    with pytest.raises(ValueError, match="synchronized_prediction_model_snapshot"):
        function(
            **_none_arguments(function),
            synchronized_and_local_prediction_mixture=SynchronizedAndLocalPredictionMixture(
                share_rate=0.1
            ),
        )


def test_prediction_rejects_mixture_with_single_model_prediction(monkeypatch):
    """現在のモデルだけで予測する経路は、混ぜる重みのownerを受け取らない（設定の検査も組合せを拒否する）。"""
    import federated_learning_experiments.runtime.observed_sample_prediction as observed_sample_prediction
    from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_prediction_model_snapshot import (
        SynchronizedPredictionModelSnapshot,
    )

    # 入力の検査は、この組合せの拒否より前に行われるので、このtestの中でだけ外す。
    monkeypatch.setattr(
        observed_sample_prediction, "_validate_observed_sample_prediction_inputs", lambda **_: None
    )
    arguments = _none_arguments(
        observed_sample_prediction.predict_observed_sample_and_update_prediction_weights
    )
    arguments["prediction_combination_strategy"] = "current_training_model_prediction"
    with pytest.raises(ValueError, match="weighted prediction combination"):
        observed_sample_prediction.predict_observed_sample_and_update_prediction_weights(
            **arguments,
            synchronized_prediction_model_snapshot=SynchronizedPredictionModelSnapshot(),
            synchronized_and_local_prediction_mixture=SynchronizedAndLocalPredictionMixture(
                share_rate=0.1
            ),
        )
