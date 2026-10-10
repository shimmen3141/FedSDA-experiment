"""クロス評価（clientでの評価と、サーバでの集計・通信量・診断の記録）を、実旧のサーバとclientに対して照合する。"""

import random
from dataclasses import replace

import pytest
import torch
from test_fedsda_run_client import (
    CROSS_EVALUATION_CLIENT_LIMIT,
    CROSS_EVALUATION_SAMPLE_LIMIT,
    assert_run_client_matches_legacy,
    assert_run_client_state_unchanged,
    run_in_both,
    snapshot_run_client_state,
)
from test_global_model_distribution import (
    DIFFERENT_LEARNING_RATES,
    assert_all_states_match_legacy_after_distribution,
    convert_legacy_global_models,
    make_client_streams,
)
from test_held_candidate_validation_progress import make_subclass_copy
from test_post_aggregation_prediction_recalibration import (
    FIFO_REPLAY_RECALIBRATION,
    run_round_with_recalibration_in_both,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping
from test_server_model_registration_and_aggregation import (
    assert_all_states_match_legacy,
    assert_server_and_client_states_unchanged,
    build_server_round_oracle,
    process_round_samples_in_both,
    snapshot_server_and_client_states,
)

import federated_learning_experiments.runtime.fedsda_run_client as run_client_module
from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    CrossEvaluationRecordStore,
)
from federated_learning_experiments.runtime.client_model_cross_evaluation import (
    ClientModelCrossEvaluation,
    ModelPairCorrectnessCounts,
    evaluate_candidate_model_on_target_model_samples,
)
from federated_learning_experiments.runtime.global_model_distribution import (
    distribute_global_models_to_clients,
)
from federated_learning_experiments.runtime.model_cross_evaluation import (
    CrossEvaluationLossSums,
    ModelCrossEvaluation,
    ModelPairUniqueCorrectnessCounts,
    cross_evaluate_global_models,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    aggregate_client_models_into_global_models,
    register_ready_client_models,
)

# 評価標本を1回に多く足す条件（評価に使える件数に、早く達する）と、評価に使う標本の上限を小さくした条件。
MORE_EVALUATION_SAMPLES = dict(added_evaluation_sample_count=8)
SMALL_CROSS_EVALUATION_SAMPLE_LIMIT = dict(cross_evaluation_sample_limit=6)


def convert_correctness_counts(legacy_counts):
    return ModelPairCorrectnessCounts(
        evaluated_sample_count=legacy_counts["n"],
        candidate_only_correct_count=legacy_counts["candidate_only_correct"],
        target_only_correct_count=legacy_counts["target_only_correct"],
        both_correct_count=legacy_counts["both_correct"],
        both_wrong_count=legacy_counts["both_wrong"],
    )


def assert_client_cross_evaluation_matches_legacy(
    client_cross_evaluation, legacy_statistics, legacy_diagnostic
):
    """clientの評価の結果を、実旧の（件数、和、2乗和）と、正誤の診断（なければNone）と照合する。"""
    assert type(client_cross_evaluation) is ClientModelCrossEvaluation
    assert (
        client_cross_evaluation.evaluated_sample_count,
        client_cross_evaluation.bounded_loss_sum,
        client_cross_evaluation.squared_bounded_loss_sum,
    ) == tuple(legacy_statistics)
    assert type(client_cross_evaluation.bounded_loss_sum) is float
    assert type(client_cross_evaluation.squared_bounded_loss_sum) is float
    if legacy_diagnostic is None:
        assert client_cross_evaluation.correctness_counts is None
        assert client_cross_evaluation.class_correctness_counts == ()
        return
    assert client_cross_evaluation.correctness_counts == convert_correctness_counts(
        legacy_diagnostic
    )
    assert client_cross_evaluation.class_correctness_counts == tuple(
        (class_id, convert_correctness_counts(legacy_class_counts))
        for class_id, legacy_class_counts in legacy_diagnostic["class_correctness"].items()
    )


def evaluate_on_client_in_both(
    *,
    run_client,
    legacy_client,
    python_random_generator,
    legacy_candidate_parameters,
    candidate_parameter_snapshot,
    target_model_id,
    compare_correctness,
):
    """実旧のclientの評価と、新のclientの評価を、同じ乱数の状態から行って照合する。戻り値: 新の結果。"""

    def evaluate_on_legacy_client():
        if compare_correctness:
            return legacy_client.evaluate_model_diagnostics(
                legacy_candidate_parameters,
                target_model_id=target_model_id,
                include_class_correctness=True,
            )
        return legacy_client.evaluate_model(legacy_candidate_parameters, target_model_id), None

    client_cross_evaluation, (legacy_statistics, legacy_diagnostic) = run_in_both(
        run_client=run_client,
        legacy_client=legacy_client,
        python_random_generator=python_random_generator,
        legacy_operation=evaluate_on_legacy_client,
        operation=lambda: run_client.evaluate_candidate_model_on_target_model_samples(
            candidate_parameter_snapshot=candidate_parameter_snapshot,
            target_model_id=target_model_id,
            compare_correctness_with_held_target_model=compare_correctness,
        ),
    )
    assert_client_cross_evaluation_matches_legacy(
        client_cross_evaluation, legacy_statistics, legacy_diagnostic
    )
    return client_cross_evaluation


CLIENT_OBSERVED_COVERAGE_BY_CONDITION = {}
# (クラス数, 概念の区間長, 標本列のseed, ラウンド数, 条件の上書き)。
CLIENT_CROSS_EVALUATION_CONDITIONS = [
    (2, 22, 11, 12, {}),
    (2, 22, 11, 12, MORE_EVALUATION_SAMPLES),
    (2, 16, 23, 12, MORE_EVALUATION_SAMPLES | SMALL_CROSS_EVALUATION_SAMPLE_LIMIT),
    (2, 10, 50, 11, MORE_EVALUATION_SAMPLES),
    (4, 22, 11, 13, MORE_EVALUATION_SAMPLES | DIFFERENT_LEARNING_RATES),
    (4, 30, 3, 13, MORE_EVALUATION_SAMPLES | SMALL_CROSS_EVALUATION_SAMPLE_LIMIT),
]


@pytest.mark.parametrize(
    "class_count,concept_block_length,stream_seed,round_count,condition_overrides",
    CLIENT_CROSS_EVALUATION_CONDITIONS,
)
def test_client_cross_evaluation_matches_real_legacy_client(
    class_count,
    concept_block_length,
    stream_seed,
    round_count,
    condition_overrides,
    monkeypatch,
    valid_run_settings_mapping,
):
    """何ラウンドか同期した状態で、各clientへ、モデルの全部の組の評価を、正誤の比較ありとなしの両方で求める。"""
    client_streams = make_client_streams(
        round_count=round_count,
        concept_block_length=concept_block_length,
        stream_seed=stream_seed,
    )
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=class_count,
        **FIFO_REPLAY_RECALIBRATION,
        **condition_overrides,
    )
    legacy_server = server_round_oracle["legacy_server"]
    python_random_generator = server_round_oracle["python_random_generator"]
    cross_evaluation_sample_limit = condition_overrides.get(
        "cross_evaluation_sample_limit", CROSS_EVALUATION_SAMPLE_LIMIT
    )
    observed_paths = set()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(round_count):
            run_round_with_recalibration_in_both(
                server_round_oracle=server_round_oracle,
                client_streams=client_streams,
                round_index=round_index,
            )
        global_model_ids = list(legacy_server.global_models)
        candidate_parameter_snapshots = dict(
            convert_legacy_global_models(legacy_server.global_models)
        )
        for run_client, legacy_client in zip(
            server_round_oracle["run_clients"], server_round_oracle["legacy_clients"], strict=True
        ):
            assert run_client.get_cross_evaluation_held_model_ids() == frozenset(
                legacy_client.get_held_model_ids()
            )
            for candidate_model_id in global_model_ids:
                # 対象は、全グローバルモデルと、保有する一時IDのモデルと、どこにもないID。
                for target_model_id in (
                    *global_model_ids,
                    *(model_id for model_id in legacy_client.models if model_id < 0),
                    99,
                ):
                    stored_sample_count = len(legacy_client.stored_data.get(target_model_id, ()))
                    training_sample_count = len(
                        legacy_client.train_data_store.get(target_model_id, ())
                    )
                    for compare_correctness in (False, True):
                        client_cross_evaluation = evaluate_on_client_in_both(
                            run_client=run_client,
                            legacy_client=legacy_client,
                            python_random_generator=python_random_generator,
                            legacy_candidate_parameters=legacy_server.global_models[
                                candidate_model_id
                            ],
                            candidate_parameter_snapshot=candidate_parameter_snapshots[
                                candidate_model_id
                            ],
                            target_model_id=target_model_id,
                            compare_correctness=compare_correctness,
                        )
                        if client_cross_evaluation.evaluated_sample_count == 0:
                            observed_paths.add("not_evaluated")
                            continue
                        observed_paths.add("evaluated")
                        if stored_sample_count > 5:
                            observed_paths.add("stored_evaluation_samples_used")
                            sampled_population_count = stored_sample_count
                        else:
                            assert target_model_id == legacy_client.current_model_id
                            observed_paths.add("current_model_training_samples_used")
                            sampled_population_count = training_sample_count
                        # 上限を超えた標本は、抜き出される。
                        assert client_cross_evaluation.evaluated_sample_count == min(
                            sampled_population_count, cross_evaluation_sample_limit
                        )
                        if (
                            client_cross_evaluation.evaluated_sample_count
                            < sampled_population_count
                        ):
                            observed_paths.add("evaluation_samples_sampled")
                        if client_cross_evaluation.correctness_counts is not None:
                            observed_paths.add("correctness_compared")
                            if len(client_cross_evaluation.class_correctness_counts) >= 2:
                                observed_paths.add("multiple_classes_compared")
                            if (
                                client_cross_evaluation.correctness_counts.candidate_only_correct_count
                                or client_cross_evaluation.correctness_counts.target_only_correct_count
                            ):
                                observed_paths.add("models_disagree")
            # 評価は、clientの状態を変えない（乱数は、上の照合で確かめている）。
            assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
    CLIENT_OBSERVED_COVERAGE_BY_CONDITION[
        (class_count, concept_block_length, stream_seed, tuple(condition_overrides))
    ] = observed_paths


def test_client_cross_evaluation_conditions_cover_required_paths():
    """上の対照が、clientの評価の経路をすべて通っていること（全条件を実行したときだけ確かめる）。"""
    if len(CLIENT_OBSERVED_COVERAGE_BY_CONDITION) < len(CLIENT_CROSS_EVALUATION_CONDITIONS):
        pytest.skip("対照の全条件を実行したときだけ確かめる")
    for required_path in (
        "not_evaluated",
        "evaluated",
        "stored_evaluation_samples_used",
        "current_model_training_samples_used",
        "evaluation_samples_sampled",
        "correctness_compared",
        "multiple_classes_compared",
        "models_disagree",
    ):
        observed_class_counts = {
            condition[0]
            for condition, observed_paths in CLIENT_OBSERVED_COVERAGE_BY_CONDITION.items()
            if required_path in observed_paths
        }
        if required_path == "stored_evaluation_samples_used":
            # 評価標本は、正式IDのモデルへ、警報の区間の標本を吸収したときだけ足される。多クラスの条件では、
            # 評価に使える件数に達しなかった（標本の選択は、クラス数によらない）。
            assert observed_class_counts, required_path
        else:
            assert observed_class_counts == {2, 4}, (required_path, observed_class_counts)


def build_cross_evaluation_server_round_oracle(*, monkeypatch, valid_run_settings_mapping):
    """再較正つきで15ラウンド進めた状態（2値、評価標本を1回に8件足す条件）。グローバルモデルは4つ。

    client 1と2は、現行でないモデル0の評価標本を、評価に使える件数（6件以上）持つ。
    """
    client_streams = make_client_streams(round_count=15, concept_block_length=16, stream_seed=23)
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
        **FIFO_REPLAY_RECALIBRATION,
        **MORE_EVALUATION_SAMPLES,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(15):
            run_round_with_recalibration_in_both(
                server_round_oracle=server_round_oracle,
                client_streams=client_streams,
                round_index=round_index,
            )
    assert len(server_round_oracle["legacy_server"].global_models) >= 3
    return server_round_oracle


def find_client_and_evaluable_non_current_target(server_round_oracle):
    """評価標本を6件以上持つ、現行でない正式IDのモデルと、そのclientの位置。"""
    for client_index, legacy_client in enumerate(server_round_oracle["legacy_clients"]):
        for model_id, stored_samples in legacy_client.stored_data.items():
            if len(stored_samples) > 5 and model_id != legacy_client.current_model_id:
                return client_index, model_id
    raise AssertionError("no client holds enough evaluation samples of a non-current model")


def test_client_cross_evaluation_without_held_target_model_matches_real_legacy_client(
    monkeypatch, valid_run_settings_mapping
):
    """対象のモデルの標本はあるが、そのモデルを保有していないとき、正誤の比較つきの評価は件数0、損失だけの評価は行う。"""
    server_round_oracle = build_cross_evaluation_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    client_index, target_model_id = find_client_and_evaluable_non_current_target(
        server_round_oracle
    )
    run_client = server_round_oracle["run_clients"][client_index]
    legacy_client = server_round_oracle["legacy_clients"][client_index]
    legacy_server = server_round_oracle["legacy_server"]
    # 両実装で、対象のモデルだけを保有から外す（標本は残す）。
    del legacy_client.models[target_model_id]
    registry = run_client.owners.held_model_training_state_registry
    registry.replace_held_model_training_states(
        held_model_training_states=tuple(
            held_state
            for held_state in registry.snapshot_ordered_held_model_training_states()
            if held_state.model_id != target_model_id
        )
    )
    assert target_model_id in run_client.get_cross_evaluation_held_model_ids()
    assert run_client.get_cross_evaluation_held_model_ids() == frozenset(
        legacy_client.get_held_model_ids()
    )
    candidate_model_id = next(
        model_id for model_id in legacy_server.global_models if model_id != target_model_id
    )
    evaluation_arguments = dict(
        run_client=run_client,
        legacy_client=legacy_client,
        python_random_generator=server_round_oracle["python_random_generator"],
        legacy_candidate_parameters=legacy_server.global_models[candidate_model_id],
        candidate_parameter_snapshot=dict(
            convert_legacy_global_models(legacy_server.global_models)
        )[candidate_model_id],
        target_model_id=target_model_id,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        torch_random_state = torch.get_rng_state().clone()
        compared_evaluation = evaluate_on_client_in_both(
            **evaluation_arguments, compare_correctness=True
        )
        assert compared_evaluation.evaluated_sample_count == 0
        # 件数0の評価は、分類器を作らない（torchの乱数を消費しない）。
        assert torch.equal(torch.get_rng_state(), torch_random_state)
        loss_only_evaluation = evaluate_on_client_in_both(
            **evaluation_arguments, compare_correctness=False
        )
        assert loss_only_evaluation.evaluated_sample_count > 0
        assert not torch.equal(torch.get_rng_state(), torch_random_state)


def get_client_cross_evaluation_arguments(*, server_round_oracle, client_index, target_model_id):
    """clientの評価の関数の、正常な引数（評価が行われる対象）。"""
    owners = server_round_oracle["run_clients"][client_index].owners
    legacy_server = server_round_oracle["legacy_server"]
    candidate_model_id = next(
        model_id for model_id in legacy_server.global_models if model_id != target_model_id
    )
    return dict(
        candidate_parameter_snapshot=dict(
            convert_legacy_global_models(legacy_server.global_models)
        )[candidate_model_id],
        target_model_id=target_model_id,
        compare_correctness_with_held_target_model=True,
        held_model_training_state_registry=owners.held_model_training_state_registry,
        model_evaluation_sample_store=owners.model_evaluation_sample_store,
        training_sample_store=owners.training_sample_store,
        current_training_model_assignment=owners.current_training_model_assignment,
        # 評価に要る件数（5件）ちょうどへ抜き出す上限。
        maximum_evaluation_sample_count=5,
        python_random_generator=server_round_oracle["python_random_generator"],
    )


def replace_candidate_parameters(arguments, change_parameter_snapshot):
    return dict(
        candidate_parameter_snapshot=change_parameter_snapshot(
            dict(arguments["candidate_parameter_snapshot"])
        )
    )


FIRST_PARAMETER_NAME = "feature_extractor.hidden_layers.0.weight"
CLIENT_CROSS_EVALUATION_OWNER_ARGUMENT_NAMES = (
    "held_model_training_state_registry",
    "model_evaluation_sample_store",
    "training_sample_store",
    "current_training_model_assignment",
)
# 条件名 -> 正常な引数から、差し替える引数を作る操作。
INVALID_CLIENT_CROSS_EVALUATION_ARGUMENT_CASES = (
    {
        "target_model_id_bool": lambda arguments: dict(target_model_id=True),
        "target_model_id_float": lambda arguments: dict(target_model_id=0.0),
        "target_model_id_text": lambda arguments: dict(target_model_id="0"),
        "compare_flag_int": lambda arguments: dict(compare_correctness_with_held_target_model=1),
        "compare_flag_none": lambda arguments: dict(
            compare_correctness_with_held_target_model=None
        ),
        "sample_limit_bool": lambda arguments: dict(maximum_evaluation_sample_count=True),
        "sample_limit_float": lambda arguments: dict(maximum_evaluation_sample_count=4.0),
        "sample_limit_zero": lambda arguments: dict(maximum_evaluation_sample_count=0),
        "random_generator_module": lambda arguments: dict(python_random_generator=random),
        "random_generator_subclass": lambda arguments: dict(
            python_random_generator=type("RandomSubclass", (random.Random,), {})(3)
        ),
        "parameters_none": lambda arguments: dict(candidate_parameter_snapshot=None),
        "parameters_not_dict": lambda arguments: replace_candidate_parameters(
            arguments, lambda parameter_snapshot: tuple(parameter_snapshot.items())
        ),
        "parameters_missing_name": lambda arguments: replace_candidate_parameters(
            arguments,
            lambda parameter_snapshot: {
                parameter_name: parameter_values
                for parameter_name, parameter_values in parameter_snapshot.items()
                if parameter_name != FIRST_PARAMETER_NAME
            },
        ),
        "parameters_extra_name": lambda arguments: replace_candidate_parameters(
            arguments, lambda parameter_snapshot: parameter_snapshot | {"other": torch.zeros(1)}
        ),
        "parameters_non_text_name": lambda arguments: replace_candidate_parameters(
            arguments, lambda parameter_snapshot: parameter_snapshot | {0: torch.zeros(1)}
        ),
        "parameters_other_shape": lambda arguments: replace_candidate_parameters(
            arguments,
            lambda parameter_snapshot: (
                parameter_snapshot
                | {FIRST_PARAMETER_NAME: parameter_snapshot[FIRST_PARAMETER_NAME][:1]}
            ),
        ),
        "parameters_float64": lambda arguments: replace_candidate_parameters(
            arguments,
            lambda parameter_snapshot: (
                parameter_snapshot
                | {FIRST_PARAMETER_NAME: parameter_snapshot[FIRST_PARAMETER_NAME].double()}
            ),
        ),
        "parameters_not_tensor": lambda arguments: replace_candidate_parameters(
            arguments,
            lambda parameter_snapshot: (
                parameter_snapshot
                | {FIRST_PARAMETER_NAME: parameter_snapshot[FIRST_PARAMETER_NAME].tolist()}
            ),
        ),
    }
    | {
        f"{owner_argument_name}_none": lambda arguments, owner_argument_name=owner_argument_name: {
            owner_argument_name: None
        }
        for owner_argument_name in CLIENT_CROSS_EVALUATION_OWNER_ARGUMENT_NAMES
    }
    | {
        f"{owner_argument_name}_subclass": lambda arguments, owner_argument_name=owner_argument_name: {
            owner_argument_name: make_subclass_copy(arguments[owner_argument_name])
        }
        for owner_argument_name in CLIENT_CROSS_EVALUATION_OWNER_ARGUMENT_NAMES
    }
)


def test_client_cross_evaluation_rejects_invalid_arguments_and_changes_no_state(
    monkeypatch, valid_run_settings_mapping
):
    """不正な入力は、乱数を消費する前に拒否する。正常な評価も、乱数のほかは何も変えない。"""
    server_round_oracle = build_cross_evaluation_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    python_random_generator = server_round_oracle["python_random_generator"]
    client_index, target_model_id = find_client_and_evaluable_non_current_target(
        server_round_oracle
    )
    run_client = server_round_oracle["run_clients"][client_index]
    arguments = get_client_cross_evaluation_arguments(
        server_round_oracle=server_round_oracle,
        client_index=client_index,
        target_model_id=target_model_id,
    )
    state_snapshot = snapshot_run_client_state(
        run_client=run_client, python_random_generator=python_random_generator
    )
    global_python_random_state = random.getstate()
    for (
        invalid_case_name,
        make_invalid_arguments,
    ) in INVALID_CLIENT_CROSS_EVALUATION_ARGUMENT_CASES.items():
        with pytest.raises((TypeError, ValueError)):
            evaluate_candidate_model_on_target_model_samples(
                **arguments | make_invalid_arguments(arguments)
            )
        assert_run_client_state_unchanged(
            state_snapshot=state_snapshot,
            run_client=run_client,
            python_random_generator=python_random_generator,
        )
        assert random.getstate() == global_python_random_state, invalid_case_name
    # 正常な評価: 標本を抜き出し（借りた乱数）、分類器を作る（torchの乱数）。ほかは変えない。
    held_model_training_states = run_client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    training_modes = [held_state.classifier.training for held_state in held_model_training_states]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        client_cross_evaluation = evaluate_candidate_model_on_target_model_samples(**arguments)
        current_snapshot = snapshot_run_client_state(
            run_client=run_client, python_random_generator=python_random_generator
        )
    assert client_cross_evaluation.evaluated_sample_count == 5
    assert client_cross_evaluation.correctness_counts.evaluated_sample_count == 5
    assert current_snapshot["python_random_state"] != state_snapshot["python_random_state"]
    assert not torch.equal(
        current_snapshot["torch_random_state"][0], state_snapshot["torch_random_state"][0]
    )
    # 借りた乱数のほかは変わらない（torchの乱数は、上のblockを出るときに元へ戻っている）。
    assert_run_client_state_unchanged(
        state_snapshot=state_snapshot
        | dict(python_random_state=current_snapshot["python_random_state"]),
        run_client=run_client,
        python_random_generator=python_random_generator,
    )
    assert training_modes == [
        held_state.classifier.training for held_state in held_model_training_states
    ]
    assert random.getstate() == global_python_random_state
    # 結果は、その後に書き換えられない値。
    with pytest.raises(AttributeError):
        client_cross_evaluation.evaluated_sample_count = 0
    assert replace(client_cross_evaluation) == client_cross_evaluation


CORRECTNESS_COUNT_NAMES = (
    "candidate_only_correct",
    "target_only_correct",
    "both_correct",
    "both_wrong",
)


def derive_legacy_cross_evaluation_diagnostics(records):
    """新の記録から、旧の3つの診断の記録（全評価、正誤の比較がある評価、クラス別）を導く。"""
    all_evaluations = [
        dict(
            round_index=record.round_index,
            client_id=record.client_id,
            candidate_model_id=record.candidate_model_id,
            target_model_id=record.target_model_id,
            n=record.evaluated_sample_count,
            sum=record.bounded_loss_sum,
            sum_sq=record.squared_bounded_loss_sum,
        )
        | dict(
            zip(
                CORRECTNESS_COUNT_NAMES,
                (-1, -1, -1, -1)
                if record.correctness_counts is None
                else record.correctness_counts,
                strict=True,
            )
        )
        for record in records
    ]
    compared_evaluations = [
        dict(
            candidate_model_id=record.candidate_model_id,
            target_model_id=record.target_model_id,
            n=record.evaluated_sample_count,
            class_correctness={
                class_id: dict(n=class_sample_count)
                | dict(zip(CORRECTNESS_COUNT_NAMES, class_counts, strict=True))
                for class_id, class_sample_count, *class_counts in record.class_correctness_counts
            },
        )
        | dict(zip(CORRECTNESS_COUNT_NAMES, record.correctness_counts, strict=True))
        for record in records
        if record.correctness_counts is not None
    ]
    class_evaluations = [
        dict(
            round_index=record.round_index,
            client_id=record.client_id,
            candidate_model_id=record.candidate_model_id,
            target_model_id=record.target_model_id,
            class_id=class_id,
            n=class_sample_count,
        )
        | dict(zip(CORRECTNESS_COUNT_NAMES, class_counts, strict=True))
        for record in records
        for class_id, class_sample_count, *class_counts in record.class_correctness_counts
    ]
    return all_evaluations, compared_evaluations, class_evaluations


def cross_evaluate_in_both(
    *, server_round_oracle, cross_evaluation_record_store, round_index, client_limit
):
    """実旧のサーバのクロス評価と、新のクロス評価を、同じ乱数の状態から行って、結果・記録・全状態を照合する。

    対象は、旧のサーバのラウンドと同じ、全clientが保有する非負のモデルID（昇順）。
    戻り値: 新の結果と、この呼出しで足された記録。
    """
    legacy_server = server_round_oracle["legacy_server"]
    legacy_clients = server_round_oracle["legacy_clients"]
    active_model_ids = sorted(
        {
            model_id
            for legacy_client in legacy_clients
            for model_id in legacy_client.models
            if model_id >= 0
        }
    )
    legacy_record_counts = (
        len(legacy_server.cross_evaluation_diagnostics),
        len(legacy_server.pair_prediction_diagnostics),
        len(legacy_server.cross_evaluation_class_diagnostics),
    )
    record_count = len(cross_evaluation_record_store.snapshot_client_cross_evaluation_records())
    model_cross_evaluation, legacy_statistics_matrix = run_in_both(
        run_client=None,
        legacy_client=None,
        legacy_clients=legacy_server.clients,
        python_random_generator=server_round_oracle["python_random_generator"],
        legacy_operation=lambda: legacy_server._cross_evaluate(
            active_model_ids, round_index=round_index
        ),
        operation=lambda: cross_evaluate_global_models(
            run_clients=server_round_oracle["run_clients"],
            global_model_repository=server_round_oracle["global_model_repository"],
            communication_volume_record_store=server_round_oracle[
                "communication_volume_record_store"
            ],
            cross_evaluation_record_store=cross_evaluation_record_store,
            cross_evaluated_model_ids=tuple(active_model_ids),
            round_index=round_index,
            maximum_evaluating_client_count_per_model=client_limit,
            python_random_generator=server_round_oracle["python_random_generator"],
        ),
    )
    assert type(model_cross_evaluation) is ModelCrossEvaluation
    assert model_cross_evaluation.cross_evaluated_model_ids == tuple(active_model_ids)
    # 損失の統計の表: 全部の組を持ち、実旧の表と同じ値。
    assert list(model_cross_evaluation.loss_sums_by_candidate_and_target_model_id) == [
        (candidate_model_id, target_model_id)
        for candidate_model_id in active_model_ids
        for target_model_id in active_model_ids
    ]
    for (
        candidate_model_id,
        target_model_id,
    ), loss_sums in model_cross_evaluation.loss_sums_by_candidate_and_target_model_id.items():
        assert type(loss_sums) is CrossEvaluationLossSums
        assert (
            loss_sums.evaluated_sample_count,
            loss_sums.bounded_loss_sum,
            loss_sums.squared_bounded_loss_sum,
        ) == tuple(legacy_statistics_matrix[candidate_model_id][target_model_id])
    # 対ごとの正誤の集計。
    legacy_pair_statistics = legacy_server._last_pair_functional_stats
    assert list(model_cross_evaluation.unique_correctness_counts_by_model_pair) == list(
        legacy_pair_statistics
    )
    for (
        model_pair,
        unique_correctness_counts,
    ) in model_cross_evaluation.unique_correctness_counts_by_model_pair.items():
        legacy_statistics = legacy_pair_statistics[model_pair]
        assert unique_correctness_counts == ModelPairUniqueCorrectnessCounts(
            evaluated_sample_count=legacy_statistics.sample_count,
            lower_id_model_only_correct_count=legacy_statistics.left_only_correct,
            higher_id_model_only_correct_count=legacy_statistics.right_only_correct,
            class_counts=tuple(
                (
                    class_id,
                    legacy_class_statistics.sample_count,
                    legacy_class_statistics.left_only_correct,
                    legacy_class_statistics.right_only_correct,
                )
                for class_id, legacy_class_statistics in legacy_statistics.class_stats.items()
            ),
        )
    # 診断の記録: この呼出しで足された記録から、旧の3つの記録を導いて照合する。
    added_records = cross_evaluation_record_store.snapshot_client_cross_evaluation_records()[
        record_count:
    ]
    all_evaluations, compared_evaluations, class_evaluations = (
        derive_legacy_cross_evaluation_diagnostics(added_records)
    )
    assert all_evaluations == legacy_server.cross_evaluation_diagnostics[legacy_record_counts[0] :]
    assert (
        compared_evaluations == legacy_server.pair_prediction_diagnostics[legacy_record_counts[1] :]
    )
    assert (
        class_evaluations
        == legacy_server.cross_evaluation_class_diagnostics[legacy_record_counts[2] :]
    )
    assert legacy_server._last_pair_prediction_diagnostics == compared_evaluations
    # サーバの全状態（通信量を含む）と、clientの全状態は、実旧と同じ。
    assert_all_states_match_legacy(server_round_oracle)
    return model_cross_evaluation, added_records


def run_round_with_cross_evaluation_in_both(
    *, server_round_oracle, cross_evaluation_record_store, client_streams, round_index, client_limit
):
    """1ラウンド: 標本処理→保留中の学習→状態の報告→登録→集約→クロス評価→配布→再較正→送信待ちの進行。

    旧は、サーバのラウンドの部品を、ラウンドと同じ順に直接呼ぶ（クロス評価の結果は、どちらの実装も使わない）。
    戻り値: クロス評価の新の結果、足された記録、クロス評価の直前に各clientが持っていた評価標本と学習データの件数。
    """
    run_clients = server_round_oracle["run_clients"]
    legacy_clients = server_round_oracle["legacy_clients"]
    legacy_server = server_round_oracle["legacy_server"]
    global_model_repository = server_round_oracle["global_model_repository"]
    communication_volume_record_store = server_round_oracle["communication_volume_record_store"]
    python_random_generator = server_round_oracle["python_random_generator"]
    process_round_samples_in_both(
        server_round_oracle=server_round_oracle,
        client_streams=client_streams,
        round_index=round_index,
    )
    legacy_server.record_client_state_summaries()
    communication_volume_record_store.record_messages(
        transfer_direction="upload", message_count=len(run_clients)
    )
    legacy_server._register_new_models(round_index)
    register_ready_client_models(
        run_clients=run_clients,
        global_model_repository=global_model_repository,
        round_index=round_index,
    )
    legacy_server.update_global_models(
        sorted(
            {
                model_id
                for legacy_client in legacy_clients
                for model_id in legacy_client.models
                if model_id >= 0
            }
        )
    )
    aggregate_client_models_into_global_models(
        run_clients=run_clients,
        global_model_repository=global_model_repository,
        communication_volume_record_store=communication_volume_record_store,
    )
    sample_counts_before_cross_evaluation = [
        dict(
            stored={
                model_id: len(samples) for model_id, samples in legacy_client.stored_data.items()
            },
            training={
                model_id: len(samples)
                for model_id, samples in legacy_client.train_data_store.items()
            },
            current_model_id=legacy_client.current_model_id,
            held_model_ids=set(legacy_client.get_held_model_ids()),
        )
        for legacy_client in legacy_clients
    ]
    model_cross_evaluation, added_records = cross_evaluate_in_both(
        server_round_oracle=server_round_oracle,
        cross_evaluation_record_store=cross_evaluation_record_store,
        round_index=round_index,
        client_limit=client_limit,
    )

    def distribute_and_recalibrate():
        distribute_global_models_to_clients(
            run_clients=run_clients,
            global_model_repository=global_model_repository,
            communication_volume_record_store=communication_volume_record_store,
            model_id_mapping={},
        )
        for run_client in run_clients:
            run_client.recalibrate_prediction_state_after_aggregation()

    def distribute_and_recalibrate_in_legacy():
        legacy_server.broadcast_models({})
        for legacy_client in legacy_clients:
            legacy_client.recalibrate_routing_after_aggregation()

    run_in_both(
        run_client=None,
        legacy_client=None,
        legacy_clients=legacy_server.clients,
        python_random_generator=python_random_generator,
        legacy_operation=distribute_and_recalibrate_in_legacy,
        operation=distribute_and_recalibrate,
    )
    for run_client, legacy_client in zip(run_clients, legacy_clients, strict=True):
        legacy_client.promote_pending_to_ready()
        run_client.advance_new_model_upload_wait_after_synchronization(round_index=round_index)
    assert_all_states_match_legacy_after_distribution(server_round_oracle)
    return model_cross_evaluation, added_records, sample_counts_before_cross_evaluation


CROSS_EVALUATION_ROUND_COUNT = 15
SERVER_OBSERVED_COVERAGE_BY_CONDITION = {}
SMALL_CROSS_EVALUATION_CLIENT_LIMIT = dict(cross_evaluation_client_limit=2)
# (クラス数, 概念の区間長, 標本列のseed, 条件の上書き)。
SERVER_CROSS_EVALUATION_CONDITIONS = [
    (2, 22, 11, MORE_EVALUATION_SAMPLES),
    (2, 16, 23, MORE_EVALUATION_SAMPLES | SMALL_CROSS_EVALUATION_SAMPLE_LIMIT),
    (2, 16, 23, MORE_EVALUATION_SAMPLES | SMALL_CROSS_EVALUATION_CLIENT_LIMIT),
    (2, 10, 50, MORE_EVALUATION_SAMPLES),
    (2, 9, 7, {}),
    (4, 22, 11, MORE_EVALUATION_SAMPLES | DIFFERENT_LEARNING_RATES),
    (4, 30, 3, MORE_EVALUATION_SAMPLES | SMALL_CROSS_EVALUATION_CLIENT_LIMIT),
    (4, 30, 3, SMALL_CROSS_EVALUATION_SAMPLE_LIMIT),
]


@pytest.mark.parametrize(
    "class_count,concept_block_length,stream_seed,condition_overrides",
    SERVER_CROSS_EVALUATION_CONDITIONS,
)
def test_cross_evaluation_matches_real_legacy_server_for_each_round(
    class_count,
    concept_block_length,
    stream_seed,
    condition_overrides,
    monkeypatch,
    valid_run_settings_mapping,
):
    """毎ラウンド、登録→集約の後（配布の前）に、クロス評価を両実装で行って照合する。"""
    client_streams = make_client_streams(
        round_count=CROSS_EVALUATION_ROUND_COUNT,
        concept_block_length=concept_block_length,
        stream_seed=stream_seed,
    )
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=class_count,
        **FIFO_REPLAY_RECALIBRATION,
        **condition_overrides,
    )
    client_limit = condition_overrides.get(
        "cross_evaluation_client_limit", CROSS_EVALUATION_CLIENT_LIMIT
    )
    sample_limit = condition_overrides.get(
        "cross_evaluation_sample_limit", CROSS_EVALUATION_SAMPLE_LIMIT
    )
    cross_evaluation_record_store = CrossEvaluationRecordStore()
    observed_paths = set()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(CROSS_EVALUATION_ROUND_COUNT):
            model_cross_evaluation, added_records, sample_counts = (
                run_round_with_cross_evaluation_in_both(
                    server_round_oracle=server_round_oracle,
                    cross_evaluation_record_store=cross_evaluation_record_store,
                    client_streams=client_streams,
                    round_index=round_index,
                    client_limit=client_limit,
                )
            )
            model_ids = model_cross_evaluation.cross_evaluated_model_ids
            if len(model_ids) >= 2:
                observed_paths.add("multiple_models_cross_evaluated")
            # 同じclientが、同じ評価する側のモデルを、複数の対象について評価した（転送は1回だけ数える）。
            if len(added_records) > len(
                {(record.candidate_model_id, record.client_id) for record in added_records}
            ):
                observed_paths.add("candidate_evaluated_on_multiple_targets_by_one_client")
            for target_model_id in model_ids:
                holding_client_count = sum(
                    target_model_id in client_sample_counts["held_model_ids"]
                    for client_sample_counts in sample_counts
                )
                evaluating_client_ids = {
                    record.client_id
                    for record in added_records
                    if record.target_model_id == target_model_id
                    and record.candidate_model_id == model_ids[0]
                }
                assert len(evaluating_client_ids) == min(holding_client_count, client_limit)
                if holding_client_count > client_limit:
                    observed_paths.add("evaluating_clients_sampled")
            for record in added_records:
                assert record.round_index == round_index
                assert (record.correctness_counts is None) == (
                    record.candidate_model_id == record.target_model_id
                    or record.evaluated_sample_count == 0
                )
                if record.evaluated_sample_count == 0:
                    observed_paths.add("client_did_not_evaluate")
                    continue
                observed_paths.add("client_evaluated")
                client_sample_counts = sample_counts[record.client_id]
                stored_sample_count = client_sample_counts["stored"].get(record.target_model_id, 0)
                if stored_sample_count > 5:
                    observed_paths.add("stored_evaluation_samples_used")
                    population_sample_count = stored_sample_count
                else:
                    assert record.target_model_id == client_sample_counts["current_model_id"]
                    observed_paths.add("current_model_training_samples_used")
                    population_sample_count = client_sample_counts["training"][
                        record.target_model_id
                    ]
                assert record.evaluated_sample_count == min(population_sample_count, sample_limit)
                if population_sample_count > sample_limit:
                    observed_paths.add("evaluation_samples_sampled")
                if record.correctness_counts is None:
                    observed_paths.add("loss_only_evaluation")
                else:
                    observed_paths.add("correctness_compared")
    SERVER_OBSERVED_COVERAGE_BY_CONDITION[
        (class_count, concept_block_length, stream_seed, tuple(condition_overrides))
    ] = observed_paths


def test_cross_evaluation_round_conditions_cover_required_paths():
    """上の対照が、要求の経路をすべて通っていること（全条件を実行したときだけ確かめる）。"""
    if len(SERVER_OBSERVED_COVERAGE_BY_CONDITION) < len(SERVER_CROSS_EVALUATION_CONDITIONS):
        pytest.skip("対照の全条件を実行したときだけ確かめる")
    for required_path in (
        "multiple_models_cross_evaluated",
        "candidate_evaluated_on_multiple_targets_by_one_client",
        "evaluating_clients_sampled",
        "client_did_not_evaluate",
        "client_evaluated",
        "stored_evaluation_samples_used",
        "current_model_training_samples_used",
        "evaluation_samples_sampled",
        "loss_only_evaluation",
        "correctness_compared",
    ):
        observed_class_counts = {
            condition[0]
            for condition, observed_paths in SERVER_OBSERVED_COVERAGE_BY_CONDITION.items()
            if required_path in observed_paths
        }
        if required_path == "stored_evaluation_samples_used":
            # 多クラスの条件では、評価標本が、評価に使える件数に達しなかった（clientの対照と同じ）。
            assert observed_class_counts, required_path
        else:
            assert observed_class_counts == {2, 4}, (required_path, observed_class_counts)


def get_cross_evaluation_arguments(server_round_oracle):
    global_model_repository = server_round_oracle["global_model_repository"]
    return dict(
        run_clients=server_round_oracle["run_clients"],
        global_model_repository=global_model_repository,
        communication_volume_record_store=server_round_oracle["communication_volume_record_store"],
        cross_evaluation_record_store=CrossEvaluationRecordStore(),
        cross_evaluated_model_ids=global_model_repository.global_model_ids,
        round_index=15,
        maximum_evaluating_client_count_per_model=3,
        python_random_generator=server_round_oracle["python_random_generator"],
    )


# 条件名 -> 正常な引数から、差し替える引数を作る操作。
INVALID_CROSS_EVALUATION_ARGUMENT_CASES = (
    {
        "clients_list": lambda arguments: dict(run_clients=list(arguments["run_clients"])),
        "clients_hold_none": lambda arguments: dict(run_clients=(*arguments["run_clients"], None)),
        "clients_hold_subclass": lambda arguments: dict(
            run_clients=(
                *arguments["run_clients"][:2],
                make_subclass_copy(arguments["run_clients"][2]),
            )
        ),
        "clients_duplicate": lambda arguments: dict(
            run_clients=(*arguments["run_clients"], arguments["run_clients"][0])
        ),
        "model_ids_list": lambda arguments: dict(
            cross_evaluated_model_ids=list(arguments["cross_evaluated_model_ids"])
        ),
        "model_ids_bool": lambda arguments: dict(cross_evaluated_model_ids=(0, True)),
        "model_ids_float": lambda arguments: dict(cross_evaluated_model_ids=(0, 1.0)),
        "model_ids_duplicate": lambda arguments: dict(cross_evaluated_model_ids=(0, 1, 0)),
        "model_ids_without_parameters": lambda arguments: dict(cross_evaluated_model_ids=(0, 99)),
        "model_ids_temporary": lambda arguments: dict(cross_evaluated_model_ids=(0, -101)),
        "round_index_bool": lambda arguments: dict(round_index=True),
        "round_index_negative": lambda arguments: dict(round_index=-1),
        "client_limit_float": lambda arguments: dict(maximum_evaluating_client_count_per_model=3.0),
        "client_limit_zero": lambda arguments: dict(maximum_evaluating_client_count_per_model=0),
        "random_generator_module": lambda arguments: dict(python_random_generator=random),
        "random_generator_subclass": lambda arguments: dict(
            python_random_generator=type("RandomSubclass", (random.Random,), {})(3)
        ),
    }
    | {
        f"{owner_argument_name}_none": lambda arguments, owner_argument_name=owner_argument_name: {
            owner_argument_name: None
        }
        for owner_argument_name in (
            "global_model_repository",
            "communication_volume_record_store",
            "cross_evaluation_record_store",
        )
    }
    | {
        f"{owner_argument_name}_subclass": lambda arguments, owner_argument_name=owner_argument_name: {
            owner_argument_name: make_subclass_copy(arguments[owner_argument_name])
        }
        for owner_argument_name in (
            "global_model_repository",
            "communication_volume_record_store",
            "cross_evaluation_record_store",
        )
    }
)


def test_cross_evaluation_rejects_invalid_arguments_before_any_update(
    monkeypatch, valid_run_settings_mapping
):
    """不正な入力は、通信量・記録・乱数を変える前に拒否する（1つの状態で、全条件を順に確かめる）。"""
    server_round_oracle = build_cross_evaluation_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    arguments = get_cross_evaluation_arguments(server_round_oracle)
    state_snapshot = snapshot_server_and_client_states(server_round_oracle)
    global_python_random_state = random.getstate()
    for (
        invalid_case_name,
        make_invalid_arguments,
    ) in INVALID_CROSS_EVALUATION_ARGUMENT_CASES.items():
        with pytest.raises((TypeError, ValueError)):
            cross_evaluate_global_models(**arguments | make_invalid_arguments(arguments))
        assert_server_and_client_states_unchanged(
            state_snapshot=state_snapshot, server_round_oracle=server_round_oracle
        )
        assert (
            arguments["cross_evaluation_record_store"].snapshot_client_cross_evaluation_records()
            == ()
        ), invalid_case_name
        assert random.getstate() == global_python_random_state, invalid_case_name


def test_cross_evaluation_stops_at_failed_client_evaluation_without_rollback(
    monkeypatch, valid_run_settings_mapping
):
    """clientの評価が途中で失敗したら、後の評価へ進まない。済んだ分の通信量と記録は残る。clientの状態は変わらない。"""
    server_round_oracle = build_cross_evaluation_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    arguments = get_cross_evaluation_arguments(server_round_oracle)
    python_random_generator = server_round_oracle["python_random_generator"]
    state_snapshot = snapshot_server_and_client_states(server_round_oracle)
    real_evaluation = run_client_module.evaluate_candidate_model_on_target_model_samples
    evaluation_call_count = 0

    def fail_at_fifth_evaluation(**evaluation_arguments):
        nonlocal evaluation_call_count
        evaluation_call_count += 1
        if evaluation_call_count == 5:
            raise RuntimeError("injected evaluation failure")
        return real_evaluation(**evaluation_arguments)

    monkeypatch.setattr(
        run_client_module,
        "evaluate_candidate_model_on_target_model_samples",
        fail_at_fifth_evaluation,
    )
    python_random_state = python_random_generator.getstate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        with pytest.raises(RuntimeError, match="injected evaluation failure"):
            cross_evaluate_global_models(**arguments)
    assert evaluation_call_count == 5
    # 済んだ4回の評価の記録が残る。通信量は、失敗した組の分まで足されている。
    records = arguments["cross_evaluation_record_store"].snapshot_client_cross_evaluation_records()
    assert len(records) == 4
    volume_before = state_snapshot["communication_volume"]
    volume_after = server_round_oracle["communication_volume_record_store"].get_state_snapshot()
    assert volume_after.downloaded_message_count > volume_before.downloaded_message_count
    assert volume_after.uploaded_message_count > volume_before.uploaded_message_count
    assert volume_after.downloaded_model_count > volume_before.downloaded_model_count
    # グローバルモデルとclientの状態（乱数を除く）は変わらない。
    python_random_generator.setstate(python_random_state)
    current_snapshot = snapshot_server_and_client_states(server_round_oracle)
    for state_name in (
        "next_global_model_id",
        "global_model_ids",
        "global_loss_statistics",
        "registration_records",
    ):
        assert current_snapshot[state_name] == state_snapshot[state_name], state_name
    for parameters, current_parameters in zip(
        state_snapshot["global_parameters"], current_snapshot["global_parameters"], strict=True
    ):
        assert all(
            torch.equal(current_parameters[parameter_name], parameter_values)
            for parameter_name, parameter_values in parameters.items()
        )
    for run_client, client_state_snapshot in zip(
        server_round_oracle["run_clients"], state_snapshot["clients"], strict=True
    ):
        assert_run_client_state_unchanged(
            state_snapshot=client_state_snapshot,
            run_client=run_client,
            python_random_generator=python_random_generator,
        )


def test_cross_evaluation_of_current_model_does_not_create_empty_training_collection(
    monkeypatch, valid_run_settings_mapping
):
    """旧は、対象が現行モデルで評価標本が足りないとき、学習データの辞書を既定値つきで読み、空の列を作る（LEGACY-017）。

    新は、状態を変えずに読む。評価の結果と、クロス評価で保有とみなすモデルIDは、どちらも同じ。
    """
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
        **FIFO_REPLAY_RECALIBRATION,
    )
    run_client = server_round_oracle["run_clients"][0]
    legacy_client = server_round_oracle["legacy_clients"][0]
    legacy_server = server_round_oracle["legacy_server"]
    # 生成直後: 現行モデル（ID 0）は、まだ学習データを持たない。
    assert dict(legacy_client.train_data_store) == {}
    assert run_client.owners.training_sample_store.snapshot_ordered_model_training_samples() == ()
    for compare_correctness in (False, True):
        client_cross_evaluation = evaluate_on_client_in_both(
            run_client=run_client,
            legacy_client=legacy_client,
            python_random_generator=server_round_oracle["python_random_generator"],
            legacy_candidate_parameters=legacy_server.global_models[0],
            candidate_parameter_snapshot=dict(
                convert_legacy_global_models(legacy_server.global_models)
            )[0],
            target_model_id=0,
            compare_correctness=compare_correctness,
        )
        assert client_cross_evaluation.evaluated_sample_count == 0
    assert dict(legacy_client.train_data_store) == {0: []}
    assert run_client.owners.training_sample_store.snapshot_ordered_model_training_samples() == ()
    assert run_client.get_cross_evaluation_held_model_ids() == frozenset(
        legacy_client.get_held_model_ids()
    )
