"""クロス評価（clientでの評価と、サーバでの集計・通信量・診断の記録）を、実旧のサーバとclientに対して照合する。"""

import random
from dataclasses import replace

import pytest
import torch
from test_fedsda_run_client import (
    CROSS_EVALUATION_SAMPLE_LIMIT,
    assert_run_client_matches_legacy,
    assert_run_client_state_unchanged,
    run_in_both,
    snapshot_run_client_state,
)
from test_global_model_distribution import (
    DIFFERENT_LEARNING_RATES,
    convert_legacy_global_models,
    make_client_streams,
)
from test_held_candidate_validation_progress import make_subclass_copy
from test_post_aggregation_prediction_recalibration import (
    FIFO_REPLAY_RECALIBRATION,
    run_round_with_recalibration_in_both,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping
from test_server_model_registration_and_aggregation import build_server_round_oracle

from federated_learning_experiments.runtime.client_model_cross_evaluation import (
    ClientModelCrossEvaluation,
    ModelPairCorrectnessCounts,
    evaluate_candidate_model_on_target_model_samples,
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
