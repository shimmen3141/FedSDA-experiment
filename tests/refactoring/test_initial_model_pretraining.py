"""初期モデルの事前学習を、実旧の事前学習（_pretrain_initial_model）と照合し、clientの組立てへつなぐ。"""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import convert_legacy_parameter_name
from test_fedsda_run_client import (
    ADAPTER_RANK,
    HIDDEN_LAYER_WIDTHS,
    LEARNING_RATE,
    MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
    WEIGHT_DECAY,
    assert_run_client_matches_legacy,
    convert_legacy_loss_statistics,
    make_concept_stream,
    make_run_client_settings,
    run_in_both,
    set_legacy_configuration,
)
from test_held_candidate_validation_progress import make_subclass_copy
from test_joint_model_parameter_update import assert_nested_state_equal
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_drift_experiment import config, experiment
from federated_drift_experiment.clients.shared_backbone import (
    ResidualAdapterRestartingSoftRoutingFedSDAClient,
)
from federated_drift_experiment.models import ResidualAdapterMLP
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.runtime.fedsda_run_client import assemble_fedsda_run_client
from federated_learning_experiments.runtime.initial_model_pretraining import (
    PretrainedInitialModel,
    pretrain_initial_model,
)

VALID_PRETRAINING_VALUES = dict(
    pretraining_sample_count=24, pretraining_epoch_count=2, pretraining_batch_sample_count=8
)


def test_pretraining_settings_accept_valid_and_boundary_values():
    assert (
        InitialModelPretrainingSettings(**VALID_PRETRAINING_VALUES).pretraining_sample_count == 24
    )
    boundary_settings = InitialModelPretrainingSettings(
        pretraining_sample_count=1, pretraining_epoch_count=0, pretraining_batch_sample_count=1
    )
    assert boundary_settings.pretraining_epoch_count == 0
    with pytest.raises(AttributeError):
        boundary_settings.pretraining_epoch_count = 3


@pytest.mark.parametrize(
    "settings_field_name,rejected_value",
    [
        *[
            (settings_field_name, rejected_value)
            for settings_field_name in VALID_PRETRAINING_VALUES
            # numpyの整数はintの派生型ではないが、共通の検査が拒否する。boolと実数も拒否する。
            for rejected_value in (True, None, "8", 2.0, -1, np.int64(8))
        ],
        ("pretraining_sample_count", 0),
        ("pretraining_batch_sample_count", 0),
    ],
)
def test_pretraining_settings_reject_invalid_values(settings_field_name, rejected_value):
    with pytest.raises(RunSettingsValidationError) as raised_error:
        InitialModelPretrainingSettings(
            **VALID_PRETRAINING_VALUES | {settings_field_name: rejected_value}
        )
    assert raised_error.value.configuration_parameter_name == settings_field_name


def make_parameter_optimizer_settings(optimizer_variant):
    if optimizer_variant == "sgd":
        return SgdParameterOptimizerSettings(learning_rate=LEARNING_RATE)
    return AdamParameterOptimizerSettings(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, adam_variant=optimizer_variant
    )


def make_pretraining_arguments(
    *,
    class_count=2,
    optimizer_variant="amsgrad",
    random_seed=7,
    pretraining_values=VALID_PRETRAINING_VALUES,
):
    """新の事前学習の引数。乱数は、seedから新しく作る（torchの乱数は、呼出し側が初期化する）。"""
    return dict(
        initial_model_pretraining_settings=InitialModelPretrainingSettings(**pretraining_values),
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=ADAPTER_RANK,
        ),
        hidden_layer_widths=HIDDEN_LAYER_WIDTHS,
        class_count=class_count,
        parameter_optimizer_settings=make_parameter_optimizer_settings(optimizer_variant),
        sample_generator=SineSampleGenerator(
            numpy_random_generator=np.random.RandomState(random_seed)
        ),
        python_random_generator=random.Random(random_seed),
    )


def run_real_legacy_pretraining(
    *, monkeypatch, class_count, optimizer_variant, random_seed, pretraining_values
):
    """実旧の事前学習を、3つの乱数を同じseedで初期化して実行する。

    戻り値: (実旧のモデル, 実旧の統計, 実行後のPythonの乱数の状態, NumPyの乱数の状態, torchの乱数の状態)。
    呼出し側の乱数は進めない。
    """
    set_legacy_configuration(monkeypatch, class_count=class_count, update_interval=2)
    for legacy_setting_name, legacy_setting_value in dict(
        DATASET="sine2",
        PRETRAIN_SAMPLES=pretraining_values["pretraining_sample_count"],
        PRETRAIN_EPOCHS=pretraining_values["pretraining_epoch_count"],
        PRETRAIN_BATCH_SIZE=pretraining_values["pretraining_batch_sample_count"],
        OPTIMIZER="sgd" if optimizer_variant == "sgd" else "adam",
        AMSGRAD=optimizer_variant == "amsgrad",
    ).items():
        assert hasattr(config, legacy_setting_name), legacy_setting_name
        monkeypatch.setattr(config, legacy_setting_name, legacy_setting_value)
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            # 旧のrunと同じ順で、3つの乱数を同じseedで初期化する。
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            legacy_model, legacy_statistics = experiment._pretrain_initial_model(ResidualAdapterMLP)
            return (
                legacy_model,
                legacy_statistics,
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state().clone(),
            )
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)


def assert_numpy_random_states_equal(numpy_random_state, legacy_numpy_random_state):
    assert numpy_random_state[0] == legacy_numpy_random_state[0]
    assert np.array_equal(numpy_random_state[1], legacy_numpy_random_state[1])
    assert numpy_random_state[2:] == legacy_numpy_random_state[2:]


def assert_pretrained_model_matches_legacy(
    *, pretrained_initial_model, legacy_model, legacy_statistics
):
    """分類器の全パラメータ、2つのoptimizerの状態、損失統計を、実旧の事前学習の結果と照合する。"""
    assert type(pretrained_initial_model) is PretrainedInitialModel
    classifier = pretrained_initial_model.classifier
    assert type(classifier) is ResidualAdapterClassifier
    legacy_state = legacy_model.state_dict()
    classifier_state = classifier.state_dict()
    assert list(classifier_state) == [
        convert_legacy_parameter_name(parameter_name) for parameter_name in legacy_state
    ]
    for parameter_name, legacy_parameter in legacy_state.items():
        assert torch.equal(
            classifier_state[convert_legacy_parameter_name(parameter_name)], legacy_parameter
        ), parameter_name
    assert_nested_state_equal(
        pretrained_initial_model.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
        legacy_model.head_optimizer.state_dict(),
    )
    assert_nested_state_equal(
        pretrained_initial_model.shared_parameter_optimizer_state.parameter_optimizer.state_dict(),
        legacy_model.backbone.optimizer.state_dict(),
    )
    assert type(pretrained_initial_model.loss_statistics) is ModelAndClassLossStatistics
    assert pretrained_initial_model.loss_statistics == convert_legacy_loss_statistics(
        legacy_statistics
    )


# (標本数, epoch数, batchの件数)。2の冪でない件数、最後のbatchが端数、batchが標本数より大きい、epoch数0、1標本。
PRETRAINING_SIZE_CASES = (
    (24, 2, 8),
    (25, 3, 7),
    (30, 2, 10),
    (23, 2, 5),
    (100, 2, 32),
    (9, 4, 50),
    (13, 0, 4),
    (1, 3, 1),
    (17, 1, 3),
    # 旧の既定の設定（最後のbatchは20件）。
    (500, 10, 32),
)


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("pretraining_sizes", PRETRAINING_SIZE_CASES)
@pytest.mark.parametrize("optimizer_variant", ("amsgrad", "standard", "sgd"))
def test_pretraining_matches_real_legacy_pretraining(
    class_count, pretraining_sizes, optimizer_variant, monkeypatch
):
    random_seed = 11 + pretraining_sizes[0]
    pretraining_values = dict(zip(VALID_PRETRAINING_VALUES, pretraining_sizes, strict=True))
    (
        legacy_model,
        legacy_statistics,
        legacy_python_random_state,
        legacy_numpy_random_state,
        legacy_torch_random_state,
    ) = run_real_legacy_pretraining(
        monkeypatch=monkeypatch,
        class_count=class_count,
        optimizer_variant=optimizer_variant,
        random_seed=random_seed,
        pretraining_values=pretraining_values,
    )
    pretraining_arguments = make_pretraining_arguments(
        class_count=class_count,
        optimizer_variant=optimizer_variant,
        random_seed=random_seed,
        pretraining_values=pretraining_values,
    )
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)
        pretrained_initial_model = pretrain_initial_model(**pretraining_arguments)
        torch_random_state = torch.get_rng_state().clone()
    assert_pretrained_model_matches_legacy(
        pretrained_initial_model=pretrained_initial_model,
        legacy_model=legacy_model,
        legacy_statistics=legacy_statistics,
    )
    # 統計は、全標本ぶん。2値のSINEなので、クラス別は多くて2つ。
    loss_statistics = pretrained_initial_model.loss_statistics
    assert loss_statistics.overall_loss_moments.observed_loss_count == pretraining_sizes[0]
    assert (
        sum(
            class_loss_moments.observed_loss_count
            for _, class_loss_moments in loss_statistics.class_loss_moments_by_class_id
        )
        == pretraining_sizes[0]
    )
    # 実行後の3つの乱数の状態。新は、借りた生成器だけを進め、全体のPython・NumPyの乱数を使わない。
    assert pretraining_arguments["python_random_generator"].getstate() == legacy_python_random_state
    assert_numpy_random_states_equal(
        pretraining_arguments["sample_generator"].numpy_random_generator.get_state(),
        legacy_numpy_random_state,
    )
    assert torch.equal(torch_random_state, legacy_torch_random_state)
    assert random.getstate() == global_python_random_state
    assert_numpy_random_states_equal(np.random.get_state(), global_numpy_random_state)


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_variant", ("amsgrad", "sgd"))
def test_pretraining_matches_real_legacy_pretraining_for_every_batch_sample_count(
    class_count, optimizer_variant, monkeypatch
):
    """batchの件数1〜33のすべてで、実旧のモデルの更新と一致する。

    新は、参加するモデルが1つの共同更新（損失×件数÷件数）で更新するので、件数ごとに確かめる。
    標本数をbatchの件数と同じにして、1 epochに、その件数のbatchを1つだけ作り、3 epoch更新する。
    """
    for batch_sample_count in range(1, 34):
        pretraining_values = dict(
            pretraining_sample_count=batch_sample_count,
            pretraining_epoch_count=3,
            pretraining_batch_sample_count=batch_sample_count,
        )
        random_seed = 100 + batch_sample_count
        legacy_model, legacy_statistics, _, _, _ = run_real_legacy_pretraining(
            monkeypatch=monkeypatch,
            class_count=class_count,
            optimizer_variant=optimizer_variant,
            random_seed=random_seed,
            pretraining_values=pretraining_values,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            pretrained_initial_model = pretrain_initial_model(
                **make_pretraining_arguments(
                    class_count=class_count,
                    optimizer_variant=optimizer_variant,
                    random_seed=random_seed,
                    pretraining_values=pretraining_values,
                )
            )
        assert_pretrained_model_matches_legacy(
            pretrained_initial_model=pretrained_initial_model,
            legacy_model=legacy_model,
            legacy_statistics=legacy_statistics,
        )


def test_pretraining_rejects_disabled_gradient_calculation_before_consuming_random_numbers():
    """勾配の計算が無効のとき、更新を行う設定は、分類器を作る前に拒否する。epoch数0は受け入れる。"""
    pretraining_arguments = make_pretraining_arguments()
    python_random_state = pretraining_arguments["python_random_generator"].getstate()
    numpy_random_state = pretraining_arguments[
        "sample_generator"
    ].numpy_random_generator.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        torch_random_state = torch.get_rng_state().clone()
        with torch.no_grad(), pytest.raises(ValueError, match="gradient calculation"):
            pretrain_initial_model(**pretraining_arguments)
        assert torch.equal(torch.get_rng_state(), torch_random_state)
        assert pretraining_arguments["python_random_generator"].getstate() == python_random_state
        assert_numpy_random_states_equal(
            pretraining_arguments["sample_generator"].numpy_random_generator.get_state(),
            numpy_random_state,
        )
        with torch.no_grad():
            pretrained_without_training = pretrain_initial_model(
                **make_pretraining_arguments(
                    pretraining_values=VALID_PRETRAINING_VALUES | dict(pretraining_epoch_count=0)
                )
            )
        assert type(pretrained_without_training) is PretrainedInitialModel
        assert type(pretrain_initial_model(**pretraining_arguments)) is PretrainedInitialModel


def test_pretraining_result_matches_client_assembly_arguments():
    """2つのoptimizerは、返した分類器の概念固有部（アダプタ→分類層）と共有部を、その順で指す。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        pretrained_initial_model = pretrain_initial_model(**make_pretraining_arguments())
    classifier = pretrained_initial_model.classifier
    for parameter_optimizer_state, expected_parameters in (
        (
            pretrained_initial_model.concept_specific_parameter_optimizer_state,
            tuple(classifier.residual_adapter.parameters())
            + tuple(classifier.classification_layer.parameters()),
        ),
        (
            pretrained_initial_model.shared_parameter_optimizer_state,
            tuple(classifier.feature_extractor.parameters()),
        ),
    ):
        assert type(parameter_optimizer_state) is ParameterOptimizerState
        assert [
            id(parameter)
            for parameter_group in parameter_optimizer_state.parameter_optimizer.param_groups
            for parameter in parameter_group["params"]
        ] == [id(parameter) for parameter in expected_parameters]
    assert classifier.feature_extractor.input_feature_count == 2
    assert classifier.class_count == 2
    with pytest.raises(AttributeError):
        pretrained_initial_model.classifier = None


def test_pretraining_with_zero_epochs_keeps_initialized_classifier_and_computes_statistics():
    pretraining_values = VALID_PRETRAINING_VALUES | dict(pretraining_epoch_count=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        pretraining_arguments = make_pretraining_arguments(pretraining_values=pretraining_values)
        initialized_classifier = ResidualAdapterClassifier(
            model_architecture_settings=pretraining_arguments["model_architecture_settings"],
            input_feature_count=2,
            hidden_layer_widths=HIDDEN_LAYER_WIDTHS,
            class_count=2,
        )
        torch.manual_seed(7)
        python_random_state = pretraining_arguments["python_random_generator"].getstate()
        pretrained_initial_model = pretrain_initial_model(**pretraining_arguments)
    for (parameter_name, parameter), (_, initialized_parameter) in zip(
        pretrained_initial_model.classifier.named_parameters(),
        initialized_classifier.named_parameters(),
        strict=True,
    ):
        assert torch.equal(parameter, initialized_parameter), parameter_name
        assert parameter.grad is None
    # shuffleも更新もしない。統計は全標本ぶん求まる。
    assert pretraining_arguments["python_random_generator"].getstate() == python_random_state
    assert (
        pretrained_initial_model.loss_statistics.overall_loss_moments.observed_loss_count
        == pretraining_values["pretraining_sample_count"]
    )
    for optimizer_state in (
        pretrained_initial_model.concept_specific_parameter_optimizer_state,
        pretrained_initial_model.shared_parameter_optimizer_state,
    ):
        assert optimizer_state.parameter_optimizer.state_dict()["state"] == {}


def make_settings_mutated_around_frozen(settings_instance, field_name, invalid_value):
    mutated_settings = replace(settings_instance)
    object.__setattr__(mutated_settings, field_name, invalid_value)
    return mutated_settings


class SineSampleGeneratorSubclass(SineSampleGenerator):
    pass


# 条件名 -> (正常な引数から、差し替える引数を作る操作, 期待する例外)。
INVALID_PRETRAINING_ARGUMENT_CASES = {
    "pretraining_settings_other_type": (
        lambda arguments: dict(initial_model_pretraining_settings=dict(VALID_PRETRAINING_VALUES)),
        TypeError,
    ),
    "pretraining_settings_subclass": (
        lambda arguments: dict(
            initial_model_pretraining_settings=make_subclass_copy(
                arguments["initial_model_pretraining_settings"]
            )
        ),
        TypeError,
    ),
    "pretraining_settings_mutated_around_frozen": (
        lambda arguments: dict(
            initial_model_pretraining_settings=make_settings_mutated_around_frozen(
                arguments["initial_model_pretraining_settings"], "pretraining_sample_count", 0
            )
        ),
        RunSettingsValidationError,
    ),
    "architecture_settings_other_type": (
        lambda arguments: dict(model_architecture_settings=object()),
        TypeError,
    ),
    "architecture_settings_mutated_around_frozen": (
        lambda arguments: dict(
            model_architecture_settings=make_settings_mutated_around_frozen(
                arguments["model_architecture_settings"], "residual_adapter_requested_rank", 0
            )
        ),
        ValueError,
    ),
    "optimizer_settings_other_type": (
        lambda arguments: dict(parameter_optimizer_settings=None),
        TypeError,
    ),
    "optimizer_settings_subclass": (
        lambda arguments: dict(
            parameter_optimizer_settings=make_subclass_copy(
                arguments["parameter_optimizer_settings"]
            )
        ),
        TypeError,
    ),
    "optimizer_settings_mutated_around_frozen": (
        lambda arguments: dict(
            parameter_optimizer_settings=make_settings_mutated_around_frozen(
                arguments["parameter_optimizer_settings"], "learning_rate", -1.0
            )
        ),
        ValueError,
    ),
    "sample_generator_other_type": (
        lambda arguments: dict(sample_generator=np.random.RandomState(7)),
        TypeError,
    ),
    "sample_generator_subclass": (
        lambda arguments: dict(
            sample_generator=SineSampleGeneratorSubclass(
                numpy_random_generator=np.random.RandomState(7)
            )
        ),
        TypeError,
    ),
    # 標本生成器が持つ乱数が、NumPyの新しい方式の生成器。
    "sample_generator_holding_other_random_generator": (
        lambda arguments: dict(
            sample_generator=SineSampleGenerator(numpy_random_generator=np.random.default_rng(7))
        ),
        TypeError,
    ),
    "random_generator_other_type": (
        lambda arguments: dict(python_random_generator=random),
        TypeError,
    ),
    "random_generator_subclass": (
        lambda arguments: dict(
            python_random_generator=type("RandomSubclass", (random.Random,), {})(7)
        ),
        TypeError,
    ),
    "hidden_layer_widths_list": (lambda arguments: dict(hidden_layer_widths=[5, 4]), ValueError),
    # 隠れ層がない（共有部にパラメータがなく、共有部のoptimizerを作れない）。
    "hidden_layer_widths_empty": (lambda arguments: dict(hidden_layer_widths=()), ValueError),
    "hidden_layer_width_zero": (lambda arguments: dict(hidden_layer_widths=(5, 0)), ValueError),
    "hidden_layer_width_bool": (lambda arguments: dict(hidden_layer_widths=(True,)), ValueError),
    "class_count_bool": (lambda arguments: dict(class_count=True), TypeError),
    "class_count_float": (lambda arguments: dict(class_count=2.0), TypeError),
    "class_count_one": (lambda arguments: dict(class_count=1), ValueError),
}


@pytest.mark.parametrize("invalid_case_name", INVALID_PRETRAINING_ARGUMENT_CASES)
def test_pretraining_rejects_invalid_arguments_before_consuming_random_numbers(invalid_case_name):
    make_invalid_arguments, expected_exception = INVALID_PRETRAINING_ARGUMENT_CASES[
        invalid_case_name
    ]
    pretraining_arguments = make_pretraining_arguments()
    invalid_arguments = make_invalid_arguments(pretraining_arguments)
    python_random_state = pretraining_arguments["python_random_generator"].getstate()
    numpy_random_state = pretraining_arguments[
        "sample_generator"
    ].numpy_random_generator.get_state()
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        torch_random_state = torch.get_rng_state().clone()
        with pytest.raises(expected_exception):
            pretrain_initial_model(**pretraining_arguments | invalid_arguments)
        assert torch.equal(torch.get_rng_state(), torch_random_state)
        assert pretraining_arguments["python_random_generator"].getstate() == python_random_state
        assert_numpy_random_states_equal(
            pretraining_arguments["sample_generator"].numpy_random_generator.get_state(),
            numpy_random_state,
        )
        assert random.getstate() == global_python_random_state
        assert_numpy_random_states_equal(np.random.get_state(), global_numpy_random_state)
        # 拒否の後に、正しい引数で実行できる。
        assert type(pretrain_initial_model(**pretraining_arguments)) is PretrainedInitialModel


@pytest.mark.parametrize("class_count", (2, 4))
def test_client_assembled_from_pretraining_matches_real_legacy_client(
    class_count, monkeypatch, valid_run_settings_mapping
):
    """新の事前学習の結果から組み立てたclientを、実旧の事前学習の結果から実__init__で作った実旧clientと照合する。"""
    random_seed = 19
    # 標本列は、旧の設定を差し替える前に作る（標本列の生成は、概念が4つある既定のdatasetを使う）。
    stream = make_concept_stream(sample_count=100, concept_block_length=22, stream_seed=11)
    legacy_model, legacy_statistics, legacy_python_random_state, _, _ = run_real_legacy_pretraining(
        monkeypatch=monkeypatch,
        class_count=class_count,
        optimizer_variant="amsgrad",
        random_seed=random_seed,
        pretraining_values=VALID_PRETRAINING_VALUES,
    )
    legacy_client = ResidualAdapterRestartingSoftRoutingFedSDAClient(
        client_id=1,
        initial_models={0: legacy_model},
        initial_stats={0: legacy_statistics},
        distance_threshold=MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
        verbose=False,
    )
    pretraining_arguments = make_pretraining_arguments(
        class_count=class_count, random_seed=random_seed
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)
        pretrained_initial_model = pretrain_initial_model(**pretraining_arguments)
    # clientは、事前学習と同じ乱数生成器を続けて使う（旧は、module全体の乱数を続けて使う）。
    python_random_generator = pretraining_arguments["python_random_generator"]
    assert python_random_generator.getstate() == legacy_python_random_state
    run_client = assemble_fedsda_run_client(
        client_id=1,
        initial_model_id=0,
        initial_classifier=pretrained_initial_model.classifier,
        initial_concept_specific_parameter_optimizer_state=pretrained_initial_model.concept_specific_parameter_optimizer_state,
        initial_shared_parameter_optimizer_state=pretrained_initial_model.shared_parameter_optimizer_state,
        initial_loss_statistics=pretrained_initial_model.loss_statistics,
        run_client_settings=make_run_client_settings(valid_run_settings_mapping, update_interval=2),
        python_random_generator=python_random_generator,
    )
    assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for sample_index, (observed_sample, concept_id) in enumerate(stream):
            run_in_both(
                run_client=run_client,
                legacy_client=legacy_client,
                python_random_generator=python_random_generator,
                legacy_operation=lambda observed_sample=observed_sample, concept_id=concept_id: (
                    legacy_client.process_one_step(
                        torch.tensor(observed_sample.feature_values, dtype=torch.float32),
                        torch.tensor([float(observed_sample.class_label)]),
                        concept_id,
                    )
                ),
                operation=lambda observed_sample=observed_sample, sample_index=sample_index, concept_id=concept_id: (
                    run_client.process_observed_sample(
                        observed_sample=observed_sample,
                        sample_index=sample_index,
                        evaluation_concept_id=concept_id,
                    )
                ),
            )
            if (sample_index + 1) % 10 == 0:
                run_in_both(
                    run_client=run_client,
                    legacy_client=legacy_client,
                    python_random_generator=python_random_generator,
                    legacy_operation=legacy_client.flush_pending_updates,
                    operation=lambda sample_index=sample_index: (
                        run_client.flush_pending_local_updates(round_index=sample_index // 10)
                    ),
                )
                assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
    assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
    # この標本列で、警報が起きている（生成直後の状態から、学習と適応が進んでいる）。
    assert (
        run_client.owners.loss_change_alarm_record_store.get_state_snapshot().alarm_sample_indices
    )
