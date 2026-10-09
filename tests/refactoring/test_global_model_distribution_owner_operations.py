"""配布の受取りのために、既存のownerへ足した操作: 全置換え、統計の一覧、サーバによる付け替えの記録、共有部のoptimizerの状態の保持。"""

import pytest
import torch
from test_alarm_adaptation_recording import make_adaptation_record
from test_global_model_repository_and_communication_volume import (
    make_loss_statistics,
    make_repository,
)
from test_held_candidate_validation_progress import make_subclass_copy

from federated_learning_experiments.evaluation.adaptation_record_store import (
    SERVER_REMAP_ADAPTATION_OUTCOME,
    AdaptationRecordStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingState,
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.shared_parameter_optimizer_state_holder import (
    SharedParameterOptimizerStateHolder,
)

OPTIMIZER_SETTINGS = SgdParameterOptimizerSettings(learning_rate=0.01)


def make_classifier():
    return ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(3,),
        class_count=2,
    )


def make_concept_specific_optimizer_state(classifier):
    return ParameterOptimizerState(
        parameters=tuple(classifier.residual_adapter.parameters())
        + tuple(classifier.classification_layer.parameters()),
        optimizer_settings=OPTIMIZER_SETTINGS,
    )


def make_held_state(model_id):
    classifier = make_classifier()
    return HeldModelTrainingState(
        model_id=model_id,
        classifier=classifier,
        concept_specific_parameter_optimizer_state=make_concept_specific_optimizer_state(
            classifier
        ),
    )


def make_registry_with_states(*held_states):
    registry = HeldModelTrainingStateRegistry()
    for held_state in held_states:
        registry.register_held_model_training_state(
            model_id=held_state.model_id,
            classifier=held_state.classifier,
            concept_specific_parameter_optimizer_state=held_state.concept_specific_parameter_optimizer_state,
        )
    return registry


def test_registry_replaces_all_held_states_in_given_order():
    original_states = (make_held_state(0), make_held_state(-101))
    registry = make_registry_with_states(*original_states)
    replacing_states = (make_held_state(3), make_held_state(0), original_states[1])
    registry.replace_held_model_training_states(held_model_training_states=replacing_states)
    held_states = registry.snapshot_ordered_held_model_training_states()
    assert tuple(held_state.model_id for held_state in held_states) == (3, 0, -101)
    assert all(
        held_state is replacing_state
        for held_state, replacing_state in zip(held_states, replacing_states, strict=True)
    )
    # 元のID 0の記録は、新しい記録に置き換わっている。一時IDの記録は、渡した同じ記録のまま。
    assert registry.get_held_model_training_state(model_id=0) is replacing_states[1]
    assert registry.get_held_model_training_state(model_id=-101) is original_states[1]
    # 空のtupleで、全部を外せる。
    registry.replace_held_model_training_states(held_model_training_states=())
    assert registry.snapshot_ordered_held_model_training_states() == ()


def make_state_with_optimizer_of_other_classifier():
    held_state = make_held_state(5)
    return HeldModelTrainingState(
        model_id=5,
        classifier=held_state.classifier,
        concept_specific_parameter_optimizer_state=make_concept_specific_optimizer_state(
            make_classifier()
        ),
    )


@pytest.mark.parametrize(
    "make_invalid_states,expected_exception",
    [
        (lambda: [make_held_state(1)], TypeError),
        (lambda: (make_held_state(1), object()), TypeError),
        (lambda: (make_held_state(1), make_subclass_copy(make_held_state(2))), TypeError),
        # 同じIDが2つ。
        (lambda: (make_held_state(1), make_held_state(1)), ValueError),
        # 別の分類器のパラメータを指すoptimizerの状態。
        (lambda: (make_held_state(1), make_state_with_optimizer_of_other_classifier()), ValueError),
        (
            lambda: (
                HeldModelTrainingState(
                    model_id=True,
                    classifier=make_classifier(),
                    concept_specific_parameter_optimizer_state=make_held_state(
                        2
                    ).concept_specific_parameter_optimizer_state,
                ),
            ),
            ValueError,
        ),
    ],
)
def test_registry_rejects_invalid_replacement_without_change(
    make_invalid_states, expected_exception
):
    original_states = (make_held_state(0), make_held_state(-101))
    registry = make_registry_with_states(*original_states)
    with pytest.raises(expected_exception):
        registry.replace_held_model_training_states(
            held_model_training_states=make_invalid_states()
        )
    held_states = registry.snapshot_ordered_held_model_training_states()
    assert tuple(held_state.model_id for held_state in held_states) == (0, -101)
    assert held_states[0].classifier is original_states[0].classifier
    assert held_states[1].classifier is original_states[1].classifier


def test_loss_statistics_store_replaces_all_statistics_in_given_order():
    loss_statistics_store = ModelAndClassLossStatisticsStore(
        initial_loss_statistics_by_model_id={
            0: make_loss_statistics(mean_loss=0.1),
            4: make_loss_statistics(mean_loss=0.4),
        }
    )
    replacing_statistics = (
        (4, make_loss_statistics(mean_loss=0.75, observed_loss_count=9)),
        (-101, make_loss_statistics(mean_loss=0.5)),
        (2, make_loss_statistics(mean_loss=0.25)),
    )
    loss_statistics_store.replace_model_loss_statistics(
        loss_statistics_by_model_id=replacing_statistics
    )
    assert loss_statistics_store.get_state_snapshot() == replacing_statistics
    # 元のID 0の統計は、なくなっている。
    assert loss_statistics_store.get_model_loss_statistics(model_id=0) is None
    loss_statistics_store.replace_model_loss_statistics(loss_statistics_by_model_id=())
    assert loss_statistics_store.get_state_snapshot() == ()


@pytest.mark.parametrize(
    "invalid_statistics,expected_exception",
    [
        ([(1, make_loss_statistics())], TypeError),
        (((1, make_loss_statistics()), [2, make_loss_statistics()]), TypeError),
        (((1, make_loss_statistics(), 3),), ValueError),
        (((1, make_loss_statistics()), (1, make_loss_statistics())), ValueError),
        (((True, make_loss_statistics()),), TypeError),
        (((1, {"n": 3}),), TypeError),
    ],
)
def test_loss_statistics_store_rejects_invalid_replacement_without_change(
    invalid_statistics, expected_exception
):
    loss_statistics_store = ModelAndClassLossStatisticsStore(
        initial_loss_statistics_by_model_id={0: make_loss_statistics(mean_loss=0.1)}
    )
    state_snapshot = loss_statistics_store.get_state_snapshot()
    with pytest.raises(expected_exception):
        loss_statistics_store.replace_model_loss_statistics(
            loss_statistics_by_model_id=invalid_statistics
        )
    assert loss_statistics_store.get_state_snapshot() == state_snapshot


def test_repository_lists_loss_statistics_in_first_set_order():
    global_model_repository = make_repository()
    assert global_model_repository.snapshot_global_model_loss_statistics() == (
        (0, make_loss_statistics()),
    )
    third_statistics = make_loss_statistics(mean_loss=0.75)
    first_statistics = make_loss_statistics(mean_loss=0.5)
    global_model_repository.set_global_model_loss_statistics(
        model_id=3, loss_statistics=third_statistics
    )
    global_model_repository.set_global_model_loss_statistics(
        model_id=1, loss_statistics=first_statistics
    )
    # 既にあるIDの置換えは、最初に置いた位置のまま。
    replaced_statistics = make_loss_statistics(mean_loss=0.125, observed_loss_count=40)
    global_model_repository.set_global_model_loss_statistics(
        model_id=0, loss_statistics=replaced_statistics
    )
    listed_statistics = global_model_repository.snapshot_global_model_loss_statistics()
    assert listed_statistics == (
        (0, replaced_statistics),
        (3, third_statistics),
        (1, first_statistics),
    )
    # 一覧は、その後の変更の影響を受けない。パラメータを持たないIDの統計も載る。
    global_model_repository.set_global_model_loss_statistics(
        model_id=7, loss_statistics=first_statistics
    )
    assert len(listed_statistics) == 3
    assert global_model_repository.global_model_ids == (0,)


def test_adaptation_store_records_server_remap_separately_from_alarm_switches():
    adaptation_record_store = AdaptationRecordStore()
    adaptation_record_store.append_adaptation_record(
        adaptation_record=make_adaptation_record(
            adaptation_sample_index=12,
            adaptation_outcome="alarm_interval_held_model_reused",
            previous_training_model_id=0,
            current_training_model_id=1,
        )
    )
    server_remap_record = make_adaptation_record(
        adaptation_sample_index=20,
        detector_name="server",
        adaptation_outcome=SERVER_REMAP_ADAPTATION_OUTCOME,
        previous_training_model_id=1,
        current_training_model_id=0,
    )
    adaptation_record_store.append_adaptation_record(adaptation_record=server_remap_record)
    state_snapshot = adaptation_record_store.get_state_snapshot()
    assert state_snapshot.adaptation_records[-1] == server_remap_record
    # サーバによる付け替えは、警報による切替の位置にも、再利用・現行適合の件数にも数えない。
    assert state_snapshot.training_model_switch_sample_indices == (12,)
    assert state_snapshot.server_remapped_sample_indices == (20,)
    assert state_snapshot.alternative_model_reuse_count == 1
    assert state_snapshot.current_model_fit_count == 0
    # IDが変わらない付け替えの記録は作れない。
    with pytest.raises(ValueError, match="switching or server remap"):
        make_adaptation_record(
            adaptation_sample_index=20,
            adaptation_outcome=SERVER_REMAP_ADAPTATION_OUTCOME,
            previous_training_model_id=1,
            current_training_model_id=1,
        )


def test_shared_optimizer_state_holder_holds_and_replaces_one_state():
    first_classifier = make_classifier()
    first_state = ParameterOptimizerState(
        parameters=tuple(first_classifier.feature_extractor.parameters()),
        optimizer_settings=OPTIMIZER_SETTINGS,
    )
    holder = SharedParameterOptimizerStateHolder(shared_parameter_optimizer_state=first_state)
    assert holder.held_shared_parameter_optimizer_state is first_state
    second_classifier = make_classifier()
    second_state = ParameterOptimizerState(
        parameters=tuple(second_classifier.feature_extractor.parameters()),
        optimizer_settings=OPTIMIZER_SETTINGS,
    )
    holder.replace_shared_parameter_optimizer_state(shared_parameter_optimizer_state=second_state)
    assert holder.held_shared_parameter_optimizer_state is second_state
    for invalid_state in (None, second_state.parameter_optimizer, make_subclass_copy(second_state)):
        with pytest.raises(TypeError):
            holder.replace_shared_parameter_optimizer_state(
                shared_parameter_optimizer_state=invalid_state
            )
        assert holder.held_shared_parameter_optimizer_state is second_state
        with pytest.raises(TypeError):
            SharedParameterOptimizerStateHolder(shared_parameter_optimizer_state=invalid_state)
    # 保持者は、optimizerの状態そのものを変えない。
    assert torch.equal(
        next(iter(second_state.parameter_optimizer.param_groups[0]["params"])),
        next(iter(second_classifier.feature_extractor.parameters())),
    )
