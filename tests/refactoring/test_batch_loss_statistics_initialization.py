"""外部batch損失の初期集計を旧登録時の数値処理へ直接照合する。"""

from types import SimpleNamespace
import warnings

import pytest
import torch

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import (
    initialize_model_and_class_loss_statistics_from_batch,
)


@pytest.mark.parametrize("loss_values, label_values, class_count", [
    ([0.25], [0], 2),
    ([0.75], [1], 2),
    ([0.2, 0.8], [1, 0], 2),
    ([0.2, 0.3, 0.8], [1, 1, 0], 2),
    ([0.0, 0.0, 0.0], [1, 0, 1], 2),
    ([1.0, 1.0, 1.0], [0, 1, 0], 2),
    ([0.5, 0.2, 0.8, 0.1], [3, 1, 3, 0], 5),
    ([0.2, 0.4, 0.6], [3, 3, 3], 5),
    ([0.10000001, 0.20000003, 0.29999998, 0.40000004, 0.50000006],
     [2, 0, 2, 1, 0], 4),
    ([1e-15, 2e-15, 3e-15, 4e-15], [2, 1, 2, 1], 4),
    ([0.4, 0.1, 0.8, 0.2, 0.6, 0.3], [1, 0, 1, 0, 1, 0], 2),
    ([0.3, 0.6, 0.2, 0.8, 0.1, 0.4], [0, 1, 0, 1, 0, 1], 2),
])
@pytest.mark.parametrize("operation", ["contiguous", "noncontiguous"])
def test_batch_initial_statistics_matches_legacy_registration(
    loss_values, label_values, class_count, operation,
):
    if operation == "contiguous":
        per_sample_bounded_losses = torch.tensor(loss_values, dtype=torch.float32, device="cpu")
        observed_class_labels = torch.tensor(label_values, dtype=torch.float32, device="cpu").reshape(-1, 1)
    else:
        per_sample_bounded_losses = torch.tensor(
            loss_values, dtype=torch.float32, device="cpu").repeat_interleave(2)[::2]
        observed_class_labels = torch.tensor(
            label_values, dtype=torch.float32, device="cpu").repeat_interleave(2).reshape(-1, 2)[:, :1]
        if len(loss_values) > 1:
            assert not per_sample_bounded_losses.is_contiguous()
            assert not observed_class_labels.is_contiguous()
    state_before_call = (per_sample_bounded_losses.clone(), observed_class_labels.clone())
    legacy_model = SimpleNamespace(
        num_classes=class_count,
        per_sample_error=lambda bx, by: per_sample_bounded_losses,
        get_params=lambda: {},
    )
    legacy_client = SimpleNamespace(
        models={}, model_stats={},
        _prepare_model_for_registration=lambda model: model,
        _record_model_compute=lambda *args: None,
    )
    # 旧singletonだけが発生させる未定義不偏分散warningをoracle内で抑制する。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        BaseClient._register_trained_new_model(
            legacy_client, temp_id=-100, new_model=legacy_model,
            bx=torch.zeros((len(loss_values), 1), dtype=torch.float32, device="cpu"),
            by=observed_class_labels, pending_ready=False,
        )
    result = initialize_model_and_class_loss_statistics_from_batch(
        per_sample_bounded_losses=per_sample_bounded_losses,
        observed_class_labels=observed_class_labels,
        class_count=class_count,
    )
    legacy_stats = legacy_client.model_stats[-100]
    assert (result.overall_loss_moments.observed_loss_count,
            result.overall_loss_moments.mean_loss,
            result.overall_loss_moments.sum_squared_loss_deviations) == (
                legacy_stats["n"], legacy_stats["mean"], legacy_stats["M2"])
    assert tuple(class_id for class_id, seed in result.class_loss_moments_by_class_id) == tuple(
        legacy_stats["class_stats"])
    for class_id, seed in result.class_loss_moments_by_class_id:
        assert (seed.observed_loss_count, seed.mean_loss, seed.sum_squared_loss_deviations) == (
            legacy_stats["class_stats"][class_id]["n"],
            legacy_stats["class_stats"][class_id]["mean"],
            legacy_stats["class_stats"][class_id]["M2"],
        )
    assert torch.equal(per_sample_bounded_losses, state_before_call[0])
    assert torch.equal(observed_class_labels, state_before_call[1])
    if len(loss_values) == 1:
        assert result.overall_loss_moments.sum_squared_loss_deviations == 0.1
        assert result.class_loss_moments_by_class_id[0][1].sum_squared_loss_deviations == 0.0


@pytest.mark.parametrize("field_name,invalid_value", [
    ("per_sample_bounded_losses", None), ("per_sample_bounded_losses", [.2,.7]),
    ("per_sample_bounded_losses", torch.tensor([.2,.7],dtype=torch.float64)),
    ("per_sample_bounded_losses", torch.tensor([0,1],dtype=torch.int64)),
    ("per_sample_bounded_losses", torch.tensor([False,True])),
    ("per_sample_bounded_losses", torch.empty(2,device="meta")),
    ("per_sample_bounded_losses", torch.tensor([.2,.7]).to_sparse()),
    ("per_sample_bounded_losses", torch.empty(0)),
    ("per_sample_bounded_losses", torch.tensor([[.2],[.7]])),
    ("per_sample_bounded_losses", torch.tensor([.2])),
    ("per_sample_bounded_losses", torch.tensor([float("nan"),.2])),
    ("per_sample_bounded_losses", torch.tensor([float("inf"),.2])),
    ("per_sample_bounded_losses", torch.tensor([-.1,.2])),
    ("per_sample_bounded_losses", torch.tensor([1.1,.2])),
    ("observed_class_labels", None), ("observed_class_labels", [[0],[1]]),
    ("observed_class_labels", torch.tensor([[0],[1]],dtype=torch.int64)),
    ("observed_class_labels", torch.tensor([[0],[1]],dtype=torch.float64)),
    ("observed_class_labels", torch.empty((2,1),device="meta")),
    ("observed_class_labels", torch.tensor([[0.],[1.]]).to_sparse()),
    ("observed_class_labels", torch.tensor([0.,1.])),
    ("observed_class_labels", torch.empty((0,1))),
    ("observed_class_labels", torch.tensor([[0.]])),
    ("observed_class_labels", torch.tensor([[0.],[.5]])),
    ("observed_class_labels", torch.tensor([[-1.],[1.]])),
    ("observed_class_labels", torch.tensor([[0.],[2.]])),
    ("observed_class_labels", torch.tensor([[float("nan")],[1.]])),
    ("observed_class_labels", torch.tensor([[float("inf")],[1.]])),
    ("class_count", True), ("class_count", 1), ("class_count", -1),
    ("class_count", 2.), ("class_count", "2"),
])
def test_batch_initial_statistics_rejects_invalid_input_without_mutation(field_name,invalid_value):
    per_sample_bounded_losses=torch.tensor([.2,.7],dtype=torch.float32,device="cpu")
    observed_class_labels=torch.tensor([[0.],[1.]],dtype=torch.float32,device="cpu")
    state_before_call=(per_sample_bounded_losses.clone(),observed_class_labels.clone())
    snapshot=invalid_value._version if isinstance(invalid_value,torch.Tensor) else None
    operation=dict(per_sample_bounded_losses=per_sample_bounded_losses,observed_class_labels=observed_class_labels,class_count=2)
    operation[field_name]=invalid_value
    with pytest.raises((TypeError,ValueError),match=field_name):
        initialize_model_and_class_loss_statistics_from_batch(**operation)
    assert torch.equal(per_sample_bounded_losses,state_before_call[0])
    assert torch.equal(observed_class_labels,state_before_call[1])
    if isinstance(invalid_value,torch.Tensor):
        assert invalid_value._version==snapshot
    if field_name=="per_sample_bounded_losses" and isinstance(invalid_value,torch.Tensor) and invalid_value.numel()==0:
        import math
        legacy_client=SimpleNamespace(models={},model_stats={},
            _prepare_model_for_registration=lambda model:model,_record_model_compute=lambda *args:None)
        legacy_model=SimpleNamespace(num_classes=2,per_sample_error=lambda bx,by:invalid_value,get_params=lambda:{})
        with pytest.warns(UserWarning):
            BaseClient._register_trained_new_model(legacy_client,-1,legacy_model,
                torch.empty((0,1)),torch.empty((0,1)),True)
        assert legacy_client.model_stats[-1]["n"]==0
        assert math.isnan(legacy_client.model_stats[-1]["mean"])
        assert legacy_client.model_stats[-1]["M2"]==.1


def test_batch_initial_statistics_preserves_independence_and_shared_state():
    from dataclasses import FrozenInstanceError
    import random
    import numpy as np
    per_sample_bounded_losses=torch.tensor([.25,.75],dtype=torch.float32,device="cpu",requires_grad=True)
    observed_class_labels=torch.tensor([[1.],[0.]],dtype=torch.float32,device="cpu")
    state_before_call=(per_sample_bounded_losses.detach().clone(),observed_class_labels.clone())
    global_python_random_state=random.getstate()
    global_numpy_random_state=np.random.get_state()
    global_torch_random_state=torch.get_rng_state().clone()
    global_default_dtype=torch.get_default_dtype()
    global_default_device=torch.get_default_device()
    global_grad_enabled=torch.is_grad_enabled()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            for operation in (torch.enable_grad,torch.no_grad):
                with operation():
                    loss_statistics=initialize_model_and_class_loss_statistics_from_batch(
                        per_sample_bounded_losses=per_sample_bounded_losses,
                        observed_class_labels=observed_class_labels,class_count=2)
                    assert torch.is_grad_enabled()==(operation is torch.enable_grad)
                    assert torch.get_default_dtype()==torch.float64
                    assert torch.get_default_device()==torch.device("meta")
        seed=initialize_model_and_class_loss_statistics_from_batch(
            per_sample_bounded_losses=per_sample_bounded_losses,observed_class_labels=observed_class_labels,class_count=2)
        assert torch.equal(per_sample_bounded_losses.detach(),state_before_call[0])
        assert torch.equal(observed_class_labels,state_before_call[1])
        assert loss_statistics==seed and loss_statistics is not seed
        assert loss_statistics.overall_loss_moments is not seed.overall_loss_moments
        assert loss_statistics.class_loss_moments_by_class_id[0][1] is not seed.class_loss_moments_by_class_id[0][1]
        with pytest.raises(FrozenInstanceError):
            loss_statistics.overall_loss_moments.mean_loss=.3
        with pytest.raises(FrozenInstanceError):
            loss_statistics.class_loss_moments_by_class_id=()
        object.__setattr__(loss_statistics.overall_loss_moments,"mean_loss",.9)
        assert seed.overall_loss_moments.mean_loss==.5
        with torch.no_grad():
            per_sample_bounded_losses.fill_(0)
            observed_class_labels.fill_(0)
        assert seed.overall_loss_moments.mean_loss==.5
        assert tuple(class_id for class_id,result in seed.class_loss_moments_by_class_id)==(0,1)
    finally:
        torch.set_default_dtype(global_default_dtype)
    assert per_sample_bounded_losses.grad is None
    assert random.getstate()==global_python_random_state
    assert np.random.get_state()[0]==global_numpy_random_state[0]
    assert np.array_equal(np.random.get_state()[1],global_numpy_random_state[1])
    assert np.random.get_state()[2:]==global_numpy_random_state[2:]
    assert torch.equal(torch.get_rng_state(),global_torch_random_state)
    assert torch.get_default_dtype()==global_default_dtype
    assert torch.get_default_device()==global_default_device
    assert torch.is_grad_enabled()==global_grad_enabled


def test_batch_initial_statistics_connects_to_store_and_baseline():
    from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore
    from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_loss_monitoring_baseline_mean_loss
    per_sample_bounded_losses=torch.tensor([.1,.3,.8],dtype=torch.float32,device="cpu")
    observed_class_labels=torch.tensor([[2.],[0.],[2.]],dtype=torch.float32,device="cpu")
    loss_statistics=initialize_model_and_class_loss_statistics_from_batch(
        per_sample_bounded_losses=per_sample_bounded_losses,observed_class_labels=observed_class_labels,class_count=4)
    store=ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={-1:loss_statistics})
    legacy_client=SimpleNamespace(models={},model_stats={},_prepare_model_for_registration=lambda model:model,
        _record_model_compute=lambda *args:None,_update_running_stats=BaseClient._update_running_stats)
    legacy_model=SimpleNamespace(num_classes=4,per_sample_error=lambda bx,by:per_sample_bounded_losses,get_params=lambda:{})
    BaseClient._register_trained_new_model(legacy_client,-1,legacy_model,
        torch.zeros((3,1)),observed_class_labels,True)
    for class_id in (2,1):
        store.record_assigned_loss(model_id=-1,observed_loss=.5,observed_class_id=class_id)
        BaseClient._update_model_stats(legacy_client,-1,.5,class_id=class_id)
        result=store.get_model_loss_statistics(model_id=-1)
        legacy_stats=legacy_client.model_stats[-1]
        assert (result.overall_loss_moments.observed_loss_count,result.overall_loss_moments.mean_loss,
                result.overall_loss_moments.sum_squared_loss_deviations)==(
                    legacy_stats["n"],legacy_stats["mean"],legacy_stats["M2"])
        assert tuple(class_id for class_id,seed in result.class_loss_moments_by_class_id)==tuple(legacy_stats["class_stats"])
        for class_id,seed in result.class_loss_moments_by_class_id:
            assert (seed.observed_loss_count,seed.mean_loss,seed.sum_squared_loss_deviations)==(
                legacy_stats["class_stats"][class_id]["n"],legacy_stats["class_stats"][class_id]["mean"],legacy_stats["class_stats"][class_id]["M2"])
    state_before_call=store.get_state_snapshot()
    assert select_loss_monitoring_baseline_mean_loss(loss_moments=result.overall_loss_moments)==result.overall_loss_moments.mean_loss
    assert store.get_state_snapshot()==state_before_call


def test_batch_initial_statistics_uses_keyword_arguments():
    with pytest.raises(TypeError):
        initialize_model_and_class_loss_statistics_from_batch(torch.tensor([.2]),torch.tensor([[0.]]),2)
