"""単一runの実行条件・乱数・呼出順を検証する。"""

import random
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from federated_learning_experiments.execution.run_random_sources import create_run_random_sources
from federated_learning_experiments.learning.models.torch_random_state_scope import (
    isolated_cpu_torch_random_state,
)


def test_run_random_sources_are_independent_and_reproduce_legacy_sequences():
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    run_random_sources = create_run_random_sources(random_seed=17)
    repeated_run_random_sources = create_run_random_sources(random_seed=17)
    expected_python_random_generator = random.Random(17)
    expected_numpy_random_generator = np.random.RandomState(17)
    assert run_random_sources.python_random_generator is not repeated_run_random_sources.python_random_generator
    assert run_random_sources.numpy_random_generator is not repeated_run_random_sources.numpy_random_generator
    assert run_random_sources.python_random_generator.random() == expected_python_random_generator.random()
    np.testing.assert_array_equal(
        run_random_sources.numpy_random_generator.uniform(size=10),
        expected_numpy_random_generator.uniform(size=10),
    )
    assert repeated_run_random_sources.python_random_generator.random() == random.Random(17).random()
    np.testing.assert_array_equal(
        repeated_run_random_sources.numpy_random_generator.uniform(size=10),
        np.random.RandomState(17).uniform(size=10),
    )
    assert random.getstate() == global_python_random_state
    np.testing.assert_equal(np.random.get_state(), global_numpy_random_state)


@pytest.mark.parametrize("fail_inside_scope", [False, True])
def test_cpu_torch_random_scope_restores_state_on_success_and_failure(fail_inside_scope):
    original_cpu_torch_random_state = torch.get_rng_state().clone()
    expected_cpu_random_generator = torch.Generator(device="cpu").manual_seed(17)
    with pytest.raises(ValueError, match="テスト内の失敗") if fail_inside_scope else nullcontext():
        with isolated_cpu_torch_random_state(random_seed=17):
            observed_cpu_random_values = torch.rand(10)
            assert torch.equal(
                observed_cpu_random_values,
                torch.rand(10, generator=expected_cpu_random_generator),
            )
            if fail_inside_scope:
                raise ValueError("テスト内の失敗")
    assert torch.equal(torch.get_rng_state(), original_cpu_torch_random_state)


def test_cpu_torch_random_scope_does_not_seed_other_devices_or_change_runtime_defaults(monkeypatch):
    non_cpu_seed_calls = []
    original_torch_thread_count = torch.get_num_threads()
    original_torch_default_dtype = torch.get_default_dtype()
    monkeypatch.setattr(torch, "manual_seed", non_cpu_seed_calls.append)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", non_cpu_seed_calls.append)
    monkeypatch.setattr(torch.xpu, "manual_seed_all", non_cpu_seed_calls.append)
    with isolated_cpu_torch_random_state(random_seed=0):
        torch.rand(3)
    assert non_cpu_seed_calls == []
    assert torch.get_num_threads() == original_torch_thread_count
    assert torch.get_default_dtype() == original_torch_default_dtype
