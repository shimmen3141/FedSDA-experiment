"""CPU torch乱数だけを単一runの間に隔離する。"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def isolated_cpu_torch_random_state(*, random_seed: int) -> Iterator[None]:
    """全出口でCPU状態を復元し、他device・thread・dtypeを変更しない。"""
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seedは全deviceへ作用するため、CPU実体だけをseed設定する。
        torch.default_generator.manual_seed(random_seed)
        yield
