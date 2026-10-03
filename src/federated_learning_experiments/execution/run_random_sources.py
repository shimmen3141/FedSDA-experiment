"""runごとに作るPython・NumPyの可変乱数源。"""

from dataclasses import dataclass
from random import Random

from numpy.random import RandomState


@dataclass(kw_only=True)
class RunRandomSources:
    """固定条件と区別し、借用する処理部だけが乱数を消費する。"""

    python_random_generator: Random
    numpy_random_generator: RandomState


def create_run_random_sources(*, random_seed: int) -> RunRandomSources:
    """呼出元のglobal乱数を使わず、旧方式の新しい実体を作る。"""
    return RunRandomSources(
        python_random_generator=Random(random_seed),
        numpy_random_generator=RandomState(random_seed),
    )
