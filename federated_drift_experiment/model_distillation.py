"""概念モデル群を機能的なteacher混合から圧縮するための値オブジェクト。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DistillationUpdate:
    """一クライアントが学習したstudentの個別部と学習標本数。"""

    personalized_params: dict
    sample_count: int


@dataclass(frozen=True)
class DistillationDifferenceStats:
    """集約後studentとteacher混合の対応あり損失差十分統計。"""

    by_target_model: dict
