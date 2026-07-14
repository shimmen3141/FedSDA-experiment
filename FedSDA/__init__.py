"""FedSDA 実験パッケージ。

主要エントリポイント:
- run_random_drift_experiment: 1回分の実験実行
- run_comparative_trials: 複数シードでの比較試行

ハイパーパラメータは FedSDA/config.py で一元管理する。
"""
from . import config
from .adwin import FullScanADWIN
from .clients import AdwinClient, BaseClient, ObliviousClient, PeriodicClient
from .data import generate_data, make_concept_schedules
from .experiment import run_random_drift_experiment
from .models import SimpleMLP
from .server import BaseServer, ClusteringServer, ClusteringServerV2
from .trials import run_comparative_trials

__all__ = [
    "config",
    "FullScanADWIN",
    "AdwinClient",
    "BaseClient",
    "PeriodicClient",
    "ObliviousClient",
    "generate_data",
    "make_concept_schedules",
    "run_random_drift_experiment",
    "SimpleMLP",
    "BaseServer",
    "ClusteringServer",
    "ClusteringServerV2",
    "run_comparative_trials",
]
