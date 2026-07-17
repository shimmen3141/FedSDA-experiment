"""FedSDA 実験パッケージ。

主要エントリポイント:
- run_random_drift_experiment: 1回分の実験実行
- run_comparative_trials: 複数シードでの比較試行

ハイパーパラメータは federated_drift_experiment/config.py で一元管理する。
"""
from . import config
from .adwin import FullScanADWIN
from .clients import (
    BaseClient,
    ClassConditionalFedSDAClient,
    EDetectorFedSDAClient,
    FedDriftClient,
    FedDriftV2Client,
    FedSDAClient,
    ObliviousClient,
)
from .data import generate_data, make_concept_schedules
from .drift_detectors import BoundedMeanEDetector
from .experiment import run_random_drift_experiment
from .models import SimpleMLP
from .servers import BaseServer, ClusteringServer, FedDriftV2Server, FedSDAV2Server, FedSDAV3Server
from .trials import run_comparative_trials

__all__ = [
    "config",
    "FullScanADWIN",
    "FedSDAClient",
    "ClassConditionalFedSDAClient",
    "EDetectorFedSDAClient",
    "BoundedMeanEDetector",
    "BaseClient",
    "FedDriftClient",
    "FedDriftV2Client",
    "ObliviousClient",
    "generate_data",
    "make_concept_schedules",
    "run_random_drift_experiment",
    "SimpleMLP",
    "BaseServer",
    "ClusteringServer",
    "FedSDAV2Server",
    "FedSDAV3Server",
    "FedDriftV2Server",
    "run_comparative_trials",
]
