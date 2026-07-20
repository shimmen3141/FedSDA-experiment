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
    ClassConditionalEDetectorFedSDAClient,
    ClassConditionalFedSDAClient,
    EDetectorFedSDAClient,
    FedDriftClient,
    FedSDAClient,
    HDDMFedSDAClient,
    ObliviousClient,
)
from .data import generate_data, make_concept_schedules
from .e_detector import BoundedMeanEDetector
from .hddm import HDDMA, HDDMW
from .e_detector_baselines import (
    EDetectorBaselineEstimator,
    EmpiricalBernsteinUCB,
    HistoricalMeanBaseline,
)
from .experiment import run_random_drift_experiment
from .models import SimpleMLP
from .servers import (
    BaseServer,
    CrossEvaluationClusteringServer,
    FedDriftServer,
    FedSDACachedServer,
    FedSDANoCachedServer,
)
from .trials import run_comparative_trials

__all__ = [
    "config",
    "FullScanADWIN",
    "FedSDAClient",
    "ClassConditionalFedSDAClient",
    "ClassConditionalEDetectorFedSDAClient",
    "EDetectorFedSDAClient",
    "HDDMFedSDAClient",
    "BoundedMeanEDetector",
    "HDDMA",
    "HDDMW",
    "EDetectorBaselineEstimator",
    "HistoricalMeanBaseline",
    "EmpiricalBernsteinUCB",
    "BaseClient",
    "FedDriftClient",
    "ObliviousClient",
    "generate_data",
    "make_concept_schedules",
    "run_random_drift_experiment",
    "SimpleMLP",
    "BaseServer",
    "CrossEvaluationClusteringServer",
    "FedSDANoCachedServer",
    "FedSDACachedServer",
    "FedDriftServer",
    "run_comparative_trials",
]
