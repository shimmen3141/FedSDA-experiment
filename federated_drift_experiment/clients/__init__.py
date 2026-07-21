"""クライアント実装パッケージ。

- BaseClient (base): 共通基底(モデル保持・統計・データストア・サーバ連携)
- FedSDAClient群 (fedsda): 検出器非依存基底 + ADWIN / e-SR / HDDM
- FedDriftClient (feddrift): FedDrift ベースライン。固定バッチ単位の損失増分検出
- ObliviousClient (oblivious): 単一モデル・無適応ベースライン

比較手法を追加する場合は、このパッケージに 1 ファイル追加して BaseClient を継承し、
ここで re-export した上で experiment.py の MODE_SPECS に登録する。
"""
from .base import BaseClient
from .fedsda import (
    ADWINFedSDAClient,
    ClassConditionalADWINFedSDAClient,
    ClassConditionalESRFedSDAClient,
    ClassConditionalHDDMAFedSDAClient,
    ESRFedSDAClient,
    FedSDAClient,
    HDDMFedSDAClient,
)
from .feddrift import FedDriftClient
from .oblivious import ObliviousClient

__all__ = [
    "BaseClient", "FedSDAClient", "ADWINFedSDAClient",
    "ClassConditionalADWINFedSDAClient", "ESRFedSDAClient",
    "ClassConditionalESRFedSDAClient",
    "ClassConditionalHDDMAFedSDAClient",
    "HDDMFedSDAClient",
    "FedDriftClient", "ObliviousClient",
]
