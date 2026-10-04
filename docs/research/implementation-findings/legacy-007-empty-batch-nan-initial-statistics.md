# LEGACY-007: 空batchでNaNの初期統計を登録する

## 状態と対象
2026-10-04、基準748c3aaのclients/base.py:286::BaseClient._register_trained_new_model。再現済み・未修正。種別は直接APIの不正空入力への堅牢性。発見specはbatch-loss-statistics-initialization。

## 再現と観測
学習/forwardなしの旧register stubを調査agentと主担当が実行した。空lossと空bx/byで全体n0/meanNaN/M2.1、class空が登録され、モデルも保持される。torch.mean(empty)のNaN平均は拒否されず、varのNaNだけが0.1へ置換される。

```python
import math
from types import SimpleNamespace
import torch
from federated_drift_experiment.clients.base import BaseClient
client = SimpleNamespace(models={}, model_stats={},
    _prepare_model_for_registration=lambda model: model,
    _record_model_compute=lambda *args: None)
model = SimpleNamespace(num_classes=2,
    per_sample_error=lambda bx, by: torch.empty(0), get_params=lambda: {})
BaseClient._register_trained_new_model(client, -1, model,
    torch.empty((0, 1)), torch.empty((0, 1)), True)
assert client.model_stats[-1]["n"] == 0
assert math.isnan(client.model_stats[-1]["mean"])
assert client.model_stats[-1]["M2"] == .1
assert -1 in client.models
```

共有venvで主担当がassert/exit0とLEGACY_EMPTY_BATCH_NAN_SEED_REPRODUCEDを確認。旧var警告は再現コマンド内だけ捕捉した。

## 影響と今回の扱い
空batchを直接渡すとNaNの初期統計を保持する。正常clientから空batch登録が起こるか、過去成果への影響は未確認。
新数値部品は非空・同じ標本数・有限loss/整数class範囲を入力契約として検査する。モデル登録を所有せず、旧production自体は変更しない。旧singleton全体M2.1は通常入力の値として維持する。

## 将来の修正と検証
担当はモデル登録時の入力検査と初期統計作成。prepare/モデル保持/送信より前に空batchなどを拒否し、例外後に外部状態が残らない契約を検討する。正常seed直接oracle、拒否後モデル/統計/pending状態、旧11/最終3goldenを検証する。旧修正spec/commitは未定。
