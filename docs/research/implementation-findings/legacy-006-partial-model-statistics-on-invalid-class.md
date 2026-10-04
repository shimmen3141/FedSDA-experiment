# LEGACY-006: 不正class入力で旧モデル全体統計が部分更新する

## 状態と対象

2026-10-04、基準748c3aaのclients/base.py:130::BaseClient._update_model_stats。状態は再現済み・未修正。種別は直接APIの不正入力への堅牢性。発見specはbounded-loss-moments。

## 再現と観測

主担当と調査agentが旧メソッドを直接実行して確認した。空model_statsでclass_id=object()を渡すと、int(class_id)がTypeErrorになる前に全体の件数と平均が更新される。最小再現:

```python
from types import SimpleNamespace
from federated_drift_experiment.clients.base import BaseClient

client = SimpleNamespace(
    model_stats={}, _update_running_stats=BaseClient._update_running_stats,
)
try:
    BaseClient._update_model_stats(client, 7, .2, class_id=object())
except TypeError:
    pass
assert client.model_stats[7]["n"] == 1
assert client.model_stats[7]["mean"] == .2
```

worktreeの既存venvで実行し、上記assertとexit 0を確認した。

## 影響・今回の扱い

直接APIで例外を捕捉して続行すると全体・クラス統計の件数がずれる可能性がある。正常clientで不正class_idが渡るか、過去成果への影響は未確認。今回の新数値部品は入力状態を変えず事前検査するが、class_id/モデル統計mapを所有しないため、この旧APIの修正には相当しない。旧productionは変更しない。

## 将来の修正と検証

担当はモデル/クラス損失統計の所有・更新。class_idと損失の検証を更新前に完了し、関連する全系列を原子的に更新する契約を別途設計する。正常の全体/class oracle、例外後状態、旧11/最終3goldenで検証する。修正spec/commitは未定。

## 新統計管理部品での対応

2026-10-04、[model-and-class-loss-statistics](../../../.kiro/specs/model-and-class-loss-statistics/README.md)で、新storeは入力classを更新前に検査し、全体と指定classの更新候補を検査してから一回保存する設計を採用。
[テスト](../../../tests/refactoring/test_model_and_class_loss_statistics.py)の`test_model_class_statistics_rejects_invalid_input_atomically`で同じ不正class入力の旧部分更新と新store非変更を対照した。後段class count overflowでも全体を保存しないことを確認。
正常系列の全体/class値は旧直接oracleへ一致。全testsは2885 passed /3 skipped、旧11/最終3golden更新なし。詳細は同specのintegration-validation.md。
これは新部品の堅牢性契約であり、旧productionを修正した記録ではない。旧修正spec/commitと正常client・過去成果への影響調査は引き続き未定。
