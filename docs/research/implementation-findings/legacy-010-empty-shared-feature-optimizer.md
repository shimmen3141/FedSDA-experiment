# LEGACY-010: 空の共有特徴抽出部でoptimizer生成が失敗

- 発見日: 2026-10-05（日本時間）。基準: 748c3aa。
- 発見spec: parameter-optimizer-construction。
- 種別: 空幅構造の受理と学習optimizer生成の境界不整合。
- 状態: 再現済み・未修正。通常実験への影響未確認。

## 観測・再現
旧models.pyのSharedFeatureBackbone(2, ())はidentityとして構築できるが、ResidualAdapterMLPへ注入すると空の共有Parameterを_build_component_optimizerへ渡し、ValueError: optimizer got an empty parameter listで失敗する。
共有venv、MPLCONFIGDIRを既存matplotlib-cacheへ設定し、次を新processで実行した。主担当の再現はexit0で期待例外を確認した。

```python
from federated_drift_experiment.models import ResidualAdapterMLP, SharedFeatureBackbone
ResidualAdapterMLP(input_dim=2, dataset="sine2", backbone=SharedFeatureBackbone(2, ()))
```

## 影響・今回の扱い
共有Parameterが空のモデルが使えないことは確認済み。既存goldenは非空隠れ層のため、通常clientと過去成果への影響は未確認。
新モデル構造specは空幅identityを受理する。今回optimizer builderは空tupleを明示拒否する責務だけで、共有optimizerの所有/省略を判断しない。
旧production/goldenは変更していない。今回の記録で修正や再実験を承認済みとは扱わない。

## 将来修正案
後続の共同/単一モデル学習specで、共有Parameterが空なら共有optimizerを生成せず個別Parameterだけを更新する案を検討する。
空共有部のforward/backward/個別optimizer step、非空構成の旧golden不変を検証する。旧実装側の修正採否と必要な再実験は未定。
