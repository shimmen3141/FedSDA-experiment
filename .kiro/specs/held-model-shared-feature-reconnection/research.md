# 調査と設計判断

## Summary
既存責務の拡張としてlight discoveryを実施。方針・steering・前spec・旧clients/shared_backbone.pyを調査した。
fable-methodの実測重視、cc-sddの要求/設計/命名/タスク/実装レビュー、naming-reviewを適用する。新外部ライブラリなし。

## 旧実装と接続先
- _share_model_backbones: 空なら無操作、最小非負IDを選び、全負ならdict先着。入力順に共有元以外だけattach_backboneする。
- attach_backbone: 共有参照を交換し、接続先に既存共有optimizerがあれば保持、個別optimizerだけ作り直す。
- 前specのattach_shared_feature_extractorは参照交換のみ。ParameterOptimizerStateは固定parameter列を借用し現在optimizerの交換を所有する。
- 現在のHeldModelTrainingBindingはoptimizerの借用参照なので、reset後に新recordを作る必要がある。

## Synthesis / Boundary Decisions
共有元選択だけを小さな独立ファイルにせず、保有モデル再接続の外側操作と借用記録を一つのmoduleに置く。
各optimizer ownerを入力から借用し、新たな一覧owner・client継承・汎用registryは作らない。
単一分類器の接続とoptimizer resetは既存公開APIを再利用する。正常モデルのみを対象とし、全input対応の事前検証後に旧順序で副作用を実行する。
予期しないreset例外で全体rollbackはしない。これは旧attach順序と一致し、通常の不正input事前拒否と区別する。
過去の借用recordは現在の接続/optimizer対応を自動更新しない。成功時の返却記録から学習記録を作る。
旧不具合正本は../../../docs/research/implementation-findings/README.md。新たな旧正常系不具合は現時点で観測していない。LEGACY-010の空共有optimizerは本specでも変更しない。
