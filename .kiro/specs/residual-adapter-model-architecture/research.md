# 調査と設計判断

## 調査範囲
既存構造の移植なのでintegration-focused light discoveryを行った。二つの既存agent threadが旧models.pyとdata/specs.pyを調査し、主担当が要求全体を統合した。
fable-method、kiro-spec-requirements/design/tasks/impl、命名review規則とrefactoring-policyを参照。新しい外部ライブラリは追加しない。

## 旧構造の直接根拠
- SharedFeatureBackboneはLinear/ReLUを宣言幅順に組む。空幅なら入力と同寸法。
- ResidualConceptAdapterはdown→ReLU→up。upの通常初期化後にzero化するため、乱数消費を省略してはいけない。
- ResidualAdapterMLPはbackbone→adapter→headを構築し、binaryならSigmoid、multiclassはIdentity。
- 注入backboneはコピーされず共有される。optimizer builderをtest-only no-opにしても構築後CPU RNGは変わらないことを調査者が実測した。
- 元データの寸法はSINE2特徴/2クラス/幅32,32、SEA3特徴/2クラス、MNIST784特徴/10クラス/幅1568。新モデルは名称を解決せず明示数値を受ける。

## 主担当によるsynthesis
共通問題はNN構造と所有/初期化順であり、学習方針ではない。extractor/adapter/classifierの三つへ責務を分け、nn.Moduleと標準state_dictを採用する。
汎用model protocol・factory・Tensor validatorやsnapshot readerは今回不要。生成前寸法検査はextractorの公開静的methodで再利用する。
共有注入は宣言だけでなく実層構造も検査する。通常forwardへの外部.to/layer置換対応は対象外。
旧重み名へのprefix変換はtest-onlyとし、productionにaliasや互換読込みを残さない。
空幅は構造として受理するが、旧optimizerの空Parameter制約を再実装しない。学習時の対応は後続で判断する。

## 保存前design gate
主担当は12条件のtraceability、三つのnn.Moduleの役割、具体path、exact依存、正常旧照合と異常拒否、共有所有、未移植範囲、既存環境で実行可能な検証を確認してPASS。孤立componentや仮定のfactoryはない。
新しい旧不具合はこの調査では観測していない。実装検証で再現した問題だけ既存findings台帳へ記録する。
