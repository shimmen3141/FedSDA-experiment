# Brief: Fixed-Share予測重みの移植

## Problem / Current State

研究者が最終FedSDAの構成を拡張・削除しやすくするためのリファクタリング。
設定基盤と単一runの供給・順序は完成したが、最終構成の予測重み更新は旧実装に残っている。
予測、データ割当、監視、候補検証を一括移植すると状態の所有と数値差の原因が追いにくい。

## Desired Outcome / Approach

まず、最終構成が使用するFixed-Share予測重みの状態・更新・集約後の損失再生を独立して移植する。
旧748c3aaのSwitchingExpertRouterをテスト側のoracleとし、重み・分散・leader・再構成/再較正の記録を照合する。
この名称はspecの対象を示し、旧Router名の互換窓口は作らない。

## Scope / Boundary Candidates

- In: model ID集合の変更、ラベル観測前の予測重み、観測済み損失での次回用重み更新、leader同率処理、損失列の時系列再生、集約後再較正の記録。
- Out: tensor予測混合、モデル推論/学習、AdaHedge/Meta-switching/probe、混合有効期間、データ割当、ClassESR、候補採否、FIFOからの損失再計算、サーバ処理、研究指標集計、CLI/保存。
- 状態は予測重みの一所有者に限定し、クライアント全体の状態を渡さない。
- 呼出元がモデルID・損失・再生順を供給する。真の概念・ドリフト位置を受け取らない。

## Upstream / Downstream / Constraints

- Upstream: configuration-foundationのPredictionCombinationSettings。single-run-executionの契約は拡張しない。
- Downstream: モデル予測の混合、client進行、集約後のFIFO損失再計算。
- 固定環境・旧goldenを保持し、新依存・旧alias・旧production importを追加しない。
- 順序・浮動小数点の演算順を維持する。無効入力の扱いは新APIの境界として明示する。
- 要求・設計・命名・tasks・実装はLunaレビューと主担当の有用指摘反映で承認する。
