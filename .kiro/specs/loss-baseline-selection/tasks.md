# 実装タスク
独立task graphレビューPASS。既存venv/pytest基盤を利用し順次実行する。

- [x] 1. 用途別の基準平均を選ぶ判断を実装する
  - 監視の欠落/0件既定と1件clip、再利用と警報後履歴の件数/ゼロ扱いを分離する。
  - 任意集計を独立コピーで検査し入力を変更しない。
  - 完了時、旧3用途のdirect oracleと境界条件が一致する。
  - _Boundary: baseline selection_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.2_

- [x] 2. 上位接続と拒否・共有状態を検証する
  - monitor/参照選択へ明示入力し、不正型/改変field拒否・入力不変・共有RNG/既定型/device/keywordを確認する。
  - 完了時、既存APIで判断が一致し副作用がない。productionの上位policyは追加しない。
  - _Depends: 1_
  - _Boundary: test integration_
  - _Requirements: 3.1, 3.2_

- [x] 3. 依存境界・全回帰と独立起動を統合検証する
  - exact moments module/宣言型のみAST許可し、private/別symbol/旧/数値lib/上位への依存を禁止注入する。
  - 完了時、全tests旧11/最終3golden不変・stdlib smoke・9条件/roadmap証拠が揃い、最終Luna GOを確認する。
  - _Depends: 1, 2_
  - _Boundary: final integration_
  - _Requirements: 3.3_

## Implementation Notes
モデル/class map、seed生成、snapshot時機と候補進行は後続。
