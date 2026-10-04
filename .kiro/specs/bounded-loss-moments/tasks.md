# 実装タスク

順次実行。既存venv・pytest・src基盤を使用する。Luna独立task graphレビューPASS、全12条件対応。

- [x] 1. 不変な損失統計と一件追加・平均標本分散を実装する
  - 空状態、非零1件seed、旧Welfordの演算順を維持する。
  - 不正値拒否・入力不変・frozen結果・Noneと真のゼロを検証する。
  - 完了時、全更新の件数・平均・偏差平方和と推定が旧直接oracleと一致する。
  - _Boundary: learning loss statistics_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 4.1_

- [x] 2. 用途別の上位接続と共有数値状態を検証する
  - 旧全体/class更新、保存平均から監視、2件以上の平均から候補履歴への明示入力をtestで確認する。
  - RNG・既定dtype/device・keyword契約を確認し、productionへ上位policyは追加しない。
  - 完了時、独立系列と上位接続が一致し共有状態が変わらない。
  - _Depends: 1_
  - _Boundary: test integration_
  - _Requirements: 4.1, 4.2_

- [ ] 3. 依存境界と全体回帰を統合検証する
  - stdlib許可・禁止依存注入、全tests・旧11/最終3golden不変、stdlibだけのsmokeを確認する。
  - 全12条件の証拠・部分完成範囲・roadmapを最終GO前に記録し、旧改善候補は共通台帳へ追跡する。
  - 完了時、全テストとLuna最終統合GOが得られ、旧productionとgoldenに差分がない。
  - _Depends: 1, 2_
  - _Boundary: final integration_
  - _Requirements: 4.3_

## Implementation Notes

- 初期batch seedの生成、監視baselineのclip、モデル/クラスID所有は本specの外。
