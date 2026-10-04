# 実装タスク
保存前の独立task graphレビューPASS。既存venv/pytest基盤で順次実行。

- [x] 1. モデル全体・クラス別の値と統計所有を実装する
  - 明示初期seed/一括置換、欠落と0件の独立参照、順序付きsnapshot、帰属損失の全体/指定class更新を実装する。
  - 正/負ID・class/None・未登録とseedの全更新値を旧直接oracleへ照合する。
  - 完了時、全体だけの件数差・非零1件seedを保持し、上位lifecycleを持たないstoreが動く。
  - _Boundary: model/class statistics_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 4.1_

- [ ] 2. 拒否の原子性・独立性と上位接続を検証する
  - ID/損失/seed/forged値と後段class overflowの拒否で全状態不変を確認する。
  - 入力map/seed/参照・別storeの独立、frozen/RNG/default型/device/keywordと、全体→baseline→monitor/参照の明示test接続を確認する。
  - 完了時、LEGACY006の旧部分更新と新拒否の差が記録され、正常系列を変えず副作用がない。
  - _Depends: 1_
  - _Boundary: test integration_
  - _Requirements: 3.1, 3.2, 3.3, 4.2_

- [ ] 3. 依存境界・全回帰と独立起動を統合検証する
  - exact数値module/2symbolだけをAST許可し、private/別symbol/別module/methods/旧/torch/NumPyを拒否する。
  - 完了時、全tests旧11/最終3golden不変・stdlib smoke・12条件/roadmap/台帳証拠と最終Luna GOが揃う。
  - _Depends: 1, 2_
  - _Boundary: final integration_
  - _Requirements: 4.3_

## Implementation Notes
統計単独の削除/全resetは旧の明確な操作がないため先取りしない。merge/ID変更、seed算出・modelpool、class上限は後続。
