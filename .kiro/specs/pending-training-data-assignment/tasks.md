# 実装タスク

- [x] 1. 観測位置の追加・状態copy・平時の容量超過解放を独立実装する
  - 既存設定から容量を受け取り、連続する位置だけを追加する。追加時に自動解放しない。
  - 旧process_one_stepへ直接投入し、容量1/3/30と開始位置0/71で解放順・保留を比較する。警報時の容量＋1件を保持できること、不正入力が状態を変更しないことを確認する。
  - 完了は対象テスト成功、Luna APPROVED、主担当の再検証で観測する。
  - _Requirements: 1.1, 1.2, 1.3, 3.3, 4.1, 4.2_
  - _Depends: 承認済み要件・設計・命名、既存Python/pytest環境とTrainingDataAssignmentSettings_
  - _Boundary: pending_training_assignment_buffer.py、同所有の対象テスト_

- [x] 2. 警報区間の非破壊分割・明示全件消費を実装する
  - 正spanをFIFO長で切り詰め、前区間・末尾区間とFIFO内開始位置を返す。空・過大spanも参照で状態を変えない。drain後も全体の標本順を継続する。
  - 旧開始位置、候補将来検証中・episode重複の全件消費、短い警報後の保持を直接照合する。LEGACY-002へ実行できる再現テストを接続する。
  - 完了は旧oracle照合・型/0span拒否・消費後連続性テスト成功、Luna APPROVEDと主担当再検証。
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 4.1, 4.2, 5.2_
  - _Depends: 1_
  - _Boundary: 同FIFO部品・対象テスト・共有LEGACY-002記録_

- [x] 3. 監視結果との位置接続・依存境界・全回帰を統合検証する
  - public監視結果の正spanを明示入力し、FIFO開始位置の切詰めを確認する。結果copy/frozen・実体間独立・共有乱数不変・keyword契約を検証する。
  - ASTをstdlibと同機能の設定だけのexact依存へ更新し、禁止注入が失敗することを確認する。全tests・旧11/最終3golden・旧importなし独立smokeを実行する。
  - 完了はLuna最終GOと主担当のfresh検証、14/14条件の証拠、部分完成範囲・後続責務・発見記録・roadmapを残す。新FedSDA全体run完成とは扱わない。
  - _Requirements: 3.3, 4.3, 5.1, 5.2_
  - _Depends: 1, 2、既存class-conditional-loss-monitoring public API_
  - _Boundary: 明示統合。対象テスト、test_single_run_dependency_boundaries.py、spec検証記録、roadmap。production監視へ依存追加しない_

