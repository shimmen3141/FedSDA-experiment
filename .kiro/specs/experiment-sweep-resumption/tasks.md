# Implementation Plan

- [x] 1. 排他と補助記録による安全な再実行を接続する
  - testを先に追加し、保存中断・旧NPZ・不一致・正常完了・排他競合と解放を検証する。
  - run開始前の補助記録、同一証拠のNPZ退避、全worker終了までの排他を既存実行へ接続する。
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3_
  - _Boundary: 実行と追加補助記録。既存artifact writerは変更しない。_
- [x] 2. 過去掃引の探索と進捗付き選択をCLIへ接続する
  - 新補助計画と旧READMEを読み、完全run集合と基準版を比較する。
  - 対話再開・新規・完了/キャンセル、非対話案内、print-plan、worker数変更、衝突しない新規名をtestで検証する。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3_
  - _Depends: 1_
  - _Boundary: 新掃引CLIのみ。_
- [x] 3. 使用手順・統合検証・独立レビューを完了する
  - 既存保存形式の不変と過去のNPZ単独の制限を文書化する。
  - 対象tests・依存境界・全pytest/golden・fresh・Ruff/Pyrightを確認し独立CLIの実装レビューを受ける。
  - _Requirements: 1.1, 2.1, 3.4, 4.1, 4.2, 4.3_
  - _Depends: 2_
  - _Boundary: docsと証拠。mainへマージしない。_

## Implementation Notes
- 2026-10-18: ユーザーは既存形式を変えない補助機能を許可。開始時はLuna high仕様レビュー待ち、実装未着手だった。source基点8b0c095。

- 仕様レビュー: Luna high PASS、Major0/Minor1。元READMEを維持し、各呼出しのpurpose等はsweep_recordへ記録する既存規約を明記。Task2で検証。Minorのため再レビュー不要。

- Task1: 11 passed。初回は子processのPYTHONPATH不足1件を修正。通常sandboxのpytest tmp ACL制約のため主担当は通常権限で検証。

- Task2時点: 追加28件＋実ソース依存境界2件=30 passed。旧掃引テストのNPZ単独は証拠なしの旧成果物として保持するようfixtureを明確化し、新規保存先の意図を--new-outputで明示。WindowsとWSLで追加28件が成功。Ruff全体・整形374ファイル・Pyright222ファイル0 error/0 warning。

- 実装レビュー1回目Major1: 既存git来歴は未追跡sourceを検査しない。追加補助記録だけに全実行Pythonの内容hashを追加し、未追跡の追加・変更・削除を検証。同じreview sessionで再確認する。レビュー対象pytestは指定tmpのACLで2 passed/28 setup errors。

- 修正後Luna high実装レビューPASS（Major/Minorなし）。権限で対象pytestは独立実行不能、境界2件とRuffは独立確認。主担当のfresh16成功ログを照合。
- 全pytest1回目は14426 passed/8 skipped/1 failed。失敗は既存CLI引数一覧の--new-output反映漏れだけ。期待値と説明を更新し単独1 passed。数値・既存artifact writerは追加変更なし。その後、8並列で全体を再実行し、下記の最終結果を確認した。

- 最終全体: 14427 passed/8 skipped、exit0、697.25秒。JUnit/logはspec.jsonのintegration_validation。旧・提案golden各testも成功。追加31件すべて成功。
- 最終検証: fresh16フロー、Linux追加31件、Ruff/整形374ファイル、Pyright222ファイル（error/warning0）、命名登録、diff check成功。
- 独立の証拠照合Luna high PASS（01a14aaf-7922-72b2-af14-cfde153460aa）。JUnit・golden・追加31件・fresh/Linux/Pyright・484パスのsource hashを照合。source hashはspec.jsonに記録。対象pytestの独立実行は権限で不能だった事実を維持し、主担当の実測証拠で確認した。
