# Implementation Plan
- [x] 1.1 研究用重み同期介入を専用テストから作る
  - R/Pの式、fallback、計数、RNGと例外時復元を実行して確認する。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1_
- [x] 2.1 専用CLIの実行・計画・保存を接続する
  - V/O直接対照とread-only計画・既存出力拒否・来歴を確認し、共通ファイル無変更を照合する。
  - _Requirements: 2.1, 2.2, 2.3, 3.1_
- [x] 3.1 小規模実験・分析と独立レビューを完了する
  - 24runを実測し、精度/通信/介入/学習不変を分析。72runのサーバ用計画とコマンドを用意する。
  - 独立実装レビュー・専用検証・共通回帰を実測し、採否未確定を記録する。
  - _Requirements: 3.1, 3.2, 3.3_

## Implementation Notes
- 初期仕様。実装未着手。Luna highによる仕様レビュー待ち。

- 仕様レビュー初回: Major1（6条件未明記）、Minor1（同期前の旧重み退避順）を採用。6条件/client数/seed/標本数と退避順を明記。同じsessionで再確認。最終パラメータ照合を可能にするmodel/key単位のdigestフィールドも明記。
- task1: fixtureに必須設定が不足した初回を修正後、stubのNotImplementedErrorでRED確認。実装後11件成功。既存Tの有効域は2以上なのでT1という設計記述を既存契約に合わせT下限2へ訂正（挙動追加なし）。
- task2: Dの実行・保存構造を独立複製し、専用19件成功。コピーした計画テストの期待run数3を4方式に修正。24run計画表示は出力先未作成を確認。製品src・共通設定・既存Dコードは未変更。
- 実装レビュー（コード契約）Luna high PASS。独立13件/Ruff確認、IO6件は権限停止し主担当19件成功を照合。任意SIM117/C408の4件は現在の研究用検査の選択範囲外かつ挙動に影響せず、実験中のscript hashを変える理由がないため不採用。結果・最終証拠は同sessionで照合予定。
- 主担当最終実測: 24run全件完了。48元ファイルhash/CRC/精度/RNG、R/P各84介入、O対12組の学習/適応/通信と最終パラメータ一致を照合。全pytest14427 passed/8 skipped/0 failure/error、旧/提案golden成功、共用fresh16flows成功。通常JUnitに新研究testなし、共通src/tests/設定/CIと既存Dのdiff空。72run計画はread-onlyで出力先未作成。詳細はintegration_validation。
- 最終結果・証拠を同実装レビューsessionで独立照合しPASS、指摘0。48ファイル/24NPZ/各84介入/12対と48概念寄与を独立再計算。全JUnit・fresh成功と72run計画を照合。独立全pytest/fresh/NN再実行なし。Pは2改善/2同値/2悪化で一律採用せず、既定・golden・候補検証は維持。全task完了。
