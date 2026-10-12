# Implementation Plan

- [x] 1. 予測前の実保有モデル数を記録する
  - 実旧全体runをbootstrap、property stubと先行testでRED→実装。作成/統合を跨ぐ予測前数と失敗標本/終端/tailを検証。
  - _Requirements: 1.1, 1.2, 1.3, 2.5_
- [x] 2. server/clientのパラメータ演算を実操作から数える
  - snapshot/property stubと先行testでRED→私有累積3+1計数。実引数観測と独立numel算式で集約/統合/診断を別々に照合。
  - 正/0/全0重み、単独/隔離、R/終端、事前検査失敗と旧状態/RNG/通信一致を確認。
  - _Depends: 1_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
- [ ] 3. 共通summaryを指標とfreshへ接続し全検証を完了する
  - 任意モデル計数と記録対応検査をtest-firstで導出へ接続。None/ゼロ分母/破損記録/読取り性、既存基本指標を確認。
  - 共用freshとexact境界、独立実装レビュー/有用指摘対応、全pytest/旧golden/Linux/品質、source identity・完了記録・commit/push。
  - _Depends: 2_
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

## Implementation Notes

- 基点`df74dec`、固定旧`748c3aa`。主担当調査/実装。cc-sdd標準6文書と仕様書群レビュー後、kiro-impl manual `--review off`で下位taskを進め、全task後に独立実装レビュー。
- 仕様書群Luna medium `/root/feddrift_summary_spec_review` PASS、Blocker/Majorなし。Minorのstore属性をnaming表へ追加する指摘は不採用: 2026-10-10ユーザー決定の範囲はmodule/class/公開関数・methodで、fieldは対象外。責務はdesignで明示済み。counter事前検査の追加もreviewerが確認。taskのtest/source開始前に判定を確認。
- 実旧oracle bootstrap: 基本指標の9ケースpassed。最終検証は下記。

- Task1: getter stubのRED後に実装。対象＋既存streamの61件passed。Task2: snapshot stubで6件RED後に計数を実装。Task3: optional引数未実装の2件RED後に接続し、対象36件passed。収集失敗はREDに数えない。終端oracleのPRETRAIN_SAMPLESを既存対照test同様に揃えた。

- 主担当の追加検証: 対象37 passed、Linuxの集計/基本指標/計測105 passed、Linux専用既存golden4 passed。共用freshは指定TMP/TEMPで16 flows成功、旧/test module importなし。既定Tempでの初回は権限エラーのため成功証拠へ数えない。names4source未登録なし、progress2/3。以降の独立レビュー/最終検証は下記。

- 独立実装レビュー: Haiku 5.5（数値/RNG/可変状態の接続のため選択）、session 24154147-ace6-4acb-ac88-233d484d5d64、実モデルclaude-haiku-5-5/is_error=false、effort mediumは指定値。未commit差分にPASS、Blocker/Majorなし。独立3434 passed、fresh16、Ruff/format325成功。全pytest/Linux/Pyright/identity/namesは未再現。raw: ../../venv/refactoring-tests/feddrift-summary-haiku-review.json。
- Minor1採用: designに「標本ごと、その標本の状態更新前」と明記。Minor3採用: 破損計数をprepare/aggregate/synchronize/finalizeで検証。Minor2不採用: terminal_snapshotの型への退避は、不正storeを注入した拒否testでも操作前後の状態を比較するため。通常storeは実列を比較し、欠落を健全値へ置換していない。Minor以下の採否/修正のみで再レビュー不要を適用。resumeは主担当の現spec再開案内でレビュー担当の変更ではない。

- Minor反映後の対象61 passed/6.05s。source/testを固定して全pytestを実行。今回の増分61は基点13432 testcaseから13493への増分と一致。変異testは既定に従い未実施。

- 最終全pytest: 13485 passed/8 skipped/8 warnings、684.62s、exit0。JUnit13493 testcase、failure0/error0。基点13424 passed/8 skippedから61件増分。skipは環境別golden1/Linux専用4/POSIX bash不在3で、Linux専用4は別途成功。warningsは既存nested tensor/旧var/TypedStorageで、新testの警告なし。
- 最終品質: Ruff check/format325成功、Pyright 0 errors/0 warnings、pip check成功、diff-check成功。names4source未登録なし。全pytest開始後のsource/test変更なし。検証commitとidentityは次の完了記録で照合する。
