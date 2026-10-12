# Implementation Plan

- [x] 1. 計測つき全体runを共通実行へ接続する
  - 実旧入口をbootstrapし、入口stubと先行testでREDを確認してから、私有checkpoint/モデルhook/準備分離/実参加者結果を実装。
  - 通常run・実旧との状態と基本指標の一致、共有部0、独立MLP算式、繰返しと大域乱数を確認。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.3_
- [x] 2. 時刻別計数と失敗時の計測解除を検証する
  - 先行testで時刻別2区分/終端のREDを確認し、差分化と全field合計を接続。Rの複数回、ゼロ同期、tail、ドリフトと各失敗stageを確認。
  - 失敗後にhook集合が復帰し正常runが同じ計数を持つことを確認。
  - _Depends: 1_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.2, 3.3_
- [x] 3. freshと依存境界を接続し、独立レビューと全検証を完了する
  - 共用freshの全体FedDrift flowに計測を接続、exact import許可と少数の禁止例を追加。
  - 独立実装レビュー/指摘対応、全pytestと既存golden、Linux対象、品質/names/progress、commit/source identity/完了記録/push。
  - _Depends: 2_
  - _Requirements: 3.1, 3.3, 3.4_

## Implementation Notes

- 基点: `286792d`。固定旧: `748c3aa`。調査/実装は主担当、仕様書群レビュー後にtaskへ進む。下位taskはkiro-impl manual `--review off`、全task後に独立実装レビュー。
- 仕様レビュー: Luna medium `/root/feddrift_measurement_spec_review`、未commitの6文書、PASS（指摘なし）。taskのtest/source開始前に確認。
- 完了: 仕様/実装の独立レビューと有用な指摘対応、全pytest/既存golden・Linux対象・品質検査、source identity。検証済み単位でcommitし通常pushする。

- Task 1: 実旧bootstrap20 passed。入口stubからNotImplementedErrorでRED確認後、11 passed。Task 2: 時刻未接続で6 failed/1 passedのRED後、対象と既存run合わせ73 passed。型/loader/途中失敗・独立の操作境界照合を追加。
- 学習batch不足時の独立test前提を既存skip契約に修正（実装は変更せず）。

- Task 3実装: 対象34+既存run44+依存境界3091=3169 passed。共用fresh16 flows（旧/test importなし）、Ruff/format324 files、Pyright0 errors/0 warnings、pip check、names --base286792d、固定旧差分なし、diff-check成功。最終全pytestとレビューは未完了。

- Linux主担当実測: 対象34+既存run44+Linux別golden4=82 passed、54.88s、exit0。独立Linux再現は未実施。

- Haiku独立実装レビュー: `2a87cfa1-699a-4c9b-8320-bdf3b9c92e14`、`claude-haiku-5-5`実モデル確認/is_error=false、effort mediumは指定値。未commit対象差分、PASS。独立3169 passed/35.59s、Ruff/format324/fresh16/diff-check成功。全pytest/WSL/Pyright/identityは未再現。rawは`../../venv/refactoring-tests/feddrift-measurement-haiku-review.json`。
- 指摘1を採用: 不正設定はすべて既存configuration_validation内で拒否されるため、RunExecutionErrorと元causeのexact型まで確認。指摘2を採用: 全dataset/optimizerケースで、準備と本実行の延べ標本数×独立MLP MAC算式を照合。追加testの仮定（dataset末尾数字=クラス数、factory constructorで即検査）を既存dataset定義/遅延検査契約へ修正。source変更なし。
- 指摘3（準備除外をfield名へ含める）と4（freshで無計測の公開入口も別実行）は任意として不採用。共通FedSDAと同じfield名/docstringで準備除外が明示され、freshを計測入口へ接続する設計が仕様レビュー済み。通常公開入口は44ケースのpytestと独立操作境界観測で照合する。共通hookの共有部例外経路は既存test責務。
- レビュー中のresume更新は主担当による本specの再開案内。レビュー担当は未変更。共有手順に従い対象specの案内としてcommitに含める。Minor以下は採否/修正のみで再レビュー不要の規則を適用。

- Minor対応のtest期待型を2回修正してもseed条件が残ったため、fable-methodの2回ルールで読取り専用の一点診断を実施。`/root/measurement_error_contract_diagnosis`が専用RunSettingsValidationErrorを確認（独立修正前1 failed/3 passed）。cause4種類をexact表で指定するtest-only修正、source変更なし。追加の完了レビューではない。

- Minor対応後の対象34 passed/16.05s、最終Ruff/format324成功。Linux対象82 passed/99.80s（全pytest並行時）、exit0。source/testはこの時点で固定し、この内容で最終全pytestを実施。

- 最終全pytest: 13424 passed/8 skipped/9 warnings、689.06s、exit0。JUnit13432 testcase、failure0/error0。基点13387 passed/8 skippedから、対象34+禁止依存3の37増分と一致。旧11/最終3goldenの成功は次のsource identityでも照合する。
- skip8件: 固定Windows回帰で代替する環境別golden1、Linux専用golden4（Linuxで成功済み）、POSIX bash不在3。warnings9件は既存testのnested tensor/旧var/TypedStorage、今回追加testの警告なし。検証source/testは最終全pytest開始後無変更。

- 検証対象commit: `3abc0bd823470b5d8980bd08d9d22db5f26c2a37`。LF正規化source hash: `4ed5fe8ff60cd725535122034c3fcb67daa02f06e01a554138f799a87d17e9d7`（428パス）。identityで固定旧差分なし/clean tree/JUnit13432・13424 passed・8 skipped/既存回帰2testcase（内部11/3golden）の成功/件数を照合。最終文書commit後も同じhashとJUnitを照合する。
- 完了判定: 全3 taskと12受入基準を実測・独立レビューで確認、GO。未完了taskなし。サーバ算術/保有モデルの分母・正規化summary/追加診断/成果物/CLIは後続specであり、このspecの未完了taskではない。
