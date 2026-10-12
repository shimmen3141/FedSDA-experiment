# Implementation Plan
- [x] 1. 成功した通常同期のモデル数を時刻ごとに記録する
  - 既存実旧oracleをbootstrapし、記録型・snapshot stubからtest-firstで実装する。
  - R回の完了時点、統計だけのID、一時モデル、失敗時・準備/集約のみ/終端の除外を観測し、旧state/RNG/通信と一致する。
  - _Requirements: 1.1, 1.2, 1.3, 3.2_
- [x] 2. 終了時の学習量とクラスタリングのモデル診断を導出する
  - 導出stubのREDから、共通集計関数と既存ownerの記録を読取りで接続する。
  - 実旧の学習量・クラスタ診断と対応し、統合/一時ID/0計数/未知対/未定義AUC・不正記録拒否で状態と乱数を変えない。
  - _Requirements: 2.1, 2.2, 2.3, 3.2_
- [x] 3. 全体run指標とfreshへ接続して検証を完了する
  - 接続testを先に書き、同期数/位置/client数との対応検査を追加する。基本指標・計算量の値は維持する。
  - 共用fresh/exact境界、独立実装レビュー/指摘対応、全pytest/品質/旧golden/names/progress/identity、commit/pushを完了する。
  - _Requirements: 3.1, 3.2, 3.3_

## Implementation Notes
- 基点1d98024、固定旧748c3aa。主担当が調査/実装し、cc-sddの要求→設計/命名→tasks→独立仕様書群レビュー→下位task manual --review off→全task後の独立実装レビューを適用。task graphのinline sanityでは隠れた前提/責務重複なし。仕様書群独立レビュー前にtaskのtest/sourceへ進まない。

- 独立仕様レビュー: GPT-6 Luna medium CLI、session 01a12ef8-9751-7cb1-815e-06ebac0045fe。ログの実モデル/effort確認、PASS・必須修正なし。対象は基点1d98024の未commit6文書。実装/テストは未実行。raw: ../../venv/refactoring-tests/feddrift-model-diagnostics-spec-review.log。
- Bootstrapは実旧基本指標9 passed。Task1はsnapshot/記録stubの14件RED→実装後19 passed。Task2は導出stubで10件RED→29 passed。Task3は指標field未接続のAttributeErrorで1件RED→接続。更新scriptのSyntaxError/テストの旧属性名とreadonly client ID注入ミスはREDに数えず修正した。
- AUC接続testの前提（実対が3件ある）を実記録1件に修正し、実旧記録型を別時刻の対として3件構成して、共通AUCの既知/未知/同率/infと2距離の配線を実旧関数へ照合。sourceの判定/数値変更なし。

- 現実測: 対象59 passed/11.45s、Ruff check/format328成功、names4source未登録なし、progress2/3。以降の独立レビュー/最終検証は下記。共通計算関数とFedSDA sourceは変更していない。

- 独立実装レビュー: Haiku 5.5（可変owner/更新順/RNGと指標配線のため選択）、session 50c2bbbb-e342-48a1-b14f-bede876656d9、実モデルclaude-haiku-5-5/is_error=false、effort mediumは指定値、未commit対象差分にPASS。独立3517 passed/83.19s、Ruff/format328、names未登録なし。fresh成功文字列は確認されたがpipeline経由のためPython終了コードは未確認。全pytest/Linux/Pyright/progress/identity/変異は未再現。resumeは対象外（主担当による現spec再開案内）。
- Minor3の観測は旧/共通と同じmodel ID単位という仕様なので変更なし。負の一時IDもclient間で同じ整数IDなら合算される制約を保持する。Minor4は成功出口のみの記録の確認で修正提案なし。Minor以下の採否のみで再レビュー不要を適用。
- 主担当freshは指定TMP/TEMPで直接実行し、exit0/16 flows/旧test importなし。Linux対象＋既存golden63 passed/46.73s。source/testをこの内容で固定し、最終全pytestと品質を実行する。基点13493 testcaseから今回59増分で13552と一致。

- 最終品質: Ruff check/format328成功、Pyright 0 errors/0 warnings、pip check成功、diff-check成功。全pytest開始後にsource/testを変更していない。変異testは既定に従い未実施。最終全pytestとidentityは下記。

- 最終全pytest: 13544 passed/8 skipped/8 warnings、886.74s、exit0。JUnit13552 testcase、failure0/error0。基点13485 passed/8 skippedから今回59増分。skipは環境別golden1/Linux専用4/POSIX bash不在3。Linux専用4は別途成功。warningsは既存nested tensor/旧var/TypedStorage、新testの警告なし。
- 3taskの記録→導出→whole/fresh接続と全要求の証拠を確認。source/testは最終全pytest開始後に変更なし。検証commit/source identityは下記。

- 検証対象commit: 2ec6efe6aadbc99618e54a899a24320d76dc7e2f。LF source hash f137fe9bbaecd477465c431e490348bb5e7cf31acc4d4fce8d3811192abd8675（432パス）。identityは固定旧748c3aa差分なし、clean tree、JUnit13552/13544 passed/8 skipped、既存回帰2testcase（内部旧11・最終3golden）の成功を照合。最終文書commit後も同hashを再確認する。
- 完了判定: GO、全3taskと全要求の証拠が揃い、未完了taskなし。保存/CLIと他の診断は後続の境界。負の一時IDも同じ整数ID単位でclient間合算する既存の集計定義を維持。独立全pytest/Linux再現は未実施、主担当実測は上記。
