# Implementation Plan

- [x] 1. clientの正式モデル反映をownerへ接続する
  - 実旧apply_server_mappingとのtestを先に書き、統計/標本/計数/現在ID/適応記録とoptimizer再構築・一時モデル保持・3RNGが一致し、不正入力が更新前に拒否されれば完了。
  - _Boundary: runtimeのclient配布/統合適用_
  - _Requirements: 1.1, 1.2, 1.3, 4.1, 4.2_
- [x] 2. 時刻準備・クラスタ判定・隔離登録を接続する
  - 距離/linkageの実旧対照と、実prepare_timestepの連続状態/通信/RNGをtestする。成熟だけ評価、client自身の加重統合、登録順/期限/記録が一致すれば完了。
  - _Boundary: methodsの分割とruntimeのprepare_
  - _Depends: 1_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 4.1, 4.2_
- [x] 3. FedAvg・全配布・R回の学習同期を接続する
  - 正/0寄与、複数client/model、R回の実旧対照を先にtestし、各段の全owner/parameter/optimizer/通信/3RNGと未観測の保証が一致すれば完了。
  - _Boundary: runtimeのround/timestep接続_
  - _Depends: 2_
  - _Requirements: 3.1, 3.2, 3.3, 4.1, 4.2_
- [x] 4. 依存境界と共用freshへserver進行を接続する
  - 新構成だけの共用freshへserver進行を接続し、新moduleのexact依存許可を登録する。対象機能test・依存境界test・共用fresh testが成功すれば、この実装taskを完了する。
  - _Boundary: 依存境界とfresh接続_
  - _Depends: 3_
  - _Requirements: 4.2_

## Implementation Notes

- task 1〜4の実装完了後に、共通引継ぎ手順に従って独立実装レビューを受ける。指摘反映後に全pytest/JUnit・Ruff/Pyright/pipを検証する。これらはtask 4の実装完了条件とは別のspec完了ゲートであり、taskのcheckboxがすべて済んでも、最終レビューと検証が通るまでspecをcompletedにしない。
- 要求/design/task graph gateを主担当inlineでPASSし委任承認。全12要求を4taskへ対応付け、kiro-impl --review offで1taskずつTDD。子エージェントなし。独立レビューは最後にCLI1回、数値/owner/RNGのためHaiku優先、週間上限継続なら所定のLuna medium。
- task 1: oracle準備のAPI名/旧model factoryの閉包を修正後、全17件がstubのNotImplementedErrorでRED。実装後、汎用snapshot検査が有限性を検査しないことを対象NaN testが検出し、配布入口へ有限性検査を追加。最多/同率/0件fallback、評価store上限超過、計数/概念/一時record保持を追加し、対象33＋既存client55で88 passed（5.59秒）。実旧適用の全owner/optimizer/3RNGを照合してtask-local VERIFIED。

- 2026-10-12ユーザー指示: task実行前の仕様書群の独立CLIレビューを追加（ユーザーの「サブエージェント」は別セッションの独立CLIを指す）。従来の主担当inline gateは履歴として残し、独立した仕様レビューの証拠とは扱わない。task 1完了・task 2途中の状態を保持し、残りのtaskの実行前に仕様レビューを実施する。
- 内蔵サブエージェントによるレビュー試行はユーザーの用語明確化により停止し、正式な承認には使用しない。その時点では仕様書群の独立CLIレビューは未実施。
- 仕様書群の独立CLIレビューをGPT-6 Luna（medium、read-only）で起動。CLIヘッダでmodel/effort/session `01a12c77-7abb-7203-ae19-05844700fab0`を確認。判定待ちであり、承認済みとは扱わない。

- 仕様レビュー1（Luna medium、CLI session `01a12c77-7abb-7203-ae19-05844700fab0`）: 仕様FAIL/手順PASS。Majorはtask 4へ実装と最終ゲートが混在、Minorはresumeの古い現在地。いずれも採用し、task 4を依存/fresh接続へ限定、最終レビュー/全体検証を別ゲートとして明記、resumeの旧記述を過去の参考と明示。任意の命名指摘は、snapshot名は公開メソッド、Random名は公開propertyであるため削除しない。Major修正の再確認待ち。

- Major修正箇所のみをLuna medium（`/root/specification_correction_review`）が再確認してPASS。task 4から最終レビュー/全体検証を分離したことで循環を解消。Minorは修正/採否記録のみで再レビューを必須としない。仕様レビューは通過し、task 2の再開が可能。実装と全体runは未完了・未承認。

- task 2: 対象98件が成功。実旧の成熟/隔離/統合/非統合を2時刻連続で操作し、全client/server状態・通信・3RNGを照合。不正時刻/設定/client集合/欠落モデル/非有限clientを記録・RNG更新前に拒否。test oracleの旧入力列は実prepareと同じsorted ID、旧観測は標本tuple列へ修正。

- task 3: 新規26件がstubのNotImplementedErrorでRED後、対象124 passed（9.91秒）。実旧の新規登録から隔離解除、複数モデル/正・0寄与/SGD・Adam/二値・多クラス、R=1/3の連続学習同期で全parameter/optimizer/owner/診断/登録来歴/通信/3RNGを照合。同期では追加観測しないことを確認。

- task 4: exact依存を4moduleと変更clientへ登録し、禁止依存12例を追加。対象/境界3203 passed（13.46秒）、追加の拒否/クロス評価集計確認後の対象130 passed（8.98秒）、最終sourceの共用fresh 1 passed（79.11秒）。公開命名の機械照合、Ruff（314 files）/Pyright/pipも成功。全実装taskは完了し、実装レビュー/全体検証は別ゲートで未完了。
- 進捗検査も新手順に合わせ、全task完了のレビュー/検証待ちを許す片方向条件へ修正。completedで未完了taskがある場合は引き続き拒否。3状態の手動検査で確認。

- 実装レビュー1（GPT-6 Luna medium、`/root/feddrift_implementation_review`）: FAIL、Major 1。全保有stateのoptimizer結合検査が配布の副作用より後となる要件1.3違反を採用。全保有stateを生成/owner更新前に検証する修正を追加。一時ID/正式IDの3ケースは修正前RED、修正後3 passed（3.41秒）。Minor以下は修正/採否記録のみとし、今回のMajor修正箇所だけ再確認する。独立検証は修正前の対象/依存境界3209 passedとRuff成功。全pytest/freshは独立実行していない。

- 実装レビューのMajor修正箇所の再確認はPASS。同じLuna reviewerが副作用前の全保有state検証と拒否時のowner/乱数不変を確認し、対象/依存境界3228 passedを独立実行。命名も設計/実態の一致を承認。spec完了判定は、この後の全pytest/JUnitと品質検査を待つ。

- 指摘反映・Major再確認後の最終検証（2026-10-12）: 全pytestは13168 passed / 4 skipped / 9 warnings、exit 0（667.16秒、-n 8 --dist loadfile）。JUnitは13172件、failure/error 0、対象149件・既存11ケース/提案3ケースgolden回帰・共用fresh smoke成功。前specの13007 passedへ対象149件と禁止依存12件を加えた数と一致。skipはPOSIX bashなし3件と既存回帰へ委ねる環境golden1件。Ruff check/format（314 files）、Pyright 0 errors/0 warnings、pip check、names、progress、diff --check成功。固定旧748c3aa・2 golden・toolsの差分なし。ログ/JUnitは`../../venv/refactoring-tests/feddrift-server-synchronization-full.log`と同名`.xml`。WSL・変異未実施。全参加者factory/終端/精度評価/全体run指標/成果物/CLI/任意相補性診断は後続で、全体run/golden移植の完了は主張しない。
