# Implementation Plan

- [x] 1. 決定済み重みを保存parameterへ適用する計算を実装する
  - 実旧server/clientの平均と、異常分割・snapshot・重みのtestを先に書く。順序を含む完全一致、独立出力、入力/勾配/3RNG不変が完了条件。
  - _Boundary: methods/feddriftのparameter平均_
  - _Requirements: 1.1, 1.3, 2.3, 3.1, 3.2_
- [x] 2. serverの統合結果を共通平均へ接続する
  - 実旧_merge_cached_clustersとの対照testを先に書く。過去重み・統計・期限・mappingと異常入力の拒否が一致すれば完了。
  - _Boundary: methods/feddriftのserver統合計算_
  - _Depends: 1_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 3.1, 3.2_
- [x] 3. 新構成だけの実行と独立レビュー・全回帰を確認する
  - exact依存・fresh計算を接続し、独立CLIの要求/設計/命名レビューと主担当の全pytest・品質検査を確認する。
  - _Boundary: 接続・検証_
  - _Depends: 2_
  - _Requirements: 3.3_

## Implementation Notes

- 要求/design/task graph gateを主担当inlineでPASS。全9要求を3taskへ対応付け、委任承認。数値部だけの分離は共通手順の例外に基づく。kiro-impl --review off、taskを1つずつTDD、子エージェントなし。全task後に独立CLI1回。数値/順序のためHaiku優先だが、前specの週間上限が続く場合は所定のLuna mediumへ代替する。
- task 1: stubによる41 failedのREDを確認。実装後のclient対照8件は旧OrderedDict→builtin dictのAPI対応漏れを補正し、41 passed（3.54秒）。実旧_weighted_modelと実client.apply_cached_merge、両optimizer/2・4クラス、全0/不均等重み、桁落ちが起きる非昇順、出力独立/入力grad/3RNG不変を確認し、task-local VERIFIED。Ruff整形済み。
- task 2: server stubによる49 failed/41 passedのREDを確認し、実装後90 passed（3.18秒）。実旧_merge_cached_clusters全結果との対照、負/欠落/大重み、0件/欠落統計、期限境界、server→共通平均→client自身の異なるsnapshotを確認。後段に不正な値があるときもparameter乗算へ到達せず、入力/3RNG不変。task-local VERIFIED。
- 接続検証: 対象90件＋依存3067件で3157 passed（7.08秒）。AST検査がmodule自体とimportしたmemberの双方を照合するため、3個のmodule許可を追加した。Ruff check/format（309ファイル）、Pyright 0 errors/0 warnings、pip check、names、diff --check成功。固定旧748c3aaの差分なし。freshは実行中、独立レビュー・全pytestは未実施。source/testを独立レビュー用commitへ固定する。
- 独立レビュー: 実装commit `8e0006184c5a477b1320a71e133bfc8ee46a306a`、起点60c4230、source hash LF `1d32af5b1538d167338643b03806414473fe8a506ac67da3b704e71588668f10`（413パス）。実モデルgpt-6-luna/reasoning effort mediumをログで確認、session `01a12c37-34c3-76c3-ae92-99e9f8f5f89e`。APPROVED、Blocker/Major/Minorなし、全9要求・命名・実旧server/clientの対応を主担当が受入。独立3157 passed・Ruff check/format（309ファイル）・names・identity・diff --check成功。全pytest/fresh/Pyright/pipは独立未実施。
- 主担当fresh CPUは1 passed（85.37秒）、旧/test moduleなしでserver統合→client自身のsnapshot平均を実行。レビュー前後の作業ツリーと固定旧差分は空、レビュー後のRuff check/format成功。全pytestは実装commitのsourceのまま実行中。レビュー依頼/出力/対象/freshログは`venv/refactoring-tests/feddrift-cached-cluster-consolidation-*`。
- 完了確認（2026-10-12）: 実装commitの全pytestは13007 passed / 4 skipped / 8 warnings、exit 0（665.61秒）。前specの12914 passedへ対象90件＋依存拒否3件を足した数と一致。JUnitは13011件、failure/error 0、対象90件・既存11ケース/提案3ケースgolden回帰・fresh smoke成功。ログ/XMLは`venv/refactoring-tests/feddrift-cached-cluster-consolidation-full.log`と同名`.xml`。全9要求・設計境界・server計算から共通平均へ/同じ重みをclient自身のsnapshotへ渡す接続を主担当が確認し、task 3/feature GOをVERIFIED。source hash/固定旧差分を再照合し、以後は完了文書だけを変更。WSL・変異・新全体runの一致は未検証。owner適用/サーバ進行の完成とは区別する。
