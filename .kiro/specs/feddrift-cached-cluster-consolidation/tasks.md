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
- [ ] 3. 新構成だけの実行と独立レビュー・全回帰を確認する
  - exact依存・fresh計算を接続し、独立CLIの要求/設計/命名レビューと主担当の全pytest・品質検査を確認する。
  - _Boundary: 接続・検証_
  - _Depends: 2_
  - _Requirements: 3.3_

## Implementation Notes

- 要求/design/task graph gateを主担当inlineでPASS。全9要求を3taskへ対応付け、委任承認。数値部だけの分離は共通手順の例外に基づく。kiro-impl --review off、taskを1つずつTDD、子エージェントなし。全task後に独立CLI1回。数値/順序のためHaiku優先だが、前specの週間上限が続く場合は所定のLuna mediumへ代替する。
- task 1: stubによる41 failedのREDを確認。実装後のclient対照8件は旧OrderedDict→builtin dictのAPI対応漏れを補正し、41 passed（3.54秒）。実旧_weighted_modelと実client.apply_cached_merge、両optimizer/2・4クラス、全0/不均等重み、桁落ちが起きる非昇順、出力独立/入力grad/3RNG不変を確認し、task-local VERIFIED。Ruff整形済み。
- task 2: server stubによる49 failed/41 passedのREDを確認し、実装後90 passed（3.18秒）。実旧_merge_cached_clusters全結果との対照、負/欠落/大重み、0件/欠落統計、期限境界、server→共通平均→client自身の異なるsnapshotを確認。後段に不正な値があるときもparameter乗算へ到達せず、入力/3RNG不変。task-local VERIFIED。
- 接続検証: 対象90件＋依存3067件で3157 passed（7.08秒）。AST検査がmodule自体とimportしたmemberの双方を照合するため、3個のmodule許可を追加した。Ruff check/format（309ファイル）、Pyright 0 errors/0 warnings、pip check、names、diff --check成功。固定旧748c3aaの差分なし。freshは実行中、独立レビュー・全pytestは未実施。source/testを独立レビュー用commitへ固定する。
