# Implementation Plan

- [x] 1. 独立モデルの平均損失による一回更新を移植する
  - stubへ実旧updateと異常系testを先に書く。loss/parameter/grad/optimizer一致、拒否時にforward/zero_gradなしを完了条件とする。
  - _Boundary: learning数値更新_
  - _Requirements: 1.1, 1.2, 1.3_
- [x] 2. 一roundのモデル別連続更新と完了量計数を接続する
  - 実旧local_trainのtestと全入力事前拒否testを先に書く。モデル順・抽出・複数round・計数・保持の一致を完了条件とする。
  - _Boundary: runtime/既存owner_
  - _Depends: 1_
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2_
- [x] 3. 依存境界・fresh接続・独立レビュー・全回帰を完了する
  - exact依存と新実装だけの実学習を追加する。独立指摘採否・全pytest・品質検査成功を完了条件とする。
  - _Boundary: 接続・検証_
  - _Depends: 2_
  - _Requirements: 3.3_

## Implementation Notes
- 要求/design/task graph gateを主担当inlineでPASS、9要求を3taskへ対応付けて委任承認。kiro-impl --review off、子エージェントなし、全task後に独立CLI1回。前specのHaiku週間上限により代替Luna mediumを予定。
- TDD: task 1のstubで12 failed（NotImplementedError）、実装後は異常系8件を含め20 passed。task 2のstubで24 failed（NotImplementedError）、実装後にtest追加位置のNameErrorを修正し58 passed。両taskのtestを実装より先に書いた。実旧update/local_train本体を代役なしで呼び、実旧updateの戻り値/入力だけを計測するwrapperでモデル順・抽出・損失を照合した。
- 主担当の接続検証: 対象58件＋依存境界3058件＋共用fresh smoke1件で3117 passed（117.08秒）。smokeでは二値/多クラスで観測・新規生成・flushに続き既存/負IDモデルをそれぞれ4回実学習し、parameter変更と学習量を確認した。Ruff check/format（304ファイル）、Pyright 0 errors、pip check、names成功。独立レビューと全pytestは未実施。
- レビュー前の構造検査補完: クラス数/分類層shape/隠れ層shape/出力活性化/特徴抽出部の5ケースを先に追加し5 failed（拒否なし、またはforward後のRuntimeError）を確認。独立モデルの実構造をzero_grad前に検査するよう追加して63 passed。実装が変わったため接続検証を取り直す。
- 最終変更後の接続検証は3122 passed（対象63件＋依存3058件＋fresh smoke1件、136.05秒）。Ruff check/format（304ファイル）・Pyright 0 errors・names成功。source/test変更はここまでで凍結して独立レビューへ出す。
- 独立レビュー: 実装commit `790ec55ff4de0053dd65b8379975d07b7b634373`、起点8d73ba6、cleanと固定旧748c3aaの差分が空を確認。source hash LF `3db919278a9f104342c1cf83fab5e190d0a0cf9f1c04b501da323f4655cb573f`（408パス）。実モデルgpt-6-luna / reasoning effort mediumをログで確認、session `01a12b7a-373e-7f23-a9c4-19f9130c77c7`。判定APPROVED、Blocker/Major/Minorなし。9要求と命名・実旧との順序を確認し、主担当が受け入れた。
- 独立実行は3121 passed（13.67秒）、Ruff check/format（304ファイル）・names・identity・diff --check成功。全pytest/fresh smoke/Pyright/pip checkは独立未実施。ログ/依頼文は`venv/refactoring-tests/feddrift-client-round-training-review-1.log`と`feddrift-client-round-training-review-prompt.txt`。主担当の全回帰を実行中。
- 完了確認（2026-10-12）: 凍結した実装commitの全pytestは12853 passed / 4 skipped / 9 warnings、exit 0（886.46秒）。JUnitは12857件、failure/error 0、対象63件・既存手法11ケース/提案構成3ケースのgolden回帰・共用fresh smoke成功。ログ/XMLは`venv/refactoring-tests/feddrift-client-round-training-full.log`と同名`.xml`。品質検査と独立レビューを合わせて主担当が受入、全3task完了。source/旧/goldenを変更せず完了文書だけを更新した。client組立・サーバ同期・全体run移植は後続であり、今回の旧部品対照と混同しない。
