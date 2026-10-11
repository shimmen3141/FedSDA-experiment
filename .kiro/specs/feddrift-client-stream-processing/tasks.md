# Implementation Plan

- [x] 1. 観測列から現在モデルの予測と検出バッチを接続する
  - stubに対する実旧process_batch/flushのtestを先に書き、二値/多クラス・跨時刻・複数発火・flushの状態と乱数が一致することを完了条件とする。
  - _Boundary: client streamと検出バッチ部品の借用接続_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1_
- [x] 2. 不正ブロックの事前拒否と記録の不変性を保証する
  - 後方不正を含めforward前の拒否、全状態/乱数保持、記録constructorのtestを先に書く。
  - test成功を完了条件とする。例外中の全block rollbackは追加しない。
  - _Boundary: 入力検査・記録_
  - _Depends: 1_
  - _Requirements: 3.2_
- [ ] 3. 共用実行の接続・独立レビュー・全回帰を確認する
  - exact依存とfresh smokeを追加し、新実装だけの実行成功、全task後の独立CLIの有用な指摘の反映、全pytestと品質検査成功を完了条件とする。
  - _Boundary: 依存境界・検証_
  - _Depends: 2_
  - _Requirements: 3.3_

## Implementation Notes
- 主担当inlineで要求/設計/task graph gateを確認してPASS。全9要求を3taskへ割当。共通手順のユーザー委任に基づき承認。task-local review off、子エージェントなし、最後の独立CLIを維持する。
- 前specで確認済みのHaiku週間上限のため、状態接続の独立レビューは代替GPT-6 Luna mediumを使用する。局所学習、サーバ/全体run、golden移植、early_stoppingは未実装のまま後続へ残す。
- 実装過程: stubのNotImplementedErrorで16 failedを確認後に実装。test側の既存API名（parameter_optimizer/assign_model_for_training）を修正して16 passed。入力検査はtask 1時点で接続したためtask 2の39件は実装後に追加した（test先行ではない）。追加時のtest挿入位置を修正後、55 passed / 2 warnings。実旧process_batch/flushを代役なしで呼び、旧予測クラスは実predictと本体呼出しを保つ計測wrapperで保存した。旧の1件batch不偏分散による警告のみ。状態/乱数・flush記録維持・後方不正のforward前拒否を確認。
- 追加検査: 保有モデルのクラス数不一致が予測より前に拒否されない1 failedを確認。バッチ部品の既存検査へ幅/クラス数の一致を移し、対象56件＋既存バッチ75件が131 passed / 5 warnings。新実装だけの共用fresh smokeは1 passed（83.04秒）。Ruff check/format（301ファイル）、Pyright 0 errors、pip check成功。独立レビューと全回帰は未実施。
