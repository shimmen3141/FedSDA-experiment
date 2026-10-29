# Implementation Plan

- [ ] 1. 逆向きの1本の流れの計算を足す（接続はまだしない）
- [ ] 1.1 予測的符号長の途中までの和とseed
  - 途中までの和（最後の要素が今の関数の値と同じ、各途中までの和が同じ順序の先頭i+1件を今の関数で測った値と同じ、runの乱数と`template_classifier`を変えない、入力の検査）と、`derive_reinitialized_learner_single_pass_initialization_seed`（値の例と検査）のtestを先に書き、REDを確かめる。`_accumulate_prequential_code_length`を、今の2つの関数の値を変えずに直す。
  - 完了条件: 新しいtestと、予測的符号長・最小記述長の選択の既存のtestが通る。依存境界の許可を登録する。
  - _Requirements: 2.2, 2.4_
  - _Boundary:_ `learning/training/prequential_code_length.py`、`runtime/alarm_interval_minimum_description_length_selection.py`（seedの関数だけ）
- [ ] 1.2 純粋な計算のkの刻み
  - `select_assignment_by_minimum_description_length`の`change_interval_step_sample_count`（刻み1と刻み5の候補、入力の辞書の検査、今の扱いの値が変わらない）のtestを先に書き、REDを確かめる。呼出し側（今の扱い）は刻み＝最小変化区間を渡す。
  - 完了条件: 新しいtestと、最小記述長の選択の既存のtestが通る。
  - _Requirements: 2.1, 2.5, 1.3_
  - _Boundary:_ `methods/fedsda/training_data_assignment/minimum_description_length_assignment_selection.py`と、その呼出し

- [ ] 2. 設定を足し、接続し、構成を足す（既定は変えない）
  - 設定の検査（値、最小記述長の選択を使わない選び方との組合せの拒否）、接続のtest（今の扱いでは途中までの和の関数を呼ばない。新しい扱いではkの候補が1件刻み、列が短いと学習者を使わず「現在のモデルのまま」、選んだkが選択の記録に残る、測定用の学習者の更新回数が学習者ごとにn回）、構成のtest（新しい構成は既定から1項目だけ違う、既定と既存の構成の値が変わらない）を先に書き、REDを確かめる。
  - 設定の項目と検査、clientからの受け渡し、新しい構成。基準版1〜15の「後に足された項目」の表へ足し、runの識別子のtestを新しい値に合わせる。
  - 完了条件: 新しいtestが通り、既存のtest（実旧との照合、基準版1〜15）が値を変えずに通る。
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.3, 2.6, 2.7, 3.1, 3.2, 4.1, 4.2, 5.2_
  - _Depends:_ 1

- [ ] 3. 研究ツールの変種R1と同じ値になることを確かめる
  - 研究ツールに、新しい構成を使う変種`every-sample-change-interval-search-configuration`を足す。照合のscript（Git管理外）で、手元の小さい条件（MNIST4・SINE2・SEA4・CIRCLE2、間隔50、seed 0、各client 1000標本程度。列Wが最小変化区間より短い警報が1回以上起きる条件を含める）で、研究ツールの変種R1とこの変種の、精度・下り・上り・予測の記録のdigest・最終のパラメータのdigest・計算回数の指標・選択の記録が一致することを確かめる。
  - 完了条件: 一致の結果（条件ごと）と、照合のscriptの場所がImplementation Notesにある。一致しない場合は、原因を特定して直すまで次へ進まない。
  - _Requirements: 5.1_
  - _Depends:_ 2

- [ ] 4. 文書を合わせる
  - モデル再利用の文書の最小記述長の節、実験の入口の構成の一覧、IMPROVE-032の状態と一覧、再開案内。既定が変わらないことを明記する。
  - 完了条件: 文書が新しい選択肢と構成を説明する。
  - _Requirements: 6.1_
  - _Depends:_ 2

- [ ] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー（GPT-6 Luna、effort medium）、指摘の反映、全pytest・Ruff・Pyright・WSL（goldenの版1〜15とこのspecのtest）・fresh smoke。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合、fresh smokeの結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_
  - _Depends:_ 3, 4

## Implementation Notes
- 背景（2026-10-29）: 研究ツールの変種R1（`default-reverse-prequential-grid1`、commit `75fcc893`）の研究室サーバでの128 run（`results/fedsda-reverse-prequential/20261029_083601_uranium-n5000`）。結果は[IMPROVE-032](../../../docs/research/improvement-candidates/improve-032-reverse-order-prequential-code-for-all-change-intervals.md)。ユーザーが、specとして実装して本規模で実験すると決めた（既定にはしない）。
