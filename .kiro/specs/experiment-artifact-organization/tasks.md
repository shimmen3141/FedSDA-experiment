# Implementation Plan

- [ ] 1. 出力先と、目的とREADME（掃引のコマンド）
  - [ ] 1.1 `collect_behavior_baseline_mismatch_descriptions`を実行のmoduleへ切り出し、`execute_experiment_sweep`がそれを使う（動作は変えない。既存のtestが通る）。
    - _Requirements: 2.5_
  - [ ] 1.2 掃引のコマンドへ`--study`・`--purpose`、出力先の既定、名前の検査、`--study`と出力先の同時の指定の拒否を足す（引数を読んだ直後に検査する）。testを先に書き、REDを確かめる（エラーの文面で理由を確かめ、正しい引数で通ることと対にする）。1.3と同じ作業の中で行う（必須にした時点で、既存の呼出しが落ちるため）。
    - _Requirements: 1.1, 1.3, 2.1_
  - [ ] 1.3 掃引のコマンドの`main`を呼ぶ既存のtest（grepで全部。subprocessで呼ぶものを含む）とfresh scriptへ`--purpose`を足す。引数の一覧のtest（`test_fedsda_default_waited_registration_and_alarm_wait.py`）へ`--study`・`--purpose`を足し、名前と注釈（「変わらない」）を実態に合わせる。終了コード2と出力先が無いことだけを見る既存のtest（MNISTが無いとき、計画が不正なとき）は、文面で理由も確かめる。
    - _Requirements: 2.1_
  - [ ] 1.4 掃引のコマンドの、基準版の確認を、出力先を作る前へ移す（`collect_behavior_baseline_mismatch_descriptions`）。testを先に書く（止まるとき、出力先の中のファイルの一覧が変わらない。READMEを作らないことは1.5で確かめる）。依存境界の許可を足す。
    - _Requirements: 2.5_
  - [ ] 1.5 `write_experiment_readme`と、実行の記録の`purpose`・`study_name`（形式の版3）を足す。testを先に書く（READMEの中身、既存のREADMEを変えない、止まるときにREADMEを作らない、記録）。`write_sweep_record`の直接の呼出しと、形式の版の期待を合わせる。依存境界の許可（`write_experiment_readme`）を足す。
    - _Requirements: 2.2, 2.3, 2.4, 2.5_
  - _Boundary:_ 1.1は単独でcommitできる。1.2〜2は、同じcommitにしてよい（比較scriptが、名前の検査とREADMEの関数を共用するため）。

- [ ] 2. 比較script
  - 出力先の既定（`results/behavior-fix-comparison/<日時>_<label>`）、`--label`・`--purpose`、名前の検査、実行の前の基準版の確認（出力先を作る前）、README、比較の記録の`purpose`・`study_name`と、記録のファイルを排他的に作ること。testを先に書く。比較scriptの`main`を呼ぶ既存のtest（基準版で止まるtestを含む。そのmonkeypatchの対象は`collect_behavior_baseline_mismatch_descriptions`へ替える）へ`--purpose`を足す。
  - _Requirements: 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 3. 分析の出力
  - 実装の前に、変更前のscriptで、固定の入力（連鎖の表に行が出るNPZを含む）から、`--chains`ありで出した標準出力を、testの期待値として埋める（設計の「表の数値を変えないことの確かめ方」）。
  - 引数を`argparse`で読み、行を集める形へ替え、既定でファイルへ書く、`--stdout`、出どころの節と、見出しの行の直後の警告。testを先に書く（表の部分が期待値と文字単位で同じ、ファイルと`--stdout`の中身が同じ、ファイルの名前と番号、出どころ、警告、入力が無いときにファイルを作らない）。分析の`main`を呼ぶ既存のtestを合わせる。
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 4. 台帳と文書
  - `docs/experiments/experiment-log.md`（規約、列、既存の3回の実行——コードは、メインの作業ツリーの`results/behavior-fix-comparison/comparison_record_*.json`から——と、未確認の120 run）、実験の入口の文書（コマンドの例の全部。1つの出力先は1つの実験に使うこと）、比較script・分析scriptのdocstringの例、AGENTS.mdの規約、再開案内。
  - _Requirements: 3.1, 3.2, 3.3, 5.1_

- [ ] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSLの対象testを通す。
  - _Requirements: 全部_
