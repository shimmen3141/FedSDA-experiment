# Implementation Plan

- [x] 1. runの来歴を要約へ残す
  - [x] 1.1 `core/run_provenance.py`（基準版の定数、`RunProvenance`、辞書との変換）と、そのtest（検査、往復、`BEHAVIOR_BASELINE_VERSION`とgoldenの`BASELINE_VERSION`の一致）を書く。依存境界の検査に、新しいmoduleを登録する（標準libraryだけ）。このtaskは単独でcommitでき、全testが通る。
    - _Requirements: 1.1, 1.2, 4.1_
  - [x] 1.2 3手法の保存の関数へ`run_provenance`を足し、形式の版を2にし、読込みで版1と2を受ける。3つのmoduleの依存境界の許可へ足す。保存の関数を呼ぶtest（設計のFile Structure Planの一覧）を追従させ、来歴の書込み・版1の読込み（版2の要約から`run_provenance`を消し、`format_version`を1にして作る）・版2の不正の拒否のtestを足す。この時点で、保存・読込みの3つのtest file（`test_fedsda_experiment_run_artifacts.py`・`test_feddrift_experiment_run_artifacts.py`・`test_fedavg_experiment_run.py`）が通る。
    - _Requirements: 1.1, 1.3_
  - _Boundary:_ 保存の関数の必須の引数は、掃引の実行と2つのコマンドまで、呼出しを全部直さないと通らない。そこで、task 1.2から2.3までを、1つのcommitにする（途中の時点では、掃引のtestが失敗してよい。commitの前に、対象のtestを全部通す）。仮の値で呼出しを埋めない。

- [x] 2. 済んだrunの判定と、止める処理
  - [x] 2.1 `find_existing_run_state`の状態`different_behavior_baseline`、`ExistingRunBehaviorBaselineMismatchError`、`execute_experiment_sweep`・`execute_planned_experiment_run`の`run_provenance`を足す（状態を全部調べてから報告と実行を始めるように、loopを分ける）。testを先に書き（同じ版→飛ばす・commitが違っても飛ばす、違う版と記録なし→何も実行・報告せずに例外、ファイルが変わらない、逐次と並列、版2の来歴の壊れた要約は`conflicting`）、REDを確かめる。掃引の実行を呼ぶtestを追従させる。executionの依存境界の許可へ足す。
    - _Requirements: 3.1, 3.2_
  - [x] 2.2 `describe_code_revision`の未commitの変更を、追跡するファイルだけにする（追跡しないファイルだけがあるとき「変更なし」のtest）。`build_run_provenance`、一覧の表の3つの固定の列、`write_sweep_record`の`run_provenance`の引数と基準版と形式の版を足す。recordsの依存境界の許可へ足す。
    - _Requirements: 1.2, 2.1, 2.2, 4.2_
  - [x] 2.3 2つのコマンド: 来歴を1回作って渡す、止まるときの警告と終了コード2（表と記録を書かない）、`--print-plan`の表示（比較scriptは理由も）、比較の記録の基準版、掃引のコマンドのmoduleの説明の終了コード。commandの依存境界の許可へ足す。testは、掃引のコマンド（`test_experiment_sweep.py`）と比較script（`test_behavior_fix_comparison_scripts.py`）の両方に足す。
    - _Requirements: 3.3, 3.4, 4.2_

- [x] 3. 文書を更新する
  - 実験の入口の文書（出力、途中で止まったとき、終了コード、未commitの変更の判定がこの変更の前後で比べられないことと、追跡しないファイルを数えないこと）、基準の文書（版を上げる手順に、`BEHAVIOR_BASELINE_VERSION`を足す）、再開案内。
  - _Requirements: 5.1_

- [ ] 4. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSLの掃引のtestを通す。
  - _Requirements: 全部_

## Implementation Notes

- task 1.1は単独でcommitした（0c9b948）。task 1.2〜3は、1つのcommitにした（設計の境界どおり）。
- 手順の逸脱: task 2.1は「testを先に書き、REDを確かめる」だったが、実装を先に書いた。代わりに、実装の後で、主要な判定を1つずつ外して、新しいtestが落ちることを確かめた（基準版の比較を外す→8件失敗、止める例外を外す→7件失敗、未追跡のファイルを数える→1件失敗、前の形式の要約で来歴の列を置かない→1件失敗。どれも元に戻した）。
- 依存境界: 掃引のmoduleは、名前つきのimportだけが許される（`import sys`は拒否される）。警告は、呼出しの時点の標準エラーへ出すため、`report_behavior_baseline_mismatch`の中で`from sys import stderr`する（moduleの読込みの時点で束縛すると、testが差し替えた標準エラーへ出ない）。
- 警告を出す関数`report_behavior_baseline_mismatch`は、掃引のコマンドのmoduleに置き、比較scriptもそれを使う（naming.mdに無い公開名。2つのコマンドで同じ文面にするため）。実行の記録の辞書を作る`describe_run_provenance_for_record`も、同じ理由で、記録のmoduleに置いた（naming.mdに無い公開名）。
- testの例の来歴は、`tests/refactoring/run_provenance_examples.py`の`TEST_RUN_PROVENANCE`。fresh scriptは、test moduleをimportしないので、同じ値を自分で作る。
- 新しいtest: `test_experiment_run_code_provenance.py`（来歴の値、goldenの版との一致）、`test_experiment_sweep.py`の末尾（同じ基準版でcommitが違えば飛ばす、違う基準版・記録なし——run設定も違う場合を含む——で逐次・並列とも何も報告・実行せずに止まる、版2の壊れた来歴は食い違い、表の3列、追跡しないファイル、コマンドの終了コード2と警告と`--print-plan`）、`test_behavior_fix_comparison_scripts.py`（比較scriptの終了コード2・警告・`--print-plan`の理由、比較の記録の来歴）、保存のtest（来歴の書込み、版1の読込み、版2の不正の拒否）。

