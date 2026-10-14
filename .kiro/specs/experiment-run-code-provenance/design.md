# Design Document

## Overview
1 runの要約へ「runの来歴」（commit、未commitの変更の有無、取れなかった理由、基準版の番号）を足し、一覧の表と実行の記録へ出す。済んだrunの判定に、基準版の一致を足し、違えば、掃引の実行の入口で、何も実行せずに例外で止める。2つのコマンドは、その例外を受けて、警告を出し、終了コード2で終わる。

## Boundary Commitments
### This Spec Owns
- 新しいmodule `core/run_provenance.py`（基準版の番号の定数、runの来歴の値、要約からの読取り）。
- 3手法の要約の保存と読込み（`runtime/{fedsda,feddrift,fedavg}_run_artifact_files.py`）: 来歴の書込みと、形式の版。
- 掃引の実行（`sweep/experiment_sweep_execution.py`）: 済んだrunの判定の状態、止める例外、来歴の受渡し。
- 掃引の記録（`sweep/experiment_sweep_records.py`）: 来歴の組立て、表の列、実行の記録の基準版、未commitの変更の数え方。
- 2つのコマンド（`sweep/experiment_sweep_command.py`、直下の`run_behavior_fix_comparison.py`）。
- testと文書。
### Out of Boundary
run設定・識別子・指標・記録の表（NPZ）の内容、固定旧、goldenの内容と生成手順、分析のscriptの出力。
### Allowed Dependencies
`core/run_provenance.py`は、標準libraryだけに依る。3手法の保存のmodule、掃引の3つのmoduleが、これをimportする（依存境界の許可へ足す）。testの基準版のhelperは、この定数をtestで読むだけ（helperの`BASELINE_VERSION`の定義は変えない）。
### Revalidation Triggers
要約のJSONの形を読むもの（掃引の表、再開の判定、比較の分析の入力の表）、保存の関数の呼出し（testを含む）、goldenの版を上げる手順。

## Architecture
### runの来歴（`core/run_provenance.py`）
- `BEHAVIOR_BASELINE_VERSION = 5`: 今のコードが満たす、期待する動作の基準版（重要動作goldenの現在の版）。goldenの版を上げるときに、一緒に上げる（testが一致を確かめる）。これは「このコードが満たす基準版」で、runの設定が選んだ扱い（比較scriptが、前の版の扱いを設定で選ぶ場合を含む）とは関係しない。
- `RunProvenance`（frozen dataclass）: `code_commit: str | None`、`has_uncommitted_changes: bool | None`、`code_revision_unavailable_reason: str | None`、`behavior_baseline_version: int`。検査: 型が厳密に合うこと。commitが文字列なら、変更の有無はbool、理由はNone。commitがNoneなら、変更の有無もNone、理由は空でない文字列。基準版は正の整数（boolを拒否）。
- `convert_run_provenance_to_mapping(run_provenance) -> dict`と`read_run_provenance(mapping) -> RunProvenance`: 要約の`run_provenance`の辞書（4つのキー。ほかのキーは拒否）との変換。読めなければ`ValueError`/`TypeError`。

### 要約の保存と読込み（3手法）
- `save_*_experiment_run(..., run_provenance: RunProvenance)`: 必須の引数を足す（型を確かめる）。要約へ`"run_provenance"`を書く（位置は`environment`の次）。
- `RUN_ARTIFACT_FORMAT_VERSION = 2`（来歴を足したので上げる）。読込みは、版1と2を受ける。版2は`run_provenance`が`read_run_provenance`で読めることを確かめる（読めなければ`ValueError`）。版1は、`run_provenance`を持たない（あれば拒否する）。
- 3つのmoduleの読込みは、同じ形で直す（今も同じ形の重複。共通化はしない）。

### 済んだrunの判定と止める処理（`sweep/experiment_sweep_execution.py`）
- `find_existing_run_state`: 今までどおり、ファイルの組（片方しかない→`conflicting`）と、読込み（読めない→`conflicting`。版2で来歴が無い・壊れているものは、読込みが拒否するので、ここに当たる）を先に調べる。読めた要約について、run設定の比較の前に、基準版を比べる（基準版の判定が先なので、版1の要約は、run設定が違っても止める側になる）。要約の`run_provenance`が無い（版1）か、`behavior_baseline_version`が`BEHAVIOR_BASELINE_VERSION`と違えば、状態`different_behavior_baseline`と、理由（要約のファイル、記録された版——無ければ「記録なし」——、今の版）を返す。基準版が同じなら、今までどおり、run設定を比べる（commitと変更の有無は見ない）。状態の一覧の説明へ足す。
- `ExistingRunBehaviorBaselineMismatchError`（`Exception`の派生）: `mismatch_descriptions: tuple[str, ...]`を持つ。
- `execute_experiment_sweep(..., run_provenance: RunProvenance)`: 必須の引数を足す。型が違えば`TypeError`、`run_provenance.behavior_baseline_version`が`BEHAVIOR_BASELINE_VERSION`と違えば`ValueError`。全部の計画のrunの状態を、進み具合の報告の前に調べ終える（今は、調べるloopの中で、飛ばしたrunを報告しているので、loopを分ける）。`different_behavior_baseline`が1つでもあれば、報告も実行もせずに、その全部の理由を持つ例外を出す。無ければ、今までどおり（`completed`→飛ばす、`conflicting`→失敗、`missing`→実行）。
- `execute_planned_experiment_run(..., run_provenance)`: 保存の関数へ渡す。並列のときも、同じ値をworkerへ渡す（frozen dataclassなのでpickleできる）。

### 記録とコマンド
- `describe_code_revision`: 未commitの変更の有無を、gitが追跡するファイルだけで判定する（`git status --porcelain --untracked-files=no`）。理由: 追跡しないファイル（コミット保留の文書ほか）があるだけで、常に「変更あり」になり、記録が役に立たないため。実行の記録の値の意味も、これに揃う。
- `build_run_provenance(*, repository_directory) -> RunProvenance`: `describe_code_revision`の結果と`BEHAVIOR_BASELINE_VERSION`から組み立てる。
- `write_run_summary_table`: 実行条件の列の次に、`code_commit`・`has_uncommitted_changes`・`behavior_baseline_version`の列を、要約の形式に依らず、常に置く（今の列は、行のキーの和集合なので、固定の列として足す）。`format_version`が1で`run_provenance`が無ければ空の欄。版2で`run_provenance`が無いか読めなければ、その要約を「読めない」として表から外す（今の扱いと同じ）。
- `write_sweep_record`: 引数`code_revision: dict`を`run_provenance: RunProvenance`に替える。記録の`code_revision`の辞書は、今のキー（`commit`・`has_uncommitted_changes`・`unavailable_reason`）のまま、来歴から作り（`code_commit`→`commit`、`code_revision_unavailable_reason`→`unavailable_reason`と読み替える）、その次に`"behavior_baseline_version"`を書く。記録の形式の版（`SWEEP_RECORD_FORMAT_VERSION`）を2へ上げる。比較scriptの記録（今は形式の版を持たない、inlineの辞書）は、`code_revision`の辞書と`behavior_baseline_version`を書く（形式の版は足さない）。
- 掃引のコマンド: moduleの説明の終了コード2に、基準版の違いで止まることを足す。実行の前に`build_run_provenance`を1回だけ作り、`execute_experiment_sweep`へ渡し、実行の記録にも同じ値を使う（commitの取り直しで、要約と記録がずれないように）。`ExistingRunBehaviorBaselineMismatchError`を受けたら、標準エラーへ、理由を1件1行の警告で`print`し（argparseの`error`は使わない。使い方の表示は要らないため）、対処（別の出力先を使う）を1行出して、2を返す（一覧の表と実行の記録は書かない。出力先のディレクトリは、既にファイルがあるので、作る処理はあっても、中身は変えない）。`--print-plan`は、新しい状態と理由を、今の形で表示する。
- 比較script: 同じ（来歴を作って渡す、例外を`main`の中で受けて警告し、2を返す——`SystemExit(main())`で終了コードになる——、`--print-plan`で理由も表示する、比較の記録へ基準版を書く）。

## 旧と違う点
要約の形式の版が2になり、`run_provenance`を持つ。一覧の表の列が3つ増える。このspecより前に作った出力先で再開すると、止まる（基準版の記録が無いため。別の出力先で実行し直す）。未commitの変更の有無は、追跡しないファイルを数えない（この変更より前の実行の記録の値とは、比べられない。文書に書く）。runの実行と、指標の値は変わらない。

## Validation Design
- 来歴の値: 正しい組合せを受け、不正な組合せ・型を拒否する。辞書との往復。
- 保存と読込み（3手法）: 要約に来歴が書かれ、読み戻せる。版1の要約が読める。版2で来歴が無い・壊れているものは拒否する。
- 済んだrunの判定: 同じ基準版→飛ばす（commitが違っても）。違う基準版・記録なし→例外で止まり、どのrunも実行されず、進み具合の報告も無く、ファイルが変わらない（済んだrunと、未実行のrunが混ざる計画で、逐次と並列の両方）。版2で来歴の壊れた要約→`conflicting`（止めない）。
- 一覧の表: 来歴の3列、版1の要約の空の欄（版1だけの表でも3列がある）。
- コマンド: 2つのコマンド（比較scriptは`tests/refactoring/test_behavior_fix_comparison_scripts.py`）が、止まるときに終了コード2・警告を出し、表と記録を書かない。`--print-plan`の表示。実行の記録の基準版。
- 基準版の一致: `BEHAVIOR_BASELINE_VERSION == BASELINE_VERSION`（golden helper）。
- 回帰: 指標の値が変わらないこと（既存の掃引・保存のtestの、直接runとの一致の比較が、来歴を除いて通る）。

## File Structure Plan
- 新規: `src/federated_learning_experiments/core/run_provenance.py`、`tests/refactoring/test_experiment_run_code_provenance.py`。
- 変更: `runtime/{fedsda,feddrift,fedavg}_run_artifact_files.py`、`sweep/experiment_sweep_execution.py`、`sweep/experiment_sweep_records.py`、`sweep/experiment_sweep_command.py`、直下の`run_behavior_fix_comparison.py`。
- testの追従（task 1.2〜2.3を1つのcommitにする中で直す）: 保存の関数の呼出し（`test_fedsda_experiment_run_artifacts.py`、`test_feddrift_experiment_run_artifacts.py`、`test_fedavg_experiment_run.py`、`test_cross_evaluation_download_deduplication.py`、`fresh_process_smoke.py`）、掃引の実行の呼出し（`test_experiment_sweep.py`、`test_feddrift_experiment_sweep.py`、`test_fedavg_experiment_sweep.py`）、形式の版のtest（版1の期待を2へ。「版2を拒否する」testは、版3を拒否するへ）、実行の記録の形式の版のtest、依存境界の許可（importを足すmoduleごとに、そのtaskで足す）。
- 文書: `docs/experiments/experiment-sweep.md`、`docs/experiments/refactoring-baseline.md`、再開案内。

## Requirements Traceability
| 要求 | 設計 |
|---|---|
| 1.1–1.3 | `RunProvenance`、保存の関数の必須の引数、形式の版2と版1の読込み、`build_run_provenance` |
| 2.1–2.2 | `write_run_summary_table`の3列 |
| 3.1–3.4 | `find_existing_run_state`の新しい状態、`execute_experiment_sweep`の例外、2つのコマンドの扱い |
| 4.1–4.2 | 一致のtest、実行の記録と比較の記録の基準版 |
| 5.1 | 文書 |
