# Design Document

## Overview
2つの実行のコマンド（掃引、比較script）に、研究の問い・ラベル・目的の引数と、出力先の既定の形を揃え、実行の前に`README.md`を置く。分析scriptは、結果を入力のディレクトリの中のファイルへ書き、先頭に出どころを書く。台帳と規約は、文書で足す。

## Boundary Commitments
### This Spec Owns
- `sweep/experiment_sweep_command.py`: `--study`・`--purpose`、出力先の既定、名前の検査、実行の前の基準版の確認の位置。
- `sweep/experiment_sweep_records.py`: READMEを作る関数、実行の記録の目的。
- `sweep/experiment_sweep_execution.py`: 基準版の違う済んだrunの理由を集める関数（`execute_experiment_sweep`と、コマンドの事前の確認で共用）。
- 直下の`run_behavior_fix_comparison.py`・`analyze_behavior_fix_comparison.py`。
- 文書: `docs/experiments/experiment-log.md`（新規）、`docs/experiments/experiment-sweep.md`、AGENTS.mdの「実験成果物と重複確認」、再開案内。
### Out of Boundary
runの要約・一覧の表・記録の表の形式、runの実行、固定旧、既存の結果の移動。
### Allowed Dependencies
新しいmoduleは無い。掃引のコマンドが、実行のmoduleの新しい関数を読む（依存境界の許可へ足す）。分析scriptは、`src`を読込みの経路へ足し、`build_run_provenance`を読む（直下のscriptなので、依存境界の検査の対象外）。
### Revalidation Triggers
コマンドの引数（サーバで使うコマンドの例）、出力先の既定、分析scriptの出力の先（標準出力からファイルへ）。

## Architecture
### 出力先（D）
- 掃引のコマンド: `--study NAME`（任意）を足す。出力先を省略したとき、`--study`があれば`results/<study>/<YYYYMMDD_HHMMSS>_<label>`、無ければ今までどおり`results/<YYYYMMDD_HHMMSS>_<label>`。日時は、今までどおり、開始の時刻（秒まで。開始の秒まで同じで、同じラベルなら、同じディレクトリになる）。`--study`と`--output-directory`を両方指定したら、引数の誤り（終了コード2）。`--label`は、出力先を指定したときは使わないが、検査は常にする（既定値`sweep`・`comparison`は通る）。
- 比較script: `--output-directory`を任意にし、`--label`（既定`comparison`）を足す。省略したとき、`results/behavior-fix-comparison/<YYYYMMDD_HHMMSS>_<label>`。研究の問いは固定（`BEHAVIOR_FIX_COMPARISON_STUDY_NAME`）。
- 名前の検査（指定された`--study`と、`--label`は常に）: `validate_plain_directory_name(name, *, argument_name)`（掃引のコマンドのmodule）。空、前後の空白、`.`・`..`、`/`・`\`・`:`を含む名前を`ValueError`で拒否する。2つのコマンドは、引数の誤りとして終了コード2で止める（何も作らない）。`--print-plan`でも検査する（表示する出力先が、実行と同じであるように）。

### 目的とREADME（B）
- `--purpose TEXT`: 実行のとき必須。前後の空白を除いて空のもの・改行（`\n`・`\r`）を含むものは拒否し、前後の空白を除いた値を記録する。`--print-plan`のときは要らない（あっても使わない）。必須の検査は、argparseの`required`でなく、`--print-plan`でないときに`main`の中で行う（終了コード2）。
- 引数の検査の順: 引数を読んだ直後（計画を作る前、MNISTの確認と来歴の作成の前）に、名前の検査、`--study`と出力先の同時の指定、（`--print-plan`でなければ）目的の検査を行う。どれかに当たれば、何も作らずに2を返す。
- 事前の確認の形: `collect_behavior_baseline_mismatch_descriptions`は理由の一覧を返す。コマンドは、空でなければ`ExistingRunBehaviorBaselineMismatchError(理由の一覧)`を作り、今の`report_behavior_baseline_mismatch`で、今と同じ文面（`warning: …`の行と`stopped before running: …`の行）を出して2を返す。済んだrunの要約を、事前の確認と`execute_experiment_sweep`で2回読むが、runの実行に比べて小さいので許す（`execute_experiment_sweep`を単独で呼ぶ場合の保護を残すため）。
- 実行の前の順（今は、出力先を作る処理——掃引のコマンドと比較scriptの`mkdir`——が、基準版の確認——`execute_experiment_sweep`の中——より前にある。これを入れ替える）: 計画を作る→（掃引のコマンドは、MNISTの確認）→来歴を作る→基準版の違う済んだrunを集める（`collect_behavior_baseline_mismatch_descriptions`。あれば、警告して2を返す。ディレクトリもREADMEも作らない）→出力先を作る→READMEを作る→掃引を実行する（`execute_experiment_sweep`は、今までどおり、自分でも同じ確認をする）。
- `write_experiment_readme(*, output_directory, purpose, study_name, started_at, command_arguments, run_provenance) -> Path | None`（記録のmodule）: `README.md`を排他的に作る（`open("x")`）。既にあれば、何もせずにNoneを返す。中身（Markdown）:
  - 見出し: 出力先のディレクトリ名。
  - 目的、研究の問い（無ければ「なし」）、作成の日時、コマンドの引数（コードブロック）、コードの版（commit——取れなければ理由——、未commitの変更の有無、基準版）。
  - 「結論」の節: 「（未記入。結論が出たら、ここと、`docs/experiments/experiment-log.md`の台帳に書く）」。
- コマンドは、作ったときは`README: <path>`、既にあったときは`README: <path> (already exists; left unchanged)`を表示する。
- 実行の記録: 掃引の`write_sweep_record`へ`purpose`と`study_name`の引数を足し、記録へ`purpose`・`study_name`を書く（どちらも必須のkeyword引数。研究の問いが無ければ`study_name=None`。記録の形式の版を3へ上げる）。比較の記録へも`purpose`と`study_name`（`behavior-fix-comparison`。出力先を指定したときも同じ）を書く。比較の記録のファイルは、今は同じ秒の再実行で上書きされうるので、掃引の記録と同じく排他的に作り、同じ名前があれば`_2`・`_3`…を付ける（比較scriptの中に書く。公開名は足さない）。開始の秒とラベルが同じ2回の実行は、同じ出力先に合流する（READMEは1回目のまま。2回目の目的は実行の記録に残る。これで足りるとする）。

### 台帳（B）
- `docs/experiments/experiment-log.md`: 冒頭に、規約（1つの目的の実験を1行。始めたときに行を足し、結論が出たら参照を書く。結果の本文は、findingや判断の記録へ書き、台帳には書かない）と列の説明。表の列は、要求3.1のとおり。
- 既存の行: `results/behavior-fix-comparison`の3回の実行（比較の記録`comparison_record_*.json`の、開始の日時・引数・`code_revision.commit`・run数から書く。これらのrunの要約は、来歴を持たない前の形式なので、分析の出どころでは「記録なし」と出る。その旨を注記する。比較の記録は基準版を持たないので、台帳の基準版は「記録なし」と書く）。研究の問いは`behavior-fix-comparison`、ディレクトリは`results/behavior-fix-comparison`（このspecより前の形。移さない）。結論の参照は、LEGACY-016の記録の該当の節。8通りのうち、全部登録×002を直した側×待つ警報の120 runは、この出力先の記録に無いので、ディレクトリを「未確認」と書く。
- 規約の追記: AGENTS.mdの「実験成果物と重複確認」に、新実装の実験の`--study`・`--purpose`と台帳の更新を1項目で足す。実験の入口の文書にも書く。

### 分析の出力（C）
- 分析scriptの使い方: `python analyze_behavior_fix_comparison.py <出力先> [--chains] [--stdout]`。
- 引数: `argparse`で読む（今は、先頭の引数を出力先として読むので、`--stdout`を先頭に置くと誤る）。`--help`は、今の「説明を出して1を返す」から、argparseの扱い（説明を出して0）に変わる。引数が無いときは、argparseの誤り（2）になる。出どころに書く「分析のコマンドの引数」は、`main`に渡された引数の列（`python`から呼ぶときは`sys.argv[1:]`）。
- 入力が無いとき（どの組合せの一覧の表も無い）: 今までどおり、その旨を出して1を返し、ファイルを作らない。
- 既定: 結果を`<出力先>/analysis_<YYYYMMDD_HHMMSS>.md`へ排他的に書く（同じ名前があれば`_2`・`_3`…）。標準出力へは、書いたファイルの場所だけを出す。`--stdout`では、同じ中身（出どころと警告を含む）を標準出力へ出し、ファイルを作らない。
- 実装: 今の`print`の並びを、行を集める形（`report_lines.append`）に替え、最後に、ファイルか標準出力へ出す。表の数値の計算は変えない。
- 見出しの直後の「出どころ」の節:
  - 作成の日時、分析のコマンドの引数、分析のコードの版（`build_run_provenance`。commit、未commitの変更の有無、基準版）、入力のディレクトリ。
  - 組合せごとの行: 組合せの名前、run数、入力のrunのcommit（異なる値を並べる。12文字に縮める）、基準版（異なる値を並べる）。値は、一覧の表の`code_commit`・`behavior_baseline_version`の列から読む。列が無い・空の行は「記録なし」。
  - 警告は、見出しの行の直後（出どころの節より前）に書く（下の条件のとき）。
  - 今の見出しの次の行（`出力先: …。組合せ: …（N run）`）は、出どころの節へ移す（本文から消す）。出どころの節は、見出しの行の次から、`## 1.`の行の前まで。
  - 警告の条件と文面: 読み込んだ全部の組合せの一覧の表の全行の基準版の値（「記録なし」を含む）が1種類でなければ、「警告: 入力のrunの基準版が揃っていない（…）。版の違う結果を比べている可能性がある」を書く。
- `read_combination`は、指標の値に加えて、来歴の2列を返す（戻り値の形を変える。呼出しは、このscriptの中だけ）。
- 分析のコードの版は、`build_run_provenance`（記録のmodule。読込みでtorchとNumPyを読む）で作る。この読込みは、出どころを作る関数の中で行い、`--help`と、入力が無いときは読まない（起動の重さは、実行のときだけ許容する）。
- 表の数値を変えないことの確かめ方: 実装の前に、変更前のscript（5ca896f）で、固定の入力（今のtestの入力と、3つの組合せと複数のデータ・集約間隔・seedを持つ入力。後者は、`--chains`の表に行が出るように、runごとの`runs/<識別子>.npz`に`adaptation.client_id`・`adaptation.adaptation_sample_index`・`adaptation.adaptation_outcome`の配列を置く）から、`--chains`ありで出した標準出力を、testの期待値として埋める（期待値の連鎖の表に行があることも確かめる）。比べる範囲は、`## 1.`の行から末尾まで（見出しと、出力先のパスを含む行は、比べない）。変更後の`--stdout`の出力の`## 1.`から末尾までが、それと文字単位で同じことを確かめる（期待値は、変更前のscriptの出力から作り、手で書かない。作り方をtestのコメントに書く）。

## 旧と違う点
掃引・比較scriptの実行に`--purpose`が要る。比較scriptの出力先が省略できる。結果のディレクトリに`README.md`ができる。分析scriptは、既定でファイルへ書く（今までの、標準出力をファイルへ移す使い方は、`--stdout`で続けられる）。表の数値、runの実行と保存は変わらない。

## Validation Design
- 出力先: `--study`あり・なしの既定、比較scriptの既定、名前の検査（不正な名前で終了コード2・何も作らない）、`--print-plan`の表示。
- 目的とREADME: `--purpose`が無い・空白・改行で終了コード2（何も作らない）、`--print-plan`では要らない、READMEの中身（目的・研究の問い・コマンド・来歴・結論の欄）、既存のREADMEを変えない、基準版で止まるときにREADMEもディレクトリの中身も変えない、実行の記録の目的。
- 分析: ファイルの名前と、同じ名前のときの番号、ファイルと`--stdout`の中身が同じ、出どころの節（run数、commit、基準版、「記録なし」）、基準版が混ざったときの警告とその位置（見出しの行の直後）、入力が無いときにファイルを作らない（要求4.5）、引数の順（`--stdout`が先頭でも読める）、表の部分が変更前のscriptの出力と文字単位で同じ（上の確かめ方）。
- 引数の誤りのtestは、終了コード2だけでなく、エラーの文面で理由（`--study`、`--purpose`、名前）を確かめ、正しい引数で通ることと対にする（今のparserは、未知の`--study`を終了コード2で止めるので、終了コードだけではREDにならない）。
- 回帰: 既存の掃引・比較・分析のtestの期待値は、引数（`--purpose`）を足すこと、掃引のコマンドの引数の一覧に`--study`が増えること（`test_fedsda_default_waited_registration_and_alarm_wait.py`の引数の一覧のtest）、比較scriptの引数の一覧（あれば）に`--label`・`--purpose`が増えること、分析の出力の先の変更（`--stdout`）以外を変えない。分析の既存のtest（標準出力の表を見るもの）は、`--stdout`を付けて呼ぶ。比較scriptの、基準版で止まるtestは、monkeypatchの対象を`execute_experiment_sweep`から`collect_behavior_baseline_mismatch_descriptions`へ替える（事前の確認が先に止めるため）。

## File Structure Plan
- 変更: `src/federated_learning_experiments/sweep/experiment_sweep_command.py`、`experiment_sweep_records.py`、`experiment_sweep_execution.py`、直下の`run_behavior_fix_comparison.py`・`analyze_behavior_fix_comparison.py`。
- test: `tests/refactoring/test_fedsda_default_waited_registration_and_alarm_wait.py`（掃引のコマンドの引数の一覧に`--study`を足す）、`tests/refactoring/test_experiment_sweep.py`（コマンドの`main`の呼出し、`write_sweep_record`の直接の呼出し、記録の形式の版の期待）、`test_behavior_fix_comparison_scripts.py`（比較scriptの`main`の呼出し、分析の`main`の呼出し）、掃引のコマンドを呼ぶほかのtest（`test_feddrift_experiment_sweep.py`・`test_fedavg_experiment_sweep.py`ほか。grepで全部を探して`--purpose`を足す）、共用のfresh script（`tests/refactoring/fresh_process_smoke.py`の掃引のコマンドの呼出し）、依存境界（`sweep/experiment_sweep_command.py`へ`collect_behavior_baseline_mismatch_descriptions`・`write_experiment_readme`、`sweep/experiment_sweep_records.py`に変更があればその名前）。
- 文書: `docs/experiments/experiment-log.md`（新規）、`docs/experiments/experiment-sweep.md`（コマンドの例の全部に`--purpose`。比較の例は出力先を省略した既定の形にし、`results/behavior-fix-comparison`直下の平らな形は、このspecより前の結果で新しい規約の対象外と注記する。分析の例に`--stdout`と既定の出力先）、`run_behavior_fix_comparison.py`と`analyze_behavior_fix_comparison.py`のdocstringの使い方の例、`AGENTS.md`、再開案内。

## Requirements Traceability
| 要求 | 設計 |
|---|---|
| 1.1–1.3 | 出力先（D）、`validate_plain_directory_name` |
| 2.1–2.5 | 目的とREADME、実行の前の順、`collect_behavior_baseline_mismatch_descriptions`、実行の記録 |
| 3.1–3.3 | 台帳 |
| 4.1–4.5 | 分析の出力 |
| 5.1 | 文書 |
