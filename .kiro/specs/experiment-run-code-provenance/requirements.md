# Requirements Document

## Project Description (Input)
- 誰が困っているか: 新実装の掃引（`run_experiment_sweep.py`）と、挙動の修正の比較（`run_behavior_fix_comparison.py`）で、実験を回して結果を比べる研究者（ユーザー）。
- 今の状況: コードの版（gitのcommitと、未commitの変更の有無）は、コマンドの実行ごとの記録（`sweep_record_*.json`・`comparison_record_*.json`）にしか残らない。1 runの要約（`runs/<識別子>.json`）と、一覧の表（`run_summary.csv`）には、どの版のコードで作ったかが無い。期待する動作の基準版（重要動作goldenの版。2026-10-14に版5）は、どこにも残らない。済んだrunを飛ばす判定は、run設定の一致だけを見るので、既定を変えずにコードの動作が変わった（基準版が上がった）後に、同じ出力先で再開すると、古い動作の結果と新しい動作の結果が、気づかれないまま混ざる。
- 何を変えるか: 1 runの要約と一覧の表に、コードの版と基準版を残す。済んだrunは、基準版が同じときだけ飛ばし、違うとき（基準版の記録が無い古い結果を含む）は、runを始める前に、警告して止める。

## Introduction
2026-10-14の会話で、ユーザーは、実験データを管理しやすくする工夫のうち、A（runのファイルにコードの版を残し、済んだrunの判定に使う）を、次に行うと決めた。B（実験の台帳）・C（分析の出力の名前と見出し）・D（研究のディレクトリの名前）は、このspecの対象外。

## Boundary Context
- 対象: 新実装の3手法（FedSDA・FedDrift・FedAvg）の1 runの要約、一覧の表、掃引の実行（済んだrunの判定と、止める処理）、2つのコマンド（掃引と、比較script）、実行の記録、期待する動作の基準版の番号の置き場所、文書。
- 対象外: 固定旧と、その出力（`run_pareto_sweep.py`ほか）、run設定と識別子の作り方（変えない）、指標の値、goldenの内容、分析のscript（`analyze_behavior_fix_comparison.py`ほか）の出力。
- 仮定（会話でユーザーが決めていない点）: 基準版の記録が無い要約（このspecより前に保存した、前の形式の、読める要約）は、「基準版が違う」として扱い、止める。基準版は、3手法に共通の1つの番号とする（FedSDAだけが変わる版でも、FedDrift・FedAvgの済んだrunは、再利用しない）。commitの違いと未commitの変更の有無では、止めない（記録するだけ）。壊れた済んだrun（ファイルが片方しかない、読めない、新しい形式なのに来歴が無いか壊れている）は、止める対象にせず、今までどおり、そのrunだけを失敗とする（上書きしない）。

## Requirements
### Requirement 1: 1 runの要約にコードの版を残す
**Objective:** As a 研究者, I want 1 runのファイルだけで、作ったコードの版がわかること, so that 結果を別の場所へ移しても、どの段階の結果かを判別できる。
#### Acceptance Criteria
1. When 掃引か比較scriptが1 runを保存したとき, the run artifact shall 要約に、gitのcommit、未commitの変更（gitが追跡するファイルの変更）の有無、期待する動作の基準版の番号を持つ。
2. If gitの情報が取れない（gitが無い、作業ツリーの最上位でない、gitが失敗した）, the run artifact shall commitと変更の有無を値なしにし、取れなかった理由と、基準版の番号を持つ（runは止めない）。
3. The run artifact shall 要約の形式の版を上げ、前の形式の要約（コードの版を持たない）も、読めるようにする。

### Requirement 2: 一覧の表にコードの版を出す
**Objective:** As a 研究者, I want 一覧の表で、runごとのコードの版が見えること, so that 表を開くだけで、版の混在に気づける。
#### Acceptance Criteria
1. The run summary table shall 1行ごとに、commit、未commitの変更の有無、基準版の番号の列を持つ（表のどの要約も前の形式でも、3つの列を置く）。
2. When 要約が前の形式（コードの版を持たない）のとき, the run summary table shall その行の3つの列を空にする（その行を表から外さない）。

### Requirement 3: 基準版が違う済んだrunでは、始める前に止める
**Objective:** As a 研究者, I want 動作の違うコードの結果が、同じ出力先で混ざらないこと, so that 比べる結果が、同じ動作のコードから出たものだと信じられる。
#### Acceptance Criteria
1. When 出力先に、計画のrunの済んだファイルがあり、その要約の基準版が、今のコードの基準版と同じとき, the sweep shall 今までどおり、そのrunを飛ばす（commitや、未commitの変更の有無が違っても、飛ばす）。
2. If 計画のrunのどれかで、出力先に要約と表がそろい、両方が読めて、その基準版が今のコードと違う、または、基準版の記録が無い（前の形式）, the sweep shall どのrunも実行・保存せずに止め、そのrunのファイルと、記録された基準版・今の基準版を、警告として出す。
3. When 2つのコマンド（掃引と比較script）が、2のために止まったとき, the command shall 0でも1でもない終了コード（引数の誤りと同じ2）で終わり、出力先のどのファイルも変えない。
4. When 計画の表示（`--print-plan`）のとき, the command shall 基準版の違う済んだrunを、それとわかる状態と理由で表示する（何も作らない）。

### Requirement 4: 実行のコードの基準版の番号を、goldenの版とずれないようにする
**Objective:** As a 開発者, I want 基準版の番号が、goldenの現在の版と一致すること, so that goldenの版を上げたときに、runに残る番号の更新を忘れない。
#### Acceptance Criteria
1. The validation shall 実行のコードが使う基準版の番号（正本）と、goldenのhelperの現在の版（`BASELINE_VERSION`）が一致することを、testで確かめる。
2. The command shall コマンドの実行ごとの記録（`sweep_record_*.json`・`comparison_record_*.json`）にも、基準版の番号を書く。

### Requirement 5: 文書を合わせる
**Objective:** As a 研究者, I want 文書が、新しい記録と止まる条件を説明すること, so that 止まったときに、どうすればよいかがわかる。
#### Acceptance Criteria
1. The documentation shall 実験の入口の文書で、要約・表・実行の記録に残るコードの版、基準版が違うときに止まることと、そのときの対処（別の出力先を使う）を書き、基準の文書で、基準版を上げるときに、実行のコードの番号も上げることを書く。
