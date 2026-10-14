# Requirements Document

## Project Description (Input)
- 誰が困っているか: 新実装の掃引（`run_experiment_sweep.py`）と、挙動の修正の比較（`run_behavior_fix_comparison.py`・`analyze_behavior_fix_comparison.py`）で、実験を回して結果を比べる研究者（ユーザー）。
- 今の状況: 結果のディレクトリ（`results/`。gitの対象外）には、目的が残らない。研究の問いごとのまとまりが無く、比較scriptは、出力先を毎回、人が決める。実験の一覧と、判断へのつながりは、gitの中のどこにも無い。分析の出力は、標準出力をファイルへ移して`behavior-fix-comparison_2.md`・`_3.md`のような連番で残していて、名前からも中身からも、どの実験のどのrun（どのコード）を集計したかがわからない。
- 何を変えるか: 結果のディレクトリを、研究の問い→日時と目的、の2段で作る（D）。コマンドが、目的などを書いたREADMEを結果のディレクトリへ自動で置き、gitの中に、1実験1行の台帳を置く（B）。分析の出力を、入力のディレクトリの中に、内容から決まる名前のファイルとして書き、先頭に出どころ（入力・run数・入力のrunのコードの版と基準版・分析のコードの版）を書く（C）。

## Introduction
2026-10-14の会話で、ユーザーは、実験データの管理の改善のうち、Aの後に、B（台帳）・C（分析の出力の名前と見出し）・D（ディレクトリの名前）を、1つのspecで進めると決めた。Bは、主担当の提案（READMEはコマンドが自動で書き、台帳はdocsに1実験1行の索引）を、ユーザーが了承した。

## Boundary Context
- 対象: 掃引のコマンド、比較script、比較の分析script、台帳の文書、実験の文書、AGENTS.mdの実験成果物の規約。
- 対象外: 固定旧と、その出力（`run_pareto_sweep.py`、`tools/`、`results/`の旧実装の結果）、runの要約と一覧の表の形式（Aで決めた）、既存の結果のディレクトリの移動や改名、`compare_sweep_results_with_legacy.py`（移行の間だけの比較）。
- 仮定（会話でユーザーが決めていない点）:
  - 目的（`--purpose`）は、実行するときに必須とする（書き忘れを防ぐため。計画の表示だけなら要らない）。そのため、これまでの掃引のコマンドの例は、`--purpose`を足す必要がある。
  - 研究の問い（`--study`）は、掃引では任意（省略したら、今までどおり`results/<日時>_<ラベル>`）。比較scriptは、研究の問いが決まっているので、`behavior-fix-comparison`に固定する。
  - 同じ出力先で再開したとき、READMEが既にあれば、変えない（人が書いた結論を消さないため）。1つの出力先は1つの実験（1つの目的）に使い、再開は同じ目的のときだけにする（文書で示す。各回の目的は、実行の記録に残る）。
  - 台帳は、人（主担当またはユーザー）が書く。コマンドは書かない（サーバの作業ツリーにgitの差分を作らないため）。

## Requirements
### Requirement 1: 結果のディレクトリを、研究の問いと、日時と目的で作る
**Objective:** As a 研究者, I want 同じ問いの実験が1か所に日時順で並ぶこと, so that 結果を探しやすく、台帳の1行と対応づけられる。
#### Acceptance Criteria
1. When 掃引のコマンドに研究の問い（`--study`）が指定され、出力先が省略されたとき, the sweep command shall 出力先を`results/<研究の問い>/<開始日時>_<ラベル>`にする。研究の問いが無ければ、今までどおり`results/<開始日時>_<ラベル>`にする。研究の問いと出力先を両方指定したら、引数の誤り（終了コード2）として止める（READMEの研究の問いと、実際の場所が食い違わないように）。
2. When 比較scriptの出力先が省略されたとき, the comparison script shall 出力先を`results/behavior-fix-comparison/<開始日時>_<ラベル>`にする（出力先を指定すれば、今までどおりそこを使う。再開は、出力先を指定して行う）。
3. If 指定された研究の問いか、ラベル（出力先を指定したときも、既定値を含めて常に）が、ディレクトリの区切り（`/`・`\`・`:`）を含む、空、前後に空白がある、`.`か`..`, the commands shall 何も作らずに、引数の誤り（終了コード2）として止める。

### Requirement 2: 結果のディレクトリに、目的と出どころを書いたREADMEを置く
**Objective:** As a 研究者, I want 結果のディレクトリだけを見て、何のための実験かがわかること, so that サーバから結果を移しても、説明が付いてくる。
#### Acceptance Criteria
1. The commands shall 実行するとき（計画の表示を除く）、目的（`--purpose`。前後の空白を除いて空でなく、改行を含まない文字列。前後の空白を除いて記録する）を必須にする。無ければ、何も作らずに、引数の誤り（終了コード2）として止める。
2. When 実行を始めるとき、出力先に`README.md`が無ければ, the commands shall 目的、研究の問い、作成の日時、コマンドの引数、コードの版（commit、未commitの変更の有無、基準版）、結論を書く欄（未記入であることと、書く場所の案内だけを持つ）を持つ`README.md`を作る。
3. When 出力先に`README.md`が既にあるとき, the commands shall それを変えない（目的が違っても上書きしない）。
4. The commands shall 実行の記録（`sweep_record_*.json`・`comparison_record_*.json`）にも、目的と研究の問いを書く。
5. If 基準版の違う済んだrunのために止まるとき（A）, the commands shall READMEも作らない（出力先のファイルを変えない、というAの約束を保つ）。

### Requirement 3: 実験の台帳を、gitの中に置く
**Objective:** As a 研究者, I want 実験の一覧と、判断へのつながりを、gitの中で追えること, so that 結果が消えても、いつ何のために何を回し、どう判断したかが残る。
#### Acceptance Criteria
1. The documentation shall `docs/experiments/experiment-log.md`に、1実験（1回のコマンドの実行の単位でなく、1つの目的の実験）を1行とする台帳を置く。列は、日付、研究の問いと結果のディレクトリ、目的、規模（構成・データ・seed・集約間隔・run数）、コード（commitと基準版）、結論の参照（findingや判断の記録へのリンク。未定なら「未」）。
2. The documentation shall 新実装の、記録の残る既存の実験（`results/behavior-fix-comparison`の3回の実行）を、台帳へ載せる。
3. The documentation shall 実験を始めたとき・結論が出たときに台帳を更新する規約を、台帳の冒頭、実験の入口の文書、AGENTS.mdの実験成果物の規約に書く。

### Requirement 4: 分析の出力を、入力のディレクトリの中に、出どころつきで書く
**Objective:** As a 研究者, I want 分析の結果のファイルが、何を集計したものかを自分で説明すること, so that 連番のファイルを取り違えず、版の混ざった集計に気づける。
#### Acceptance Criteria
1. When 分析scriptを実行したとき, the analysis script shall 結果を、入力のディレクトリの中の`analysis_<作成日時>.md`へ書き（同じ名前があれば上書きせず、番号を付ける）、そのファイルの場所を標準出力へ出す。`--stdout`を指定したときは、同じ中身（出どころと警告を含む）を、ファイルを作らずに標準出力へ出す（出す先だけが違う）。
2. The analysis script shall 結果の先頭に、作成の日時、分析のコマンドの引数、分析のコードの版（commit、未commitの変更の有無、基準版）、入力のディレクトリ、組合せごとのrun数と、その入力のrunのcommitと基準版（一覧の表の列から。列が無い・空なら「記録なし」）を書く。
3. If 集計に使う全部のrunの基準版が1つでない（読み込んだ全部の組合せの一覧の表の全行が対象。「記録なし」を含む）, the analysis script shall 見出しの行の直後（出どころの節より前）に、その旨の警告の行を書く（集計は続ける）。
4. The analysis script shall 集計の値（表の数値）を変えない（変更前のscriptが同じ入力から出す表の部分と、文字単位で同じ）。
5. If 入力のディレクトリに、どの組合せの一覧の表も無い, the analysis script shall 今までどおり、その旨を出して1を返し、ファイルを作らない。

### Requirement 5: 文書を合わせる
**Objective:** As a 研究者, I want 文書が新しい引数と置き場所を説明すること, so that サーバでのコマンドを正しく書ける。
#### Acceptance Criteria
1. The documentation shall 実験の入口の文書と、比較script・分析scriptの説明で、`--study`・`--label`・`--purpose`、出力先の形、README、分析の出力の置き場所を説明し、コマンドの例に`--purpose`を足す。
