# Requirements Document

## Project Description (Input)
- 誰が困っているか: 実装のたびに全pytestで完了を判定している研究者と主担当。
- 今の状況: 全pytest（並列12 worker）は約46分かかる。そのうち、重要動作goldenのtest（`tests/refactoring/test_important_behavior_regression_goldens.py`、306件）だけで45.4分かかり、全体の下限になっている。並列の配り方は「同じファイルのtestは同じprocess」なので、このファイルの全ケース×全版（基準版1〜9、173組）の実行が、1つのprocessで順に行われている。また、版をまたいでrun設定が同じ「ケース×版」があり（173組のうち、違う設定は134組）、同じrunを何度も実行している。
- 何を変えるか: goldenのtestのファイルを、ケースの名前で、いくつかの組へ分けて別のprocessへ配る（構成の照合のファイルと同じ仕組み）。同じprocessの中では、同じrun設定のrunの実行結果を使い回す。照合の内容と結果は変えない。

## Introduction
2026-10-20の会話で、ユーザーはgoldenの照合の時間を短くすることを決めた。最初は「同じ設定のケースを使い回す」案だったが、数えると、使い回せるのは軽いFedDrift・FedAvgのケースが中心で、時間はおよそ1割しか減らない見込みだった。一方、このファイルが1つのprocessで45分かかっていることが全体の下限なので、ケースで分けて配る方が効く。2つを合わせて行う。

## Boundary Context
- 対象: `tests/conftest.py`の並列の配り方、`tests/refactoring/test_important_behavior_regression_goldens.py`のtestの分け方（版の差の説明のtestを、ケースごとのtestに分け直す）、`tests/refactoring/behavior_golden_cases.py`の実行の使い回し、文書（共通引継ぎ手順の全pytestの所要時間）。
- 対象外: source（`src/`）、goldenのファイル、照合の内容（比べる項目と許す差）、goldenの生成の手順（2つの独立processで生成する）、ほかのtestのファイル。

## Requirements
### Requirement 1: goldenのtestを、ケースの名前で組へ分けて配る
#### Acceptance Criteria
1. When 全pytestを`--dist loadfile`の並列で実行する, the test infrastructure shall goldenのtestのファイルのうち、ケースの名前を持つtestを、ケースの名前で決まる組へ分け、組ごとに同じprocessで実行する（ケースの名前を持たないtestは、今と同じく、ファイルの組で実行する）。
2. The test infrastructure shall 同じケースの名前を持つtest（全版の数値の照合と、版の差の説明）を、同じ組に入れる（同じprocessの中で、ケースの実行結果を使い回すため）。
3. The test infrastructure shall 組の分け方を、testで確かめる（同じケースのtestが同じ組になる。重いケースが別々の組に入る）。

### Requirement 2: 版の差の説明のtestを、ケースごとのtestに分け直す
#### Acceptance Criteria
1. The validation shall 1つのtestの中で全ケースを回している版の差の説明のtest（版5→6、版7→8、版8→9）を、ケースごとのtestに分け直し、各ケースについて、今と同じ項目を、今と同じ規則で比べる。
2. The validation shall 「少なくとも1つのケースで差がある」という確認（版5→6の予測の記録、版7→8の下りの減少、版8→9の統合）を、その経路を通ると分かっている特定のケースで、そのケースのtestとして確かめる（比較が空にならないことを保つ）。

### Requirement 3: 同じrun設定のrunの実行結果を使い回す
#### Acceptance Criteria
1. The validation shall 同じprocessの中で、同じ完全なrun設定（保存の形）を持つ「ケース×版」のrunを1回だけ実行し（完全なrun設定が1つでも違えば、別に実行する）、その実行結果（runの記録・指標・パラメータ・版の差の事象の数）を使い回す。経路の記録（`coverage`）と必須の経路の確認、比べる形の設定は、版ごとに計算する。
2. The validation shall goldenの生成の手順（2つの独立processで、それぞれ全ケースを実行して比べる）では、使い回しをしない（独立に実行することが生成の確認の意味なので）。

### Requirement 4: 結果を変えずに速くなることを確かめる
#### Acceptance Criteria
1. The validation shall 変更の後の全pytestが、変更の前と同じtestの合否（件数は、分け直したtestの分だけ増減する）で通ることを確かめる。goldenの照合のtestの全部が成功する。
2. The validation shall 変更の前と後の、全pytestの所要時間と、goldenのtestのファイルの時間を測って記録する。
