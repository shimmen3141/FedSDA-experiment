# 独立レビューと検証記録

主担当はClaude Code。レビュー担当の選択は共通引継ぎ手順に従う。実行環境はWindowsの基準環境（Python 3.13、torch 2.12.1+cpu）。

## 要求r1・設計r1・命名r1・tasks r1（Luna、session `01a11d4d-d7b4-7e10-a912-0da032effa49`）— REQUIREMENTS: APPROVED / DESIGN: REJECTED / NAMING: APPROVED / TASKS: APPROVED

- 選択: 文書と命名の通常のレビューなのでGPT-6 Luna（`model_reasoning_effort="medium"`、read-only、ephemeral。ログのmodel行と`reasoning effort: medium`を確認）。代替は行っていない。依頼文には、4文書のLF hash、確認項目（検査がすべて吸収の更新より前にあるか、「読む→吸収→解放」で吸収の拒否のときに何も変わらないか、旧との順序の違いで最終状態が変わる経路がないか、解放がないときも吸収を空で呼ぶ判断、実旧をoracleにする方法、共用scriptへの流れの追加）を入れた。
- 指摘（設計、Minor）: 解放がない場合も吸収を空の標本列で呼ぶ設計だと、吸収は標本が空でも保有モデルを取得するため、現在の学習帰属のモデルが未登録なら`KeyError`になる。旧の警報のない分岐は、解放対象がなければモデルを参照しない。要求1.2とも例外時の扱いが揃っていない。→ 事実と確認して採用。解放する標本がなければ吸収を呼ばずに空のtupleを返す形へ改めた（設計revision2。下書きのsourceへ早期returnを足し、testへ「解放がないとき吸収が呼ばれない」ことの確認を足した）。
- 他の確認項目（旧との最終状態の一致、検査の位置、oracle、共用script、命名、tasks）は問題なしと報告された。レビュー担当はtestを実行していない。

## 設計r2（Luna、別session `01a11d52-a285-78b0-9b45-f211b64e0cbe`）— DESIGN r2: APPROVED

指摘なし。解放対象がなければ入力検査と解放位置の確認の後に空のtupleを返して吸収を呼ばないことが、下書きのsource・要求1.2・旧の挙動と整合し、解放がない場合に吸収へ渡すだけのownerを検査しない点も要求2.3の範囲と矛盾しないと報告された。レビュー担当はtestを実行していない。

- 手順上の事実: 命名の事前登録のため、sourceとtestをリポジトリ外で下書きし、`git archive HEAD`で作った作業ツリーの複製へ置いてWindowsの基準環境のPythonで実行した（worktreeのsource・testは変更していない。設計r2の内容で、新test 44件と共用scriptを実行するtestが成功、Ruff・Pyright成功、汎用の変異toolで17/20・未検出3種は等価と判断）。`spec_checks.py names`は複製で報告なし。
- 外部証拠: 元checkoutの`venv/refactoring-tests/released-pending-sample-assignment-spec-review.md`/`.log`、`-design-r2-review.md`/`.log`。

## Task 1 — 実施記録

- testを先に適用した時点（sourceは未変更）: 保留位置のownerのtestは、足した読取りの操作のtest 2条件が失敗（2 failed / 67 passed）。新しいmoduleのtestは、moduleがないための収集失敗なので記録していない（新しい規則）。
- GREEN（commit `86cd7ed`）: 新test 44、保留位置のownerのtest 69、依存境界、共用scriptを実行するtest、保持と進行のtestを合わせて3215 passed。Ruff・Pyright成功。worktreeのファイルは、設計r2の再レビューでLunaが読んだ下書きとbyteが同じ。
- レビューの前に、汎用の変異toolを実行した（確定の関数17/20、保留位置のownerの読取りの操作3/3）。
- commit `86cd7ed`の全pytestは10077 passed / 3 skipped（下の反映より前のcommitの値。判定には使わない）。

### Haiku 1回目（session `0d8bb960-cb90-481d-a019-cb118fa5c38f`、source/test commit `86cd7ed`）— TASK 1: CHANGES_REQUESTED

選択: 複数のownerの更新順序を扱う実装のレビューなのでClaude Haiku 5.5（`--effort medium`指定、JSONの`is_error=false`と`modelUsage`の`claude-haiku-5-5`で確認。実効effortは出力されないので指定値）。代替は行っていない。読取り専用で、testとgitを実行していない。実装にBlocker・Majorはないと報告された。確認したと報告されたこと: 状態を変えるのは吸収の更新と解放だけで全検査の後にある、通常の入力で解放が失敗する具体例はない、旧との最終状態（標本の並び、概念計数、損失統計とクラス別統計）が一致する、読取りの操作が状態を変えず解放と同じ位置を返す、止めた旧の処理は対照の対象を変えない、共用scriptの既存の確認は残っている。確認していないと報告されたもの: `_validate_buffered_alarm_observations`との比較、`build_absorption_oracle`の本体、変異tool、gitとtestの実行、固定旧実装とgoldenが変わっていないこと。

指摘と採否:

1. Minor: 証拠文書の「どの入力でも同じ例外」は誤り。型の不正と並びの不一致が同時にある入力では、検査を後へ移した変異で例外の型が変わる（元はTypeError、変異ではValueError）。既存の拒否testは不正を1件ずつしか入れないので、この差を検出できず、等価とした理由が成り立たない。→ 事実と確認して採用。型の検査が並びの検査より先に拒否することを確かめるtest（2条件）を足し、該当の2種は検出されるようになった（19/20）。証拠文書を訂正した。残る1種（並びの検査そのものを後へ移す）は、移した先との間に例外を出す処理がなく、等価のまま。
2. Minor: 現在の学習帰属のモデルが保有されていない状態で解放が起きるときの拒否のtestがない。→ 採用。拒否条件へ1件足した（吸収が`LookupError`で拒否し、保留を含む全状態が不変）。
3. 任意: 要求2.3の「解放しない保留標本の中身は検査しない」は、全要素の型と位置のintを検査している実装・設計と文言がずれる。→ 要求は変更しない。設計4節が検査の対象を正確に書いており（型と位置は全要素、標本と概念IDは解放する分だけ）、要求2.3の「中身」は標本と概念IDを指す、と読む。次に要求を改訂する機会があれば「標本と概念IDは検査しない」と書く。
4. 任意: 吸収が途中で失敗しないことは損失評価の入力検査に依存しており、吸収の仕様に書かれていない。→ 本specの範囲外（吸収のspec）。記録だけ残す。

反映後の実測（commit `cbf38b6`。sourceは変更なし）: 新test 47 passed。確定の関数の変異19/20、復元後47 passed。命名表へtest名を足した（revision2に含める。revision2は未レビューで、Task 2・3の依頼で再レビューを受ける）。

### Haiku 2回目（別session `88f989b5-4b43-4de4-8689-07d4d8dce19a`、source/test commit `cbf38b6`）— TASK 1: APPROVED

1回目とは別のsession（`--effort medium`指定、JSONの`modelUsage`で実モデルを確認）。読取り専用で、testとgitを実行していない。1回目の指摘1・2は解消と報告された: 追加したtestの2条件は、型の検査が抜けたり後ろへ移ったりすれば`ValueError`になって失敗する。残る未検出1種の等価の判断は実コードから導ける（代入は検査を通った入力で例外を出さず、移した先は早期returnと吸収より前）。未保有モデルの拒否条件は吸収が`KeyError`（`LookupError`の派生型）で更新より前に拒否し、「本処理自身の検査」のtestの対象から正しく外れている。Blocker・Major・Minorなし。確認していないと報告されたもの: testとgitの実行、変異の`report.json`、設計4節・要求・命名、`build_absorption_oracle`の本体、特徴数の違う標本で損失評価が出す例外の型、固定旧実装とgoldenの差分、全pytest、共用script。

任意の指摘1件と採否: 要素の位置がintの派生型で、かつ並びが不一致の入力での検査の順は確かめていない。→ 不採用。レビュー担当の報告どおり、この組合せを分ける変異はなく（要素の検査のloopを後へ移す変異は、並びの検査より前のままなので作られない）、単一の不正の条件で要素の検査は覆われている。

## 命名r2・Task 2・Task 3 1回目（Haiku代替、session `dce2dbec-dab8-45f8-8135-3811fc31497b`、HEAD `331bba0`）— NAMING r2: APPROVED / TASK 2: CHANGES_REQUESTED / TASK 3: APPROVED

選択と代替: 証拠の照合と再実行が中心なので優先はGPT-6 Luna。Lunaへ依頼したが（session `01a11d66-e928-79b3-82be-21e0c15ddf05`）、利用上限に達していて応答が得られなかった（CLIのエラー: 「You've hit your usage limit ... try again at Oct 14th, 2026 12:54 PM」）。共通引継ぎ手順の代替規則によりClaude Haiku 5.5へ依頼した（`--effort medium`指定、JSONの`is_error=false`と`modelUsage`の`claude-haiku-5-5`で確認）。この回のHaikuは読取り専用で、対象test・共用script・Ruff・`spec_checks.py`を独立に実行していない（全pytestの判定基準が求める独立実行は、下の2回目で行う）。

- 命名r2: 追加した2つのtest名を含め、命名表がtest・sourceの定義と一致すると報告された。
- Task 3: JUnitの集計（10083 testcase、failure 0、error 0、skip 3）、旧回帰2件のtestcaseの存在とfailureなし、logの件数、件数の内訳、spec.jsonのhashと統合検証の表の一致を照合したと報告された（hashの計算はしていない）。
- Task 2: 変異の`report.json`（19/20、復元前後のhash一致、復元後47 passed。読取りの操作3/3）と、未検出1種の等価の判断を確認したうえで、下の指摘1。

指摘と採否:

1. Task 2、Minor: 変異toolの「消す」は関数の最上位の文だけが対象で、loopの中の2つの検査（要素の型、位置のint）は「消す」変異が作られていない。証拠文書の「各検査を消す…が含まれる」は範囲を超えている。→ 事実と確認して採用。toolへloopの本体の中の文を消す変異を足し、実行し直した（21/22。足された2種は検出）。証拠文書を訂正した。
2. Task 3、任意: 本specは2 module（新しいruntimeのmoduleと、保留位置のowner）に触れるので「小さいspec」の例外に当たらず、feature最終の判定は別sessionで受けること。→ そのとおり。別sessionへ依頼する（tasks.mdのTask 3は「Task 1のレビューにBlocker・Majorが残らなければ同じ依頼で」と書いているが、module数の条件を満たさないので適用しない）。
3. 命名、任意: 追加した2つのtest名が命名の承認より先に実装された事実をreview.mdにも残すこと。→ 採用（この記録）。設計r2で足したtest名と、Task 1の指摘で足したtest名は、実装の後に命名r2として再レビューを受けた。
4. Task 3、任意: 統合検証の「旧11」「3golden」の説明がない。→ 不採用。これまでの全specで使っている呼び方（旧11ケースの回帰goldenと、最終構成3ケースの回帰golden）で、共通引継ぎ手順の全pytestの基準にも同じ語がある。

## Task 2 2回目と独立実行（Haiku代替、別session `0bbf737a-b50e-493a-924e-8e1e55bf446c`、HEAD `aec8402`）— TASK 2: APPROVED

1回目とは別のsession（`--effort medium`指定、JSONの`modelUsage`で実モデルを確認）。全pytestの判定基準が独立レビュー担当に求める独立実行を行えるよう、検証コマンドだけを許可して起動した（pytest、Ruff、共用script、`spec_checks.py`、`git status`・`git diff`。共通引継ぎ手順の「testを実行させるときは対象と権限を別途限定する」に当たる。それ以外のコマンドとファイルの変更は許可していない）。指摘なし。

- 1回目の指摘は解消と報告された: `report.json`は22件、loopの中の文を消す2件が検出、未検出は等価な1件だけ、復元前後のhashが一致、toolの追加部分は説明どおりの変異を作る、証拠文書の件数が一致。確定の関数の検査・処理で、どの変異でも覆われていないものはないと報告された。
- レビュー担当が独立に実行したもの: 対象の4 test file（3160 passed）、共用scriptの単独実行（16の流れ、成功）、Ruff check/format、`spec_checks.py identity --rev cbf38b6`（4段階の承認hash、固定旧差分、作業ツリー、source hash、JUnit集計、旧回帰2件がすべてOK）、`git status --short`（実行の前後とも空）。
- 独立実行していないもの: 変異tool（sourceを書き換えるため`report.json`を読んだ）、全pytest。

全3taskを完了とした。命名r2・Task 2・Task 3は、Lunaが利用上限で使えないため、Claude Haiku 5.5が代替して承認した。

## feature最終レビュー（Haiku代替、別session `585569da-9763-4b00-8f50-1976f1e5ec6d`、HEAD `882b5b1`）— FEATURE FINAL: GO

選択と代替: 優先はGPT-6 Lunaだが、利用上限（10月14日まで）で使えないため、Claude Haiku 5.5が代替した（`--effort medium`指定、JSONの`modelUsage`で実モデルを確認）。これまでのどのレビューとも別のsession。照合コマンドだけを許可して起動した（`spec_checks.py`、`git status`・`git diff`）。本specは2 moduleに触れるので、Task 3と同じ依頼で最終判定を受ける扱いは適用していない。Blocker・Majorなし。要求9項目は、2.2を「概ね充足」（下の指摘2）、他を「充足」と判定された。

レビュー担当が独立に実行したもの: `spec_checks.py identity --rev cbf38b6`（承認hash、固定旧差分、作業ツリー、source hash、JUnit集計、旧回帰2件がOK）、`progress`（3/3、phase一致）、`git status`（空）、`src`・`tests`と固定旧の差分の確認。pytest・Ruff・Pyright・共用script・変異toolは実行していない（対象test・共用script・RuffはTask 2の2回目のレビュー担当が独立に実行済み）。

指摘と採否:

1. Minor（記録）: 統合検証とspec.jsonの変異の件数が「19/20」のままで、最終の実測（21/22）と合わない。→ 採用。両方を21/22へ直した。
2. Minor: クラス数2で観測ラベルが負の標本は、損失評価を通り、吸収の更新loopの中の損失統計の記録で`ValueError`になるので、その時点で1件目の標本追加と概念計数が済んでいる（要求2.2と設計4節が成り立たない）、という指摘。→ 不採用（前提が事実と違う）。主担当が、吸収のoracleで3標本の2件目のラベルを−1にして実行して確かめた: 新の吸収は、損失評価の入力検査が`ValueError`（観測ラベルは0以上クラス数未満が必要）で拒否し、標本・割当概念計数・損失統計は何も変わらない。損失評価は全標本について更新より前に行われるので、負のラベルは更新loopへ到達しない。確認に使ったscriptは元checkoutの`venv/refactoring-tests/released-pending-sample-assignment-final-review-probe-negative-label.py`（Git管理外）。同じ入力を実旧`_absorb_into_store`へ渡すと、例外なく3件とも吸収され、クラス別統計にクラス−1の項目ができる（旧の契約外入力での挙動。不正なclass入力での旧の統計の扱いはLEGACY-006に記録がある）。
3. 任意: 特徴数の違う標本の拒否条件の期待例外が`(RuntimeError, ValueError)`と広い。→ 変更しない。吸収のspecが決める例外で、本specは「拒否され、何も変わらない」ことだけを確かめる。
4. 任意: tasks.mdのTask 3の書き方が、共通引継ぎ手順の「1 module」の条件より緩い。→ 承認済みの文書は書き換えない。本specは別sessionで最終判定を受けた（この記録）。
5. 任意（開示済みの逸脱）: 追加したtest名が命名r2の承認より先に実装された。→ 記録済み。

本specを完了とした。
