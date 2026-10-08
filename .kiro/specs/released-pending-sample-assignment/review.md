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
