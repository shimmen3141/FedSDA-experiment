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
