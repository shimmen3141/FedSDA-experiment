# 独立レビューと採否

要求revision1の主担当確認: 数値ID、EARS、主要正常/拒否/借用/境界/接続を確認した。要求の正本を保存し、独立Luna承認待ち。後続の設計・命名・tasks・実装・feature GOは承認を得てから進む。

実GPT-6 Luna /root/luna_session_spec の要求r1はNEEDS_FIXES。事前検査の範囲と途中失敗の原子性の限定を採用してr2へ修正した。借用の責任とbinding/可変部品を要求前提に明記。必須matrixと具体的preflight項目は設計で固定する指摘も採用予定。

同担当の要求r2 APPROVED。続いて設計r1・命名r1ともAPPROVED。検査条件/有限matrix/順序/borrowと所有/依存symbolsを照合した。非阻害のexact Tensorによる入力制限への注意は、本factoryの契約として明示する方針を維持する（旧全入力の互換は要件にしない）。設定・モデル・区間の借用を曖昧にしないことを優先し、aliasや暗黙変換を追加しない。

## Task graph sanity

別fresh実GPT-6 Luna /root/luna_session_graph の3task案はNEEDS_FIXES。拒否と観測、ASTと全回帰を分ける指摘を採用し5tasksへ修正。Task1は既存public部品4つの接続と共有oracleの正常matrixであり、正常比較をさらに分割する必要はないと判断した。

二回目はgraph自体が妥当との確認と、逐次Depends省略・Boundary名称の表記修正のみのNEEDS_FIXES。graphを再設計せずその機械的修正を採用し、同担当が実graphの阻害点なしをPASSと最終確認した。第三の設計草案cycleは行っていない。正式tasks r1を保存。設計r2では検証groupの旧Task番号を除去し、5taskとの対応を明示した（契約変更なし）。

正式レビューで設計r2 APPROVED、tasks r1はDepends明記を求めNEEDS_FIXES。主担当はこの指摘を不採用とした。task-generation規則は逐次順で依存を表しDo not over-annotateとし、前のgraph担当がDepends省略を求めた事実にも反するため。草案を再変更せずこの根拠を示して同担当へ再判定を依頼した。

同担当が規則と実graphを読み直し、隣接specの冗長表記を規範と誤認したと撤回、tasks r1 APPROVED。追加修正なし。要求r2/設計r2/命名r1/tasks r1の承認hashをspec.jsonへ保存してTask1へ進む。

命名r2のTask1追加3名は同実GPT-6 Luna担当APPROVED、指摘なし。source/test作成前に承認hashを保存して担当へ開始を通知した。

## Task1

新source作成前RED: ModuleNotFoundError/collection1error/1.92s/exit1。正常36＋最終6batch32＋履歴8＋未採番/空保留/負ID4の対象54passed。実旧_begin_forward_validationの学習を有効にし、全候補parameter/grad/optimizer型defaultsstate・参照値/順序・履歴平均・実epoch引数合算/学習計数・torch/Python/NumPyRNG・borrow/保有/統計/帰属不変を照合した。主担当も54passed/4.34s/exit0を確認。

別fresh実GPT-6 Luna /root/luna_session_task1 APPROVED、独立54passed/4.10s、対象Ruff check/formatと承認hashを確認、指摘なし。REDの独立再実行はしていない。拒否/観測接続/AST/fresh/全回帰は後続。

命名r3のTask2局所名・一時変異script名は実装前に実GPT-6 Luna /root/luna_session_spec APPROVED。finallyで元byteへ復元する注意は採用する。

Task2担当がexact型拒否用の動的fixture型名4つを未承認の文字列で先に作った。source変更はなく、新しいclass宣言がないことを理由に承認を省略した点は手順逸脱として記録する。命名r4へ追記し後続の完了レビュー前に独立確認する。実装前承認済みとは扱わない。

実GPT-6 Luna /root/luna_session_spec の命名r4はAPPROVED、実testの用法とexact→isinstanceの退行検出目的を確認し、改名不要。事後承認という扱いを維持して進める。

## Task2

test追加のみでsource補修なし。初回nested.shape比較の4失敗はtest準備の非対応であり、is_nested分岐で比較して修正した。productionのREDとは扱わない。復元後166passed/1nested warning/6.73s、主担当も166passed/5.92sとsource差分なし・復元hashを照合した。

4変異はstub72failed/order_swap45failed/omit_training30failed/live_reference54failed、全てexit1。Git管理外の証拠は../../venv/refactoring-tests/session_start_mutation_evidence.py、session-start-task2-mutation-evidence.jsonと4logs。各finally/終端で元byteを復元しSHA256 `ad4af70509aca22d063a239df6bee70c94c87b8a11a8f739b7fc100efc3b00fc`一致、source_bytes_restored=true。非公開統計をtest準備で破損させる3条件は保証対象外からpublic getの再検査を観測する補助testと区別した。

別fresh実GPT-6 Luna /root/luna_session_task2 APPROVED。独立166passed/6.05s、diff/契約/JSON/log/hashを確認。変異scriptは独立再実行せず、事後命名承認と保証外fixtureを区別。必須修正なし。観測/AST/fresh/全回帰は未実施。
