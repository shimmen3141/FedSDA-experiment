# 独立レビューと採否

## 要求

GPT-6 Luna（/root/luna_single_run_3_1_review）の要求revision1判定NEEDS_FIXES。「最良」を観測最小値と読む曖昧さの指摘を採用し、閾値を厳密に超える改善として最後に採用した値のみ復元することをrevision2へ明記した。閾値ちょうどは非改善。0epochの乱数差は設計/検証へ明記する提案を採用する。revision2を再レビューする。

要求revision2はAPPROVED、命名revision1もAPPROVED（同実GPT-6 Lunaセッション）。roundとepoch0の差を設計/testへ維持するコメントを採用した。design revision1へ反映し設計レビューを依頼した。

設計revision1はAPPROVED。命名表のfloat表記が設計のint/float受理と異なる指摘を採用し、命名revision2で注釈と実行時受理型を区別して統一した。名前や役割の変更はない。

命名revision2もAPPROVED。Task1のtest準備名をrevision3へ追記してレビューする。

## Task graph sanity

別の実GPT-6 Lunaセッション`/root/luna_epoch_task_graph`によるindependent sanity。最初の4task案はNEEDS_FIXES。epoch実装と大量条件を一taskに詰める指摘を採用し、固定反復/事前検査をTask2、公開入口/早期停止をTask3へ分離した。ASTと全回帰は約2分の既存全suite・品質commandを実行する明示的統合task5とした。

一回修正後もTask3分割のNEEDS_FIXES。主担当は入口のみ先行するとearlyの未実装分岐が残る点と、preflight/固定/拒否matrixはTask2済み、Task3は約60行の同一アルゴリズムである点から追加分割を不採用とした。草案を再修正せずこの根拠で再判定を求め、同レビュー担当が実在する分割blockerなしと確認してPASS。最終草案5tasksを保存する。ユーザーの有用指摘のみ採用する方針を適用し、主担当が判定を書換えていない。

正式tasks revision1と命名revision3はGPT-6 Luna（/root/luna_single_run_3_1_review）APPROVED。全10条件を5tasksへ対応づけた。実装開始前に4段階の承認hashをspec.jsonへ保存した。

## Task1

source作成前RED: 設定module未存在のModuleNotFoundError、1error/exit1。GREEN: 対象26passed/0.07s、Ruff check成功。実装前に別の未承認test関数名settings_structureを一時作ったが、source作成前に削除して承認済settings_contractへ統合した。未承認のtest先取りの事実をレビュー担当へ提示した。最終src/testの正式名は承認済みのもの。

別の実GPT-6 Lunaセッション`/root/luna_epoch_task1`でAPPROVED。独立実行26passed/exit0、Ruffと境界/RED/値域/必須/不変recordを確認、指摘なし。全pytestはTask5で主担当が実行予定。レビュー時のpytest cache権限warningは対象コード由来ではない。Task2/3追加命名revision4は実装前レビュー待ち。

命名revision4も実GPT-6 Luna（/root/luna_single_run_3_1_review）APPROVED、指摘なし。Task2開始前に承認hashを記録した。

命名revision5でoptimizer group/parameter位置/学習開始RNGの3名を区別し、実GPT-6 Luna APPROVED（同セッション、指摘なし）。承認後に仮の名前使い回しを除去して検証した。公開APIの変更なし。

## Task2

実装module作成前REDはModuleNotFoundError/exit1。固定反復の54条件（class2/4×3optimizer×epoch0/1/3×N/batch=7/3・11/5・4/8）で全parameter/grad/optimizer/RNG/counters/実batch順が実旧に一致。履歴ありpreflight拒否/受理43、record1、設定26を含む対象124passed。主担当も124passed/8.18s/exit0、Ruff対象と全src Pyright0errors/0warningsを確認。nested prototypeは不正tensorを作るtest準備の警告。主担当の対象実行はMPLCONFIGDIR指定漏れにより終了時matplotlib一時dirのACL cleanup例外が出たが、pytestのexit0と件数は正常。全回帰では共通手順どおり明示する。公開入口/earlyはTask3のため未実装。

別の実GPT-6 Lunaセッション`/root/luna_epoch_task2` APPROVED、独立対象124passed/exit0、Ruff・境界・承認hash確認。指摘なし。全回帰は未実施、レビュー担当はPyrightを独立再実行していない。

命名revision6は実GPT-6 Luna（/root/luna_single_run_3_1_review）APPROVED、指摘なし。loss記録wrapper2名とTask4接続test名は承認後に実装する。

## Task3

公開入口の追加前REDはImportError: cannot import name train_candidate_classifier_epochs、exit1。方式別216（2class×3optimizer×3方式×epoch0/1/4×N1/3/5/11）と最良snapshotのみ復元/実旧損失差の厳密閾値6条件、既存124を含む346passed。主担当も346passed/1nested warning/9.37s/exit0、全src Pyright0errors/0warningsを確認した。

差替えscriptはGit管理外`../../venv/refactoring-tests/candidate_epoch_training_red_evidence.py`、証拠はcandidate-epoch-training-red-evidence.json。stub/復元省略/optimizer reset/extra randperm/閾値<=の5条件を全てexit1で検出し、元byteへ復元した。元/復元/主担当が照合したruntime LF hashは`1d67010544b048c123b7151b854974a6fb0ac18c3bfd6c0e2af9b99c73760394`で一致。ここまで継続更新・AST・全回帰は未実施。

別の実GPT-6 Lunaセッション`/root/luna_epoch_task3` APPROVED。独立346passed/exit0、Ruff/実diff/境界/RED/5変異のJSONとhashを確認し指摘なし。差替えscriptは独立実行していない。レビュー担当側のmatplotlib一時dir cleanup例外とpytest exit0を区別する。主担当のMPLCONFIGDIR明示の対象再実行ではcleanup例外なし。全suiteは後続Task5。

## Task4

??revision7??GPT-6 Luna?/root/luna_single_run_3_1_review?APPROVED???????????RNG?????????test?????????????????????batch????parameter/grad/optimizer/RNG/?????????????6?????352passed?fresh CPU smoke????/????????module import??????

???GPT-6 Luna????? /root/luna_epoch_task4 APPROVED???352passed?smoke?Ruff?diff?????????30epoch?patience1/3??????1e-4/10??24??????????24passed/346deselected/3.19s?delta10?patience+1?????????????????????APPROVED????????24?????????????????????????production??????session/???run?????
