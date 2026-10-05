# レビューと採否

## Requirements 第一回

GPT-6 Luna判定FAIL。特徴の適合次元/ラベル数/クラス範囲、非対応PCGradの事前拒否の明記を採用した。
無条件shallをEARS違反とする指摘は不採用。ローカルears-format.mdの「5. Ubiquitous Requirements」がThe [system] shall形式を正式に許可しているため。検証/状態契約は無条件形式が適切。
同Lunaの第二回判定PASS。次元/標本数/クラス範囲/非対応方式とUbiquitous EARSの整合を再確認。主担当採用し、修正済みrequirementsのhashを承認する。

## Design / naming revision 1

同GPT-6 Luna判定PASS。検査/更新順・空列/空共有/凍結・soft target・optimizer参照順・実旧照合可能性、名前と借用状態の区別を確認。特段の修正指摘なし。主担当採用し、最新版hash/revisionを承認する。

## Task graph 保存前

主担当gate: 全17要件のnumeric ID・二production/実旧oracle/拒否/AST/証拠を順次三taskへ対応付け、観測可能な完了条件を確認。
同GPT-6 Luna第一回FAIL: 要件2.4の非空共有に対するoptimizer欠落を有効/無効両方で検証する記載を採用。空共有へのoptimizer誤指定も追加。
修正draftの独立第二回PASS。保存前sanityの方式はindependent_reused_thread。fresh thread作成ではなく実Luna既存threadを再利用。主担当が採用してtasks保存/hash承認する。

## Naming revision 2

Task1の二test関数名を追加。同GPT-6 Luna判定PASS。実旧比較と操作順観測の役割に一致し、production API変更なし。主担当採用してrevision/hash承認した。

## Task 1

worker実RED: missing moduleの1 collection error/exit1/18.91s。GREEN26 passed/exit0。
同GPT-6 Lunaのkiro-review判定APPROVED。独立対象pytest26 passed、境界/import/secret/TODO/diffcheckを確認、修正指摘なし。
主担当fresh対象pytestも26 passed/exit0。24数値条件で実旧3step全Parameter/grad/optimizerstate/損失が完全一致し、2条件で共有forward一回とzero/backward/step順を観測。Task2/3は未完了のまま。

## Naming revision 3

Task2のtest名・snapshot helperと局所変数、特徴/ラベル共用validator引数training_tensorを追加。同GPT-6 Luna判定PASS、名前から検査対象の役割を区別できると確認。主担当採用してrevision/hash承認。公開入力record名は変えない。

## Naming revision 4

主担当がFieldをparameterと呼ぶtest-localの混同を見つけ、training_batch_fieldを提案。同GPT-6 Luna判定PASS。NN Parameterと区別する改善を採用し、承認後にtest-localだけ機械的改名した。

## Task 2

worker対象77 passed/exit0。初回76 passed/1 failedは不正groupのstate_dictをtest snapshotへ使った問題で、実state/id列を直接独立保存して修正。production欠落ではなくfakeREDは要求しない。
同GPT-6 Lunaのkiro-review判定APPROVED。独立対象77 passed、拒否不変性/空列/空共有/凍結/soft target/複数group/環境/境界を確認、修正指摘なし。
主担当fresh対象も77 passed/exit0。41事前拒否条件はzero/step未呼出とwarm state/sentinelgrad/値保持を確認。空共有は独立loss/stepと全値/grad/state一致、既存LEGACY-010の旧生成失敗とは区別した。
4.3のrollback非保証は保証範囲の明示で、部分変更を必須にする要件ではない。snapshot rollbackを実装していないこととdesignを確認し、特定の部分変更を要求する追加testは不要と判断した。
