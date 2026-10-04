# レビューと採否

## 要求作成前の主担当gate
数値ID9条件、EARS、正常三方式・同率/空・コピー/拒否/環境・上位接続と検証を確認してPASS。
初期値snapshotを返す一責務に限定し、モデル操作・学習は含めない。要求へ実装技術を埋め込んでいない。
以降の承認はユーザー委任のGPT-6 Lunaレビューと主担当による採否で行う。

## 要求Lunaレビュー1と採否
NEEDS_FIXES: 2.1の契約外が抽象的で、未知方式・欠落ID・無効損失・異なるparameter構造の拒否を一意に決められない。
主担当は採用。観測可能な拒否条件と、名前挿入順だけの差は受理することを2.1へ明記した。修正後に再レビューする。

## 要求Lunaレビュー2と採否
NEEDS_FIXES: 「不正な名前」の型と空文字の扱いが未定義。採用し、空でない文字列、値は数値・真偽配列（スカラーと空形状も可）と明記した。

## 要求Luna最終レビュー
PASS: 名前/値の型、0次元と空形状/空集合の区別、順序差の受理、三方式とコピーまでの範囲が一意に検証可能。
主担当は追加指摘なしのPASSを採用し、最新requirementsのhashを承認する。

## 設計保存前の主担当gate
9条件の対応、2productionファイルの責務・exact依存・前提環境・入力と例外・検証の実行性を確認してPASS。
方針の一般化/既存採用/簡素化をresearch.mdに記録し、状態を持たないsnapshot作成に統一した。

## 設計・命名Lunaレビュー1と採否
NEEDS_FIXES: 旧helperで先頭コピー可能なuint16/uint32/uint64が設計の整数dtype一覧から漏れていた。採用し一覧に追加した。
input_nameとvalidated_parameter_snapshots_by_model_idは適切との指摘を採用し追加表へ記録。parameter_namesのproduction意味も明示した。標準pytest fixture名の過剰列挙は不要とし増やさない。

## 設計・命名Luna最終レビュー
PASS: dtype/順序/全検査/copy/CPU dense/一責務とexact依存が整合し、追加名の役割も区別できる。
主担当はPASSを採用し、designとnaming revision1の最新hashを承認する。

## Taskgraph保存前レビュー
主担当gate: 9条件の割当、1→2→3のDepends、共有ファイルのため(P)なし、既存venv/validator前提、実測可能な完了状態を確認してPASS。
GPT-6 Luna（独立・既存thread再利用）はPASS。3taskが一責務と明示test-only/統合境界に沿い、隠れた前提・漏れ・過大taskなし。
主担当は指摘なしのPASSを採用してtasks.mdを保存し、内容hashを承認する。

## Task1
実装者READY_FOR_REVIEW。REDは設定module未実装のcollection 1error/3.56s/exit1、GREEN18 passed/3.20s/exit0。
実旧helperと三方式/先着同率/空/fallback/評価外/丸め/15dtype/Shared風key/逆key順/スカラー/空形状/noncontiguousを照合し、基本storage独立性も検証。
LunaはAPPROVED、対象18 passed/3.19s/exit0、placeholder/secretなし、指定3files境界内、生成/評価/学習の先取りなし。指摘なし。
主担当fresh検証18 passed/3.14s/exit0とsource直接読みでVERIFIED。高度拒否・grad/環境・候補評価接続とAST/fullは後続のままTask1を完了する。

## 命名revision2
Task2のtest-only検証名と同意味の既存名を追加。production名・役割は不変。
LunaはPASS、拒否/keyword/環境/評価mean接続/非有限平均の分離とkeyword入力dict/別返却の意味が明確との判断。主担当は指摘なしのPASSを採用し最新hash/revision2を承認する。
保存時に生じた末尾空行だけを削除してhashを再計算した。命名・役割・revisionは変えていない。

## Task2
実装者READY_FOR_REVIEW。test-only追加40件、RED非該当。58 passed/1 warning/4.02s/exit0。
後段未選択入力/forged設定/ID・loss・構造・有限性を拒否、aliasの解消と双方向copy、leaf/nonleaf grad detach、meta contextでCPU/dtype/RNG/grad/default保持、keyword/frozen/default無しを確認。
公開評価のreference平均だけを明示対応し、再利用不適合な最小lossモデルの初期値を選び旧helperへ照合。
有限float32max2件から旧inf/新ValueError/入力不変を同条件実測し、主担当がLEGACY009へ記録。旧未修正・通常/過去影響未確認と区別する。
Luna APPROVED、58 passed/3.91s、production変更なし、指摘なし。qint8 fixture deepcopyのTypedStorage非推奨警告1件はlibrary/test由来と確認。
主担当fresh58 passed/1 warning/4.09s/exit0とdiffcheck成功、src差分なしでVERIFIED。Task3 AST/fullは未完了のままTask2を完了する。

## Task3の実測証拠
AST禁止14/許可7を先行追加しRED8 failed/209 passed/0.55s/exit1。exact2module境界へ更新後、対象58+AST217=275 passed/1 warning/4.55s/exit0。
fresh CPU smokeはCANDIDATE_PARAMETER_INITIALIZATION_SMOKE_PASS/exit0、旧package importなし。
全testsは3182 passed/3 skipped/1 warning/226.33s/exit0。旧11/最終3goldenを含み、旧production/golden/比較testは748c3aaと差分なし。
全suite開始後production/testコードは変更なし。design/tasks/LEGACY009文書末尾の空行だけを除去しhashを更新した。契約・名前・役割を変えていない。
9条件と実測の対応、配置と未移植範囲はintegration-validation.mdに記録。Luna taskreview/最終gateは続いて行う。

## Task3レビューと完了
Luna APPROVED。対象/AST275 passed/exit0、全3182 passed/3 skipped/exit0、placeholder/secretなし、exact依存と指定境界内、RED証拠を確認。指摘なし、最終feature GOは別gateとした。
主担当は指摘なしの承認を採用。自ら取得した全回帰・smoke・275件の出力と、その後source/testコード変更なし、diffcheck/旧基準差分なしを照合してVERIFIED。Task3を完了する。
