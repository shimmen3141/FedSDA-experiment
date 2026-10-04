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
