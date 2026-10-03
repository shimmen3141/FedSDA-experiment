# レビューと検証

## 要件: PASS・承認

- レビュー者: `/root/luna_single_run_3_1_review`（GPT-6 Luna、既存thread再利用）。
- 対象: brief、requirements、spec.json、README、旧SwitchingExpertRouter、既承認PredictionCombinationSettings。
- 6群22条件と境界を確認してPASS。
- 採用: 1.2で不正入力時に全計数も不変、2.3で学習割当変更だけでは重み・分散を消去しないと明記。要件数は変えず境界を精密化した。
- ユーザーの全段階Lunaレビュー委任に基づき、主担当が有用指摘の反映を確認して承認した。
- threadのclose APIがなく、既存threadを再利用した。新規独立threadとは扱わない。

## 設計・命名revision 1: PASS・承認

- 同じGPT-6 Lunaが設計・命名・調査、要件全22条件、旧演算、既存設定と境界checkerを直接レビューした。
- 値域・検証用fsum・演算本体sum・時間尺度の先行検査・空列・計数・責務が一貫しているとしてPASS。
- 採用: 状態表は読取property名、backing fieldは先頭_付きprivate属性と明記。API名は変更しない。
- 主担当が要件網羅と入力・状態・依存境界を再確認し、有用指摘反映後の設計・命名を承認した。

## Task graph: PASS・承認

- 同じGPT-6 Lunaが未保存draftと要件・設計・task規則を直接レビューした。共有controllerなので順次実行、task 4で依存検査へ明示統合する順序を確認してPASS。
- 採用: task 4にroadmap完成範囲の更新と既存・最終構成の回帰テスト名を明記。task 1～3の段階検証とtask 4の全src検証を明記した。
- 主担当の網羅確認: 全22条件、4実行タスク、既存Python/pytest再利用、隠れた環境前提なし。実装前の命名revision 1・設計hash一致を確認する。

## Task 1: APPROVED / VERIFIED

- TDD: 実装前に対象テストがModuleNotFoundError・collection error 1件、exit 1。実装後、新29ケースと既存設定1259ケースが通った。
- Lunaが実ファイルと要件・設計をレビューし、同じ対象コマンドで1288 passed / 1.44s / exit 0。指摘なし、APPROVED。
- 主担当が最終ファイルを読み、`../../venv/Scripts/python.exe -m pytest tests/refactoring/test_fixed_share_prediction_weights.py tests/refactoring/test_run_settings_validation.py -q -p no:cacheprovider`を再実行: 1288 passed / 1.25s / exit 0。
- 命名revision 1、所有状態、負ID・空/重複/列挙例外・copy・読取property・条件再検証を確認してVERIFIED。更新・再生はtask 2～3、全src依存統合はtask 4のまま。

## Task 2: APPROVED / VERIFIED

- TDD RED: 未実装update/selectのAttributeErrorで29 failed / 1287 passed。GREEN: 1316 passed / 3.21s。
- Luna: 実diffと旧updateの演算順を直接照合、共有TMP/TEMP/MPLCONFIGDIRを使った対象・設定テスト1316 passed / 3.22s / exit 0、指摘なしAPPROVED。
- 主担当: 同じ最終対象コマンドを再実行、1316 passed / 3.08s / exit 0。時間尺度2/30/10000の各操作後に全状態完全一致、不正入力時の非ゼロ計数保持を確認してVERIFIED。
- RED時のmatplotlib一時ディレクトリ終了処理のsandbox権限エラーは、既存共有TMP/MPLCONFIGDIR指定で解消。goldenや依存は変更していない。

## Task 3: APPROVED / VERIFIED

- 主担当が既承認task 3をmanual TDDで実装。RED: 未実装replay/resetで17 failed / 1316 passed / exit 1。GREEN: 1333 passed。keywordと列挙例外の補強後1335 passed / 3.54s。
- Lunaが実diff、旧再生・reset、全行検証と計数を直接レビュー。対象・設定テスト1335 passed / 3.20s / exit 0、指摘なしAPPROVED。
- 主担当が同じ最終対象コマンドを再実行し、1335 passed / exit 0を確認してVERIFIED。
- 全列先行検証・途中集合差・空列・集約回数/標本数・明示reset・不正後段/列挙途中例外を確認。依存境界と全回帰はtask 4。
