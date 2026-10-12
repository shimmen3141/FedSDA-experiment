# Implementation Plan

- [x] 1. 完全な条件から計測つきrunと全指標を返す
  - 実行・参加者・指標条件を束ね、部分型の再検証とバッチ/集約間隔の対応をモデル生成前に確認する。
  - 既存の計測入口と指標導出に接続し、実旧の基本指標/状態と呼出し側RNG不変を確認する。
  - 不正な条件で実行が始まらず、有効な条件では完全な結果が返るtestが成功する。
  - _Boundary: 完全設定と実行_
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. runの記録を時刻とclientに対応した表へ変換する
  - 標本・バッチ・概念変更・切替・適応と、モデル登録/クラスタ/クロス評価/終了時学習量の表を接続する。
  - 通常時刻のモデル数と計数、準備/終端の計数を別に扱う。空列と欠測を数値/Unicode配列として保持する。
  - 実旧/既存ownerとの対応、全列のdtype/形/行順、状態/RNG不変を確かめるtestが成功する。
  - _Boundary: 記録表_
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3. 新形式の保存・読込みを接続し単一run全体を検証する
  - 完全条件/全指標/環境をJSON、記録表をpickle不要のNPZへ保存する。
  - 不正な名前/既存成果物を書込み前に拒否し、保存段ごとの失敗で作成物を掃除する。
  - loaderの版/手法/単純なNPZ名の検査とroundtrip、fresh新経路/依存境界を接続する。
  - 一つの公開入口から実行・保存・読込みでき、既存回帰/品質検査が成功する。
  - _Boundary: 保存・読込みと単一run統合_
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3_

## Implementation Notes

- 要求/設計/taskのinline gate: 数値ID・全要求対応・責務分離・既存前提・観測可能な完了条件を確認。task graphは1→2→3、通常taskのローカルレビューはoff、仕様書群と全task後の独立CLIレビューを各1回行う。
- Luna medium仕様レビューPASS（UTF-8再確認）。最初のsessionは日本語を読めておらず承認に使わない。Minor1採用: loaderは版/手法/表名を検証し、条件/指標/環境の型復元・完全性検証はしない（既存FedSDAと同じ辞書読込み境界）。再レビュー不要。任意の全列契約testはtask 2で実施。bootstrap34 passed/7.23s。最初の-kは34 deselected/exit1で証拠に数えない。

- Task 1: stubで13 RED→13 passed/5.13s。実旧入口とsine2/sea4/mnist2/同期なしの値・状態・RNGを確認。ローカルreview off、kiro-verify-completionのtask証拠確認済み。

- Task 2: 空表stubで3 RED→全17 passed/5.58s。初回NameErrorと編集scriptのsubstringエラー（書込み前）はREDに数えない。unknown概念の注入は既存全体run契約に反するため正常testから除き、一時負IDと診断None/空列を確認。

- Task 3: save/load stubで35 RED→52 passed/6.53s。対象＋依存3143 passed/13.37s。fresh直接exit0/16 flows（FedDrift実行→保存→読込みを追加）、Ruff/format333、names4source未登録なし、Pyright0。sandbox内のPyright182 missing importsは権限によるため固定venv参照の権限付き実行で確認。全pytestは独立レビュー後。

- Linux対象＋既存4golden: 56 passed/16.32s、exit0。WSLのsandbox内アクセス拒否後、通常権限付き実行で確認。旧goldenの値/ファイルは変更なし。新成果物の全指標は共通形式、CLI/presetは後続。

- 独立Haiku実装レビュー1: session 3bc24570-d18d-4991-bf73-47032378e6a5、modelUsage claude-haiku-5-5/is_error=false、medium指定、FAIL（Major1、Blockerなし）。独立3236 passed/18.82s、fresh直接exit0/16flows、Ruff/format333、names/progress成功、全pytest/Linux/Pyright/変異未実施。対象Git状態変更なし。
- Major1採用: 概念変更の2列の値を実旧extract_true_drift_events（bootstrap {0:[2],1:[1]} 成功）へ照合し、非空条件と未処理末尾だけに変化がある条件を追加。Minor2は設計/既存共通指標と同じ全生成列という境界を要求へ明記（意味の変更なし、ユーザーへの再確認不要）。Minor3は再導出を主要対応の検査として保持し、要約との不一致を保存前に拒否する方針で補う。任意4は旧/FedSDAと同じファイル名境界のため不採用、OS固有の禁止文字はOS例外となり作成物を掃除する。任意5は規約の命名対象外のため不採用。Major修正のみ再確認し、Minorの別レビューは置かない。

- Major修正の概念列4ケースpassed/4.12s。主担当の追加点: 表builderのexact完全設定拒否と要約不一致拒否を2 RED（AttributeError/未拒否）→実装。再導出は既存の主要対応の検査として意図的に保持するため、Minor3の再導出削除案は不採用、懸念の要約/表不一致は保存前拒否で対応。新規testは合計55件。

- 独立Haiku修正再確認: session db1e7985-5196-4ff3-89c2-721f77feefbb、実モデルclaude-haiku-5-5/is_error=false、medium指定、PASS。Major1解消、必須指摘なし。独立3146 passed/11.78sとRuff/format333、Git状態不変。fresh/names/全pytest/Linux/Pyright/変異は再実行していない。主担当修正後Linux59 passed/17.89s、Pyright0/pip check成功。

- 最終品質: source/testを固定後、Ruff check/format333、Pyright0、pip check成功、diff-check/固定旧748c3aa差分なし。修正後fresh直接exit0/16flows。変異testは既定に従い未実施。全pytestとsource identityは下記で記録する。

- ???pytest: 13599 passed / 8 skipped / 9 warnings?674.78s?exit0?JUnit13607??failure/error?0??test_regression?test_proposed_regression????source/test??????????????Linux59??fresh16flows???????
