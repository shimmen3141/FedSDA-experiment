# Implementation Plan

- [x] 1. 設定と写しのowner
  - [x] 1.1 設定の項目と許す値。作る箇所の全部へ`current_local_parameters`を明示する。goldenのhelperの「後に足された項目」へ、版1〜5の値を足す（設定の保存の形が変わるので、同じ変更で）。FedSDAのrunの識別子を期待するtestを合わせる。設定の検査のtest（`tests/refactoring/test_run_settings_validation.py`の、文字列の項目ごとに許す値を並べた表）へ、新しい項目を足す。testを先に書く。
    - _Requirements: 1.1, 3.1, 3.2_
  - [x] 1.2 写しのowner（取る・選ぶ）。testを先に書く（設計のValidation Designの2）。依存境界の許可へ足す。
    - _Requirements: 1.2, 1.3_

- [x] 2. 予測とclientへの接続、構成
  - [x] 2.1 予測の関数（結合と、現在のモデルだけの両方）と標本の処理の省略可能な引数（ownerの検査の表へは入れず、Noneか正しい型かを別に確かめる）。testを先に書く（設計のValidation Designの5・6と、予測の関数の単体の確認）。
    - _Requirements: 1.3, 1.4_
  - [x] 2.2 clientの属性としての写しのowner（ownerの束`FedsdaRunClientOwners`へは入れない）と、配布の後の写し、標本の処理への受渡し。testを先に書く（設計のValidation Designの3・4: 3つのモデルの構造での診断との一致、検出力）。
    - _Requirements: 1.2, 1.5_
  - [x] 2.3 構成名と組み立て、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`への登録、構成の一覧を使うtest。
    - _Requirements: 2.1, 2.2_

- [x] 3. 文書
  - 実験の入口の文書（構成の表）、研究バックログの8、再開案内、完了specの一覧。
  - _Requirements: 4.1_

- [x] 4. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest（goldenの版1〜5を含む）・Ruff・Pyright・WSLの対象test。手元で、診断と同じ条件のrun（sea2・circle2・sine2、seed 0、間隔50、標本5000）が、診断の記録（`fedsda_distributed_prediction_probe.json`）の、`dataset`が同じで`seed`が0、`variant`が`distributed_prediction`の`accuracy`と、新しいrunの`prediction_accuracy`が厳密に（等号で）一致することを確かめる（診断は標本5000——scriptの`PROBE_N`の既定——で走らせた）（`default`の`accuracy`は、今の`fedsda`の値と一致する想定）（完了の条件に含める。一致しなければ、原因を調べてから完了とする）。
  - _Requirements: 全部_

## Implementation Notes

- 手順の逸脱: 写しのowner（task 1.2）と、予測の関数への接続（task 2.1）は、testより先に実装した。代わりに、主な処理を1つずつ壊して、新しいtestが落ちることを確かめた（分類器ごとに別々に写す、同じオブジェクトかを見ない、共有特徴をID昇順で最初のモデルで計算する、配布の後に写さない、現在のモデルだけの予測で写しを使わない。5つとも1件以上失敗。最後の1つは、最初は検出できなかったので、`fedsda-single-model-prediction`の構造でも診断と照合するtestを足してから確かめた。どれも元に戻した）。
- 診断との一致: 小さい条件（sine2、client 3、標本400、間隔50、seed 1。同期の後に作ったモデルを含む予測が起きる）で、4つの構成（提案構成、共有部を持たない構造、adapterを持たない構造、現在のモデルだけで予測する構成）の全部で、予測の記録の全部と精度が、testの中で書き直した診断の差し替えと一致した。本規模（sea2・circle2・sine2、seed 0、client 10、標本5000、間隔50）でも、新しい構成の精度が診断の記録の`distributed_prediction`と、既定の精度が`default`と、6つとも厳密に一致した（`../../venv/refactoring-tests/synchronized-prediction-full-scale-check.txt`）。
- 予測の関数の型の検査: 写しのownerの型を、ほかの入力の検査より前に確かめる順にした（標本の処理と揃えた）。
- 依存境界: 新しいmoduleの許可（`copy`・`copy.deepcopy`・`ModelArchitectureClassifier`・`HeldModelTrainingStateRegistry`とそれぞれのmodule）と、予測・標本の処理・clientの`SynchronizedPredictionModelSnapshot`（予測は`ModelArchitectureClassifier`も）を足した。
- 実装レビュー（独立Haiku 5.5 medium、1回）: APPROVED（Minor 3）。1: 要求1.4のtestが設計の記述（監視の関数を包む）と違う→設計の記述を、実装した検査（標本の処理の中の損失の評価へ渡る分類器が写しでないこと）に合わせて直した。2: 研究バックログの言い回し→直した。3: tasks.mdとspec.jsonの記録→このnotesとspec.json。
- 検証: 全pytest 14347 passed / 8 skipped（失敗0。JUnit `../../venv/refactoring-tests/synchronized-prediction-full.xml`）、Ruff成功、Pyright 0件。WSLで、新しいtest・goldenの版1〜5・前のspecのtestの231件が通った。レビューの後の修正は文書だけ。
