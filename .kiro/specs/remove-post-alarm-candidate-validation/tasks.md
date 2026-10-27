# Implementation Plan

依存: task 1 → 2 → 3 → 4 → 5 → 6。実装はGPT-6.1-sol（task 1〜5）。版15のgoldenの生成、全pytest、WSL、独立レビューは主担当（task 6）。
各taskの完了条件に共通: 版1〜14の残すcaseの数値の照合（`tests/refactoring/test_important_behavior_regression_goldens.py`）が通る。

- [ ] 1. 残す比較と共有の部品を確定する
  - 候補の学習を新しいモジュールへ移し、呼び出し側と研究用の道具のpatchの先を直す。goldenの照合に「外すcase」「除く項目」の仕組みを足す（この段では、外すcaseは候補検証ありの方針のcase、除く項目は空）。
  - 外すcaseの一覧が、保存済みのgoldenの設定の記録の方針の値と一致するtest。
  - _Requirements: 2.2, 4.1, 4.2_

- [ ] 2. 構成と設定の入口を閉じる
  - 構成・方針の値・設定の項目・制約を消す。検証ありの設定を組み立てていたtest・helper・fixtureを、外すか検証なしへ直す（旧実装との照合の扱いは要求4.5）。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.5_

- [ ] 3. 実行と状態を消す
  - _Requirements: 2.1, 2.3_

- [ ] 4. 記録・指標・保存の形を消す
  - goldenの除く項目の一覧を確定する（一覧にない差は失敗）。
  - _Requirements: 3.1, 3.2, 3.3, 4.2, 5.1, 5.2_

- [ ] 5. 研究用の道具・文書・版15
  - 研究用の道具の候補検証のモード・入口を外し、残る専用の検証を通す。文書（要求6.2・6.3）。`BEHAVIOR_BASELINE_VERSION = 15`と版15のcaseの一覧、版15と版14の比較のtest（版15のファイルが無い間は失敗してよい。報告に書く）。
  - _Requirements: 4.3, 4.4, 6.1, 6.2, 6.3_

- [ ] 6. 確認（主担当）
  - 版15のgolden（Windows・Linux）を作り、4.4のtestを通す。全pytest、WSLのgolden、pyright、ruff。独立の実装レビュー1回。
