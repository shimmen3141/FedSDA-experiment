# Implementation Plan

依存: task 1 → 2 → 3 → 4 → 5 → 6。実装はGPT-6.1-sol（task 1〜5）。版15のgoldenの生成、全pytest、WSL、独立レビューは主担当（task 6）。

各taskの完了条件に共通:
- `tests/`の全体が通る（壊れたtestを次のtaskへ送らない。そのtaskの変更で成り立たなくなったtestは、そのtaskの中で要求4.5の規則に従って直すか消し、記録に書く）。
  sandboxの制約で実行できないtestは、名前と理由を記録に書き、主担当が通常の環境で確かめる。
- 版1〜14の残すcaseの数値の照合（`tests/refactoring/test_important_behavior_regression_goldens.py`）と、要求4.7の乱数の状態の照合が通る。
- そのtaskで消した名前について、要求5.3の参照の検査（`git grep`）で残りが無い。

- [ ] 1. 残す比較と共有の部品を確定する（まだ何も消さない）
  - 候補の学習を新しいモジュールへ移し、呼び出し側と、調査の報告の5の表で挙げた研究用の道具のpatchの先（`tools/research_s3_contribution/`、`tools/research_output_layer_learner/`ほか）を直す。
  - goldenの照合に「外すcase」「除く項目」の仕組みを足す（外すcaseは要求4.6、除く項目はこの段では空）。外すcaseの一覧のtest（要求4.6）。
  - 要求4.7の乱数の状態のfixtureを、このtaskの開始時のHEADのコードで作り、照合のtestを足す。
  - _Requirements: 2.2, 4.1, 4.2, 4.6, 4.7_

- [ ] 2. 構成と設定の入口を閉じる
  - 構成・方針の値・設定の項目・制約を消す。検証ありの設定を組み立てていたtest・helper・fixtureを、要求4.5の規則で扱う。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.5_

- [ ] 3. 実行と状態を消す
  - _Requirements: 2.1, 2.3, 4.5_

- [ ] 4. 記録・指標・保存の形を消す
  - goldenの除く項目の一覧を確定する（一覧にない差は失敗）。
  - _Requirements: 3.1, 3.2, 3.3, 4.2, 5.1, 5.2_

- [ ] 5. 研究用の道具・文書・版15
  - 研究用の道具の候補検証のモード・入口を外し、調査の報告の5の表の道具ごとに、残るモードの専用の検証（`checks*.py`）が通ることを確かめて記録する。
  - 文書（要求6.2・6.3）。`BEHAVIOR_BASELINE_VERSION = 15`と版15のcaseの一覧（要求4.3）、版15と版14の比較のtest（版15のファイルが無い間は失敗してよい。記録に書く）。
  - 要求5.3の参照の検査を、消した全部の名前で行う。
  - _Requirements: 4.3, 4.4, 5.3, 6.1, 6.2, 6.3_

- [ ] 6. 確認（主担当）
  - 版15のgolden（Windows・Linux）を作り、4.4のtestを通す。全pytest、WSLのgolden、pyright、ruff。独立の実装レビュー1回。
