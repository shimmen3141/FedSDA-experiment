# Implementation Plan

- [ ] 1. 構成とtestを足す
  - 要求2.1・2.2のtestを先に書き、REDを確かめてから、構成を足し、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、基準版1〜16のgoldenの照合（Windows）が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3_
- [ ] 2. 文書と検証の記録
  - 要求3.1の文書。実装レビュー（独立CLI）と、関係するtestの結果をImplementation Notesに記録する。
  - _Requirements: 3.1_
  - _Depends:_ 1
- [ ] 3. 本規模の追加の実行と集計
  - commit・pushの後、主担当が内部確認をしてコマンドを渡す（6データは追加分だけ、新しい4データは構成要素の比較の全構成と一緒。要求3.2）。結果が届いたら、同じstudyの`fedsda`と対で、精度（平均±SD、勝ち/負け）、通信、MAC、警報の回数、新しいモデルの選択の回数、MNIST・fmnistの概念ごとの精度（概念0以外が88%未満のrun、NEW-006の指標）を集計し、NEW-006と実験の台帳に記録する。
  - _Requirements: 3.2_
  - _Depends:_ 2

## Implementation Notes
- Task 1・2（実装: GPT-6.1-sol）: 構成`fedsda-without-warm-started-learner`（`experiment_configurations.py`）、新しいtest`test_fedsda_without_warm_started_learner_configuration.py`（設定の差と登録、固定条件の全体run、学習者1つでln 2を足さない部品のtest。実装前2 failed・1 passed、実装後3 passed）、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界への登録、既存のtest 2ファイルの構成の一覧の追従、文書（`experiment-sweep.md`、`model-reuse.md`、NEW-006）。固定条件（`fedsda-sine2-adoption-without-validation`のケースの条件）で、新しいモデルが17回選ばれ、`warm_started_learner`は無い（代わりの条件は使っていない）。既存の19構成の完全な設定と識別子は変更前と同じ（Solの確認）。
- 主担当の確認: 新しいtest、構成・掃引・識別子・依存境界・設定の検査のtest、基準版1〜16のgoldenの照合（9ファイル、Windows、12並列）が4,091 passed / 8 skipped / 失敗0（815秒、JUnit `../../venv/refactoring-tests/without-warm-targeted.xml`）。Ruff成功。Pyright（設定の範囲）はSolの実行で0 errors。全pytestは設計どおり行わない（変更は構成の追加だけ）。
