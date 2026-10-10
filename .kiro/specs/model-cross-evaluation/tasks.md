# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 記録と設定
- [x] 1.1 診断の記録のownerと、束の標本の上限を足す
  - 診断の記録（記録の型と検査、追加、写し）と、スカラーの設定のfieldを足す。束を作る箇所（既存のtestと共用script）を直す。新しいmoduleの依存の許可集合を登録する。
  - 完了: 記録のownerの単独のtest（順、検査、不正で不変）、束のtest、依存境界のsuiteが成功する。
  - _Boundary: CrossEvaluationRecordStore、ClientCrossEvaluationRecord、FedsdaRunClientScalarSettings_
  - _Requirements: 3.3, 3.4, 6.2_

- [ ] 2. クロス評価
- [x] 2.1 clientの評価を、実旧のclientの評価との対照つきで実装する
  - 旧の設定の差し替えへ、評価標本の追加の件数と、評価の標本の上限を足す。同期したラウンドの途中の状態で、実旧のclientの`evaluate_model`・`evaluate_model_diagnostics`と照合する対照testを先に書く。
  - 評価の関数、結果の型、clientの2つの操作を実装する。依存の許可集合を登録する。
  - 完了: clientの対照が、2値・多クラスで、モデルの全部の組について、結果と乱数の一致を示す。拒否のtest、評価が変えない状態のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1_
  - _Boundary: evaluate_candidate_model_on_target_model_samples、ClientModelCrossEvaluation、ModelPairCorrectnessCounts、FedsdaRunClientの2つの操作_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4.3, 5.2, 6.2_
- [ ] 2.2 サーバのクロス評価を、実旧のサーバのクロス評価との対照つきで実装する
  - ラウンドの対照（登録→集約→クロス評価→配布→再較正）を先に書く。クロス評価の関数と結果の型を実装する。依存の許可集合を登録する。
  - 完了: ラウンドの対照が、表・対の集計・3つの診断の記録・通信量・乱数・clientの状態の一致を示し、要求4.2の経路を通ったことを確かめるtestが成功する。拒否と途中の失敗のtest、依存境界のsuiteが成功する。
  - _Depends: 2.1_
  - _Boundary: cross_evaluate_global_models、ModelCrossEvaluation、CrossEvaluationLossSums、ModelPairUniqueCorrectnessCounts_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 5.1, 5.3, 5.4, 6.2_

- [ ] 3. 新実装だけの確認
- [ ] 3.1 共用のfresh process scriptで、新規モデルが登録されたラウンドにクロス評価を行う
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、集約の後にクロス評価を行い、表の形と、通信量と記録が足されたことを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Depends: 2.2_
  - _Boundary: 共用のfresh process script_
  - _Requirements: 6.1_

- [ ] 4. 検証
- [ ] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.1_
  - _Requirements: 4.1, 6.1, 6.2_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
