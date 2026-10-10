# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 検査と計測
- [x] 1.1 旧の計数を、外側からの数え直しと照合するtestを書く
  - goldenの3ケースで、旧の全体runを、hookつきで実行し、旧の計数の合計と照合する。
  - 完了: 検査のtestが成功する（旧実装は変えない）。
  - _Boundary: test_legacy_computation_count_audit_
  - _Requirements: 1.1, 1.2_

- [x] 1.2 モデルの計算を、計測の区間の間、外側から数える
  - 計数の照合（手計算）、対象外のmodule、区間の後と例外の後、入れ子、差、結果と乱数を変えないことのtestを先に書く。
  - 完了: 計測のtestと、依存境界のsuiteが成功する。
  - _Boundary: model_computation_measurement_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 6.2_

- [x] 1.3 clientが、検出器の計算の計数を保持する
  - ownerのtestと、clientの全状態の新旧照合への、実旧の検出器の計数との照合を先に書く。
  - 完了: ownerのtest、clientのtest、依存境界のsuiteが成功する。
  - _Depends: なし_
  - _Boundary: LossMonitoringComputationCountStore、FedsdaRunClient_
  - _Requirements: 3.1, 3.2, 6.2_

- [x] 2. 重複した計算の解消
- [x] 2.1 有界損失の評価へ、3つの入口を足す
  - 既存の関数との一致、共有部を使い回した損失の一致、検査と拒否のtestを先に書く。
  - 完了: 有界損失の評価のtestが成功する。
  - _Boundary: classifier_bounded_loss_evaluation_
  - _Requirements: 4.4_

- [x] 2.2 クロス評価・警報時の区間の準備・集約後の再較正の、重複した順伝播をなくす
  - 3つの対照のtestへ、順伝播の標本数が、実旧の同じ処理の計数と一致することの確認を、先に足す（失敗を確かめる）。
  - 完了: 3つのmoduleの既存のtest（実旧との対照、拒否）と、足した確認が成功する。
  - _Depends: 1.2, 2.1_
  - _Boundary: client_model_cross_evaluation、alarm_training_interval_preparation、post_aggregation_prediction_recalibration_
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 3. 計測つきの全体runと指標
- [ ] 3.1 計測つきの全体runを実行し、計算量の指標を導出して、goldenの7項目と照合する
  - 指標の導出のtestを、33指標のすべてを照合する形へ、先に直す。計測つきの全体runのtest。
  - 計測つきの全体run、指標の導出の2項目を実装する。共用scriptへ足す。
  - 完了: 指標の導出のtest（Windowsでは、goldenの33指標の照合を含む）、計測つきの全体runのtest、共用script、依存境界のsuiteが成功する。
  - _Depends: 1.2, 1.3, 2.2_
  - _Boundary: fedsda_measured_run_execution、fedsda_run_metric_derivation_
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2_

- [ ] 4. 検証
- [ ] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.1_
  - _Requirements: 4.4, 5.3, 6.1, 6.2_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
