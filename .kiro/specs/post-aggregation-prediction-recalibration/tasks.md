# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [ ] 1. 診断証拠の操作
- [ ] 1.1 単一の診断証拠へ、集約後の再生・再始動と、3つの計数を足す
  - 実旧のAdaHedgeと、通常の観測・概念操作による再始動・集約後の再生・集約後の再始動を混ぜた列で照合するtestを先に書く。照合のhelperへ、3つの計数を足す。
  - 完了: 足した操作の対照（空の列、途中でモデル集合が変わる列、モデルが1つの列を含む）と、不正な列で証拠と計数が変わらないことのtest、診断証拠の既存のtestが成功する。
  - _Boundary: AdaHedgeDiagnosticEvidence_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 4.3, 5.2_

- [ ] 2. 再較正
- [ ] 2.1 clientの再較正を、実旧のサーバのラウンドとの対照つきで実装する
  - 旧の設定の差し替えへ、再較正の方式を足す（既定は`none`のまま）。配布の対照のoracleで、旧は`run_round(round_index, clustering_enabled=False)`、新は登録→集約→配布→全clientの再較正、を行うラウンドの対照を先に書く。
  - 再較正の関数（検査、損失の列、globalの診断証拠の再生、真の概念別の診断証拠の再始動、Fixed-Shareの再生）と、clientの操作を実装する。依存の許可集合を登録する。
  - 完了: ラウンドの対照が、2値・多クラス、学習率の2つの設定が違う条件で、ラウンドごとに全状態と乱数の一致を示し、要求4.2の経路を通ったことを確かめるtestが成功する。clientの再較正だけの対照、拒否のtest、再較正が変えない状態のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1_
  - _Boundary: recalibrate_prediction_state_after_aggregation、PostAggregationPredictionRecalibration、FedsdaRunClient.recalibrate_prediction_state_after_aggregation_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 4.1, 4.2, 5.1, 5.3, 5.4, 6.2_

- [ ] 3. 新実装だけの確認
- [ ] 3.1 共用のfresh process scriptで、配布の後に全clientの再較正を行う
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、ラウンドごとに配布の後で全clientの再較正を行い、空でない列での再較正が1回以上あることを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Depends: 2.1_
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
