# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [ ] 1. 計算とowner
- [ ] 1.1 判定の計算と、判定の基準の型を、実旧の関数との対照つきで実装する
  - 実旧の、Wilsonの下限・クラス別の同時信頼下限・average linkageへ、同じ入力を与えて照合するtestを先に書く。
  - 完了: 3つの関数の対照（境界の入力を含む）と、拒否、判定の基準の型のtest、依存境界のsuiteが成功する。
  - _Boundary: model_clustering_calculations_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 6.3, 8.2_
- [ ] 1.2 診断の記録のowner、グローバルモデルを外す操作、clientの割当概念の計数を足す
  - 完了: 記録のownerの単独のtest（順、検査、不正で不変）、外す操作のtest（パラメータと統計が外れ、次の正式IDと来歴は変わらない。持たないIDは拒否）、clientの操作のtest、依存境界のsuiteが成功する。
  - _Boundary: ModelClusteringRecordStore、GlobalModelRepository、FedsdaRunClient_
  - _Requirements: 3.1, 3.2, 4.4, 8.2_

- [ ] 2. クラスタリングと統合、同期
- [ ] 2.1 クラスタリングと統合、サーバの1ラウンドの同期を、実旧のサーバのラウンドとの対照つきで実装する
  - ラウンドの対照（旧は`run_round(round_index, clustering_enabled=…)`、新は同期の関数）を先に書く。クラスタリングの診断の記録を、実旧の来歴の観測と照合するhelperを書く。
  - クラスタリングと統合の関数、同期の関数、結果の型を実装する。依存の許可集合を登録する。
  - 完了: ラウンドの対照が、2値・多クラスで、ラウンドごとに、全状態・診断の記録・乱数の一致を示し、要求6.2の経路を通ったことを確かめるtestが成功する。拒否と途中の失敗のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1, 1.2_
  - _Boundary: cluster_and_consolidate_global_models、ModelConsolidation、synchronize_models_in_server_round、ServerRoundSynchronization_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 6.1, 6.2, 7.1, 7.2, 7.3, 8.2_

- [ ] 3. 新実装だけの確認
- [ ] 3.1 共用のfresh process scriptのサーバの代役を、同期の関数の呼出しにする
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、ラウンドごとに同期の関数を呼び、クラスタリングが1回以上行われ、記録が足されたことを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Depends: 2.1_
  - _Boundary: 共用のfresh process script_
  - _Requirements: 8.1_

- [ ] 4. 検証
- [ ] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.1_
  - _Requirements: 6.1, 8.1, 8.2_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
