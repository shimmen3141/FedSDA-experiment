# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 既存のownerと束への追加
- [x] 1.1 ownerの操作を足す
  - 保有モデルの全置換え、損失統計の全置換え、グローバルの損失統計の一覧、サーバによる統合の適応記録（結果種別と位置の列）、共有部のoptimizerの状態の保持者を足す。新しいmoduleの依存の許可集合を登録する。
  - 完了: 足した操作の単独のtest（順、検査、不正で不変）と、変更したmoduleの既存のtest、依存境界のsuiteが成功する。
  - _Boundary: HeldModelTrainingStateRegistry、ModelAndClassLossStatisticsStore、GlobalModelRepository、AdaptationRecordStore、SharedParameterOptimizerStateHolder_
  - _Requirements: 2.1, 2.4, 3.1, 3.3, 6.2_
- [x] 1.2 clientが、共有部のoptimizerの状態を保持者で持ち、作り直すモデルのoptimizerの設定を束で受け取る
  - ownerの記録の共有部のoptimizerの状態を保持者にし、共有部と構造の参照を、現在の学習帰属のモデルから読む。束へ、作り直すモデルのoptimizerの設定を足す（既存の設定と同じ種類であることを確かめる）。束を作る箇所と、共有部のoptimizerの状態を読む箇所の、既存のtestと共用scriptを直す。
  - 完了: clientと束の既存のtest（実旧clientとの対照を含む）、事前学習・登録と集約のtest、共用script、依存境界のsuiteが、変更の後も成功する。束の新しいfieldのtestが成功する。
  - _Boundary: FedsdaRunClient、FedsdaRunClientOwners、FedsdaRunClientSettings_
  - _Requirements: 3.3, 3.4_

- [x] 2. 受取りと配布
- [x] 2.1 clientの受取りを、実旧のclientの受取りとの対照つきで実装する
  - 登録と集約の対照のoracleの状態から、実旧のclientの`apply_server_mapping`と新の受取りへ、同じID対応とグローバルモデルを与えて照合する対照testを先に書く。
  - 入力の検査、統計の選択、分類器とoptimizerの状態の生成、適応記録、統計・評価標本・学習データ・計数の付け替え、保有モデルの置換えとつなぎ直し、現在の学習帰属の付け替え、clientの操作を実装する。依存の許可集合を登録する。
  - 完了: ID対応の受取りの対照が、現在の学習帰属が付け替わる場合・複数のモデルが1つへ集まる場合・評価標本が上限を超える場合で、全状態と乱数の一致を示す。拒否のtest、受取りが変えない状態のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1, 1.2_
  - _Boundary: apply_global_model_distribution、GlobalModelDistributionApplication、FedsdaRunClient.apply_global_model_distribution_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.2, 5.1, 5.2, 5.3, 5.5, 6.2_
- [x] 2.2 サーバの配布を実装し、登録→集約→配布のラウンドを実旧と照合する
  - 下りの通信量の記録と、全clientへの受渡しを実装する。ラウンドの対照（標本処理→保留中の学習→登録→集約→配布→送信待ちの進行）を追加する。依存の許可集合を登録する。
  - 完了: ラウンドの対照が、2値・多クラス、複数の標本列、学習率の2つの設定が違う条件で、ラウンドごとに、配布の後のサーバとclientの全状態と乱数の一致を示し、要求4.3の経路を通ったことを確かめるtestが成功する。配布の拒否と途中の失敗のtest、依存境界のsuiteが成功する。
  - _Depends: 2.1_
  - _Boundary: distribute_global_models_to_clients_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.1, 4.3, 5.1, 5.4, 5.5, 6.2_

- [ ] 3. 新実装だけの確認
- [ ] 3.1 共用のfresh process scriptで、ラウンドごとに登録・集約・配布を行う
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、ラウンドごとに登録・集約・配布を行い、配布の後に、全clientが全グローバルモデルを同じ値で保有し、1つの共有部につながっていることと、下りの通信量が足されたことを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
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
