# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [ ] 1. サーバが持つ状態のowner
- [ ] 1.1 通信量の記録のownerを実装する
  - 上り・下りの、モデル転送数、メッセージ数、パラメータの値の数、バイト数を数える。依存の許可集合を登録する。
  - 完了: ownerの単独のtest（各操作の加算、値の数とバイト数、各不正入力の拒否と不変）と、依存境界のsuiteが成功する。
  - _Boundary: CommunicationVolumeRecordStore、CommunicationVolumeSnapshot_
  - _Requirements: 2.1, 2.2, 2.3, 7.2_
- [ ] 1.2 グローバルモデルのownerを実装する
  - 初期モデル、採番、モデルIDごとのパラメータと統計（写しで受け渡す）、登録の来歴を持つ。依存の許可集合を登録する。
  - 完了: ownerの単独のtest（初期状態、採番、写しの独立、設定した順、来歴の昇順と上書き、各不正入力の拒否と不変）と、依存境界のsuiteが成功する。
  - _Boundary: GlobalModelRepository、GlobalModelRegistrationRecord_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 7.2_

- [ ] 2. 登録と集約
- [ ] 2.1 新規モデルの登録を、実旧のサーバとの対照つきで実装する
  - 実旧の事前学習・サーバ・client 3つと、新のclient 3つ・ownerを作り、ラウンドごとに照合する対照testの土台を先に作る。
  - 入力の検査と、client順の採番・来歴の記録・正式IDの確認を実装する。
  - 完了: 対照testで、登録の後の、次の正式ID・来歴・各clientの全状態が実旧と一致する（集約を含む全項目の一致は2.2の完了で確かめる）。拒否のtest、登録の途中の失敗のtestが成功する。
  - _Depends: 1.2_
  - _Boundary: register_ready_client_models、RegisteredClientModel_
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.3, 6.4_
- [ ] 2.2 clientのモデルの集約を実装する
  - 対象のIDの決定、clientごとの参加モデル、共有部と概念固有部の重み付き平均、統計の平均、通信量、全部の計算の後の反映を実装する。依存の許可集合を登録する。
  - 完了: 対照testが、2値・多クラス、複数の標本列で、ラウンドごとに、グローバルモデルのIDの順・全パラメータ・損失統計、次の正式ID、来歴、通信量の全項目、集約の件数、各clientの全状態、乱数の状態の一致を示し、要求5.2の経路を通ったことを確かめるtestが成功する。拒否のtest、計算の途中の失敗で何も変わらないことのtest、依存境界のsuiteが成功する。
  - _Depends: 1.1, 2.1_
  - _Boundary: aggregate_client_models_into_global_models、ClientModelAggregation_
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2, 6.1, 6.2, 6.4, 7.2_

- [ ] 3. 新実装だけの確認
- [ ] 3.1 共用のfresh process scriptで、ラウンドごとに登録と集約を行う
  - clientの流れの、何もしないサーバの代役を、登録と集約を行う代役に替える（配布は行わない）。
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、複数のclientの標本処理の後で登録と集約をラウンドごとに行い、グローバルモデルが更新されたことと、上りの通信量が足されたことを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Depends: 2.2_
  - _Boundary: 共用のfresh process script_
  - _Requirements: 7.1_

- [ ] 4. 検証
- [ ] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.1_
  - _Requirements: 5.1, 7.1, 7.2_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
