# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. サーバの操作
- [x] 1.1 サーバのownerの記録、組立て、3つの操作を実装する
  - サーバの操作のtest（軽量メッセージ、同期の関数の呼出しと結果の保持、終端で不変、組立てと操作の拒否）を先に書く。依存の許可集合を登録する。
  - 完了: サーバのtestと、依存境界のsuiteが成功する。
  - _Boundary: FedsdaRunServer、FedsdaRunServerOwners、assemble_fedsda_run_server_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.2, 6.2_

- [x] 2. 参加者の初期準備と全体run
- [x] 2.1 設定の束とfactoryを、実旧の全体runとの対照つきで実装する
  - 旧の設定の差し替えへ、全体runの条件を足す。旧の全体の流れを、旧の部品を旧の順に呼んで実行するhelperと、新の全体run（test専用の中継で、真の概念を渡す）との対照testを先に書く。
  - 設定の束、factoryを実装する。依存の許可集合を登録する。
  - 完了: 全体runの対照が、複数のseedと条件で、概念列・観測列・サーバの全状態・診断の記録・各clientの全状態・runの乱数の最終状態の一致を示し、要求4.2の経路を通ったことを確かめるtestが成功する。真の概念なしの全体run、繰返し、束とfactoryの拒否のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1_
  - _Boundary: FedsdaRunParticipantSettings、FedsdaRunParticipantFactory_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 5.1, 5.3, 6.2_

- [x] 3. 新実装だけの確認
- [x] 3.1 共用のfresh process scriptで、factoryと実行の枠による全体runを実行する
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、全体runを実行し、区間の進行の件数と、サーバとclientの状態の対応を確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
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
