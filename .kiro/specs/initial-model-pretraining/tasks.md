# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [ ] 1. 事前学習
- [ ] 1.1 事前学習の条件の設定型を実装する
  - 標本数、epoch数、batchの件数を、既存の検査の仕組みで宣言する。依存の許可集合を登録する。
  - 完了: 設定のtest（各値の境界、型と範囲の不正）と、依存境界のsuiteが成功する。
  - _Boundary: InitialModelPretrainingSettings_
  - _Requirements: 3.1, 4.2_
- [ ] 1.2 事前学習の関数を、実旧の事前学習との対照つきで実装する
  - 実`_pretrain_initial_model`と新の関数を、同じseedで実行して照合する対照testを先に書く（2値・多クラス、標本数・epoch数・batchの件数の組、optimizerの種類）。
  - 引数の検査、分類器と2つのoptimizerの状態の生成、標本の生成、epochごとのshuffleとbatchごとの更新、統計の逐次更新、結果の記録を実装する。依存の許可集合を登録する。
  - 完了: 対照testが、全条件で、全パラメータ・2つのoptimizerの状態・損失統計・実行後の3つの乱数の状態の一致を示す。拒否のtest（各不正で、3つの乱数の状態が変わらない）、結果の形のtest、依存境界のsuiteが成功する。
  - _Boundary: pretrain_initial_model、PretrainedInitialModel_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 3.2, 3.3, 4.2_

- [ ] 2. clientの組立てへの接続
- [ ] 2.1 事前学習の結果からclientを組み立て、実旧のclientと照合する
  - 事前学習の結果を`assemble_fedsda_run_client`へ渡し、実旧の事前学習の結果から実`__init__`で作った実旧のclientと、生成直後と、続く標本列・ラウンド境界の処理の後に照合するtestを追加する（sourceの変更が要らないことを確かめる）。
  - 完了: 生成直後と、標本列の処理の後の全状態と乱数が、実旧と一致する。
  - _Depends: 1.2_
  - _Boundary: pretrain_initial_modelとassemble_fedsda_run_clientの接続_
  - _Requirements: 2.2_
- [ ] 2.2 共用のfresh process scriptの、clientの流れの初期モデルを、事前学習で作る
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、事前学習→clientの組立て→実行の枠での標本列の処理、を最後まで行って成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Boundary: 共用のfresh process script_
  - _Requirements: 4.1_

- [ ] 3. 検証
- [ ] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.2_
  - _Requirements: 2.1, 4.1, 4.2_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
