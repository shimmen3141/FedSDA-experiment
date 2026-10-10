# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [ ] 1. 読込みと生成
- [ ] 1.1 MNISTの学習用データの読込み
  - 実旧の読込みとの一致、ファイルがないとき・形式が不正なときの拒否、使い回し、置き場所のtestを先に書く。
  - 完了: 読込みのtestと、依存境界のsuiteが成功する。
  - _Boundary: mnist_training_data_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [ ] 1.2 MNISTの生成器、datasetの定義、生成器を作る関数
  - 実旧の生成との対照、ラベルの交換、概念IDの拒否、定義のtestを先に書く。
  - 完了: 生成のtest、定義のtest、合成データの生成のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1_
  - _Boundary: MnistSampleGenerator、dataset_definitions、observed_sample_generation、ExperimentRunConditions_
  - _Requirements: 1.1, 1.2, 1.3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 2. 全体run
- [ ] 2.1 mnist2・mnist4の小さい条件で、全体runを実旧と照合する
  - 全体runの対照へ、mnist2・mnist4の条件を先に足す（クラス数は、datasetの定義から）。
  - 完了: 全体runの対照（合成データの条件と、足した条件）が成功する。
  - _Depends: 1.2_
  - _Requirements: 4.2, 4.3, 4.5_

- [ ] 2.2 mnist2で、goldenの33指標と31の離散列を照合する
  - 指標の導出のtestのfixtureへ、mnist2を足す（旧の定義の学習率と隠れ層の幅を、設定の束へ渡す）。共用scriptへ、mnist2・mnist4の全体runを足す。
  - 完了: 指標の導出のtest（sine2・sea2・mnist2。Windowsでは、goldenの照合を含む）、共用scriptが成功する。
  - _Depends: 2.1_
  - _Requirements: 4.1, 4.4_

- [ ] 3. 検証
- [ ] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.2_
  - _Requirements: 4.5_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
