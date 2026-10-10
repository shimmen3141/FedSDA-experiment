# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [ ] 1. datasetの定義と生成
- [ ] 1.1 datasetの定義と、観測標本・概念列の型の一般化
  - 定義のtest（実旧との一致、拒否）と、型のtestを先に書く。
  - 完了: 定義のtest、型のtest、依存境界のsuiteが成功する。
  - _Boundary: dataset_definitions、ObservedSample、ClientConceptTrace_
  - _Requirements: 1.1, 1.2, 1.3, 3.1, 3.2_

- [ ] 1.2 SEAとCIRCLE-2の生成器、生成器を作る関数、概念列の生成の一般化
  - 実旧の生成・概念列との対照のtestを先に書く。
  - 完了: 生成のtest、概念列のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1_
  - _Boundary: SeaSampleGenerator、CircleSampleGenerator、observed_sample_generation、random_concept_schedule_generation_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.3_

- [ ] 2. 全体run
- [ ] 2.1 実行の枠・factory・事前学習・clientが、datasetの定義に従う
  - 全体runの対照のhelperへdatasetを渡せるようにし、sea2・sea4・circle2の対照を先に書く。
  - 完了: 全体runの対照（sine2の9条件と、足した条件）、実行の枠・factory・事前学習・clientのtest、依存境界のsuiteが成功する。
  - _Depends: 1.2_
  - _Boundary: single_run_execution、fedsda_run_participant_factory、initial_model_pretraining、fedsda_run_client_
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.2, 5.3_

- [ ] 2.2 sea2で、goldenの33指標と31の離散列を照合する
  - 指標の導出のtestのfixtureを、datasetごとに回せる形にする。共用scriptへ、sine2以外の全体runを足す。
  - 完了: 指標の導出のtest（sine2とsea2。Windowsでは、goldenの照合を含む）、共用scriptが成功する。
  - _Depends: 2.1_
  - _Requirements: 5.1, 5.4, 5.5_

- [ ] 3. 検証
- [ ] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.2_
  - _Requirements: 4.4, 5.1, 5.5_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
