# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. datasetの定義と生成
- [x] 1.1 datasetの定義と、観測標本・概念列の型の一般化
  - 定義のtest（実旧との一致、拒否）と、型のtestを先に書く。
  - 完了: 定義のtest、型のtest、依存境界のsuiteが成功する。
  - _Boundary: dataset_definitions、ObservedSample、ClientConceptTrace_
  - _Requirements: 1.1, 1.2, 1.3, 3.1, 3.2_

- [x] 1.2 SEAとCIRCLE-2の生成器、生成器を作る関数、概念列の生成の一般化
  - 実旧の生成・概念列との対照のtestを先に書く。
  - 完了: 生成のtest、概念列のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1_
  - _Boundary: SeaSampleGenerator、CircleSampleGenerator、observed_sample_generation、random_concept_schedule_generation_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.3_

- [x] 2. 全体run
- [x] 2.1 実行の枠・factory・事前学習・clientが、datasetの定義に従う
  - 全体runの対照のhelperへdatasetを渡せるようにし、sea2・sea4・circle2の対照を先に書く。
  - 完了: 全体runの対照（sine2の9条件と、足した条件）、実行の枠・factory・事前学習・clientのtest、依存境界のsuiteが成功する。
  - _Depends: 1.2_
  - _Boundary: single_run_execution、fedsda_run_participant_factory、initial_model_pretraining、fedsda_run_client_
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.2, 5.3_

- [x] 2.2 sea2で、goldenの33指標と31の離散列を照合する
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

### 実装（task 1.1〜2.2）

- 手順からの逸脱: task 1.1・1.2・2.1のsourceの一般化（定義、SEA・CIRCLEの生成器、型の緩和、実行の枠・factory・事前学習・clientの追随）を、新しいtestより先に書いた。testは、その後で書き、実旧をoracleにして照合した（下の「testが見つけたこと」のとおり、後から書いたtestで、sourceの不足を1件見つけて直した）。
- 設計からの追加: `is_observed_sample_generator_of_dataset`（design.md・naming.mdへ追記）。factoryは、実行条件のdataset名と違うdatasetの生成器（sea2の条件へSINEの生成器、sea2の条件へsea4の生成器）を、乱数を進める前に拒否する。型の一般化だけでは、この組合せが通ってしまうことを、既存のtest（factoryの拒否）が検出した。
- 依存境界: `data/dataset_definitions.py`と`data/observed_sample_generation.py`は、moduleごとの許可集合へ登録した。`data/sea/`・`data/circle/`は、`data/sine/`と同じ、data層の規則（NumPyとdata層の中だけ）に従う。実行設定（`execution/*_settings.py`）から使える`data`のmoduleへ、`data.dataset_definitions`を足した。
- clientの組立ての「初期モデルの特徴数が2であること」の検査は、やめた（特徴数は、datasetが決める）。対応するtestの1ケース（`classifier_with_other_feature_count`）を削除した。分類器と標本の特徴数の不一致は、順伝播の入力の検査が拒否する。

### 照合の結果（Windowsの基準環境）

- 定義: 4 datasetの特徴数・概念数・クラス数が、実旧の`DATASET_SPECS`と一致。
- 生成: 4 dataset×全概念×2 seedで、1500件ずつ、実旧の`generate_data`と、特徴（float32）・ラベル・生成後の乱数の状態が一致。観測列（`build_data_streams`）も一致。
- 概念列: 4 dataset×2 seed×2件数×2確率で、実旧の`make_concept_schedules`と、概念列・乱数の状態が一致（sea4は、4概念すべてを通る条件を含む）。
- 全体run（小さい条件）: sea2・sea4・circle2の6条件で、サーバ・全client・診断の記録・乱数の最終状態が、実旧と一致。条件は、新旧の全体runを54条件（3 dataset×6 seed×3条件。全部一致）進めて、新しいモデルの登録・統合・複数モデルでの終了を通るものを選んだ。sea2の1条件は、sine2の条件が通らない「別の保有モデルの再利用」の判定を通る。
- goldenの条件（sea2、600件）: 33指標と31の離散列が、実旧の`run_random_drift_experiment`と一致し、Windows用のgolden（sea2）とも一致。この条件は、警報のたびに現行モデルを維持し、モデルが1つのまま終わる（旧の回帰testも、経路を求めるのはsine2だけ）。そのため、経路の検査（複数モデル、統合ほか）は、sine2のときだけ行う。
- 共用script: sea2・sea4・circle2の全体runを、2回ずつ実行して同じ結果になる（旧実装とtestのmoduleを読み込まない）。

### testが見つけたこと

- 全体runの対照のhelper（`assert_run_client_matches_legacy`）が、2特徴の入力を固定で使っていた。特徴数を、実旧の（差し替えた）datasetの定義から取る形へ直した（sine2では、同じ入力のまま）。
- 指標の導出のtestの、sine2の条件に固有の定数（標本数4500、ラウンド数30ほか）を、datasetの条件から計算する形へ直した。

### 未検証・残る制約

- mnist2・mnist4は、未移植（次のspec）。実行条件のdataset名としては受け取るが、実行設定の検査とfactoryが拒否する。
- Linuxでの照合は、していない（Linux用のgoldenは、後のspec）。
- CIRCLE-2の「円周ちょうど」の標本（ラベル0）は、浮動小数点で作れないので、単体では確かめていない（実旧と同じ式`> 0`で、乱数からの標本は実旧と一致）。
