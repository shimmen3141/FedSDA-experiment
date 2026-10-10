# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. 読込みと生成
- [x] 1.1 MNISTの学習用データの読込み
  - 実旧の読込みとの一致、ファイルがないとき・形式が不正なときの拒否、使い回し、置き場所のtestを先に書く。
  - 完了: 読込みのtestと、依存境界のsuiteが成功する。
  - _Boundary: mnist_training_data_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 1.2 MNISTの生成器、datasetの定義、生成器を作る関数
  - 実旧の生成との対照、ラベルの交換、概念IDの拒否、定義のtestを先に書く。
  - 完了: 生成のtest、定義のtest、合成データの生成のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1_
  - _Boundary: MnistSampleGenerator、dataset_definitions、observed_sample_generation、ExperimentRunConditions_
  - _Requirements: 1.1, 1.2, 1.3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 2. 全体run
- [x] 2.1 mnist2・mnist4の小さい条件で、全体runを実旧と照合する
  - 全体runの対照へ、mnist2・mnist4の条件を先に足す（クラス数は、datasetの定義から）。
  - 完了: 全体runの対照（合成データの条件と、足した条件）が成功する。
  - _Depends: 1.2_
  - _Requirements: 4.2, 4.3, 4.5_

- [x] 2.2 mnist2で、goldenの33指標と31の離散列を照合する
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

### 実装（task 1.1〜2.2）

- 手順: 読込み・生成・定義のtest（task 1.1・1.2）を先に書き、失敗（収集の失敗）を確かめてから、sourceを書いた。全体runの対照（2.1）とgoldenの照合（2.2）は、sourceの後に条件を足した（sourceは、task 1.2で全部そろうため。条件を足した時点では、照合が通るかどうかは分かっていなかった）。
- 主担当の決定（ユーザーが覆せる）: 新実装は、MNISTのファイルを取得しない。置いてあるファイルだけを読み、なければ、ファイル名と`FDE_MNIST_DATA_DIR`を書いた例外で拒否する（research.mdの「Decision」）。旧実装がある間は、旧の読込みを1回呼べば、取得できる。
- 設計どおり: 画素をuint8のまま持ち、256個のfloatの表から標本のtupleを作る。置き場所は、環境変数と既定（旧と同じ規則）。読んだデータは、解決したディレクトリごとに使い回す（書込み不可）。
- 依存境界: `data/mnist/`の2つのmoduleは、moduleごとの許可集合へ登録した（名前単位のimport。`urllib`ほかの取得の手段を、importできない）。
- 実行条件のdataset名の許す値は、datasetの定義と同じ6つになった。「実行条件は受け取るが、定義がない」datasetは、なくなった（実行設定とfactoryの、定義の有無の検査は、残してある）。

### 照合の結果（Windowsの基準環境）

- 読込み: 6万件の画素（旧と同じ変換をした値）とラベルが、実旧の`load_mnist`と一致。置き場所の規則も、実旧の`default_data_dir`と一致。
- 生成: mnist2・mnist4×全概念×2 seedで、400件ずつ、実旧の`generate_data`と、特徴（float32）・ラベル・生成後の乱数の状態が一致。
- 全体run（小さい条件）: mnist2・mnist4の4条件で、サーバ・全client・診断の記録・乱数の最終状態が、実旧と一致。条件は、新旧の全体runを48条件（2 dataset×6 seed×4条件。全部一致）進めて、登録・統合・複数モデルでの終了・別の保有モデルの再利用を通るものを選んだ。MNISTの小さい条件は、隠れ層の幅を(48,)にした（合成データの小さい条件の幅(5, 4)では、モデルが増えない）。mnist4は、4概念すべてを通る。
- goldenの条件（mnist2: client 2、100件、隠れ層(1568,)、学習率1e-3）: 33指標と31の離散列が、実旧の`run_random_drift_experiment`と一致し、Windows用のgolden（mnist2）とも一致。これで、goldenの3ケース（sine2・sea2・mnist2）の全部を、新実装で照合できる。この条件は、候補の採用と棄却、学習帰属の切替を通り、グローバルモデルは1つのまま終わる。
- 共用script: mnist2・mnist4の全体runを、2回ずつ実行して同じ結果になる。

### testの変更で申告すること

- 指標の導出のtestの、経路の検査を、datasetごとの期待の表（`GOLDEN_CONDITION_PATHS_BY_DATASET_NAME`）へまとめた（sine2とsea2の期待は、前と同じ）。clientの数3の定数と、事前学習のepoch数は、goldenの条件から取る形へ直した。「標本あたりの値×標本数＝合計」の検査は、浮動小数点の丸めで成り立たない条件があるので、「標本あたりの値＝合計÷標本数」へ直した。
- 共用scriptの精度の下限を、0.5から、偶然の正解率（1／クラス数）へ直した（MNISTは10クラス）。
- 全体runの対照の、モデルの出力を比べる入力を、特徴数に合わせて作る形へ直した（3特徴までは、前と同じ値）。

### 未検証・残る制約

- Linuxでの照合は、していない（Linux用のgoldenは、次のspec）。
- 旧の既定の規模（client 10×5000件）でのMNISTのrunは、実行していない（メモリは、標本あたりの実測からの計算: 約316MB）。観測標本の持ち方の見直しは、[IMPROVE-012](../../../docs/research/improvement-candidates/improve-012-hold-observed-sample-features-as-array.md)。
- datasetごとの学習率と隠れ層の幅の既定（mnistは1e-3、(1568,)）は、新実装の中に持っていない（呼出し側が、設定の束へ渡す）。
