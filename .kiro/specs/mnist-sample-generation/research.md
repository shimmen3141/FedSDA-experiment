# Research & Design Decisions

## Summary

- **Feature**: `mnist-sample-generation`
- **Discovery Scope**: Extension（datasetを足す）
- **Key Findings**:
  - 前のspec（synthetic-dataset-sample-generation）で、実行の枠・factory・事前学習・clientは、特徴数・クラス数・概念数を、datasetの定義から取る形になっている。MNISTは、定義と生成器を足せば、全体runが動く。
  - 旧の生成の乱数の消費は、標本1件につき、NumPyの`randint(0, 件数, size=1)`の1回だけ。
  - 旧のdatasetの定義の学習率（1e-3）は、通常の学習率と、新規モデルの学習率の両方に優先する。新では、設定の束の2つのoptimizerの設定へ、呼出し側が1e-3を渡す。

## Research Log

### 旧の読込みと生成

- **Sources Consulted**: federated_drift_experiment/data/mnist.py（`default_data_dir`・`_ensure_file`・`_read_images`・`_read_labels`・`load_mnist`・`apply_mnist_concept`・`sample_mnist`）、data/streams.py（`generate_data`）、data/specs.py、models.py（`spec.learning_rate`の使い方: 39・118・194・249・261・316・377行）、tests/test_proposed_regression.py（`CASES`の`mnist2`）。
- **Findings**:
  - 画像: gzipのIDX。先頭16バイト（識別の数値2051、件数、行、列）の後に、uint8の画素。`astype(float32) / 255.0`。ラベル: 先頭8バイト（識別の数値2049、件数）の後に、uint8。
  - `generate_data(concept_id)`は、`sample_mnist(concept_id, 1)`の、画像1件をfloat32のtensor、ラベルをfloat32のtensor（要素1つ）にする。
  - 取得: ファイルがなければ、`urllib`で取得する。
- **Implications**: 生成器は、借りた`RandomState`の`randint(0, 件数, size=1)`を1回引く。

### 784次元の観測標本の負担（Windowsの基準環境で実測、2026-10-10）

- 特徴ごとに`float(...)`を作る場合: 1標本あたり約25KB、約530マイクロ秒。旧の既定の規模（client 10×標本5000＝5万件）で、約1.26GB。
- 画素の値（0〜255）ごとのfloatを256個だけ作り、標本のtupleは、その参照を並べる場合: 1標本あたり約6.3KB、約110マイクロ秒。5万件で、約316MB。
- uint8の画素を、256個のfloat32の表で引いた値は、旧の`astype(float32) / 255.0`と、6万件×784の全部で一致した。
- clientが、標本1件を1行のtensorへ直す時間は、約57マイクロ秒。
- **Implications**: 画素はuint8のまま持ち（約47MB。float32なら約188MB）、256個のfloatの表から、標本のtupleを作る。

## Design Decisions

### Decision: ファイルは取得しない

- **Alternatives Considered**: 旧と同じく、なければ取得する——実験のコードが、実行中にネットワークへ出る。取得の失敗・途中のファイル・並列の実行での競合を、扱う必要がある。
- **Selected Approach**: 置いてあるファイルだけを読む。なければ、足りないファイルの名前と、`FDE_MNIST_DATA_DIR`を書いた例外で拒否する。
- **Rationale**: 読込みは、副作用のない操作になる。data層の依存は、標準ライブラリとNumPyだけで済む。
- **Follow-up**: 取得の手順は、旧実装を外すときに、文書か、別の道具として残す（旧実装がある間は、旧の読込みを1回呼べば、取得できる）。ユーザーが、新実装でも取得が要ると判断すれば、覆せる。

### Decision: 置き場所は、環境変数と既定（旧と同じ規則）で決め、実行条件には入れない

- **Alternatives Considered**: 実行条件・実行設定へ、ディレクトリを入れる——置き場所は、計算機ごとに違い、実験の条件ではない（同じ条件のrunが、別の設定に見える）。
- **Selected Approach**: data層の関数が、`FDE_MNIST_DATA_DIR`、なければリポジトリ直下の`data/mnist`を返す。生成器を作る関数が、これを使う。
- **Rationale**: 旧と同じ場所のファイルを、そのまま使える。

### Decision: 読んだデータは、processの中で使い回す

- **Selected Approach**: 解決したディレクトリごとに、読んだ結果を保持する（moduleの中の辞書）。配列は、書込み不可にする。
- **Rationale**: runごとに生成器を作るたびに、ファイルを読み直さない。書込み不可なので、使い回しても、runの間で影響しない。

### Decision: 学習率と隠れ層の幅は、定義に入れない

- **Selected Approach**: `DatasetDefinition`は、特徴数・概念数・クラス数だけのまま。goldenの照合のtestは、設定の束へ、旧の定義の値（(1568,)、1e-3）を渡す。
- **Follow-up**: 完全なrun設定を決めるspecで、datasetごとの既定の置き場所を決める（再開案内へ書く）。

### Synthesis

- **Build vs. Adopt**: SEAの生成器の形（借りた`RandomState`と概念数を持ち、標本1件を返す）を使う。
- **Simplification**: 取得、検証用データ（t10k）の読込みは、作らない。

## Risks & Mitigations

- 5万件で約316MBを、runごとに持つ — 並列14 workersで約4.4GB。観測標本の持ち方（配列）の見直しは、改善候補へ記録する。
- 置き場所にファイルがない環境（新しい計算機）で、MNISTのrunが止まる — 例外に、ファイル名と環境変数を書く。

## References

- [synthetic-dataset-sample-generation](../synthetic-dataset-sample-generation/)（datasetの定義、生成器の共通の型）、[fedsda-run-metric-derivation](../fedsda-run-metric-derivation/)（goldenの照合のtest）。
