# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAを評価する研究者（ユーザー）。FedSDAは今の6データ（SINE2、CIRCLE2、SEA2、SEA4、MNIST2、MNIST4）で調整してきたので、調整に使っていないデータでの性能を比べたい。
- 何を変えるか: Fashion-MNISTのラベルの入れ替えのドリフト（`fmnist2`・`fmnist4`）と、MNISTの回転のドリフト（`rotmnist2`・`rotmnist4`）を、全手法（FedSDA・FedDrift・FedAvg）の新しいデータとして足す。
- 守ること: この実験の中だけの解釈で作らない。既存研究の実装を根拠にし、そこからずれない（2026-10-27のユーザー指示「既存研究の実装からずれないように進めてください」）。

## Introduction
先行実装の調査は`results/dataset-surveys/20261027-fmnist-rotation/dataset-survey.md`（gitの管理外。ユーザー指示でコミットしない）。要点:
- ドリフトを入れた共通の完成済みのデータセットは無く、各研究が公開の元データから生成する。そのため根拠の実装を1つに決める。
- ラベルの入れ替え: FedDriftの公式コード（`FedDrift/fedml_api/data_preprocessing/MNIST/data_loader_cont.py`の`generate_sample`）と論文の付録の定義
  （概念0は元のラベル、概念1・2・3は1↔2・3↔4・5↔6）。FedCCFA（NeurIPS 2024）のFashion-MNISTの入れ替えのクラスの組も同じ。今の`mnist2`・`mnist4`の実装（`data/mnist/mnist_sample_generation.py`）と同じ定義。
- 回転: FedDriftの公式リポジトリのcommit `18e7fbe`（「added rotated MNIST dataset」）と`62ad4e2`の同じ関数。画像を28×28に並べて`np.rot90(x2d, k=rotation_k)`（反時計回りにk×90°）してから平らに戻す。
  ラベルは変えない。今のHEADにも、この3行がコメントとして残る。概念と角度の対応は要求2.1。4角度はIFCA（NeurIPS 2020）のRotated MNIST（0/90/180/270°）と同じ。
- 元データ: Fashion-MNIST（Zalando、MIT）の公式の配布のIDX形式のファイル。MNISTと同じ形式・同じファイル名。手元の`data/fashion-mnist/`に置き、公式のMD5と一致を確認した（gitの管理外）。

根拠の実装から変えるのは、画像の元（MNIST→Fashion-MNIST）と、概念の割り当て（ラベルの入れ替え→回転）だけにする。
全手法で共通の条件（概念の時系列、clientへの分け方、標本の選び方、ネットワーク）は、今の`mnist2`・`mnist4`と同じにし、開示する差として文書に書く。
既存の6データの振る舞いと値は変えない。

## Requirements

### Requirement 1: Fashion-MNISTのラベルの入れ替え（`fmnist2`・`fmnist4`）
1.1 The data shall `fmnist2`（概念2つ）と`fmnist4`（概念4つ）を足す。入力784、クラス10。
1.2 The generator shall Fashion-MNISTの学習用の画像（60000件）から、今の`mnist2`・`mnist4`と同じ規則（乱数の使い方を含む）で1件を選び、概念cが1以上なら、ラベル`2*c-1`と`2*c`を入れ替える（c=1で1↔2、c=2で3↔4、c=3で5↔6。今の`mnist_sample_generation.py`の`first_swapped_digit = 2 * concept_id - 1`と同じ）。
1.3 The data loader shall Fashion-MNISTのIDXファイル（`train-images-idx3-ubyte.gz`、`train-labels-idx1-ubyte.gz`）を、MNISTと同じ読み方で読む。置き場所は、環境変数`FDE_FASHION_MNIST_DATA_DIR`があればその値、なければリポジトリ直下の`data/fashion-mnist`。取得はしない（MNISTと同じ）。
1.4 The data loader shall 読んだファイルが置き場所に無い・形式が違うときは、MNISTと同じ形で拒否する。

### Requirement 2: MNISTの回転（`rotmnist2`・`rotmnist4`）
2.1 The data shall `rotmnist2`（概念2つ: 0°・180°）と`rotmnist4`（概念4つ: 0°・90°・180°・270°）を足す。入力784、クラス10。
  角度の根拠（公式FedDriftの回転の2つの版）: commit `18e7fbe`の2概念の版は「the second concept, which is a 180 deg rotation (k=2)」（2つ目の概念を`generate_sample(…, 2)`）。
  commit `62ad4e2`で概念の数を一般化し、変化点の行列の値kを`generate_sample(…, k)`へそのまま渡す（概念0〜3が0/90/180/270°）。
  2概念は、公式が明示した180°にする（IFCAのRotated CIFARの2クラスタも0°・180°）。4概念は`62ad4e2`の対応にする（IFCAのRotated MNISTの4角度と同じ）。
2.2 The generator shall MNISTの学習用の画像から、今の`mnist2`・`mnist4`と同じ規則で1件を選び、概念の角度の回数r（`rotmnist2`は概念0→0、概念1→2、`rotmnist4`は概念k→k）だけ、
  画素を28×28（行優先）に並べて反時計回りに90°回し（`numpy.rot90(x, k=r)`と同じ結果）、行優先で784へ戻す。ラベルは変えない。
2.3 The design shall FedDriftの公式コードの回転（commit `18e7fbe`・`62ad4e2`の該当の行）と、画素の並び（行優先）・回転の向き・概念と角度の対応が同じであることを、根拠の行とともに書く。

### Requirement 3: 全手法の設定
3.1 The settings shall 新しい4データで、FedSDA・FedDrift・FedAvgのネットワークと学習率を、今の`mnist2`・`mnist4`と同じにする（MLP、学習率0.001。FedSDAの最終構成の設定、FedDriftの既定、FedAvgの既定）。
   FedAvgの`dataset_name.startswith("mnist")`のような名前の前方一致で設定を選ぶ箇所は、新しい名前でも同じ設定になるよう、明示の対応に直す。
3.2 The FedDrift default settings shall 新しい4データのδの既定を0.10にする（公式コードの`default_deltas`の`MNIST`の値。新しいデータの公式の値は無いので、元が同じ画像のMNISTの値を使う。設計書と文書に開示する）。
3.3 The sweep shall 新しい4データを、掃引の`--datasets`で指定でき、実行の重さの見積り（`experiment_sweep_execution.py`のdatasetの重み）でMNISTと同じに扱う。

### Requirement 4: 既存の値の維持と検証
4.1 The goldens shall 版1〜14の全caseの値を変えない。`BEHAVIOR_BASELINE_VERSION`は上げない（既存のデータの振る舞いは変わらない）。
4.2 The tests shall 次を確かめる: (i) `fmnist`の入れ替えが概念ごとに1↔2・3↔4・5↔6だけを変える。(ii) `rotmnist`の画素が、同じ画像の`numpy.rot90(pixels.reshape(28, 28), k=r).reshape(784)`（rは要求2.2の回数）と一致し、ラベルが変わらない。
   (iii) 同じseedで、`fmnist`と`mnist`が同じ位置の画像を選ぶ（乱数の使い方が同じ）、`rotmnist`と`mnist`の概念0の標本が一致する。(iv) 置き場所の環境変数と既定の置き場所、ファイルが無いときの拒否。
   (v) 3手法の設定が、新しい4データでMNISTと同じネットワーク・学習率になり、FedDriftのδの既定が0.10になる。
   (vi) 全体runが新しい4データで3手法とも完了する（各データ1 run、10 client、各client 100標本、集約間隔50。Fashion-MNISTのファイルが無い環境ではskipし、理由を出す）。
   (vii) 掃引の`--datasets`で4データを指定でき（計画の作成）、実行の重さの見積りが同じ概念の数のMNISTと同じになる。

### Requirement 5: 文書
5.1 The docs shall `docs/experiments/dataset-characteristics.md`ほかのデータの説明に、新しい4データの定義・根拠の実装・開示する差を足す（今の状態の正本）。
   調査の記録（gitの管理外）は参照せず、根拠の実装（公式FedDriftのファイル・commit、IFCA・FedCCFAの論文）を直接書く。
5.2 The docs shall Fashion-MNISTのファイルの取得の手順（公式のURLとMD5）を書く（uraniumなど別の環境で置くため）。
5.3 The docs shall `docs/reference/options.md`を直接編集せず、必要なら`python -m tools.generate_option_docs`で再生成する。
