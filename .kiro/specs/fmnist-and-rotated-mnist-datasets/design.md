# Design Document

## Overview
新しい4データを、今のMNISTの生成の経路に足す。画像の読み込みはMNISTの読み方を共通にして置き場所だけを分け、生成器はMNISTの生成器の「概念の割り当て」だけを替える。
既存の6データの経路と値は変えない。

## 根拠（公式FedDrift: 手元の`FedDrift/`、`fedml_api/data_preprocessing/MNIST/data_loader_cont.py`）
- 今のHEAD（`e4ac32fe`）の`MNIST_Data.generate_sample(num_sample, rotation_k)`: `rotation_k`が1・2・3のとき、ラベルの1↔2・3↔4・5↔6を入れ替える。
  同じ`if rotation_k != 0:`の中に、コメントとして`x2d = np.reshape(x, (28, 28))`、`rx2d = np.rot90(x2d, k=rotation_k)`、`x = rx2d.flatten()`が残る。
- commit `18e7fbe`（2021-10-26）で回転のMNISTを足し、`62ad4e2`で概念の数を一般化し、`37fd772`でラベルの入れ替えに替えた（各commitの記録）。主担当の確認（`git show <commit>:fedml_api/data_preprocessing/MNIST/data_loader_cont.py`）:
  - `18e7fbe`: `generate_sample`の181〜183行が`x2d = np.reshape(x, (28, 28))`、`rx2d = np.rot90(x2d, k=rotation_k)`、`x = rx2d.flatten()`。ラベル`y`は変えない。
    データの生成（74〜83行）は、変化点の前を`generate_sample(num_sample, 0)`、後を「Get samples for the second concept, which is a 180 deg rotation (k=2)」として`generate_sample(num_sample, 2)`。
  - `62ad4e2`: 変化点の行列の値`k = change_point[it][c]`を`generate_sample(sample_per_client_iter, k)`へ渡す（76〜77行）。回転は同じ（169行の`np.rot90(x2d, k=rotation_k)`）。
  実装担当は、上の行を自分でも確かめ、違いがあれば実装の前に止めて報告する。
- `np.reshape(x, (28, 28))`は行優先（C順）。`np.rot90(m, k)`は、最初の2軸の面で反時計回りにk×90°回す。`flatten()`は行優先。
- 今の`MnistTrainingData.pixel_values`はIDXの画像の行優先の784画素（`mnist_training_data.py`の読み方。IDX形式は画像を行の順に格納する）。
- 公式のMNISTの784画素の並び: 公式の`data/MNIST/README.md`は、元のデータを「training set images: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz」ほかのIDXファイルとし、
  「The input of the model is a flattened 784-dimensional (28 × 28) image」と書く（FedProxの論文の前処理のデータ。Google Driveの配布のzip）。IDXの画像を平らにしたものなので行優先とみる。
  平らにする処理のコード自体は公式リポジトリに無く、配布のzipも取得していないので、この点は文書の根拠の限界として書く。
- よって、同じ並びで`numpy.rot90(pixels.reshape(28, 28), k=r).reshape(784)`とすれば、公式と同じ回転になる。

## 生成
- `fmnist2`・`fmnist4`: 今の`MnistSampleGenerator`と同じ処理（画像の位置の選択に乱数を1回使い、概念cでラベル`2*c-1`↔`2*c`。c=1で1↔2、c=2で3↔4、c=3で5↔6）を、Fashion-MNISTの学習用データで行う。
- `rotmnist2`・`rotmnist4`: 同じ位置の選択の後、概念の90°の回数r（`rotmnist2`は概念0→0・概念1→2、`rotmnist4`は概念k→k）だけ画素を回す。ラベルは変えない。概念0は`mnist`と同じ標本になる。
- 実装の形（生成器の型を分けるか、概念の割り当ての方式を引数にするか）は実装担当が決めてよい。ただし、既存の`mnist2`・`mnist4`の値（乱数の消費を含む）を変えない。
- 回転は、28×28の配列を回して並べ直す画素の並べ替えなので、画素の値（0〜255、`_FEATURE_VALUE_BY_PIXEL_VALUE`の変換）は変わらない。並べ替えの添字の表を概念ごとに先に作り、標本ごとに使い回してよい。

## 読み込み
- MNISTのIDXの読み方（`mnist_training_data.py`）を、画像の元（MNIST・Fashion-MNIST）ごとの置き場所で使えるようにする。Fashion-MNISTは、環境変数`FDE_FASHION_MNIST_DATA_DIR`、なければリポジトリ直下の`data/fashion-mnist`。
- 読んだ結果のprocessの中での使い回し（`_LOADED_TRAINING_DATA_BY_DIRECTORY`）は、置き場所ごとに分ける（MNISTとFashion-MNISTが混ざらない）。
- 公式のファイルとMD5（`http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/`）: train-images `8d4fb7e6c68d591d4c3dfef9ec88bf0d`、train-labels `25c81989df183df01b3e8a0aad5dffbe`。
  読み込みでMD5を検査するかは、MNISTの今の扱いに合わせる（MNISTで検査していなければ、検査しない。文書に取得の手順とMD5を書く）。

## 設定
- `experiment_run_conditions.py`の許される値、`dataset_definitions.py`、`observed_sample_generation.py`の生成器の表に4データを足す。
- FedSDA（`fedsda_final_configuration_run_settings.py`のデータ別の表）、FedDrift（`feddrift_default_run_settings.py`のネットワーク・学習率とδの表）、FedAvg（`fedavg_run_settings.py`）で、
  4データの値を`mnist2`・`mnist4`と同じにする。FedAvgの`startswith("mnist")`は、データの名前の明示の集合（MNISTの画像を使う6データ: mnist2、mnist4、fmnist2、fmnist4、rotmnist2、rotmnist4）に直す。
  既存の値は変えない。
- FedDriftのδの既定: 4データとも0.10（公式の`default_deltas`の`MNIST`）。
- 掃引の重さの見積り（`experiment_sweep_execution.py` 300行付近）: `fmnist*`・`rotmnist*`を、同じ概念の数の`mnist*`と同じ重みにする。
- 事前学習（全手法で共通、概念0の標本）は今のまま。`fmnist`の事前学習はFashion-MNISTの概念0、`rotmnist`はMNISTの0°の標本になる。

## 開示する差（文書に書く）
- 標本の選び方: 公式FedDriftは、seed 100で並べ替えた順に使い、使い切ったら先頭へ戻る。こちらは無作為の位置（今のMNISTと同じ）。
- 概念の時系列とclientへの分け方: 公式は時刻の境界の変化点の行列、こちらはランダムな変化点（全手法で共通、今のデータと同じ）。
- ネットワーク: 公式FedDriftの回転のMNISTの版はCNNも足した（commit `18e7fbe`の記録）。FedCCFA・IFCAもCNN。こちらはMLP（今のMNISTと同じ）。精度は論文の値と直接比べない。
- Fashion-MNISTの元のデータの入れ替えのクラスの意味: 1 Trouser↔2 Pullover、3 Dress↔4 Coat、5 Sandal↔6 Shirt（FedCCFAと同じ組）。
- 回転の2概念は0°・180°（公式の`18e7fbe`の2概念の版。IFCAのRotated CIFARも同じ）。`62ad4e2`の一般化の版で2概念の行列（値0・1）を与えると0°・90°になるが、公式が2概念として明示したのは180°なので、そちらに従う。

## Validation Design
要求4.2の(i)〜(vii)。(ii)は、Fashion-MNISTのファイルを使わずMNISTだけで確かめられる。(vi)の全体runの規模は要求4.2の(vi)のとおり。
取得の手順の文書には、公式のURL（HTTP。配布元のREADMEのとおり）と、取得の後のMD5の照合を必須として書く。
版1〜14のgoldenが変わらないこと（`tests/refactoring/test_important_behavior_regression_goldens.py`）。

## 範囲の外
- 回転のFashion-MNIST、CIFAR、FMoW、CNN。
- 新しいデータでの本規模の実験（この仕様の後。FedSDAは基準版14の既定のまま、調整しない）。
